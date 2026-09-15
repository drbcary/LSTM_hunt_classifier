
import os
import re
import glob
from datetime import timedelta
import warnings
import copy

warnings.filterwarnings('ignore')

try:
    import PySide6.QtWidgets, PySide6.QtGui
    if not hasattr(PySide6.QtGui, 'QApplication'):
        PySide6.QtGui.QApplication = PySide6.QtWidgets.QApplication
except ImportError:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
)

try:
    from scipy import stats as sp_stats
except ImportError:
    sp_stats = None

from LSTM_model_training import HuntingBiLSTM
from VidLabeler import create_hunting_video_clip



# ==========================================
# 1. Feature Engineering Helpers 
# (Must match training logic exactly)
# ==========================================

import dataset_utils

# The processing functions and configuration are now imported from dataset_utils.py

# ==========================================
# 4. Logic Refinement & Smoothing (UPDATED)
# ==========================================

def frames_to_time(frames, video_fps):
    seconds = frames / video_fps
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    else:
        return f"00:{minutes:02}:{seconds:02}"

_EMPIRICAL_TRANS_MAT = None

def viterbi_decode(probs, transition_matrix, transition_weight=1.0):
    """
    Decodes the most likely state sequence using Viterbi algorithm in log space.
    transition_weight > 1.0 makes off-diagonal transitions less likely.
    """
    num_frames, num_states = probs.shape
    
    if transition_weight != 1.0:
        trans_mat = transition_matrix.copy()
        for i in range(num_states):
            non_diag = [j for j in range(num_states) if j != i]
            trans_mat[i, non_diag] /= transition_weight
        row_sums = trans_mat.sum(axis=1, keepdims=True)
        trans_mat = trans_mat / row_sums
    else:
        trans_mat = transition_matrix
        
    log_probs = np.log(np.clip(probs, 1e-10, 1.0))
    log_trans = np.log(np.clip(trans_mat, 1e-10, 1.0))
    
    dp = np.zeros((num_frames, num_states))
    backpointers = np.zeros((num_frames, num_states), dtype=int)
    
    dp[0] = log_probs[0]
    
    for t in range(1, num_frames):
        for s in range(num_states):
            trans_probs = dp[t-1] + log_trans[:, s]
            best_prev = np.argmax(trans_probs)
            dp[t, s] = trans_probs[best_prev] + log_probs[t, s]
            backpointers[t, s] = best_prev
            
    path = np.zeros(num_frames, dtype=int)
    path[-1] = np.argmax(dp[-1])
    for t in range(num_frames-2, -1, -1):
        path[t] = backpointers[t+1, path[t+1]]
        
    return path

def extract_events_from_path(path):
    events = []
    is_active = (path > 0)
    
    bounded = np.concatenate(([0], is_active.astype(int), [0]))
    diffs = np.diff(bounded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    
    for s, e in zip(starts, ends):
        block = path[s:e]
        
        pursuit_idx = np.where(block == 2)[0]
        if len(pursuit_idx) == 0:
            continue # Ignored: No pursuit state reached
            
        pursuit = s + pursuit_idx[0]
        
        eat_idx = np.where(block == 3)[0]
        if len(eat_idx) > 0:
            capture = s + eat_idx[0]
        else:
            capture = e 
            
        events.append({
            'enter': s,
            'pursuit': pursuit,
            'capture': capture,
            'capture_end': e,
            'remarks': 'HMM Decoded'
        })
        
    return events

def refine_predictions(probs, preds, cricket_visible_feature=None, transition_weight=5.0):
    """
    Replaces old logic with a strict chronological HMM Viterbi Decoder.
    """
    global _EMPIRICAL_TRANS_MAT
    if _EMPIRICAL_TRANS_MAT is None:
        try:
            _EMPIRICAL_TRANS_MAT = dataset_utils.calculate_empirical_transition_matrix()
            print("Loaded empirical transition matrix from training data.")
        except Exception as e:
            print(f"Warning: Failed to calculate empirical transition matrix: {e}")
            _EMPIRICAL_TRANS_MAT = np.array([
                [0.999, 0.001, 0.0,   0.0  ],
                [0.05,  0.94,  0.01,  0.0  ],
                [0.05,  0.0,   0.94,  0.01 ],
                [0.05,  0.0,   0.0,   0.95 ]
            ])
            
    path = viterbi_decode(probs, _EMPIRICAL_TRANS_MAT, transition_weight=transition_weight)
    events = extract_events_from_path(path)
    return events

# ==========================================
# 5. Visualization Helpers
# ==========================================

def events_to_sequence(events, num_frames):
    seq = np.zeros(num_frames, dtype=int)
    for event in events:
        enter = int(event['enter'])
        pursuit = int(event['pursuit']) if event.get('pursuit', -1) != -1 else enter
        capture = int(event['capture'])
        capture_end = event.get('capture_end', capture + dataset_utils.EATING_BUFFER)
        
        enter = max(0, min(num_frames, enter))
        pursuit = max(enter, min(num_frames, pursuit))
        capture = max(pursuit, min(num_frames, capture))
        eating_end = max(capture, min(num_frames, int(capture_end)))
        
        if pursuit > enter: seq[enter:pursuit] = 1 
        if capture > pursuit: seq[pursuit:capture] = 2 
        if eating_end > capture: seq[capture:eating_end] = 3 
            
    return seq

def plot_inference_results(probs, preds, refined_events=None, label_path=None, save_path=None, title="Inference Results", label_format='lucas', dlc_path=None):
    num_frames = len(preds)
    frames = np.arange(num_frames)
    
    # Auto-detect if probs contains multi-window data
    if isinstance(probs, dict) and 'windows' in probs:
        final_probs = probs['final']
        window_probs = probs['windows']
        is_multi_window = True
    else:
        final_probs = probs
        is_multi_window = False
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    
    classes = ['Background', 'Cricket Present', 'Pursuit', 'Eating']
    colors = ['lightgray', 'orange', 'red', 'green']
    
    for i, cls_name in enumerate(classes):
        if is_multi_window:
            # Plot individual sliding windows (faint)
            for w_data in window_probs:
                w_start, w_end, w_p = w_data['start'], w_data['end'], w_data['probs']
                w_frames = np.arange(w_start, w_end)
                ax1.plot(w_frames, w_p[:, i], color=colors[i], linewidth=0.8, alpha=0.2)
            
            # Plot the final averaged consensus (bold)
            ax1.plot(frames, final_probs[:, i], label=cls_name, color=colors[i], linewidth=2.0, alpha=0.9)
        else:
            # Standard single array plot
            ax1.plot(frames, final_probs[:, i], label=cls_name, color=colors[i], linewidth=1.5, alpha=0.8)
    
    ax1.set_ylabel("Probability")
    ax1.set_title(f"{title} - Model Confidence")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    
    ax2.step(frames, preds, where='post', label='Raw Model Prediction', 
             color='blue', linewidth=1.5, alpha=0.6, linestyle='--')
    
    if refined_events:
        refined_seq = events_to_sequence(refined_events, num_frames)
        ax2.step(frames, refined_seq, where='post', label='Refined Logic', 
                 color='magenta', linewidth=1.5, alpha=0.9)

    if label_path and os.path.exists(label_path):
        try:
            if label_format == 'lucas':
                valid_hunts = dataset_utils.parse_lucas_labels(label_path)
            elif label_format == 'dan':
                if dlc_path is None:
                    print("Warning: dlc_path required to match session for Dan format. Ground truth not plotted.")
                    valid_hunts = []
                else:
                    dan_hunts = dataset_utils.load_dan_labels(os.path.dirname(label_path))
                    base = os.path.basename(dlc_path)
                    import re
                    # Match Animal_D(day)_HS(session)
                    match = re.search(r'([a-zA-Z]+\d+)_D(\d+)_HS(\d+)', base, re.IGNORECASE)
                    if match:
                        animal = match.group(1).upper()
                        session = int(match.group(3))
                        valid_hunts = dan_hunts.get((animal, session), [])
                    else:
                        print(f"Warning: Could not parse animal/session from {base}. Ground truth not plotted.")
                        valid_hunts = []
            else:
                valid_hunts = []

            true_seq = np.zeros(num_frames)
            
            for row in valid_hunts:
                enter = row['enter']
                capture = row['capture']
                pursuit_start = row['pursuit_start']
                cons_end = row['consumption_end']
                
                eating_end = cons_end if cons_end is not None else capture + dataset_utils.EATING_BUFFER
                
                enter = max(0, min(num_frames, enter))
                pursuit_start = max(0, min(num_frames, pursuit_start))
                capture = max(0, min(num_frames, capture))
                eating_end = max(0, min(num_frames, eating_end))
                
                if pursuit_start > enter: true_seq[enter:pursuit_start] = 1
                if capture > pursuit_start: true_seq[pursuit_start:capture] = 2
                if eating_end > capture: true_seq[capture:eating_end] = 3
            
            ax2.step(frames, true_seq, where='post', label='Manual Labels', 
                     color='black', linewidth=4, alpha=0.3, zorder=0)
                     
        except Exception as e:
            print(f"Error loading labels: {e}")

    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(classes)
    ax2.set_ylabel("Class")
    ax2.set_xlabel("Frame Index")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close() 
        print(f"Plot saved: {save_path}")
    else:
        plt.show()

# ==========================================
# 6. Excel Generation
# ==========================================

def generate_analyzed_excel(events, output_path, video_fps):
    data_rows = []
    
    for i, event in enumerate(events):
        enter = event['enter']
        pursuit = event['pursuit']
        capture = event['capture']
        capture_end = event.get('capture_end', capture + dataset_utils.EATING_BUFFER)
        remarks = event.get('remarks', '')
        
        latency_frames = pursuit - enter if pursuit != -1 else 0
        pursuit_dur_frames = capture - pursuit if pursuit != -1 else 0
        total_frames = capture - enter
        
        include_status = 'Include'
        if remarks:
            include_status = 'Review'
        
        row = {
            'Cricket_Number': i,
            'Enter_Frame': enter,
            'Enter_Time': frames_to_time(enter, video_fps),
            'Pursuit_Frame': pursuit,
            'Capture_Frame': capture,
            'Capture_Time': frames_to_time(capture, video_fps),
            'Consumption_End_Frame': capture_end,
            'Consumption_Time': frames_to_time(capture_end, video_fps),
            'Time_to_Capture_Frames': total_frames,
            'Time_to_Capture': frames_to_time(total_frames, video_fps),
            'Pursuit_Time_Frames': pursuit_dur_frames,
            'Pursuit_Time': frames_to_time(pursuit_dur_frames, video_fps),
            'Latency_to_Attack_Frames': latency_frames,
            'Latency_to_Attack': frames_to_time(latency_frames, video_fps),
            'Remarks': remarks,
            'Include?': include_status
        }
        data_rows.append(row)
        
    df = pd.DataFrame(data_rows)
    try:
        if output_path.endswith('.csv'):
            df.to_csv(output_path, index=False)
        else:
            df.to_excel(output_path, index=False)
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Error saving {output_path}: {e}")

# ==========================================
# 7. Main Batch Processor
# ==========================================

def batch_process_dlc_folder(folder_path, model_path, device, label_folder=None, label_format=None, window_size=6000, step_size=1000, output_subdir="excel_files", plot_subdir="plots", prob_subdir=None):
    # 1. Load Model
    print(f"Loading Model from {model_path}...")
    model = HuntingBiLSTM(input_dim=22, hidden_dim=256, num_layers=3, num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. Prepare Output Dirs
    if output_subdir:
        output_dir = os.path.join(folder_path, output_subdir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = folder_path
        
    if plot_subdir:
        plot_dir = os.path.join(folder_path, plot_subdir)
        os.makedirs(plot_dir, exist_ok=True)
    else:
        plot_dir = folder_path
        
    if prob_subdir:
        prob_dir = os.path.join(folder_path, prob_subdir)
        os.makedirs(prob_dir, exist_ok=True)
    else:
        prob_dir = folder_path
    
    # 3. Find Files
    dlc_files = [f for f in glob.glob(os.path.join(folder_path, "*.csv")) if not f.endswith('_probs.csv')]
    print(f"Found {len(dlc_files)} DLC files.")
    
    for dlc_file in dlc_files:
        filename = os.path.basename(dlc_file)
        
        # Universal cleanup: strip DLC output suffix and file extension
        clean_base = re.sub(r'DLC_resnet50.*\.csv$', '', filename, flags=re.IGNORECASE)
        if clean_base == filename:
            clean_base = filename[:-4] # Fallback if regex doesn't match
            
        output_filename = f"{clean_base}_analyzed.xlsx"
        plot_title = clean_base
        
        # Determine label format
        curr_label_format = label_format
        if curr_label_format is None:
            if clean_base.startswith('LW'):
                curr_label_format = 'lucas'
            else:
                curr_label_format = 'dan'
            
        output_path = os.path.join(output_dir, output_filename)
        print(f"Processing {filename} -> {output_filename}...")
        
        try:
            # B. Run Model via sliding windows (using matching settings from single inference)
            probs, preds = run_inference_on_video(dlc_file, model, device, 
                                                  window_size=window_size, step_size=step_size, 
                                                  return_all_windows=True)
            
            if probs is None: continue
            
            final_probs = probs['final'] if isinstance(probs, dict) else probs
            
            # Save Probabilities
            prob_path = os.path.join(prob_dir, f"{clean_base}_probs.csv")
            prob_df = pd.DataFrame(final_probs, columns=['Background', 'Stalking', 'Pursuit', 'Eating'])
            prob_df.to_csv(prob_path, index=False)
                
            # C. Clean Predictions with HMM
            events = refine_predictions(final_probs, preds)
            
            # D. Plotting
            plot_filename = output_filename.replace('.xlsx', '.png')
            plot_path = os.path.join(plot_dir, plot_filename)
            
            label_path = None
            if label_folder:
                 candidate = os.path.join(label_folder, output_filename)
                 if os.path.exists(candidate):
                     label_path = candidate
                 else:
                     potential_labels = glob.glob(os.path.join(label_folder, "*.xlsx"))
                     
                     if curr_label_format == 'lucas':
                         # Match animal and hunt number
                         match = re.search(r'(LW\d+)[-_]hunt[-_]?(\d+)', clean_base, re.IGNORECASE)
                         if match:
                             animal = match.group(1).lower()
                             hunt_num = match.group(2)
                             for pl in potential_labels:
                                 pl_lower = os.path.basename(pl).lower().replace('_', '-')
                                 if animal in pl_lower and f"hunt-{hunt_num}" in pl_lower:
                                     label_path = pl
                                     break
                                     
                     elif curr_label_format == 'dan':
                         # Match just the animal. plot_inference_results will filter by session internally.
                         match = re.search(r'([a-zA-Z]+\d+)_D\d+_HS\d+', clean_base, re.IGNORECASE)
                         if match:
                             animal = match.group(1).lower()
                             for pl in potential_labels:
                                 if animal in os.path.basename(pl).lower():
                                     label_path = pl
                                     break
            
            plot_inference_results(probs, preds, refined_events=events, 
                                   label_path=label_path, save_path=plot_path, 
                                   title=plot_title, label_format=curr_label_format,
                                   dlc_path=dlc_file)
            
            # E. Generate Excel
            _, _, video_fps = dataset_utils.get_video_params(dlc_file)
            if len(events) > 0:
                generate_analyzed_excel(events, output_path, video_fps)
            else:
                print(f"  -> No hunts detected in {filename}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()

def run_inference_on_video(dlc_path, model, device, use_sliding_window=True, window_size=2000, step_size=500, return_all_windows=False, use_max_prob=False):
    """
    Runs the trained model on a full video DLC file using an overlapping sliding window.
    Returns:
        probs: (N, 4) numpy array of class probabilities
        preds: (N,) numpy array of predicted class indices
    """
    print(f"Processing {dlc_path}...")
    
    # 1. Process Data
    video_width, _, video_fps = dataset_utils.get_video_params(dlc_path)
    processed_data = dataset_utils.load_and_process_dlc(dlc_path, video_fps, filter_poops=True, poop_dist_thresh=6, poop_thresh=0.8)
    if processed_data is None:
        return None, None
        
    rat_head, rat_mid, rat_butt, rat_tail, cricket_pos = processed_data
    features = dataset_utils.generate_features(rat_head, rat_mid, rat_butt, rat_tail, cricket_pos, video_width, video_fps)
    num_frames = len(features)
    
    # 2. Inference
    model.eval()
    full_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    
    if use_sliding_window:
        total_probs = np.zeros((num_frames, 4), dtype=np.float32)
        counts = np.zeros((num_frames, 1), dtype=np.float32)
        starts = list(range(0, num_frames, step_size))
        
        all_window_probs = [] if return_all_windows else None
        
        with torch.no_grad():
            for start in starts:
                end = min(start + window_size, num_frames)
                
                if (end - start) < 10:
                    continue
                    
                chunk = full_tensor[start:end].unsqueeze(0)
                lengths = torch.tensor([end - start]).cpu()
                
                logits = model(chunk, lengths)
                probs = torch.softmax(logits, dim=2)[0].cpu().numpy()
                
                if return_all_windows:
                    all_window_probs.append({
                        'start': start,
                        'end': end,
                        'probs': probs.copy()
                    })
                
                # Apply a Gaussian taper to down-weight the edges of the window
                # where the BiLSTM lacks context, and prioritize the center.
                chunk_len = end - start
                mu = (chunk_len - 1) / 2.0
                sigma = chunk_len / 3.0
                x = np.arange(chunk_len)
                gaussian_window = np.exp(-0.5 * ((x - mu) / sigma) ** 2).reshape(-1, 1)
                
                tapered_probs = probs * gaussian_window
                
                if use_max_prob:
                    current_max = total_probs[start:end]
                    total_probs[start:end] = np.maximum(current_max, tapered_probs)
                else:
                    total_probs[start:end] += tapered_probs
                    counts[start:end] += gaussian_window
                
        if use_max_prob:
            final_probs = total_probs
            # Note: raw max probabilities across windows might not sum to exactly 1.0 per frame,
            # but preserving the raw max is usually desired for this analysis.
        else:
            counts[counts == 0] = 1.0
            final_probs = total_probs / counts
        
        
        preds = np.argmax(final_probs, axis=1)
        
        if return_all_windows:
            return {'final': final_probs, 'windows': all_window_probs}, preds
            
        return final_probs, preds
    else:
        # Standard full-sequence inference
        with torch.no_grad():
            chunk = full_tensor.unsqueeze(0)
            lengths = torch.tensor([num_frames]).cpu()
            logits = model(chunk, lengths)
            final_probs = torch.softmax(logits, dim=2)[0].cpu().numpy()
            preds = np.argmax(final_probs, axis=1)
            
        if return_all_windows:
            all_window_probs = [{'start': 0, 'end': num_frames, 'probs': final_probs.copy()}]
            return {'final': final_probs, 'windows': all_window_probs}, preds
            
        return final_probs, preds



# ==========================================
# 8. Model Accuracy Evaluation & Metrics
# ==========================================

EVAL_CLASS_NAMES = ['Background', 'Cricket Present', 'Pursuit', 'Eating']
EVAL_CLASS_COLORS = ['#7f8c8d', '#f39c12', '#e74c3c', '#27ae60']

def calc_pearson_correlation(x, y):
    """Calculates Pearson r, R^2, and approximate p-value safely."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 2:
        return 0.0, 0.0, 1.0
    x = x[valid]
    y = y[valid]
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 0.0, 1.0
    if sp_stats is not None:
        try:
            r, p = sp_stats.pearsonr(x, y)
            return float(r), float(r**2), float(p)
        except Exception:
            pass
    r = float(np.corrcoef(x, y)[0, 1])
    n = len(x)
    if abs(r) >= 1.0:
        p = 0.0
    else:
        t_stat = r * np.sqrt((n - 2) / max(1e-8, 1 - r**2))
        p = float(2.0 * (1.0 - 0.5 * (1.0 + np.tanh(abs(t_stat) / 1.5))))
    return r, float(r**2), p

def match_hunt_events(true_hunts, pred_events, fps=30.0, video_name=""):
    """
    Matches manual ground truth hunts with predicted hunt events using temporal interval overlap.
    Returns:
        matched_records: list of dicts with timing details and errors for true positives
        unmatched_true: list of missed manual hunts (false negatives)
        unmatched_pred: list of spurious predicted hunts (false positives)
    """
    matched_records = []
    
    if len(true_hunts) == 0 and len(pred_events) == 0:
        return matched_records, [], []
    if len(true_hunts) == 0:
        unmatched_pred = [{**p, 'video': video_name} for p in pred_events]
        return matched_records, [], unmatched_pred
    if len(pred_events) == 0:
        unmatched_true = [{**t, 'video': video_name} for t in true_hunts]
        return matched_records, unmatched_true, []
        
    candidates = []
    for t_idx, t_hunt in enumerate(true_hunts):
        t_enter = t_hunt['enter']
        t_capture = t_hunt['capture']
        for p_idx, p_event in enumerate(pred_events):
            p_enter = p_event['enter']
            p_capture = p_event['capture']
            
            overlap = max(0, min(t_capture, p_capture) - max(t_enter, p_enter))
            union = max(t_capture, p_capture) - min(t_enter, p_enter) + 1e-6
            iou = overlap / union
            
            capture_diff_frames = abs(p_capture - t_capture)
            is_valid_match = (iou > 0.05) or (overlap > 0) or (capture_diff_frames <= (2.0 * fps) and p_enter <= (t_capture + fps))
            
            if is_valid_match:
                score = iou + max(0, 1.0 - (capture_diff_frames / (5.0 * fps)))
                candidates.append((score, t_idx, p_idx, iou))
                
    candidates.sort(key=lambda x: x[0], reverse=True)
    matched_t = set()
    matched_p = set()
    
    for score, t_idx, p_idx, iou in candidates:
        if t_idx not in matched_t and p_idx not in matched_p:
            matched_t.add(t_idx)
            matched_p.add(p_idx)
            
            t_hunt = true_hunts[t_idx]
            p_event = pred_events[p_idx]
            
            t_enter = t_hunt['enter']
            t_pursuit = t_hunt['pursuit_start']
            t_capture = t_hunt['capture']
            
            p_enter = p_event['enter']
            p_pursuit = p_event['pursuit']
            p_capture = p_event['capture']
            
            rec = {
                'video': video_name,
                'iou': iou,
                'true_enter_frame': t_enter,
                'pred_enter_frame': p_enter,
                'enter_error_sec': (p_enter - t_enter) / fps,
                'true_pursuit_frame': t_pursuit,
                'pred_pursuit_frame': p_pursuit,
                'pursuit_error_sec': (p_pursuit - t_pursuit) / fps,
                'true_capture_frame': t_capture,
                'pred_capture_frame': p_capture,
                'capture_error_sec': (p_capture - t_capture) / fps,
                'true_duration_sec': max(0, t_capture - t_enter) / fps,
                'pred_duration_sec': max(0, p_capture - p_enter) / fps,
                'true_pursuit_sec': max(0, t_capture - t_pursuit) / fps,
                'pred_pursuit_sec': max(0, p_capture - p_pursuit) / fps,
                'true_latency_sec': max(0, t_pursuit - t_enter) / fps,
                'pred_latency_sec': max(0, p_pursuit - p_enter) / fps
            }
            matched_records.append(rec)
            
    unmatched_true = [{**true_hunts[i], 'video': video_name} for i in range(len(true_hunts)) if i not in matched_t]
    unmatched_pred = [{**pred_events[j], 'video': video_name} for j in range(len(pred_events)) if j not in matched_p]
    
    return matched_records, unmatched_true, unmatched_pred

def load_and_align_evaluation_data(folder_path, label_folder, label_format=None, output_subdir="excel_files", prob_subdir="probs"):
    """
    Scans folder_path, loads saved predictions (_probs.csv, _analyzed.xlsx),
    matches manual labels in label_folder, and returns aligned evaluation data per video.
    """
    output_dir = os.path.join(folder_path, output_subdir) if output_subdir else folder_path
    prob_dir = os.path.join(folder_path, prob_subdir) if prob_subdir else folder_path
    
    dlc_files = [f for f in glob.glob(os.path.join(folder_path, "*.csv")) if not f.endswith('_probs.csv')]
    if len(dlc_files) == 0:
        raise FileNotFoundError(f"No DLC tracking CSV files found in {folder_path}")
        
    dan_labels_cache = None
    aligned_data = []
    
    for dlc_file in dlc_files:
        filename = os.path.basename(dlc_file)
        clean_base = re.sub(r'DLC_resnet50.*\.csv$', '', filename, flags=re.IGNORECASE)
        if clean_base == filename:
            clean_base = filename[:-4]
            
        curr_label_format = label_format
        if curr_label_format is None:
            if clean_base.startswith('LW'):
                curr_label_format = 'lucas'
            else:
                curr_label_format = 'dan'
                
        # 1. Find probability file
        prob_candidates = [
            os.path.join(prob_dir, f"{clean_base}_probs.csv"),
            os.path.join(folder_path, f"{clean_base}_probs.csv"),
        ]
        prob_file = next((p for p in prob_candidates if os.path.exists(p)), None)
        if prob_file is None:
            print(f"Warning: No probability file found for {clean_base}. Skipping this video.")
            continue
            
        # 2. Find analyzed excel file
        excel_candidates = [
            os.path.join(output_dir, f"{clean_base}_analyzed.xlsx"),
            os.path.join(folder_path, f"{clean_base}_analyzed.xlsx")
        ]
        excel_file = next((p for p in excel_candidates if os.path.exists(p)), None)
        
        # Parse predicted events
        pred_events = []
        if excel_file and os.path.exists(excel_file):
            try:
                pred_df = pd.read_excel(excel_file)
                for _, row in pred_df.iterrows():
                    enter = int(row['Enter_Frame'])
                    capture = int(row['Capture_Frame'])
                    latency = int(row['Latency_to_Attack_Frames']) if 'Latency_to_Attack_Frames' in row and pd.notna(row['Latency_to_Attack_Frames']) else 0
                    pursuit = enter + latency
                    cons_end = row.get('Consumption_End_Frame')
                    capture_end = int(cons_end) if pd.notna(cons_end) else capture + dataset_utils.EATING_BUFFER
                    pred_events.append({
                        'enter': enter,
                        'pursuit': pursuit,
                        'capture': capture,
                        'capture_end': capture_end
                    })
            except Exception as e:
                print(f"Warning: Error reading {excel_file}: {e}")
                
        # 3. Match ground truth manual labels
        true_hunts = []
        label_matched = False
        
        if label_folder and os.path.exists(label_folder):
            if curr_label_format == 'lucas':
                candidate_direct = os.path.join(label_folder, f"{clean_base}_analyzed.xlsx")
                if os.path.exists(candidate_direct):
                    true_hunts = dataset_utils.parse_lucas_labels(candidate_direct)
                    label_matched = True
                else:
                    match = re.search(r'(LW\d+)[-_]hunt[-_]?(\d+)', clean_base, re.IGNORECASE)
                    if match:
                        animal = match.group(1).lower()
                        hunt_num = match.group(2)
                        potential_labels = glob.glob(os.path.join(label_folder, "*.xlsx"))
                        for pl in potential_labels:
                            pl_lower = os.path.basename(pl).lower().replace('_', '-')
                            if animal in pl_lower and f"hunt-{hunt_num}" in pl_lower:
                                true_hunts = dataset_utils.parse_lucas_labels(pl)
                                label_matched = True
                                break
            elif curr_label_format == 'dan':
                if dan_labels_cache is None:
                    dan_labels_cache = dataset_utils.load_dan_labels(label_folder)
                match = re.search(r'([a-zA-Z]+\d+)_D(\d+)_HS(\d+)', clean_base, re.IGNORECASE)
                if match:
                    animal = match.group(1).upper()
                    session = int(match.group(3))
                    key = (animal, session)
                    if key in dan_labels_cache:
                        true_hunts = dan_labels_cache[key]
                        label_matched = True
                if not label_matched:
                    match_animal = re.search(r'([a-zA-Z]+\d+)', clean_base, re.IGNORECASE)
                    if match_animal:
                        animal = match_animal.group(1).upper()
                        for (a, s), h_list in dan_labels_cache.items():
                            if a == animal:
                                true_hunts = h_list
                                label_matched = True
                                break
                                
        if not label_matched:
            print(f"Warning: No matching manual label found for {clean_base}. Skipping video.")
            continue
            
        # 4. Load probabilities
        prob_df = pd.read_csv(prob_file)
        prob_cols = ['Background', 'Stalking', 'Pursuit', 'Eating']
        if all(c in prob_df.columns for c in prob_cols):
            probs_array = prob_df[prob_cols].values
        else:
            probs_array = prob_df.values[:, :4]
        num_frames = len(probs_array)
        
        preds_raw = np.argmax(probs_array, axis=1)
        preds_refined = events_to_sequence(pred_events, num_frames)
        
        # Build dense ground truth sequence
        y_true = np.zeros(num_frames, dtype=int)
        for row in true_hunts:
            enter = int(row['enter'])
            pursuit_start = int(row['pursuit_start'])
            capture = int(row['capture'])
            cons_end = row.get('consumption_end')
            eating_end = int(cons_end) if cons_end is not None and pd.notna(cons_end) else capture + dataset_utils.EATING_BUFFER
            enter = max(0, min(num_frames, enter))
            pursuit_start = max(0, min(num_frames, pursuit_start))
            capture = max(0, min(num_frames, capture))
            eating_end = max(0, min(num_frames, eating_end))
            if pursuit_start > enter: y_true[enter:pursuit_start] = 1
            if capture > pursuit_start: y_true[pursuit_start:capture] = 2
            if eating_end > capture: y_true[capture:eating_end] = 3
            
        _, _, video_fps = dataset_utils.get_video_params(dlc_file)
        
        aligned_data.append({
            'video_name': clean_base,
            'dlc_file': dlc_file,
            'fps': video_fps,
            'num_frames': num_frames,
            'probs': probs_array,
            'y_raw': preds_raw,
            'y_refined': preds_refined,
            'y_true': y_true,
            'true_hunts': true_hunts,
            'pred_events': pred_events
        })
        
    print(f"Successfully aligned {len(aligned_data)} video sessions with manual labels.")
    return aligned_data

# ==========================================
# 9. Professional Plotting Functions
# ==========================================

def plot_confusion_matrices_and_f1(all_true, all_raw, all_refined, save_path=None):
    """
    Plots side-by-side normalized Confusion Matrices (Raw vs HMM Refined)
    and a grouped bar chart of Precision, Recall, and F1 across classes.
    """
    fig = plt.figure(figsize=(18, 5.5))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1.2], figure=fig, wspace=0.32)
    
    classes = EVAL_CLASS_NAMES
    
    # 1. Raw Model Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    cm_raw = confusion_matrix(all_true, all_raw, labels=[0, 1, 2, 3])
    cm_raw_norm = cm_raw.astype(float) / np.maximum(cm_raw.sum(axis=1, keepdims=True), 1e-10) * 100.0
    
    im1 = ax1.imshow(cm_raw_norm, cmap='Blues', vmin=0, vmax=100)
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Recall (% of True Frames)', fontsize=10)
    
    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    ax1.set_xticklabels(classes, rotation=25, ha='right', fontsize=9.5)
    ax1.set_yticklabels(classes, fontsize=9.5)
    ax1.set_xlabel('Predicted Class', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Ground Truth Class', fontsize=11, fontweight='bold')
    ax1.set_title('Raw Predictions Confusion Matrix', fontsize=12, fontweight='bold', pad=10)
    
    for i in range(4):
        for j in range(4):
            val = cm_raw_norm[i, j]
            count = cm_raw[i, j]
            color = "white" if val > 50 else "black"
            ax1.text(j, i, f"{val:.1f}%\n({count:,})", ha='center', va='center', color=color, fontsize=8.5, fontweight='bold')
            
    # 2. Refined HMM Confusion Matrix
    ax2 = fig.add_subplot(gs[0, 1])
    cm_ref = confusion_matrix(all_true, all_refined, labels=[0, 1, 2, 3])
    cm_ref_norm = cm_ref.astype(float) / np.maximum(cm_ref.sum(axis=1, keepdims=True), 1e-10) * 100.0
    
    im2 = ax2.imshow(cm_ref_norm, cmap='YlGn', vmin=0, vmax=100)
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Recall (% of True Frames)', fontsize=10)
    
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels(classes, rotation=25, ha='right', fontsize=9.5)
    ax2.set_yticklabels(classes, fontsize=9.5)
    ax2.set_xlabel('Predicted Class', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Ground Truth Class', fontsize=11, fontweight='bold')
    ax2.set_title('HMM Refined Confusion Matrix', fontsize=12, fontweight='bold', pad=10)
    
    for i in range(4):
        for j in range(4):
            val = cm_ref_norm[i, j]
            count = cm_ref[i, j]
            color = "white" if val > 50 else "black"
            ax2.text(j, i, f"{val:.1f}%\n({count:,})", ha='center', va='center', color=color, fontsize=8.5, fontweight='bold')
            
    # 3. Per-Class Precision / Recall / F1 Bar Chart
    ax3 = fig.add_subplot(gs[0, 2])
    p_raw, r_raw, f1_raw, _ = precision_recall_fscore_support(all_true, all_raw, labels=[0, 1, 2, 3], zero_division=0)
    p_ref, r_ref, f1_ref, _ = precision_recall_fscore_support(all_true, all_refined, labels=[0, 1, 2, 3], zero_division=0)
    
    macro_raw_f1 = f1_score(all_true, all_raw, average='macro', zero_division=0)
    macro_ref_f1 = f1_score(all_true, all_refined, average='macro', zero_division=0)
    acc_raw = accuracy_score(all_true, all_raw) * 100.0
    acc_ref = accuracy_score(all_true, all_refined) * 100.0
    
    x = np.arange(4)
    w = 0.38
    
    ax3.bar(x - w/2, f1_raw, width=w, label='Raw Model F1', color='#4a90e2', alpha=0.9, edgecolor='black', linewidth=0.6)
    ax3.bar(x + w/2, f1_ref, width=w, label='Refined HMM F1', color='#2ecc71', alpha=0.9, edgecolor='black', linewidth=0.6)
    
    for i in range(4):
        ax3.text(x[i] - w/2, f1_raw[i] + 0.02, f"{f1_raw[i]:.2f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        ax3.text(x[i] + w/2, f1_ref[i] + 0.02, f"{f1_ref[i]:.2f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        
    ax3.set_xticks(x)
    ax3.set_xticklabels(classes, rotation=20, ha='right', fontsize=9.5)
    ax3.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
    ax3.set_ylim(0, 1.15)
    ax3.set_title('Per-Class F1 Score Comparison', fontsize=12, fontweight='bold', pad=10)
    ax3.legend(loc='upper right', framealpha=0.9)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Summary Info Box
    summary_text = (
        f"Macro F1 (Raw): {macro_raw_f1:.3f} | (Refined): {macro_ref_f1:.3f}\n"
        f"Frame Accuracy: {acc_raw:.1f}% (Raw) | {acc_ref:.1f}% (Refined)"
    )
    ax3.text(0.03, 0.95, summary_text, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#bdc3c7', alpha=0.9))
             
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if save_path.endswith('.png'):
            plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)

def plot_multiclass_roc_and_pr_curves(all_true, all_probs, save_path=None):
    """
    Plots multi-class One-vs-Rest ROC curves and Precision-Recall curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    classes = EVAL_CLASS_NAMES
    colors = EVAL_CLASS_COLORS
    
    # 1. ROC Curves
    for c in range(4):
        binary_true = (all_true == c).astype(int)
        if np.sum(binary_true) == 0:
            continue
        fpr, tpr, _ = roc_curve(binary_true, all_probs[:, c])
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=colors[c], linewidth=2.2, label=f"{classes[c]} (AUC = {roc_auc:.3f})")
        
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.2, alpha=0.5, label='Random Chance (AUC = 0.500)')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
    ax1.set_title('One-vs-Rest ROC Curves', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='lower right', framealpha=0.9, fontsize=9.5)
    ax1.grid(True, alpha=0.25, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 2. Precision-Recall Curves
    for c in range(4):
        binary_true = (all_true == c).astype(int)
        if np.sum(binary_true) == 0:
            continue
        precision, recall, _ = precision_recall_curve(binary_true, all_probs[:, c])
        ap = average_precision_score(binary_true, all_probs[:, c])
        prevalence = np.mean(binary_true)
        ax2.plot(recall, precision, color=colors[c], linewidth=2.2, label=f"{classes[c]} (AP = {ap:.3f})")
        ax2.axhline(prevalence, color=colors[c], linestyle=':', alpha=0.4, linewidth=1.0)
        
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax2.set_title('Precision-Recall Curves (Multi-class)', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=9.5)
    ax2.grid(True, alpha=0.25, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if save_path.endswith('.png'):
            plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)

def plot_hunt_detection_and_timing_errors(event_summary, matched_df, save_path=None):
    """
    Plots hunt-event detection performance breakdown (TP, FP, FN)
    and transition timing error distributions (Enter, Pursuit, Capture offsets in seconds).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={'width_ratios': [1, 1.4]})
    
    # 1. Event Detection Breakdown Bar Chart
    categories = ['Detected Hunts\n(True Positives)', 'False Alarms\n(False Positives)', 'Missed Hunts\n(False Negatives)']
    counts = [event_summary['true_positives'], event_summary['false_positives'], event_summary['false_negatives']]
    bar_colors = ['#27ae60', '#e74c3c', '#e67e22']
    
    bars = ax1.bar(categories, counts, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.8, width=0.55)
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15, f"{count}",
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
                 
    ax1.set_ylabel('Number of Hunt Events', fontsize=11, fontweight='bold')
    ax1.set_title('Hunt Event Detection Performance', fontsize=12, fontweight='bold', pad=10)
    ax1.grid(axis='y', alpha=0.25, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Callout badge for Detection Rates
    badge_text = (
        f"Sensitivity / Recall: {event_summary['recall']*100:.1f}%\n"
        f"Precision: {event_summary['precision']*100:.1f}%\n"
        f"Hunt Event F1: {event_summary['f1_score']*100:.1f}%\n"
        f"Ground Truth Hunts: {event_summary['total_true']}\n"
        f"Predicted Hunts: {event_summary['total_pred']}"
    )
    ax1.text(0.04, 0.95, badge_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.6', facecolor='#f4f6f7', edgecolor='#bdc3c7', alpha=0.9))
             
    # 2. Timing Error Distributions
    if len(matched_df) > 0:
        timing_cols = ['enter_error_sec', 'pursuit_error_sec', 'capture_error_sec']
        labels = ['Hunt Enter', 'Pursuit Start', 'Capture']
        data_to_plot = [matched_df[c].dropna().values for c in timing_cols]
        
        box = ax2.boxplot(data_to_plot, patch_artist=True, tick_labels=labels,
                          boxprops=dict(facecolor='#d5dbdb', color='black', alpha=0.7),
                          medianprops=dict(color='#c0392b', linewidth=2.0),
                          whiskerprops=dict(color='black', linewidth=1.2),
                          capprops=dict(color='black', linewidth=1.2),
                          flierprops=dict(marker='o', markersize=4, alpha=0.3))
                          
        # Overlay jittered points
        for idx, d in enumerate(data_to_plot):
            jitter = np.random.normal(0, 0.05, size=len(d))
            ax2.scatter(np.ones(len(d))*(idx+1) + jitter, d, color='#2980b9', alpha=0.5, s=25, zorder=3)
            
            mae = np.mean(np.abs(d))
            med = np.median(d)
            ax2.text(idx + 1, ax2.get_ylim()[0] if ax2.get_ylim()[0] > -10 else -5,
                     f"MAE: {mae:.2f}s\nMed: {med:+.2f}s",
                     ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))
                     
        ax2.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
        ax2.set_ylabel('Timing Error in Seconds (Predicted - Manual)', fontsize=11, fontweight='bold')
        ax2.set_title('Transition Timing Accuracy for Matched Hunts', fontsize=12, fontweight='bold', pad=10)
        ax2.grid(axis='y', alpha=0.25, linestyle='--')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
    else:
        ax2.text(0.5, 0.5, "No matched hunts to evaluate timing errors.", ha='center', va='center')
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if save_path.endswith('.png'):
            plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)

def plot_duration_correlations(matched_df, save_path=None):
    """
    Plots correlation scatter plots with unity line y=x and linear regression fit
    for Total Hunt Duration, Pursuit Duration, and Attack Latency.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics_info = [
        ('true_duration_sec', 'pred_duration_sec', 'Total Hunt Duration (Enter -> Capture)', '#2980b9'),
        ('true_pursuit_sec', 'pred_pursuit_sec', 'Pursuit Duration (Pursuit -> Capture)', '#e74c3c'),
        ('true_latency_sec', 'pred_latency_sec', 'Attack Latency (Enter -> Pursuit)', '#f39c12'),
    ]
    
    for ax, (true_col, pred_col, title, color) in zip(axes, metrics_info):
        if len(matched_df) > 1 and true_col in matched_df and pred_col in matched_df:
            x = matched_df[true_col].values
            y = matched_df[pred_col].values
            
            # Identity line y=x
            max_val = max(np.max(x), np.max(y)) * 1.1 if len(x) > 0 and len(y) > 0 else 10
            max_val = max(max_val, 2.0)
            ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, alpha=0.5, label='Identity (y = x)')
            
            # Scatter points
            ax.scatter(x, y, color=color, alpha=0.7, edgecolor='black', s=45, linewidth=0.6, label='Hunts')
            
            # Regression line
            r, r2, p = calc_pearson_correlation(x, y)
            if len(x) >= 2 and np.std(x) > 0:
                slope, intercept = np.polyfit(x, y, 1)
                line_x = np.linspace(0, max_val, 100)
                ax.plot(line_x, slope * line_x + intercept, color='#2c3e50', linewidth=1.8,
                        label=f"Fit (y = {slope:.2f}x + {intercept:.2f})")
                        
            p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
            stats_text = f"R² = {r2:.3f}\nPearson r = {r:.3f}\n{p_str}\nN = {len(x)}"
            ax.text(0.05, 0.92, stats_text, transform=ax.transAxes, fontsize=9.5,
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='#fdfefe', edgecolor='#bdc3c7', alpha=0.9))
                    
            ax.set_xlim(0, max_val)
            ax.set_ylim(0, max_val)
            ax.set_xlabel('Ground Truth Manual (s)', fontsize=10.5, fontweight='bold')
            ax.set_ylabel('Model Predicted (s)', fontsize=10.5, fontweight='bold')
            ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.25, linestyle='--')
            ax.legend(loc='lower right', framealpha=0.9, fontsize=8.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        else:
            ax.text(0.5, 0.5, "Insufficient matched hunt data", ha='center', va='center')
            ax.set_title(title, fontsize=11, fontweight='bold')
            
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if save_path.endswith('.png'):
            plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)

def plot_per_video_performance(per_video_df, save_path=None):
    """
    Plots horizontal bar chart showing Macro F1 and Event Detection Rate per video.
    """
    if len(per_video_df) == 0:
        return
        
    fig, ax = plt.subplots(figsize=(13, max(4.5, len(per_video_df) * 0.45)))
    y_pos = np.arange(len(per_video_df))
    h = 0.38
    
    ax.barh(y_pos - h/2, per_video_df['refined_f1'], height=h, label='Frame Macro F1 (Refined)', color='#27ae60', alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.barh(y_pos + h/2, per_video_df['detection_recall'], height=h, label='Hunt Detection Rate (Sensitivity)', color='#2980b9', alpha=0.85, edgecolor='black', linewidth=0.5)
    
    # Average vertical lines
    mean_f1 = per_video_df['refined_f1'].mean()
    mean_det = per_video_df['detection_recall'].mean()
    ax.axvline(mean_f1, color='#27ae60', linestyle='--', linewidth=1.2, alpha=0.8, label=f"Avg F1: {mean_f1:.2f}")
    ax.axvline(mean_det, color='#2980b9', linestyle=':', linewidth=1.2, alpha=0.8, label=f"Avg Detection: {mean_det:.2f}")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(per_video_df['video_name'], fontsize=9.5)
    ax.set_xlabel('Performance Score (0.0 - 1.0)', fontsize=11, fontweight='bold')
    ax.set_xlim(0.0, 1.05)
    ax.set_title('Cross-Video Performance Breakdown', fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='lower right', framealpha=0.9, fontsize=9.5)
    ax.grid(axis='x', alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if save_path.endswith('.png'):
            plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)

def plot_master_evaluation_dashboard(all_true, all_raw, all_refined, all_probs, event_summary, matched_df, save_path=None):
    """
    Creates a master 6-panel executive summary dashboard combining the core evaluation figures.
    """
    fig = plt.figure(figsize=(20, 13))
    gs = GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.28)
    classes = EVAL_CLASS_NAMES
    colors = EVAL_CLASS_COLORS
    
    # --- PANEL A: Refined Confusion Matrix ---
    axA = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(all_true, all_refined, labels=[0, 1, 2, 3])
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1e-10) * 100.0
    imA = axA.imshow(cm_norm, cmap='YlGn', vmin=0, vmax=100)
    plt.colorbar(imA, ax=axA, fraction=0.046, pad=0.04).set_label('Recall (%)', fontsize=9.5)
    axA.set_xticks(range(4))
    axA.set_yticks(range(4))
    axA.set_xticklabels(classes, rotation=20, ha='right', fontsize=9)
    axA.set_yticklabels(classes, fontsize=9)
    axA.set_xlabel('Predicted Class', fontsize=10.5, fontweight='bold')
    axA.set_ylabel('Ground Truth Class', fontsize=10.5, fontweight='bold')
    axA.set_title('A. Refined Predictions Confusion Matrix', fontsize=12, fontweight='bold', pad=8)
    for i in range(4):
        for j in range(4):
            val = cm_norm[i, j]
            axA.text(j, i, f"{val:.1f}%\n({cm[i, j]:,})", ha='center', va='center',
                     color="white" if val > 50 else "black", fontsize=8, fontweight='bold')
                     
    # --- PANEL B: Per-Class F1 Score Comparison ---
    axB = fig.add_subplot(gs[0, 1])
    _, _, f1_raw, _ = precision_recall_fscore_support(all_true, all_raw, labels=[0, 1, 2, 3], zero_division=0)
    _, _, f1_ref, _ = precision_recall_fscore_support(all_true, all_refined, labels=[0, 1, 2, 3], zero_division=0)
    x = np.arange(4)
    w = 0.38
    axB.bar(x - w/2, f1_raw, width=w, label='Raw F1', color='#4a90e2', alpha=0.9, edgecolor='black', linewidth=0.5)
    axB.bar(x + w/2, f1_ref, width=w, label='Refined F1', color='#2ecc71', alpha=0.9, edgecolor='black', linewidth=0.5)
    for i in range(4):
        axB.text(x[i] - w/2, f1_raw[i] + 0.02, f"{f1_raw[i]:.2f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
        axB.text(x[i] + w/2, f1_ref[i] + 0.02, f"{f1_ref[i]:.2f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
    axB.set_xticks(x)
    axB.set_xticklabels(classes, rotation=20, ha='right', fontsize=9)
    axB.set_ylabel('F1 Score', fontsize=10.5, fontweight='bold')
    axB.set_ylim(0, 1.15)
    axB.set_title('B. Per-Class F1 (Raw vs Refined)', fontsize=12, fontweight='bold', pad=8)
    axB.legend(loc='upper right', framealpha=0.9, fontsize=9)
    axB.grid(axis='y', alpha=0.25, linestyle='--')
    axB.spines['top'].set_visible(False)
    axB.spines['right'].set_visible(False)
    
    # --- PANEL C: Precision-Recall Curves ---
    axC = fig.add_subplot(gs[0, 2])
    for c in range(4):
        bin_t = (all_true == c).astype(int)
        if np.sum(bin_t) > 0:
            p_curve, r_curve, _ = precision_recall_curve(bin_t, all_probs[:, c])
            ap = average_precision_score(bin_t, all_probs[:, c])
            axC.plot(r_curve, p_curve, color=colors[c], linewidth=2.0, label=f"{classes[c]} (AP={ap:.3f})")
    axC.set_xlim([0.0, 1.0])
    axC.set_ylim([0.0, 1.05])
    axC.set_xlabel('Recall', fontsize=10.5, fontweight='bold')
    axC.set_ylabel('Precision', fontsize=10.5, fontweight='bold')
    axC.set_title('C. Multiclass Precision-Recall Curves', fontsize=12, fontweight='bold', pad=8)
    axC.legend(loc='upper right', framealpha=0.9, fontsize=8.5)
    axC.grid(True, alpha=0.25, linestyle='--')
    axC.spines['top'].set_visible(False)
    axC.spines['right'].set_visible(False)
    
    # --- PANEL D: Hunt Event Detection Breakdown ---
    axD = fig.add_subplot(gs[1, 0])
    cats = ['Detected Hunts\n(TP)', 'False Alarms\n(FP)', 'Missed Hunts\n(FN)']
    cnts = [event_summary['true_positives'], event_summary['false_positives'], event_summary['false_negatives']]
    b_colors = ['#27ae60', '#e74c3c', '#e67e22']
    bars = axD.bar(cats, cnts, color=b_colors, alpha=0.85, edgecolor='black', linewidth=0.6, width=0.55)
    for b, ct in zip(bars, cnts):
        axD.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1, f"{ct}", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    axD.set_ylabel('Hunt Count', fontsize=10.5, fontweight='bold')
    axD.set_title('D. Hunt Event Detection Performance', fontsize=12, fontweight='bold', pad=8)
    axD.grid(axis='y', alpha=0.25, linestyle='--')
    axD.spines['top'].set_visible(False)
    axD.spines['right'].set_visible(False)
    badge = (f"Sensitivity / Recall: {event_summary['recall']*100:.1f}%\n"
             f"Precision: {event_summary['precision']*100:.1f}%\n"
             f"Hunt Event F1: {event_summary['f1_score']*100:.1f}%")
    axD.text(0.04, 0.95, badge, transform=axD.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f4f6f7', edgecolor='#bdc3c7', alpha=0.9))
             
    # --- PANEL E: Transition Timing Errors (seconds) ---
    axE = fig.add_subplot(gs[1, 1])
    if len(matched_df) > 0:
        timing_cols = ['enter_error_sec', 'pursuit_error_sec', 'capture_error_sec']
        t_labels = ['Hunt Enter', 'Pursuit Start', 'Capture']
        data_plot = [matched_df[col].dropna().values for col in timing_cols]
        axE.boxplot(data_plot, patch_artist=True, tick_labels=t_labels,
                    boxprops=dict(facecolor='#d5dbdb', color='black', alpha=0.7),
                    medianprops=dict(color='#c0392b', linewidth=2.0))
        for idx, d in enumerate(data_plot):
            jit = np.random.normal(0, 0.05, size=len(d))
            axE.scatter(np.ones(len(d))*(idx+1) + jit, d, color='#2980b9', alpha=0.45, s=20, zorder=3)
            mae = np.mean(np.abs(d))
            axE.text(idx + 1, axE.get_ylim()[0] if axE.get_ylim()[0] > -8 else -4,
                     f"MAE: {mae:.2f}s", ha='center', va='bottom', fontsize=8, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))
        axE.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
        axE.set_ylabel('Timing Error (s) [Pred - True]', fontsize=10.5, fontweight='bold')
        axE.set_title('E. Transition Timing Accuracy', fontsize=12, fontweight='bold', pad=8)
        axE.grid(axis='y', alpha=0.25, linestyle='--')
        axE.spines['top'].set_visible(False)
        axE.spines['right'].set_visible(False)
        
    # --- PANEL F: Total Hunt Duration Scatter Correlation ---
    axF = fig.add_subplot(gs[1, 2])
    if len(matched_df) > 1:
        x = matched_df['true_duration_sec'].values
        y = matched_df['pred_duration_sec'].values
        max_v = max(np.max(x), np.max(y)) * 1.1 if len(x) > 0 and len(y) > 0 else 10
        axF.plot([0, max_v], [0, max_v], 'k--', linewidth=1.5, alpha=0.5, label='Identity (y=x)')
        axF.scatter(x, y, color='#2980b9', alpha=0.7, edgecolor='black', s=45, linewidth=0.6, label='Hunts')
        r, r2, p = calc_pearson_correlation(x, y)
        if len(x) >= 2 and np.std(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            line_x = np.linspace(0, max_v, 100)
            axF.plot(line_x, slope * line_x + intercept, color='#e74c3c', linewidth=1.8, label='Linear Fit')
        p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        axF.text(0.05, 0.92, f"R² = {r2:.3f} | r = {r:.3f}\n{p_str} | N = {len(x)}",
                 transform=axF.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#fdfefe', edgecolor='#bdc3c7', alpha=0.9))
        axF.set_xlim(0, max_v)
        axF.set_ylim(0, max_v)
        axF.set_xlabel('Ground Truth Manual Duration (s)', fontsize=10.5, fontweight='bold')
        axF.set_ylabel('Model Predicted Duration (s)', fontsize=10.5, fontweight='bold')
        axF.set_title('F. Total Hunt Duration Agreement', fontsize=12, fontweight='bold', pad=8)
        axF.legend(loc='lower right', framealpha=0.9, fontsize=8.5)
        axF.grid(True, alpha=0.25, linestyle='--')
        axF.spines['top'].set_visible(False)
        axF.spines['right'].set_visible(False)
        
    fig.suptitle("Rat Hunt Model - Performance & Accuracy Evaluation Dashboard", fontsize=16, fontweight='bold', y=0.98)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if save_path.endswith('.png'):
            plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Master Dashboard saved: {save_path}")
    plt.close(fig)

def export_evaluation_metrics(excel_path, frame_metrics_df, event_summary, timing_stats_df, per_video_df, matched_df):
    """
    Exports all numerical evaluation metrics into an organized multi-sheet Excel workbook.
    """
    try:
        with pd.ExcelWriter(excel_path) as writer:
            # Sheet 1: Executive Summary
            summary_data = [
                {'Metric': 'Frame Classification Accuracy (Refined)', 'Value': f"{frame_metrics_df.loc['Refined_Overall', 'Accuracy']*100:.2f}%"},
                {'Metric': 'Frame Macro F1 Score (Refined)', 'Value': f"{frame_metrics_df.loc['Refined_Overall', 'F1_Score']:.4f}"},
                {'Metric': 'Frame Macro F1 Score (Raw)', 'Value': f"{frame_metrics_df.loc['Raw_Overall', 'F1_Score']:.4f}"},
                {'Metric': 'Hunt Event Detection Sensitivity / Recall', 'Value': f"{event_summary['recall']*100:.2f}%"},
                {'Metric': 'Hunt Event Detection Precision', 'Value': f"{event_summary['precision']*100:.2f}%"},
                {'Metric': 'Hunt Event Detection F1 Score', 'Value': f"{event_summary['f1_score']:.4f}"},
                {'Metric': 'Total Ground Truth Hunts', 'Value': event_summary['total_true']},
                {'Metric': 'Total Predicted Hunts', 'Value': event_summary['total_pred']},
                {'Metric': 'True Positives (Detected Hunts)', 'Value': event_summary['true_positives']},
                {'Metric': 'False Positives (False Alarms)', 'Value': event_summary['false_positives']},
                {'Metric': 'False Negatives (Missed Hunts)', 'Value': event_summary['false_negatives']},
            ]
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 2: Frame-Level Metrics
            frame_metrics_df.to_excel(writer, sheet_name='Frame_Metrics')
            
            # Sheet 3: Timing Accuracy
            timing_stats_df.to_excel(writer, sheet_name='Timing_Accuracy', index=False)
            
            # Sheet 4: Per-Video Breakdown
            per_video_df.to_excel(writer, sheet_name='Per_Video_Breakdown', index=False)
            
            # Sheet 5: Matched Hunts Raw Detail
            matched_df.to_excel(writer, sheet_name='Matched_Hunts_Detail', index=False)
            
        print(f"Metrics report exported: {excel_path}")
    except Exception as e:
        print(f"Error exporting metrics Excel: {e}")

# ==========================================
# 10. Main Pipeline: Batch Process & Evaluate
# ==========================================

def batch_process_and_evaluate(
    folder_path,
    model_path,
    label_folder,
    device=None,
    label_format=None,
    window_size=6000,
    step_size=1000,
    output_subdir="excel_files",
    plot_subdir="plots",
    prob_subdir="probs",
    eval_subdir="evaluation_results",
    rerun_inference=True
):
    """
    Runs batch inference on a DLC video folder (if rerun_inference=True or files missing),
    loads the saved predicted labels and probabilities, aligns them with manual ground-truth labels,
    computes rigorous frame-level and hunt-level accuracy metrics, and generates publication-grade plots.
    
    Args:
        folder_path: Path to directory containing DLC CSV files.
        model_path: Path to trained PyTorch BiLSTM model weights (.pth).
        label_folder: Directory containing manual ground-truth label spreadsheets.
        device: torch.device (defaults to CUDA if available, else CPU).
        label_format: 'lucas', 'dan', or None (auto-detect based on video filenames).
        window_size: Sliding window size in frames for BiLSTM inference.
        step_size: Step size in frames for sliding window inference.
        output_subdir: Subdirectory for predicted event Excel files.
        plot_subdir: Subdirectory for single-video timeline plots.
        prob_subdir: Subdirectory for probability CSV files.
        eval_subdir: Subdirectory to save evaluation plots and metrics report.
        rerun_inference: If True, executes batch_process_dlc_folder. If False, reuses existing saved outputs.
        
    Returns:
        results: Dictionary containing all computed metrics, DataFrames, and output file paths.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    eval_dir = os.path.join(folder_path, eval_subdir)
    os.makedirs(eval_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # Step 1: Execute Batch Inference (if requested)
    # -------------------------------------------------------------
    if rerun_inference:
        print("\n=======================================================")
        print("  STEP 1: Running Model Batch Inference on DLC Folder")
        print("=======================================================")
        batch_process_dlc_folder(
            folder_path=folder_path,
            model_path=model_path,
            device=device,
            label_folder=label_folder,
            label_format=label_format,
            window_size=window_size,
            step_size=step_size,
            output_subdir=output_subdir,
            plot_subdir=plot_subdir,
            prob_subdir=prob_subdir
        )
    else:
        print("\n[Info] Reusing existing inference outputs (rerun_inference=False).")
        
    # -------------------------------------------------------------
    # Step 2: Load and Align Predictions with Ground Truth
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("  STEP 2: Aligning Predictions with Ground Truth Labels")
    print("=======================================================")
    aligned_data = load_and_align_evaluation_data(
        folder_path=folder_path,
        label_folder=label_folder,
        label_format=label_format,
        output_subdir=output_subdir,
        prob_subdir=prob_subdir
    )
    
    if len(aligned_data) == 0:
        raise RuntimeError("No aligned data could be formed between predictions and manual labels.")
        
    # -------------------------------------------------------------
    # Step 3: Compute Comprehensive Performance Metrics
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("  STEP 3: Computing Frame & Event Accuracy Metrics")
    print("=======================================================")
    
    all_true = np.concatenate([d['y_true'] for d in aligned_data])
    all_raw = np.concatenate([d['y_raw'] for d in aligned_data])
    all_refined = np.concatenate([d['y_refined'] for d in aligned_data])
    all_probs = np.vstack([d['probs'] for d in aligned_data])
    
    # Match hunt events across videos
    all_matched = []
    all_unmatched_true = []
    all_unmatched_pred = []
    per_video_records = []
    
    for d in aligned_data:
        v_matched, v_unmatched_t, v_unmatched_p = match_hunt_events(
            d['true_hunts'], d['pred_events'], fps=d['fps'], video_name=d['video_name']
        )
        all_matched.extend(v_matched)
        all_unmatched_true.extend(v_unmatched_t)
        all_unmatched_pred.extend(v_unmatched_p)
        
        v_true = d['y_true']
        v_ref = d['y_refined']
        v_f1 = f1_score(v_true, v_ref, average='macro', zero_division=0)
        v_num_true = len(d['true_hunts'])
        v_num_pred = len(d['pred_events'])
        v_recall = len(v_matched) / v_num_true if v_num_true > 0 else (1.0 if v_num_pred == 0 else 0.0)
        
        per_video_records.append({
            'video_name': d['video_name'],
            'num_frames': d['num_frames'],
            'fps': d['fps'],
            'true_hunts': v_num_true,
            'pred_hunts': v_num_pred,
            'matched_hunts': len(v_matched),
            'detection_recall': v_recall,
            'refined_f1': v_f1
        })
        
    per_video_df = pd.DataFrame(per_video_records)
    matched_df = pd.DataFrame(all_matched)
    
    # Event Detection Summary
    total_true = len(all_matched) + len(all_unmatched_true)
    total_pred = len(all_matched) + len(all_unmatched_pred)
    tp = len(all_matched)
    fp = len(all_unmatched_pred)
    fn = len(all_unmatched_true)
    
    ev_recall = tp / total_true if total_true > 0 else 0.0
    ev_prec = tp / total_pred if total_pred > 0 else 0.0
    ev_f1 = (2 * ev_prec * ev_recall / (ev_prec + ev_recall)) if (ev_prec + ev_recall) > 0 else 0.0
    
    event_summary = {
        'total_true': total_true,
        'total_pred': total_pred,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'recall': ev_recall,
        'precision': ev_prec,
        'f1_score': ev_f1
    }
    
    # Frame Metrics DataFrame
    p_raw, r_raw, f1_raw, supp = precision_recall_fscore_support(all_true, all_raw, labels=[0, 1, 2, 3], zero_division=0)
    p_ref, r_ref, f1_ref, _ = precision_recall_fscore_support(all_true, all_refined, labels=[0, 1, 2, 3], zero_division=0)
    
    frame_metrics_data = {}
    for idx, cname in enumerate(EVAL_CLASS_NAMES):
        frame_metrics_data[f"Raw_{cname}"] = {'Precision': p_raw[idx], 'Recall': r_raw[idx], 'F1_Score': f1_raw[idx], 'Support_Frames': supp[idx], 'Accuracy': np.nan}
        frame_metrics_data[f"Refined_{cname}"] = {'Precision': p_ref[idx], 'Recall': r_ref[idx], 'F1_Score': f1_ref[idx], 'Support_Frames': supp[idx], 'Accuracy': np.nan}
        
    frame_metrics_data['Raw_Overall'] = {
        'Precision': np.mean(p_raw),
        'Recall': np.mean(r_raw),
        'F1_Score': f1_score(all_true, all_raw, average='macro', zero_division=0),
        'Support_Frames': len(all_true),
        'Accuracy': accuracy_score(all_true, all_raw)
    }
    frame_metrics_data['Refined_Overall'] = {
        'Precision': np.mean(p_ref),
        'Recall': np.mean(r_ref),
        'F1_Score': f1_score(all_true, all_refined, average='macro', zero_division=0),
        'Support_Frames': len(all_true),
        'Accuracy': accuracy_score(all_true, all_refined)
    }
    frame_metrics_df = pd.DataFrame.from_dict(frame_metrics_data, orient='index')
    
    # Timing Errors Summary
    timing_rows = []
    if len(matched_df) > 0:
        for col, label in [('enter_error_sec', 'Hunt Enter'), ('pursuit_error_sec', 'Pursuit Start'), ('capture_error_sec', 'Capture')]:
            vals = matched_df[col].dropna().values
            timing_rows.append({
                'Transition': label,
                'MAE_Seconds': float(np.mean(np.abs(vals))),
                'Median_Error_Seconds': float(np.median(vals)),
                'Mean_Error_Seconds': float(np.mean(vals)),
                'Std_Error_Seconds': float(np.std(vals)),
                'IQR_Seconds': float(np.percentile(vals, 75) - np.percentile(vals, 25)) if len(vals) > 0 else 0.0,
                'N_Hunts': len(vals)
            })
    timing_stats_df = pd.DataFrame(timing_rows)
    
    # -------------------------------------------------------------
    # Step 4: Generate Professional Evaluation Plots
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("  STEP 4: Generating Publication-Grade Visualizations")
    print("=======================================================")
    
    fig_cm_path = os.path.join(eval_dir, "01_confusion_matrices_and_f1.png")
    plot_confusion_matrices_and_f1(all_true, all_raw, all_refined, save_path=fig_cm_path)
    
    fig_roc_path = os.path.join(eval_dir, "02_multiclass_roc_and_pr_curves.png")
    plot_multiclass_roc_and_pr_curves(all_true, all_probs, save_path=fig_roc_path)
    
    fig_det_path = os.path.join(eval_dir, "03_hunt_detection_and_timing_errors.png")
    plot_hunt_detection_and_timing_errors(event_summary, matched_df, save_path=fig_det_path)
    
    fig_corr_path = os.path.join(eval_dir, "04_behavioral_metric_correlations.png")
    plot_duration_correlations(matched_df, save_path=fig_corr_path)
    
    fig_vid_path = os.path.join(eval_dir, "05_per_video_performance.png")
    plot_per_video_performance(per_video_df, save_path=fig_vid_path)
    
    fig_dash_path = os.path.join(eval_dir, "00_model_evaluation_dashboard.png")
    plot_master_evaluation_dashboard(all_true, all_raw, all_refined, all_probs, event_summary, matched_df, save_path=fig_dash_path)
    
    # -------------------------------------------------------------
    # Step 5: Export Metrics to Excel
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("  STEP 5: Exporting Numerical Reports to Excel")
    print("=======================================================")
    excel_report_path = os.path.join(eval_dir, "evaluation_metrics.xlsx")
    export_evaluation_metrics(
        excel_path=excel_report_path,
        frame_metrics_df=frame_metrics_df,
        event_summary=event_summary,
        timing_stats_df=timing_stats_df,
        per_video_df=per_video_df,
        matched_df=matched_df
    )
    
    # -------------------------------------------------------------
    # Console Executive Summary
    # -------------------------------------------------------------
    print("\n" + "="*65)
    print("             MODEL EVALUATION EXECUTIVE SUMMARY")
    print("="*65)
    print(f"Total Videos Evaluated       : {len(aligned_data)}")
    print(f"Total Frames Analyzed        : {len(all_true):,}")
    print(f"Refined Frame Accuracy       : {frame_metrics_df.loc['Refined_Overall', 'Accuracy']*100:.2f}%")
    print(f"Refined Frame Macro F1       : {frame_metrics_df.loc['Refined_Overall', 'F1_Score']:.4f} (Raw: {frame_metrics_df.loc['Raw_Overall', 'F1_Score']:.4f})")
    print(f"Hunt Event Detection Rate    : {event_summary['recall']*100:.1f}% ({tp}/{total_true} true hunts detected)")
    print(f"Hunt Detection Precision     : {event_summary['precision']*100:.1f}% ({fp} false alarms)")
    print(f"Hunt Event Detection F1      : {event_summary['f1_score']:.4f}")
    if len(timing_stats_df) > 0:
        for _, row in timing_stats_df.iterrows():
            print(f"Timing MAE ({row['Transition']:<14}) : {row['MAE_Seconds']:.2f} sec (Median: {row['Median_Error_Seconds']:+.2f}s)")
    print(f"\nAll plots and Excel report saved to: {eval_dir}")
    print("="*65 + "\n")
    
    return {
        'frame_metrics': frame_metrics_df,
        'event_summary': event_summary,
        'timing_stats': timing_stats_df,
        'per_video': per_video_df,
        'matched_hunts': matched_df,
        'eval_dir': eval_dir,
        'figures': {
            'dashboard': fig_dash_path,
            'confusion_matrices': fig_cm_path,
            'roc_pr_curves': fig_roc_path,
            'event_detection': fig_det_path,
            'duration_correlations': fig_corr_path,
            'per_video': fig_vid_path
        }
    }


if __name__ == "__main__":

    # Usage Example:
    # dlc_file = "path/to/video_dlc.csv"
    # label_file = "path/to/labels.xlsx"
    # probs, preds = run_inference_on_video(dlc_file, best_model, device)
    # plot_inference_results(probs, preds, label_path=label_file)

    # example video
    dlc_file = r"\\research.brandeis.edu\turrigiano-lab\DPL\DATA\XPro\CONTROL\XP03_D4_HS4DLC_resnet50_MasterRatCrickHuntMar24shuffle1_1010000.csv"
    # label_file = r"\\research.brandeis.edu\turrigiano-lab\LucasWahl\Excel files\Exp 8 - naive wt vs naive scn2a\Exp 8 - naive wt vs naive scn2a\test_analyzed\scn2a\LW134-Hunt-1_analyzed.xlsx"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HuntingBiLSTM(input_dim=22, hidden_dim=256, num_layers=3, num_classes=4).to(device) 
    # Model loading settings
    model_path = r"C:\Users\bcary\Documents\Work\CODE\CricketCode\LSTM_hunt_label\models\dan+lucas_0.15chunk2iters\BGadd_feat22_dim256_lay3_ep30_valf1=0.8431.pth"
    
    # -------------------------------------------------------------
    # Example: Running the new Batch Evaluation Pipeline
    # -------------------------------------------------------------
    eval_results = batch_process_and_evaluate(
        folder_path=r"\\research.brandeis.edu\turrigiano-lab\DPL\DATA\XPro\CONTROL",
        model_path=model_path,
        label_folder=r"\\research.brandeis.edu\turrigiano-lab\DPL\DATA\XPro\XPRO_BEHAVIOR\CTL",
        device=device,
        label_format=None, # Auto-detects 'lucas' or 'dan'
        window_size=6000,
        step_size=1000,
        output_subdir="excel_files",
        plot_subdir="plots",
        prob_subdir="probs",
        eval_subdir="evaluation_results",
        rerun_inference=True # Set to False to quickly re-plot without rerunning inference
    )

    
    model.load_state_dict(torch.load(model_path, map_location=device))

    probs, preds = run_inference_on_video(dlc_file, model, device, step_size = 1000,
                                        window_size = 6000, return_all_windows=True, use_max_prob=False)

    # video_width, _, video_fps = dataset_utils.get_video_params(dlc_file)
    # res = dataset_utils.load_and_process_dlc(dlc_file, video_fps)
    # rh, rm, rb, rt, cp = res
    # features = dataset_utils.generate_features(rh, rm, rb, rt, cp, video_width, video_fps)
    # cricket_vis = features[:, 10]

    refined_events = refine_predictions(probs['final'], preds)

    # plot_inference_results(probs, preds, refined_events=refined_events, label_path=label_file,
    #                     label_format='dan', dlc_path=dlc_file)
    # plot_inference_results(probs, preds, refined_events=refined_events, label_path=label_file,
    #                        label_format='lucas', dlc_path=dlc_file)
    plot_inference_results(probs, preds, refined_events=refined_events)
    # plot_inference_results(probs, preds)
    plt.show()    
    

    video_path = r"\\research.brandeis.edu\turrigiano-lab\DPL\DATA\XPro\CONTROL\XP03_D4_HS4.avi"
    output_dir = r"C:\Users\bcary\Documents\Work\CODE\CricketCode\LSTM_hunt_label\vids"
    output_path = os.path.join(output_dir, f"{os.path.basename(video_path)}_analyzed.avi")
    start_frame = 18000
    end_frame = 21000
    create_hunting_video_clip(video_path, probs['final'], preds, start_frame, end_frame, output_path,
                              trailing_frames=300, dlc_path=dlc_file, circle_radius=4, include_blurry_cricket=True,
                              poop_dist_thresh=6, filter_poops=True, plot_poops=True)


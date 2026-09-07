
from matplotlib import pyplot
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os
import re
import glob
from datetime import timedelta
import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from datetime import timedelta


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
        pursuit = int(event['pursuit'])
        capture = int(event['capture'])
        capture_end = event.get('capture_end', capture + dataset_utils.EATING_BUFFER)
        
        enter = max(0, min(num_frames, enter))
        pursuit = max(0, min(num_frames, pursuit))
        capture = max(0, min(num_frames, capture))
        eating_end = min(num_frames, int(capture_end))
        
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
        df.to_excel(output_path, index=False)
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Error saving Excel {output_path}: {e}")

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
    dlc_files = glob.glob(os.path.join(folder_path, "*.csv"))
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



if __name__ == "__main__":

    # Usage Example:
    # dlc_file = "path/to/video_dlc.csv"
    # label_file = "path/to/labels.xlsx"
    # probs, preds = run_inference_on_video(dlc_file, best_model, device)
    # plot_inference_results(probs, preds, label_path=label_file)

    # dlc_file = r"Z:\LucasWahl\Exp 5 - Scn2a mut observer data\Scn2a observer\hunt_crop\LW79-hunt-1-09062024094954_huntPerfDLC_resnet50_Lucas_HuntObsMar6shuffle1_2430000.csv"
    
    # LW134 hunt 1
    dlc_file = r"\\research.brandeis.edu\turrigiano-lab\LucasWahl\Exp 8 - Re-do of naive Wt and naive Scn2a data\hunt_crop\scn2a\LW134-hunt-1-03112025102959_huntPerfDLC_resnet50_Lucas_HuntObsMar6shuffle1_2430000.csv"
    label_file = r"\\research.brandeis.edu\turrigiano-lab\LucasWahl\Excel files\Exp 8 - naive wt vs naive scn2a\Exp 8 - naive wt vs naive scn2a\test_analyzed\scn2a\LW134-Hunt-1_analyzed.xlsx"

    # example 
    dlc_file = r"\\research.brandeis.edu\turrigiano-lab\DPL\DATA\PV-Cre\unused_animals\PV01_D1_HS1DLC_resnet50_MasterRatCrickHuntMar24shuffle1_1010000.csv"
    label_file = r"\\research.brandeis.edu\turrigiano-lab\DPL\DATA\PV-Cre\CricketHuntTimes_PV\CricketHuntTimes_PV01.xlsx"

    # just DLC
    dlc_file = r"\\research.brandeis.edu\turrigiano-lab\DPL\DATA\XPro\XPRO\XP06_D1_HS3DLC_resnet50_MasterRatCrickHuntMar24shuffle1_1010000.csv"
    label_file = None


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HuntingBiLSTM(input_dim=22, hidden_dim=256, num_layers=3, num_classes=4).to(device) 
    # Model loading settings
    model_path = r"C:\Users\bcary\Documents\Work\CODE\CricketCode\LSTM_hunt_label\models\dan+lucas_0.15chunk2iters\BGadd_feat22_dim256_lay3_ep30_valf1=0.8431.pth"
    
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

    # video_path = r"\\research.brandeis.edu\turrigiano-lab\DPL\DATA\PV-Cre\unused_animals\PV14_D1_HS2.avi"
    video_path = r"\\research.brandeis.edu\turrigiano-lab\DPL\DATA\XPro\XPRO\XP06_D1_HS3.avi"
    output_dir = r"C:\Users\bcary\Documents\Work\CODE\CricketCode\LSTM_hunt_label\vids"
    output_path = os.path.join(output_dir, f"{os.path.basename(video_path)}_analyzed.avi")
    start_frame = 11500
    end_frame = 13500
    create_hunting_video_clip(video_path, probs['final'], preds, start_frame, end_frame, output_path,
                              trailing_frames=300, dlc_path=dlc_file, circle_radius=4, include_blurry_cricket=True,
                              poop_dist_thresh=6, filter_poops=True, plot_poops=False)

    batch_process_dlc_folder(
        folder_path=r"\\research.brandeis.edu\turrigiano-lab\DPL\DATA\PV-Cre\unused_animals", 
        model_path=model_path, 
        device=device,
        window_size=6000,
        step_size=1000,
        label_format='dan',
        label_folder=r"\\research.brandeis.edu\turrigiano-lab\DPL\DATA\PV-Cre\CricketHuntTimes_PV",
        output_subdir=None,
        plot_subdir="plots",
        prob_subdir=None
    )

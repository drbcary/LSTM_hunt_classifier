
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

VIDEO_WIDTH = 514
VIDEO_FPS = 30.0
RAT_LIKELIHOOD_THRESH = 0.82
CRICKET_LIKELIHOOD_THRESH = 0.9
MAX_INTERP_GAP = 10 
EATING_BUFFER = 300 # Used for reconstructing ground truth labels

# ==========================================
# 3. Data Processing Helpers
# ==========================================
def calculate_angle_3pts(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.sum(ba*bc, axis=1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_bearing(start_pt, end_pt):
    return np.arctan2(end_pt[:, 1] - start_pt[:, 1], end_pt[:, 0] - start_pt[:, 0])

def load_and_process_dlc(filepath):
    try:
        df = pd.read_csv(filepath, header=[0, 1, 2], index_col=0)
    except: return None

    scorer = df.columns.get_level_values(0)[0]
    
    def get_part(part_name, thresh=0.0):
        if part_name not in df[scorer]:
            return np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))
        x = df[scorer][part_name]['x'].copy()
        y = df[scorer][part_name]['y'].copy()
        conf = df[scorer][part_name]['likelihood'].copy()
        mask = conf < thresh
        x[mask] = np.nan
        y[mask] = np.nan
        x = x.interpolate(method='linear', limit=MAX_INTERP_GAP, limit_direction='both')
        y = y.interpolate(method='linear', limit=MAX_INTERP_GAP, limit_direction='both')
        return x.values, y.values, conf.values

    nose_x, nose_y, _ = get_part('nose', RAT_LIKELIHOOD_THRESH)
    head_x, head_y, _ = get_part('head', RAT_LIKELIHOOD_THRESH)
    mid_x, mid_y, _   = get_part('mid_body', RAT_LIKELIHOOD_THRESH)
    butt_x, butt_y, _ = get_part('butt', RAT_LIKELIHOOD_THRESH)
    tail_x, tail_y, _ = get_part('endtail', RAT_LIKELIHOOD_THRESH)

    rat_head = np.column_stack((head_x, head_y))
    rat_nose = np.column_stack((nose_x, nose_y))
    mask_head = np.isnan(rat_head[:,0])
    rat_head[mask_head] = rat_nose[mask_head]
    rat_mid = np.column_stack((mid_x, mid_y))
    rat_butt = np.column_stack((butt_x, butt_y))
    rat_tail = np.column_stack((tail_x, tail_y))

    c_x, c_y, c_conf = get_part('cricket', 0.0)
    bc_x, bc_y, bc_conf = get_part('blurry_cricket', 0.0)
    final_cx = np.full_like(c_x, np.nan)
    final_cy = np.full_like(c_y, np.nan)
    c_valid = c_conf > CRICKET_LIKELIHOOD_THRESH
    bc_valid = bc_conf > CRICKET_LIKELIHOOD_THRESH
    final_cx[bc_valid] = bc_x[bc_valid]
    final_cy[bc_valid] = bc_y[bc_valid]
    final_cx[c_valid] = c_x[c_valid]
    final_cy[c_valid] = c_y[c_valid]
    final_cx = pd.Series(final_cx).interpolate(limit=MAX_INTERP_GAP).values
    final_cy = pd.Series(final_cy).interpolate(limit=MAX_INTERP_GAP).values
    cricket_pos = np.column_stack((final_cx, final_cy))
    
    return rat_head, rat_mid, rat_butt, rat_tail, cricket_pos

def generate_features(rat_head, rat_mid, rat_butt, rat_tail, cricket_pos):
    def dist(p1, p2): return np.linalg.norm(p1 - p2, axis=1)
    d_head_cricket = dist(rat_head, cricket_pos) / VIDEO_WIDTH
    d_head_mid = dist(rat_head, rat_mid) / VIDEO_WIDTH
    d_mid_tail = dist(rat_mid, rat_tail) / VIDEO_WIDTH
    d_head_tail = dist(rat_head, rat_tail) / VIDEO_WIDTH
    angle_spine = calculate_angle_3pts(rat_head, rat_mid, rat_tail) / 180.0
    rat_orientation = calculate_bearing(rat_mid, rat_head)
    target_vector = calculate_bearing(rat_head, cricket_pos)
    angle_diff = target_vector - rat_orientation
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    angle_diff = angle_diff / np.pi
    
    def get_derivs(pos_array):
        vel_vec = np.diff(pos_array, axis=0, prepend=pos_array[0:1]) * VIDEO_FPS
        speed = np.linalg.norm(vel_vec, axis=1) / VIDEO_WIDTH
        speed = np.clip(speed, 0, 2.0)
        acc_vec = np.diff(vel_vec, axis=0, prepend=vel_vec[0:1]) * VIDEO_FPS
        accel = np.linalg.norm(acc_vec, axis=1) / VIDEO_WIDTH
        return speed, accel
    rat_speed, rat_accel = get_derivs(rat_head)
    cricket_speed, cricket_accel = get_derivs(cricket_pos)
    
    cricket_visible = np.where(np.isnan(d_head_cricket), 0.0, 1.0)
    
    feature_stack = np.column_stack([
        d_head_cricket, d_head_mid, d_mid_tail, d_head_tail,
        angle_spine, angle_diff, rat_speed, rat_accel,
        cricket_speed, cricket_accel, cricket_visible
    ])
    mask_inv = 1.0 - cricket_visible
    feature_stack[mask_inv == 1.0, 0] = 0.0 
    feature_stack[mask_inv == 1.0, 5] = 0.0
    return np.nan_to_num(feature_stack, nan=0.0)

# ==========================================
# 4. Logic Refinement & Smoothing (UPDATED)
# ==========================================

def frames_to_time(frames):
    seconds = frames / VIDEO_FPS
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    else:
        return f"00:{minutes:02}:{seconds:02}"

def find_contiguous_blocks(mask):
    bounded = np.concatenate(([0], mask.astype(int), [0]))
    diffs = np.diff(bounded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return list(zip(starts, ends))

def refine_predictions(probs, preds, cricket_visible_feature):
    """
    Robustly segments separate hunts and prevents overlap.
    """
    num_frames = len(preds)
    
    # 1. Clean raw predictions
    smooth_preds = pd.Series(preds).rolling(7, center=True).median().fillna(0).values.astype(int)
    
    # 2. Identify PURSUIT Anchors (Class 2)
    is_pursuit = (smooth_preds == 2)
    pursuit_blocks = find_contiguous_blocks(is_pursuit)
    
    # Merge close pursuit blocks
    merged_pursuits = []
    if pursuit_blocks:
        curr_start, curr_end = pursuit_blocks[0]
        for i in range(1, len(pursuit_blocks)):
            next_start, next_end = pursuit_blocks[i]
            if next_start - curr_end < 60:
                curr_end = next_end 
            else:
                merged_pursuits.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end
        merged_pursuits.append((curr_start, curr_end))
    
    valid_anchors = []
    for s, e in merged_pursuits:
        if (e - s) < 10: continue 
        # Check if probability is high enough to be real
        if np.max(probs[s:e, 2]) < 0.4: continue
        valid_anchors.append((s, e))
        
    events = []
    last_hunt_end = 0 
    
    for i, (p_start, p_end) in enumerate(valid_anchors):
        
        # --- A. DETERMINE ENTER (STALKING START) ---
        enter_frame = p_start
        scan_limit = max(last_hunt_end, p_start - 2000) # Look back ~66s
        
        if p_start > scan_limit:
            pre_window_preds = smooth_preds[scan_limit:p_start]
            stalk_blocks = find_contiguous_blocks(pre_window_preds == 1)
            
            if stalk_blocks:
                global_stalk_blocks = [(s + scan_limit, e + scan_limit) for s, e in stalk_blocks]
                current_head = p_start
                
                # Consolidate backward
                for sb_start, sb_end in reversed(global_stalk_blocks):
                    gap_start = sb_end
                    gap_end = current_head
                    gap_len = gap_end - gap_start
                    
                    # Connect if gap is small OR if gap has "soft stalking" probability
                    is_connected = False
                    if gap_len < 150: 
                        is_connected = True
                    elif gap_len < 600:
                        # Check probability of Stalking (Class 1) in the gap
                        # If avg prob > 0.2, it's likely a continuous stalk that dipped in confidence
                        gap_stalk_prob = np.mean(probs[gap_start:gap_end, 1])
                        if gap_stalk_prob > 0.2:
                            is_connected = True
                            
                    if is_connected:
                        enter_frame = sb_start
                        current_head = sb_start
                    else:
                        break
        
        # --- B. DETERMINE CAPTURE (EATING START) ---
        # Ensure we DO NOT cross into the next pursuit anchor
        next_anchor_start = valid_anchors[i+1][0] if (i + 1) < len(valid_anchors) else num_frames
        
        capture_frame = p_end 
        
        # Look forward for Eating (Class 3)
        # Stop looking if we hit the next hunt
        scan_fwd_limit = min(next_anchor_start, p_end + 2000) 
        
        if scan_fwd_limit > p_end:
            post_window_preds = smooth_preds[p_end:scan_fwd_limit]
            eat_blocks = find_contiguous_blocks(post_window_preds == 3)
            
            if eat_blocks:
                # If eating starts reasonably soon
                first_eat_start = eat_blocks[0][0] + p_end
                if (first_eat_start - p_end) < 150:
                    # Valid eating detected.
                    # We take the start of eating as the capture time
                    capture_frame = first_eat_start
                    
                    # Check if this eating block runs into the next anchor
                    # If the eating block ends exactly where the next pursuit starts, 
                    # it's suspicious but technically allowable as a "catch immediately followed by theft/chase"
                    # But usually, we just want to ensure we don't label the next pursuit as eating.
                    # Our 'scan_fwd_limit' guarantees we don't look past the next anchor.
                
        # --- C. FINAL VALIDATION ---
        remarks = ""
        confidence = np.mean(probs[p_start:p_end, 2])
        
        if (capture_frame - enter_frame) < 15:
            remarks = "Extremely Short Event"
        elif confidence < 0.5:
            remarks = "Low Confidence"
            
        events.append({
            'enter': enter_frame,
            'pursuit': p_start,
            'capture': capture_frame,
            'remarks': remarks
        })
        
        # Update boundary: Next hunt cannot start before this one ends
        last_hunt_end = capture_frame + 10 
            
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
        
        enter = max(0, min(num_frames, enter))
        pursuit = max(0, min(num_frames, pursuit))
        capture = max(0, min(num_frames, capture))
        eating_end = min(num_frames, capture + EATING_BUFFER)
        
        if pursuit > enter: seq[enter:pursuit] = 1 
        if capture > pursuit: seq[pursuit:capture] = 2 
        if eating_end > capture: seq[capture:eating_end] = 3 
            
    return seq

def plot_inference_results(probs, preds, refined_events=None, label_path=None, save_path=None, title="Inference Results"):
    num_frames = len(preds)
    frames = np.arange(num_frames)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    
    classes = ['Background', 'Cricket Present', 'Pursuit', 'Eating']
    colors = ['lightgray', 'orange', 'red', 'green']
    
    for i, cls_name in enumerate(classes):
        ax1.plot(frames, probs[:, i], label=cls_name, color=colors[i], linewidth=1.5, alpha=0.8)
    
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
            labels_df = pd.read_excel(label_path)
            # valid_hunts = labels_df[labels_df['Include?'] == 'Include']
            valid_hunts = labels_df
            true_seq = np.zeros(num_frames)
            
            for _, row in valid_hunts.iterrows():
                enter = int(row['Enter_Frame'])
                capture = int(row['Capture_Frame'])
                
                # Estimate pursuit/eating for visual comparison if exact cols missing
                pursuit_start = enter + int(row['Latency_to_Attack_Frames'])
                eating_end = capture + EATING_BUFFER
                
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

def generate_analyzed_excel(events, output_path):
    data_rows = []
    
    for i, event in enumerate(events):
        enter = event['enter']
        pursuit = event['pursuit']
        capture = event['capture']
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
            'Enter_Time': frames_to_time(enter),
            'Capture_Frame': capture,
            'Capture_Time': frames_to_time(capture),
            'Time_to_Capture_Frames': total_frames,
            'Time_to_Capture': frames_to_time(total_frames),
            'Pursuit_Time_Frames': pursuit_dur_frames,
            'Pursuit_Time': frames_to_time(pursuit_dur_frames),
            'Latency_to_Attack_Frames': latency_frames,
            'Latency_to_Attack': frames_to_time(latency_frames),
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

def batch_process_dlc_folder(folder_path, model_path, device, label_folder=None):
    # 1. Load Model
    print(f"Loading Model from {model_path}...")
    # Updated to your current best model size
    model = HuntingBiLSTM(input_dim=11, hidden_dim=256, num_layers=3, num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. Prepare Output Dirs
    output_dir = os.path.join(folder_path, "excel_files")
    plot_dir = os.path.join(folder_path, "plots")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # 3. Find Files
    dlc_files = glob.glob(os.path.join(folder_path, "*.csv"))
    print(f"Found {len(dlc_files)} DLC files.")
    
    for dlc_file in dlc_files:
        filename = os.path.basename(dlc_file)
        
        # NEW REGEX STRATEGY:
        # 1. Try strict Observe-Hunt format: LW39-observe-LW40-hunt-1...
        match_complex = re.search(r'(LW\d+)[-_]observe[-_](LW\d+)[-_]hunt[-_]?(\d+)', filename, re.IGNORECASE)
        
        # 2. Fallback to simple Hunt format: LW40-hunt-1...
        match_simple = re.search(r'(LW\d+)[-_]hunt[-_]?(\d+)', filename, re.IGNORECASE)
        
        if match_complex:
            obs_animal = match_complex.group(1)
            hunt_animal = match_complex.group(2)
            hunt_num = match_complex.group(3)
            output_filename = f"{obs_animal}-observe-{hunt_animal}-hunt-{hunt_num}_analyzed.xlsx"
            plot_title = f"{obs_animal} observing {hunt_animal} (Hunt {hunt_num})"
            
        elif match_simple:
            hunt_animal = match_simple.group(1)
            hunt_num = match_simple.group(2)
            output_filename = f"{hunt_animal}-hunt-{hunt_num}_analyzed.xlsx"
            plot_title = f"{hunt_animal} (Hunt {hunt_num})"
            
        else:
            print(f"Skipping {filename}: Could not parse Animal ID/Hunt pattern.")
            continue
            
        output_path = os.path.join(output_dir, output_filename)
        print(f"Processing {filename} -> {output_filename}...")
        
        try:
            # A. Load Data
            res = load_and_process_dlc(dlc_file)
            if res is None: continue
            rh, rm, rb, rt, cp = res
            features = generate_features(rh, rm, rb, rt, cp)
            
            # B. Run Model
            seq_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            lengths = torch.tensor([len(features)]).cpu()
            
            with torch.no_grad():
                logits = model(seq_tensor, lengths)
                probs = torch.softmax(logits, dim=2).cpu().numpy()[0]
                preds = torch.argmax(logits, dim=2).cpu().numpy()[0]
                
            # C. Clean Predictions
            cricket_vis = features[:, 10]
            events = refine_predictions(probs, preds, cricket_vis)
            
            # D. Plotting
            plot_filename = output_filename.replace('.xlsx', '.png')
            plot_path = os.path.join(plot_dir, plot_filename)
            
            label_path = None
            if label_folder:
                 # Try to find exact match first
                 candidate = os.path.join(label_folder, output_filename)
                 if os.path.exists(candidate):
                     label_path = candidate
                 else:
                     # Fallback: try to match partial name if format differs
                     # e.g. if manual file is LW40-hunt-1 but output is LW39-observe-LW40...
                     pass 
            
            plot_inference_results(probs, preds, refined_events=events, 
                                   label_path=label_path, save_path=plot_path, 
                                   title=plot_title)
            
            # E. Generate Excel
            if len(events) > 0:
                generate_analyzed_excel(events, output_path)
            else:
                print(f"  -> No hunts detected in {filename}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()

def run_inference_on_video(dlc_path, model, device):
    """
    Runs the trained model on a full video DLC file.
    Returns:
        probs: (N, 4) numpy array of class probabilities
        preds: (N,) numpy array of predicted class indices
    """
    print(f"Processing {dlc_path}...")
    
    # 1. Process Data
    processed_data = load_and_process_dlc(dlc_path)
    if processed_data is None:
        return None, None
        
    rat_head, rat_mid, rat_butt, rat_tail, cricket_pos = processed_data
    features = generate_features(rat_head, rat_mid, rat_butt, rat_tail, cricket_pos)
    
    # 2. Prepare Tensor
    # Batch size = 1, Length = full video
    seq_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    lengths = torch.tensor([len(features)]).cpu() # Length needed for pack_padded
    
    # 3. Run Model
    model.eval()
    with torch.no_grad():
        logits = model(seq_tensor, lengths)
        probs = torch.softmax(logits, dim=2)
        preds = torch.argmax(probs, dim=2)
    
    # 4. Convert to Numpy (Strip batch dimension)
    probs_np = probs[0].cpu().numpy()
    preds_np = preds[0].cpu().numpy()
    
    return probs_np, preds_np



if __name__ == "__main__":

    # Usage Example:
    # dlc_file = "path/to/video_dlc.csv"
    # label_file = "path/to/labels.xlsx"
    # probs, preds = run_inference_on_video(dlc_file, best_model, device)
    # plot_inference_results(probs, preds, label_path=label_file)

    # dlc_file = r"Z:\LucasWahl\Exp 5 - Scn2a mut observer data\Scn2a observer\hunt_crop\LW79-hunt-1-09062024094954_huntPerfDLC_resnet50_Lucas_HuntObsMar6shuffle1_2430000.csv"
    
    #LW134 hunt 1
    dlc_file = r"Z:\LucasWahl\Exp 8 - Re-do of naive Wt and naive Scn2a data\hunt_crop\scn2a\LW134-hunt-1-03112025102959_huntPerfDLC_resnet50_Lucas_HuntObsMar6shuffle1_2430000.csv"
    label_file = r"Z:\LucasWahl\Excel files\Exp 8 - naive wt vs naive scn2a\Exp 8 - naive wt vs naive scn2a\test_analyzed\scn2a\LW134-Hunt-1_analyzed.xlsx"

    #example 
    dlc_file = r"Z:\BrianCary\DATA\CricketHunt\dan\DLC_ex_vids\BL71_72-210224-155924_Crickets_BJL-210303-113352_Cam1_fr1-60396_lightDLC_resnet50_DanCrick_LightJun6shuffle1_1130000.csv"
    label_file = r""


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HuntingBiLSTM(input_dim=11, hidden_dim=256, num_layers=3, num_classes=4).to(device) 
    model.load_state_dict(torch.load('BGadd_feat11_dim256_lay3_ep50_valf1=0.8865.pth', map_location=device))

    probs, preds = run_inference_on_video(dlc_file, model, device)

    res = load_and_process_dlc(dlc_file)
    rh, rm, rb, rt, cp = res
    features = generate_features(rh, rm, rb, rt, cp)
    cricket_vis = features[:, 10]
    refined_events = refine_predictions(probs, preds, cricket_vis)

    plot_inference_results(probs, preds, refined_events=refined_events, label_path=label_file)
    plot_inference_results(probs, preds, label_path=label_file)
    plot_inference_results(probs, preds, refined_events=refined_events)
    # plot_inference_results(probs, preds)

    video_path = r"Z:\LucasWahl\Exp 8 - Re-do of naive Wt and naive Scn2a data\hunt_crop\scn2a\LW134-hunt-1-03112025102959_huntPerfDLC_resnet50_Lucas_HuntObsMar6shuffle1_2430000_labeled.mp4"
    output_path = r"LW134-hunt-1_ex_vid2.avi"
    start_frame = 20330
    end_frame = 21900
    create_hunting_video_clip(video_path, probs, preds, start_frame, end_frame, output_path, trailing_frames=300)

    batch_process_dlc_folder(
        folder_path=r"Z:\LucasWahl\Visual attention tracking project\Exp 5 - Scn2a mut observer data\hunt_crop", 
        model_path="BGadd_feat11_dim256_lay3_ep50_valf1=0.8865.pth", 
        device=device
    )

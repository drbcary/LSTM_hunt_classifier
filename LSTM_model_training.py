import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from typing import Optional, Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

# Compatibility patch for debugpy / pydevd Qt event loop integration with PySide6
try:
    import PySide6.QtWidgets, PySide6.QtGui
    if not hasattr(PySide6.QtGui, 'QApplication'):
        setattr(PySide6.QtGui, 'QApplication', PySide6.QtWidgets.QApplication)
except ImportError:
    pass

import matplotlib.pyplot as plt
import os
import re
import glob
import warnings
import copy
import random
from tqdm import tqdm

# Suppress chaining warnings from pandas
warnings.filterwarnings('ignore')

# ==========================================
# 1. Configuration & Constants
# ==========================================
import dataset_utils

# File Filtering Settings
VERBOSE = True

# ==========================================
# 2. Helper Functions (Data Loading)
# ==========================================

# ==========================================
# 3. Data Processing
# ==========================================

def is_valid_training_sequence(
    targets: np.ndarray,
    min_transition_prop: float = 0.10,
    min_transition_frames: int = 30
) -> bool:
    """
    Validates whether a target sequence is appropriate for training:
      - Pure background sequences (all zeros) are valid.
      - Sequences containing hunt behaviors (classes > 0) MUST NOT be monolithic single-label blocks.
      - Must contain at least two classes, with non-dominant classes making up at least min_transition_prop
        (default 10%) of the sequence AND at least min_transition_frames (default 30 frames).
    """
    if len(targets) == 0:
        return False
    if np.all(targets == 0):
        return True

    unique_classes, counts = np.unique(targets, return_counts=True)
    if len(unique_classes) < 2:
        return False

    total_len = len(targets)
    max_count = np.max(counts)
    secondary_count = total_len - max_count

    if secondary_count < min_transition_frames:
        return False
    if (secondary_count / total_len) < min_transition_prop:
        return False

    return True

def snap_chunk_to_transition(
    start_idx: int,
    end_idx: int,
    full_targets: np.ndarray,
    min_transition_prop: float = 0.10,
    min_transition_frames: int = 30
) -> Tuple[int, int]:
    """
    Given a chunk [start_idx, end_idx], if it contains hunt behaviors but fails
    transition adequacy (e.g., monolithic single-hunt label), snaps/shifts the window
    towards the nearest behavioral transition boundary in full_targets so that the
    transition is captured with adequate context. Prevents unnecessary data dropping.
    """
    chunk_len = end_idx - start_idx
    if chunk_len <= 0:
        return start_idx, end_idx

    chunk_targ = full_targets[start_idx:end_idx]
    if is_valid_training_sequence(chunk_targ, min_transition_prop, min_transition_frames):
        return start_idx, end_idx

    # Only attempt snapping if there is some hunt behavior; pure background is already valid
    if np.all(chunk_targ == 0):
        return start_idx, end_idx

    total_frames = len(full_targets)
    transition_pts = np.where(np.diff(full_targets) != 0)[0] + 1
    if len(transition_pts) == 0:
        return start_idx, end_idx

    desired_margin = max(min_transition_frames + 15, int(min_transition_prop * 1.5 * chunk_len))

    # Find closest transition point to the window center
    center = (start_idx + end_idx) / 2.0
    dists = np.abs(transition_pts - center)
    best_t = transition_pts[np.argmin(dists)]

    # If best_t is before or near the start, shift window left so best_t is well inside
    if best_t <= start_idx + desired_margin:
        new_start = max(0, best_t - desired_margin)
        new_end = min(total_frames, new_start + chunk_len)
        if (new_end - new_start) < chunk_len and new_start > 0:
            new_start = max(0, new_end - chunk_len)
    elif best_t >= end_idx - desired_margin:
        new_end = min(total_frames, best_t + desired_margin)
        new_start = max(0, new_end - chunk_len)
        if (new_end - new_start) < chunk_len and new_end < total_frames:
            new_end = min(total_frames, new_start + chunk_len)
    else:
        new_start = start_idx
        new_end = end_idx

    new_targ = full_targets[new_start:new_end]
    if is_valid_training_sequence(new_targ, min_transition_prop, min_transition_frames):
        return new_start, new_end

    # Secondary attempt: try the nearest transition before center and nearest after center
    before_pts = transition_pts[transition_pts <= center]
    if len(before_pts) > 0:
        t_before = before_pts[-1]
        s1 = max(0, t_before - desired_margin)
        e1 = min(total_frames, s1 + chunk_len)
        if is_valid_training_sequence(full_targets[s1:e1], min_transition_prop, min_transition_frames):
            return s1, e1

    after_pts = transition_pts[transition_pts > center]
    if len(after_pts) > 0:
        t_after = after_pts[0]
        e2 = min(total_frames, t_after + desired_margin)
        s2 = max(0, e2 - chunk_len)
        if is_valid_training_sequence(full_targets[s2:e2], min_transition_prop, min_transition_frames):
            return s2, e2

    return start_idx, end_idx

def create_dataset_sequences(paired_files, paradigm="rat", target_fps=30.0, continuous_cricket_confidence=False):
    all_sequences = []
    all_targets = []
    total_hunts_processed = 0
    shortened_eating_hunts = []
    chunked_long_hunts = []
    total_snapped_chunks = 0
    max_eating_frames = int(60 * target_fps)
    MAX_HUNT_SEQ_LEN = 5000
    
    if VERBOSE: print(f"Processing {len(paired_files)} files (Balanced Mode, Paradigm: {paradigm}, Target FPS: {target_fps}, Continuous Cricket: {continuous_cricket_confidence})...")
    iterator = tqdm(paired_files, desc="Processing Videos", unit="vid", disable=not VERBOSE)
    
    for pair in iterator:
        try:
            video_width, video_height, video_fps = dataset_utils.get_video_params(
                pair['dlc_path'],
                video_path=pair.get('video_path'),
                verbose=False
            )
            
            is_mouse = (paradigm == "mouse" or pair.get('format') in ('diane', 'diane_parts'))
            if is_mouse:
                mouse_cfg = MouseParadigmConfig()
                mouse_cfg.continuous_cricket_confidence = continuous_cricket_confidence
                pose_data = dataset_utils.load_mouse_dlc(pair['dlc_path'], video_fps, mouse_cfg=mouse_cfg)
                if pose_data is None: continue
                full_features, _ = dataset_utils.generate_mouse_features(pose_data, video_width, video_fps, config_obj=mouse_cfg)
            else:
                processed_data = dataset_utils.load_and_process_dlc(pair['dlc_path'], video_fps)
                if processed_data is None: continue
                rat_head, rat_mid, rat_butt, rat_tail, cricket_pos = processed_data
                full_features = dataset_utils.generate_features(rat_head, rat_mid, rat_butt, rat_tail, cricket_pos, video_width, video_fps)
            
            # --- Standardize to Canonical Target FPS (e.g. 20 FPS -> 30 FPS) ---
            if abs(video_fps - target_fps) > 0.1:
                full_features = dataset_utils.resample_feature_matrix(full_features, video_fps, target_fps)
                pair_hunts = dataset_utils.scale_hunt_events_fps(pair['hunts'], video_fps, target_fps)
            else:
                pair_hunts = pair['hunts']

            total_frames = len(full_features)
            
            # Build full ground-truth targets for this video
            full_targets = np.zeros(total_frames, dtype=int)
            for hunt_data in pair_hunts:
                enter = hunt_data['enter']
                capture = hunt_data['capture']
                pursuit_start = hunt_data['pursuit_start']
                cons_end = hunt_data['consumption_end']
                first_app = hunt_data.get('first_approach', None)
                
                if enter >= capture or pursuit_start > capture: continue
                if first_app is not None:
                    first_app = max(enter, min(pursuit_start, first_app))

                if cons_end is not None:
                    eating_end_frame = min(cons_end, capture + max_eating_frames)
                else:
                    eating_end_frame = capture + min(dataset_utils.EATING_BUFFER, max_eating_frames)
                    
                if is_mouse:
                    # 5-class targets: 0=BG, 1=Cricket Present, 2=First Approach, 3=Final Approach, 4=Eating
                    if first_app is not None and first_app > enter:
                        full_targets[max(0, enter):min(total_frames, first_app)] = 1
                        if pursuit_start > first_app:
                            full_targets[max(0, first_app):min(total_frames, pursuit_start)] = 2
                    else:
                        if pursuit_start > enter:
                            full_targets[max(0, enter):min(total_frames, pursuit_start)] = 2
                    if capture > pursuit_start:
                        full_targets[max(0, pursuit_start):min(total_frames, capture)] = 3
                    if eating_end_frame > capture:
                        full_targets[max(0, capture):min(total_frames, eating_end_frame)] = 4
                else:
                    # 4-class targets: 0=BG, 1=Stalking/Approach, 2=Pursuit, 3=Eating
                    if pursuit_start > enter:
                        full_targets[max(0, enter):min(total_frames, pursuit_start)] = 1
                    if capture > pursuit_start:
                        full_targets[max(0, pursuit_start):min(total_frames, capture)] = 2
                    if eating_end_frame > capture:
                        full_targets[max(0, capture):min(total_frames, eating_end_frame)] = 3

            # Decide if this video is fully chunked
            if random.random() < dataset_utils.FULL_VIDEO_CHUNK_PROPORTION:
                for _ in range(dataset_utils.FULL_VIDEO_CHUNK_ITERATIONS):
                    current_idx = 0
                    while current_idx <= total_frames - dataset_utils.FULL_VIDEO_CHUNK_MIN:
                        chunk_size = random.randint(dataset_utils.FULL_VIDEO_CHUNK_MIN, dataset_utils.FULL_VIDEO_CHUNK_MAX)
                        end_idx = min(total_frames, current_idx + chunk_size)
                        
                        if (end_idx - current_idx) >= dataset_utils.FULL_VIDEO_CHUNK_MIN:
                            adj_start, adj_end = snap_chunk_to_transition(
                                current_idx, end_idx, full_targets, min_transition_prop=0.10, min_transition_frames=30
                            )
                            if (adj_start, adj_end) != (current_idx, end_idx):
                                total_snapped_chunks += 1
                                
                            seq_features = full_features[adj_start:adj_end]
                            seq_targets = full_targets[adj_start:adj_end]
                            
                            if len(seq_features) >= dataset_utils.FULL_VIDEO_CHUNK_MIN and is_valid_training_sequence(seq_targets):
                                all_sequences.append(seq_features)
                                all_targets.append(seq_targets)
                            
                        current_idx = end_idx
                continue
            
            # --- BG CHUNK LIMITER ---
            # We want background, but not infinite background.
            bg_chunks_generated = 0
            last_event_end = 0 
            
            for hunt_data in pair_hunts:
                enter = hunt_data['enter']
                capture = hunt_data['capture']
                pursuit_start = hunt_data['pursuit_start']
                cons_end = hunt_data['consumption_end']
                first_app = hunt_data.get('first_approach', None)
                
                # 1. GAP HANDLING (With Limits & Boundary Clamping)
                gap_start = min(total_frames, max(0, last_event_end))
                gap_end = min(total_frames, max(0, enter - dataset_utils.CONTEXT_BUFFER))
                
                # Only add background if we haven't hit the limit for this video
                if (gap_end - gap_start) > dataset_utils.MIN_BG_LEN and bg_chunks_generated < dataset_utils.MAX_BG_CHUNKS_PER_VIDEO:
                    gap_attempts = 0
                    current_idx = gap_start
                    
                    while current_idx < (gap_end - dataset_utils.MIN_BG_LEN) and gap_attempts < 2:
                        chunk_size = random.randint(dataset_utils.MIN_BG_LEN, dataset_utils.MAX_BG_LEN)
                        end_idx = min(gap_end, current_idx + chunk_size)
                        
                        chunk_features = full_features[current_idx:end_idx]
                        if len(chunk_features) >= dataset_utils.MIN_BG_LEN:
                            all_sequences.append(chunk_features)
                            all_targets.append(np.zeros(len(chunk_features), dtype=int))
                            bg_chunks_generated += 1
                            gap_attempts += 1
                            
                        current_idx = end_idx + random.randint(100, 300) # Jump forward

                # 2. HUNT HANDLING
                if enter >= capture or pursuit_start > capture: continue
                if first_app is not None:
                    first_app = max(enter, min(pursuit_start, first_app))
                total_hunts_processed += 1
                
                rand_buffer = random.randint(50, 400)
                
                # Cap maximum eating duration to prevent massive sequence lengths (e.g. 60s max consumption window)
                if cons_end is not None:
                    if (cons_end - capture) > max_eating_frames:
                        shortened_eating_hunts.append({
                            'pair_id': pair.get('id', 'unknown'),
                            'original_eating_frames': cons_end - capture,
                            'shortened_eating_frames': max_eating_frames,
                        })
                    eating_end_frame = min(cons_end, capture + max_eating_frames)
                else:
                    eating_end_frame = capture + min(dataset_utils.EATING_BUFFER, max_eating_frames)
                
                seq_start = max(0, min(total_frames, enter - rand_buffer))
                seq_end = max(0, min(total_frames, eating_end_frame + rand_buffer))
                if seq_end <= seq_start: continue
                
                seq_features = full_features[seq_start:seq_end]
                seq_len = len(seq_features)
                if seq_len < 10: continue 
                
                seq_targets = full_targets[seq_start:seq_end]
                
                # If a hunt sequence is exceptionally long (> 5000 frames / > 2.7 min),
                # use Dual-Anchored partitioning to preserve entry and exit transitions
                MAX_HUNT_SEQ_LEN = 5000
                if seq_len > MAX_HUNT_SEQ_LEN:
                    num_chunks = 0
                    # Chunk 1: Dual-anchored entry window [0 : MAX_HUNT_SEQ_LEN]
                    chunk1_feat = seq_features[0:MAX_HUNT_SEQ_LEN]
                    chunk1_targ = seq_targets[0:MAX_HUNT_SEQ_LEN]
                    if len(chunk1_feat) >= dataset_utils.MIN_BG_LEN and is_valid_training_sequence(chunk1_targ):
                        all_sequences.append(chunk1_feat)
                        all_targets.append(chunk1_targ)
                        num_chunks += 1

                    # Chunk 2: Dual-anchored exit window [seq_len - MAX_HUNT_SEQ_LEN : seq_len]
                    chunk2_feat = seq_features[seq_len - MAX_HUNT_SEQ_LEN:seq_len]
                    chunk2_targ = seq_targets[seq_len - MAX_HUNT_SEQ_LEN:seq_len]
                    if len(chunk2_feat) >= dataset_utils.MIN_BG_LEN and is_valid_training_sequence(chunk2_targ):
                        all_sequences.append(chunk2_feat)
                        all_targets.append(chunk2_targ)
                        num_chunks += 1

                    # If hunt is extremely long (> 2 * MAX_HUNT_SEQ_LEN, e.g. > 10,000 frames),
                    # extract an intermediate chunk centered on internal transitions
                    if seq_len > (2 * MAX_HUNT_SEQ_LEN):
                        seq_transitions = np.where(np.diff(seq_targets) != 0)[0] + 1
                        mid_transitions = [t for t in seq_transitions if MAX_HUNT_SEQ_LEN // 2 < t < (seq_len - MAX_HUNT_SEQ_LEN // 2)]
                        for mid_t in mid_transitions:
                            m_start = max(0, mid_t - MAX_HUNT_SEQ_LEN // 2)
                            m_end = min(seq_len, m_start + MAX_HUNT_SEQ_LEN)
                            m_feat = seq_features[m_start:m_end]
                            m_targ = seq_targets[m_start:m_end]
                            if len(m_feat) >= dataset_utils.MIN_BG_LEN and is_valid_training_sequence(m_targ):
                                all_sequences.append(m_feat)
                                all_targets.append(m_targ)
                                num_chunks += 1
                                break

                    chunked_long_hunts.append({
                        'pair_id': pair.get('id', 'unknown'),
                        'total_frames': seq_len,
                        'approach_frames': max(0, pursuit_start - enter),
                        'attack_frames': max(0, capture - pursuit_start),
                        'num_chunks': num_chunks,
                    })
                else:
                    if is_valid_training_sequence(seq_targets):
                        all_sequences.append(seq_features)
                        all_targets.append(seq_targets)
                
                last_event_end = seq_end
                
        except Exception as e:
            if VERBOSE: tqdm.write(f"Error: {e}")
            continue
            
    # Strict validation filter: eliminate any 0-length, corrupted, or monolithic single-hunt sequences
    valid_sequences = []
    valid_targets = []
    rejected_monolithic_count = 0
    for seq, targ in zip(all_sequences, all_targets):
        if len(seq) >= 10 and len(seq) == len(targ):
            if is_valid_training_sequence(targ):
                valid_sequences.append(seq)
                valid_targets.append(targ)
            else:
                rejected_monolithic_count += 1

    if VERBOSE:
        print("\n--- Hunt Duration & Chunking Diagnostics ---")
        print(f"Total Hunt Events Processed: {total_hunts_processed}")
        if total_snapped_chunks > 0:
            print(f"  [BOUNDARY SNAPPING] Successfully snapped {total_snapped_chunks} chunks to behavioral boundaries to satisfy transition requirements and preserve data.")
        if rejected_monolithic_count > 0:
            print(f"  [TRANSITION FILTER] Excluded {rejected_monolithic_count} sequences that were monolithic single-hunt labels or lacked >=10% transition context.")
        if shortened_eating_hunts:
            print(f"  [SHORTENED EATING] {len(shortened_eating_hunts)} hunts had eating duration capped at {max_eating_frames} frames ({max_eating_frames/target_fps:.1f}s):")
            for h in shortened_eating_hunts[:5]:
                print(f"    - {h['pair_id']}: original eating {h['original_eating_frames']} frames ({h['original_eating_frames']/target_fps:.1f}s) -> capped to {h['shortened_eating_frames']} frames")
            if len(shortened_eating_hunts) > 5:
                print(f"    ... (+{len(shortened_eating_hunts)-5} more)")
        else:
            print("  [SHORTENED EATING] 0 hunts had eating duration capped.")

        if chunked_long_hunts:
            print(f"  [CHUNKED HUNTS] {len(chunked_long_hunts)} hunts exceeded MAX_HUNT_SEQ_LEN ({MAX_HUNT_SEQ_LEN} frames / {MAX_HUNT_SEQ_LEN/target_fps:.1f}s) and were partitioned via dual-anchored entry/exit windows (no behavioral transition data lost):")
            for h in chunked_long_hunts[:5]:
                print(f"    - {h['pair_id']}: total {h['total_frames']} frames (approach: {h['approach_frames']}f, attack: {h['attack_frames']}f) -> {h['num_chunks']} transition chunks")
            if len(chunked_long_hunts) > 5:
                print(f"    ... (+{len(chunked_long_hunts)-5} more)")
        else:
            print("  [CHUNKED HUNTS] 0 hunts exceeded MAX_HUNT_SEQ_LEN.")
        print("--------------------------------------------\n")

    return valid_sequences, valid_targets

# ==========================================
# 4. Model & Training Utils
# ==========================================

class HuntingBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=0.4)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        logits = self.fc(output)
        return logits

def validate_epoch(model, loader, criterion, device, num_classes=4):
    model.eval()
    val_loss = 0
    all_preds = []
    all_targs = []
    val_batches = 0
    
    with torch.no_grad():
        for seqs, targs, lens in loader:
            if seqs is None: continue
            seqs = seqs.to(device)
            targs = targs.to(device)
            
            logits = model(seqs, lens)
            
            # Calc Loss
            loss = criterion(logits.view(-1, num_classes), targs.view(-1))
            val_loss += loss.item()
            val_batches += 1
            
            # Calc Metrics (Mask padding out)
            probs = torch.softmax(logits, dim=2)
            preds = torch.argmax(probs, dim=2)
            
            # Iterate batch to mask padding correctly
            for i in range(len(lens)):
                length = int(lens[i])
                all_preds.extend(preds[i, :length].cpu().numpy())
                all_targs.extend(targs[i, :length].cpu().numpy())
                
    avg_loss = val_loss / max(1, val_batches)
    
    # Calculate F1 Score with explicit labels to ensure full class consistency
    labels = list(range(num_classes))
    f1 = f1_score(all_targs, all_preds, labels=labels, average='macro', zero_division=0)
    per_class_f1 = f1_score(all_targs, all_preds, labels=labels, average=None, zero_division=0)
    
    return avg_loss, f1, per_class_f1


def plot_training_history(history, paradigm="mouse", num_classes=None, save_dir=None):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(20, 5)) # Wider figure for 4 plots
    
    if num_classes is None:
        num_classes = 5 if paradigm == "mouse" else 4

    if num_classes == 5:
        classes = ['Background', 'Cricket Present', 'First Approach', 'Final Approach', 'Eating']
    elif paradigm == "mouse":
        classes = ['Background', 'First Approach', 'Final Approach', 'Eating']
    else:
        classes = ['Background', 'Stalking', 'Pursuit', 'Eating']
    
    # 1. Loss
    plt.subplot(1, 4, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r--', label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. F1 Score
    plt.subplot(1, 4, 2)
    plt.plot(epochs, history['val_f1'], 'g-', label='Val F1 (Macro)')
    plt.title('Validation F1')
    plt.xlabel('Epochs')
    plt.grid(True, alpha=0.3)
    
    # 3. Per Class F1
    plt.subplot(1, 4, 3)
    per_class = np.array(history['val_per_class_f1'])
    for i in range(num_classes):
        if i < per_class.shape[1]:
            plt.plot(epochs, per_class[:, i], label=classes[i])
    plt.title(f'Per-Class F1 ({paradigm.capitalize()})')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Gradient Norms
    plt.subplot(1, 4, 4)
    plt.plot(epochs, history['grad_norm_avg'], 'purple', label='Avg Grad Norm')
    plt.plot(epochs, history['grad_norm_max'], 'orange', linestyle='--', label='Max Grad Norm')
    plt.title('Gradient Norms')
    plt.xlabel('Epochs')
    plt.ylabel('Norm')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=200, bbox_inches='tight')
    plt.show()


def evaluate_and_plot(model, loader, device, num_plots=5, save_path=None, paradigm="mouse", num_classes=None):
    """
    Plots model predictions vs ground truth for Hunting Model.
    Mouse 5-Class: 0=Background, 1=Cricket Present, 2=First Approach, 3=Final Approach, 4=Eating
    Rat 4-Class:   0=Background, 1=Stalking, 2=Pursuit, 3=Eating
    """
    model.eval()
    found_plots = 0
    
    if num_classes is None:
        num_classes = 5 if paradigm == "mouse" else 4

    # Visualization Configuration
    if num_classes == 5:
        class_names = ['Background', 'Cricket Present', 'First Approach', 'Final Approach', 'Eating']
        class_colors = ['gray', 'deepskyblue', 'orange', 'red', 'green']
    elif paradigm == "mouse":
        class_names = ['Background', 'First Approach', 'Final Approach', 'Eating']
        class_colors = ['gray', 'orange', 'red', 'green']
    else:
        class_names = ['Background', 'Stalking', 'Pursuit', 'Eating']
        class_colors = ['gray', 'blue', 'orange', 'green']
    
    print(f"Generating {num_plots} validation plots...")
    
    with torch.no_grad():
        for seqs, targs, lens in loader:
            if found_plots >= num_plots:
                break
            
            # Move to GPU
            seqs = seqs.to(device)
            targs = targs.to(device)
            
            # Inference
            logits = model(seqs, lens) # Shape: [Batch, SeqLen, num_classes]
            probs = torch.softmax(logits, dim=2) # Probabilities (0-1)
            preds = torch.argmax(probs, dim=2)   # Hard Class Predictions
            
            # Move to CPU for plotting
            probs = probs.cpu().numpy()
            preds = preds.cpu().numpy()
            targs = targs.cpu().numpy()
            lens = lens.cpu().numpy()
            
            batch_size = seqs.shape[0]
            
            for i in range(batch_size):
                if found_plots >= num_plots:
                    break
                
                length = int(lens[i])
                if length < 50: continue # Skip tiny sequences/padding artifacts
                
                # Slice data to actual sequence length (remove padding)
                curr_probs = probs[i, :length, :]
                curr_preds = preds[i, :length]
                curr_targs = targs[i, :length]
                
                # Create 2-panel Plot
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
                
                frames = np.arange(length)
                
                # --- PANEL 1: PROBABILITIES ---
                for cls_idx in range(num_classes):
                    ax1.plot(frames, curr_probs[:, cls_idx], 
                             label=class_names[cls_idx], 
                             color=class_colors[cls_idx], 
                             linewidth=2, alpha=0.8)
                
                ax1.set_title(f"Example {found_plots + 1}: Model Confidence ({paradigm.capitalize()} Paradigm)")
                ax1.set_ylabel("Probability")
                ax1.legend(loc='upper right', framealpha=0.9)
                ax1.grid(True, alpha=0.3)
                
                # --- PANEL 2: SEGMENTATION (PRED VS TRUTH) ---
                # Plot Ground Truth
                ax2.step(frames, curr_targs, where='post', label='Ground Truth', 
                         color='black', linewidth=3, alpha=0.6)
                
                # Plot Prediction
                ax2.step(frames, curr_preds, where='post', label='Prediction', 
                         color='blue', linestyle='--', linewidth=2)
                
                # Formatting
                ax2.set_yticks(list(range(num_classes)))
                ax2.set_yticklabels(class_names)
                ax2.set_xlabel("Frame Index (Includes Pre/Post Buffers)")
                ax2.set_ylabel("Behavior Class")
                ax2.legend(loc='upper left', framealpha=0.9)
                ax2.grid(True, alpha=0.3)
                
                # Highlight active terminal strike/pursuit region in Truth for visual anchoring
                active_class_idx = 3 if num_classes == 5 else 2
                active_mask = (curr_targs == active_class_idx)
                if np.any(active_mask):
                     ax2.fill_between(frames, -0.5, num_classes - 0.5, where=active_mask, 
                                      color='red', alpha=0.1, label=f'True {class_names[active_class_idx]} Zone')

                plt.tight_layout()
                if save_path:
                    os.makedirs(save_path, exist_ok=True)
                    plot_filename = os.path.join(save_path, f"eval_plot_{found_plots + 1}.png")
                    plt.savefig(plot_filename, dpi=200, bbox_inches='tight')
                    plt.close(fig)
                else:
                    plt.show()
                
                found_plots += 1


def generate_verification_plots(
    paired_files: List[Dict[str, Any]],
    target_fps: float = 30.0,
    save_dir: Optional[str] = None,
    num_example_videos: int = 2,
    verbose: bool = True
):
    """
    Generates pre-training verification plots comparing manual 5-class ground truth
    behavioral annotations against extracted features:
      1) Mouse speed overlayed with 5-class behavior steps
      2) 3-point distal tail angle overlayed with 5-class behavior steps
    Evaluates across num_example_videos (covering all hunts in each video).
    """
    if not paired_files:
        return

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    class_names = ['Background', 'Cricket Present', 'First App', 'Final App', 'Eating']
    class_colors = {
        0: ('#808080', 0.0),   # Background (no fill)
        1: ('#00bfff', 0.18),  # Cricket Present (cyan)
        2: ('#ffa500', 0.22),  # First Approach (orange)
        3: ('#ff3333', 0.30),  # Final Approach (red)
        4: ('#228b22', 0.18),  # Eating (forest green)
    }

    selected_pairs = paired_files[:num_example_videos]
    if verbose:
        print(f"\n[VERIFICATION] Generating pre-training diagnostic plots for {len(selected_pairs)} example videos...")

    for vid_idx, pair in enumerate(selected_pairs):
        try:
            dlc_path = pair['dlc_path']
            video_path = pair.get('video_path')
            video_width, video_height, video_fps = dataset_utils.get_video_params(
                dlc_path, video_path=video_path, verbose=False
            )
            pose_data = dataset_utils.load_mouse_dlc(dlc_path, video_fps)
            if pose_data is None:
                continue

            features, manifest = dataset_utils.generate_mouse_features(pose_data, video_width, video_fps)
            feat_names = manifest.feature_names

            # Standardize FPS if needed
            if abs(video_fps - target_fps) > 0.1:
                features = dataset_utils.resample_feature_matrix(features, video_fps, target_fps)
                pair_hunts = dataset_utils.scale_hunt_events_fps(pair['hunts'], video_fps, target_fps)
            else:
                pair_hunts = pair['hunts']

            total_frames = len(features)
            speed_idx = feat_names.index("predator_speed") if "predator_speed" in feat_names else None
            tail_candidates = ["distal_3pt_tail_angle", "distal_3pt_angle", "distal_tail_angle", "tail_tip_speed"]
            tail_idx = None
            for tn in tail_candidates:
                if tn in feat_names:
                    tail_idx = feat_names.index(tn)
                    break

            if tail_idx is None and verbose:
                print(f"  [VERIFICATION WARNING] Tail angle feature not found among {tail_candidates} in manifest features.")

            # Filter valid hunts
            valid_hunts = []
            for h in pair_hunts:
                enter = h['enter']
                capture = h['capture']
                pursuit = h['pursuit_start']
                first_app = h.get('first_approach') or enter
                cons_end = h.get('consumption_end') or (capture + int(15 * target_fps))
                if enter >= capture or pursuit > capture:
                    continue
                first_app = max(enter, min(pursuit, first_app))
                valid_hunts.append({
                    'enter': enter,
                    'first_approach': first_app,
                    'pursuit_start': pursuit,
                    'capture': capture,
                    'consumption_end': cons_end
                })

            first_dlc = dlc_path[0] if isinstance(dlc_path, (list, tuple)) else dlc_path
            if not valid_hunts:
                if verbose:
                    print(f"  [VERIFICATION] Video {vid_idx+1} ({os.path.basename(first_dlc)}) has no valid hunts.")
                continue

            pair_label = pair.get('id') or os.path.splitext(os.path.basename(first_dlc))[0]
            if len(pair_label) > 35:
                pair_label = pair_label[:35] + "..."

            buffer = int(2.0 * target_fps) # 2-second pre- and post-buffer
            n_hunts = len(valid_hunts)

            # Create Speed Figure and Tail Angle Figure
            fig_specs = [
                ("Mouse Speed", speed_idx, "Speed (normalized)", "royalblue", "speed_overlay"),
                ("Distal 3-pt Tail Angle", tail_idx, "Angle (deg / 180)", "darkviolet", "tail_angle_overlay"),
            ]

            for feat_title, feat_col, y_label, trace_color, file_suffix in fig_specs:
                if feat_col is None:
                    if verbose:
                        print(f"  [VERIFICATION WARNING] Skipping '{feat_title}' plot (feature not found in manifest).")
                    continue

                fig, axes = plt.subplots(n_hunts, 1, figsize=(14, max(4, 2.8 * n_hunts)), sharex=False, squeeze=False)
                fig.suptitle(
                    f"Pre-Training Verification: {feat_title} vs 5-Class Manual Behavior\nSession {vid_idx+1}: {pair_label} ({n_hunts} Hunts)",
                    fontsize=12, fontweight='bold', y=0.99
                )

                for h_idx, hunt in enumerate(valid_hunts):
                    ax = axes[h_idx, 0]
                    h_start = max(0, hunt['enter'] - buffer)
                    h_end = min(total_frames, hunt['consumption_end'] + buffer)
                    h_len = h_end - h_start
                    if h_len < 5:
                        continue

                    feat_trace = features[h_start:h_end, feat_col]
                    frames = np.arange(h_len)
                    time_sec = frames / target_fps

                    # Ground truth 5-class target construction
                    h_targets = np.zeros(h_len, dtype=int)
                    local_enter = max(0, hunt['enter'] - h_start)
                    local_first = max(0, min(h_len, hunt['first_approach'] - h_start))
                    local_pursuit = max(0, min(h_len, hunt['pursuit_start'] - h_start))
                    local_capture = max(0, min(h_len, hunt['capture'] - h_start))
                    local_cons = max(0, min(h_len, hunt['consumption_end'] - h_start))

                    if local_first > local_enter:
                        h_targets[local_enter:local_first] = 1
                    if local_pursuit > local_first:
                        h_targets[local_first:local_pursuit] = 2
                    if local_capture > local_pursuit:
                        h_targets[local_pursuit:local_capture] = 3
                    if local_cons > local_capture:
                        h_targets[local_capture:local_cons] = 4

                    # Left axis: Feature trace
                    ax.plot(time_sec, feat_trace, color=trace_color, linewidth=1.8, label=feat_title)
                    ax.set_ylabel(y_label, color=trace_color, fontsize=9)
                    ax.tick_params(axis='y', labelcolor=trace_color)
                    ax.set_xlabel("Time from hunt start (s)", fontsize=9)
                    ax.grid(True, alpha=0.25)

                    # Right axis: 5-Class Manual Behavior Step
                    ax2 = ax.twinx()
                    ax2.step(time_sec, h_targets, where='post', color='black', linewidth=2.0, label='Manual 5-Class Label')
                    ax2.set_yticks([0, 1, 2, 3, 4])
                    ax2.set_yticklabels(class_names, fontsize=8)
                    ax2.set_ylabel("Behavior Class", color='black', fontsize=9)
                    ax2.set_ylim(-0.5, 4.5)

                    # Shaded behavior epochs
                    for c_idx, (c_col, c_alpha) in class_colors.items():
                        if c_alpha > 0:
                            mask = (h_targets == c_idx)
                            if np.any(mask):
                                ax2.fill_between(time_sec, -0.5, 4.5, where=mask, color=c_col, alpha=c_alpha)

                    ax.set_title(
                        f"Hunt {h_idx+1}: Frames {hunt['enter']} - {hunt['consumption_end']} (Duration: {(hunt['consumption_end']-hunt['enter'])/target_fps:.1f}s)",
                        fontsize=10, loc='left'
                    )

                plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.96))
                if save_dir:
                    out_fname = os.path.join(save_dir, f"verification_session_{vid_idx+1}_{file_suffix}.png")
                    plt.savefig(out_fname, dpi=200, bbox_inches='tight')
                    plt.close(fig)
                    if verbose:
                        print(f"  [VERIFICATION] Saved: {out_fname}")
                else:
                    plt.show()

        except Exception as e:
            if verbose:
                print(f"  [VERIFICATION] Error plotting verification session {vid_idx+1}: {e}")

    if verbose and save_dir:
        print(f"[VERIFICATION] Pre-training verification plots saved to: {save_dir}\n")


# ==========================================
# 5. Main Execution
# ==========================================

def main(
    paradigm: str = "rat",
    target_fps: float = 30.0,
    datasets_config: Optional[Dict[str, Any]] = None,
    diane_excel_path: Optional[str] = None,
    diane_dlc_dir: Optional[str] = None,
    diane_video_dir: Optional[str] = None,
    diane_format: str = "diane",
    tentative_approach_strategy: str = "engagement",
    recursive: Optional[bool] = None,
    verbose: bool = True,
    pretrained_weights_path: Optional[str] = None,
    model_save_dir: Optional[str] = None,
    continuous_cricket_confidence: bool = False
):
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cap VRAM usage to ~80% to allow for video playback/other tasks without throttling
    if device.type == 'cuda':
        try:
            torch.cuda.set_per_process_memory_fraction(0.8, device=0)
        except Exception as e:
            if VERBOSE: print(f"Could not cap VRAM: {e}")
            
    if verbose: print(f"Training on: {device} (Paradigm: {paradigm}, Canonical Target FPS: {target_fps}, Continuous Cricket: {continuous_cricket_confidence})")
    
    if datasets_config is not None:
        if recursive is not None:
            for cfg in datasets_config.values():
                cfg['recursive'] = recursive
        pairs = dataset_utils.pair_dataset_files(datasets_config=datasets_config, verbose=verbose)
    elif diane_excel_path and diane_dlc_dir:
        pairs = dataset_utils.pair_diane_dataset(
            master_excel_path=diane_excel_path,
            dlc_dir=diane_dlc_dir,
            video_dir=diane_video_dir,
            tentative_approach_strategy=tentative_approach_strategy,
            recursive=bool(recursive),
            split_parts=(diane_format == 'diane_parts'),
            verbose=verbose
        )
    elif paradigm == "mouse":
        if recursive is not None:
            for cfg in dataset_utils.MOUSE_DATASETS.values():
                cfg['recursive'] = recursive
        pairs = dataset_utils.pair_dataset_files(datasets_config=dataset_utils.MOUSE_DATASETS, verbose=verbose)
    else:
        if recursive is not None:
            for cfg in dataset_utils.DATASETS.values():
                cfg['recursive'] = recursive
        pairs = dataset_utils.pair_dataset_files(datasets_config=dataset_utils.DATASETS, verbose=verbose)
    if not pairs: return

    # Pre-training verification plots when verbose=True (Mouse paradigm: Speed and 3-pt Tail Angle vs 5-class manual labels)
    if verbose and paradigm == "mouse":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        verif_dir = os.path.join(script_dir, 'figs', 'verification_plots')
        generate_verification_plots(pairs, target_fps=target_fps, save_dir=verif_dir, num_example_videos=2, verbose=verbose)

    # 2. Data (Standardized to target_fps across mixed 20/30 FPS videos)
    X, y = create_dataset_sequences(pairs, paradigm=paradigm, target_fps=target_fps, continuous_cricket_confidence=continuous_cricket_confidence)
    if len(X) == 0: return

    # Print Stats
    lengths = [len(seq) for seq in X]
    feat_dim = X[0].shape[1]
    print(f"\n--- Sequence Length Statistics ---")
    print(f"Total Sequences: {len(lengths)}")
    print(f"Min: {min(lengths)}, Max: {max(lengths)}")
    print(f"Avg: {np.mean(lengths):.1f}")
    print(f"Feature Dimension: {feat_dim}")
    print(f"Standardized Target FPS: {target_fps}")
    print(f"----------------------------------\n")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    bg_ct = sum(1 for i in y_train if np.sum(i) == 0)
    print(f'BG X prop: {bg_ct/len(y_train):.3f}')

    class HuntDataset(Dataset):
        def __init__(self, seqs, targs): self.seqs, self.targs = seqs, targs
        def __len__(self): return len(self.seqs)
        def __getitem__(self, index): return torch.tensor(self.seqs[index], dtype=torch.float32), torch.tensor(self.targs[index], dtype=torch.long)

    def collate(batch):
        batch = [b for b in batch if len(b[0]) > 0 and len(b[1]) > 0]
        if not batch:
            return None, None, None
        seqs, targs = zip(*batch)
        lens = torch.tensor([len(s) for s in seqs])
        seqs_pad = pad_sequence(list(seqs), batch_first=True, padding_value=0)
        targs_pad = pad_sequence(list(targs), batch_first=True, padding_value=-100) 
        return seqs_pad, targs_pad, lens

    train_loader = DataLoader(HuntDataset(X_train, y_train), batch_size=12, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(HuntDataset(X_test, y_test), batch_size=12, shuffle=False, collate_fn=collate)

    # 3. Model
    num_classes = 5 if paradigm == "mouse" else 4
    h_dim = 256
    num_layers = 3
    model = HuntingBiLSTM(input_dim=feat_dim, hidden_dim=h_dim, num_layers=num_layers, num_classes=num_classes).to(device) 
    
    def _load_model_weights(target_model, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            target_model.load_state_dict(ckpt['model_state_dict'])
            if 'target_fps' in ckpt and VERBOSE:
                print(f"Loaded checkpoint trained with target_fps={ckpt['target_fps']}")
        else:
            target_model.load_state_dict(ckpt)

    # Load pretrained weights if requested and available
    if pretrained_weights_path and os.path.exists(pretrained_weights_path):
        try:
            _load_model_weights(model, pretrained_weights_path)
            if VERBOSE: print(f"Loaded pretrained weights from: {pretrained_weights_path}")
        except Exception as e:
            if VERBOSE: print(f"Could not load pretrained weights ({e}), initializing fresh model.")
    elif paradigm == "rat" and feat_dim == 22:
        default_rat_ckpt = r"C:\Users\bcary\Documents\Work\CODE\CricketCode\LSTM_hunt_label\models\dan+lucas_0.15chunk2iters\BGadd_feat22_dim256_lay3_ep38_valf1=0.8270.pth"
        if os.path.exists(default_rat_ckpt):
            try:
                _load_model_weights(model, default_rat_ckpt)
                if VERBOSE: print(f"Loaded default rat checkpoint: {default_rat_ckpt}")
            except Exception as e:
                if VERBOSE: print(f"Could not load default rat checkpoint: {e}")

    # Loss Weights
    if paradigm == "mouse":
        # 5 Classes: [0: Background, 1: Cricket Present, 2: First Approach, 3: Final Approach, 4: Eating]
        weights = torch.tensor([0.2, 1.5, 1.5, 2.5, 0.4], dtype=torch.float32).to(device)
    else:
        # 4 Classes: [0: Background, 1: Stalking, 2: Pursuit, 3: Eating]
        weights = torch.tensor([0.25, 1.5, 2.0, 0.4], dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-100) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    
    # Scheduler: If validation F1 doesn't improve for n epochs, cut LR in half.
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='max', factor=0.5, patience=4, verbose=True
    # )
    # Initial cycle of 15 epochs, doubling each time (15 -> 30 -> 60 epochs), floor at 1e-5:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-5
    )

    # 4. Training Loop with History
    history = {
        'train_loss': [], 'val_loss': [], 'val_f1': [], 
        'val_per_class_f1': [], 'lr': [],
        'grad_norm_avg': [], 'grad_norm_max': []
    }
    best_val_f1 = 0.0

    num_epochs = 200
    if VERBOSE: print("Starting training...")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        epoch_grad_norms = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not VERBOSE)
        
        for seqs, targs, lens in pbar:
            if seqs is None: continue
            seqs, targs = seqs.to(device), targs.to(device)
            optimizer.zero_grad()
            logits = model(seqs, lens)
            loss = criterion(logits.view(-1, num_classes), targs.view(-1))
            loss.backward()

            # --- TRACKING GRADIENTS ---
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            epoch_grad_norms.append(grad_norm.item())

            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        # End of Epoch Validation
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss, val_f1, val_per_class = validate_epoch(model, val_loader, criterion, device, num_classes=num_classes)
        
        # scheduler.step(val_f1)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = copy.deepcopy(model)

            # Save Model with rich metadata payload
            save_path_dir = model_save_dir or os.path.join("models", f"{paradigm}_models")
            os.makedirs(save_path_dir, exist_ok=True)
            model_save_name = f'HuntLSTM_{paradigm}_feat{feat_dim}_dim{h_dim}_lay{num_layers}_ep{epoch}_valf1={best_val_f1:.4f}.pth'
            checkpoint_payload = {
                'model_state_dict': best_model.state_dict(),
                'target_fps': target_fps,
                'feat_dim': feat_dim,
                'paradigm': paradigm,
                'num_classes': num_classes,
                'hidden_dim': h_dim,
                'num_layers': num_layers,
                'best_val_f1': best_val_f1,
                'epoch': epoch,
                'continuous_cricket_confidence': continuous_cricket_confidence,
            }
            torch.save(checkpoint_payload, os.path.join(save_path_dir, model_save_name))
            tqdm.write(f"--> New Best Model Saved! (F1: {best_val_f1:.4f}, FPS: {target_fps})")

        # Update History
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(val_f1)
        history['val_per_class_f1'].append(val_per_class)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Gradient Stats
        history['grad_norm_avg'].append(np.mean(epoch_grad_norms))
        history['grad_norm_max'].append(np.max(epoch_grad_norms))

        if VERBOSE:
            print(f"Epoch {epoch+1}: Training Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val F1={val_f1:.4f}")
            print(f"Grad Norm -> Avg: {np.mean(epoch_grad_norms):.2f} | Max: {np.max(epoch_grad_norms):.2f}")
            
            if current_lr:
                print(f'LR: {current_lr:.6f}')

            script_dir = os.path.dirname(os.path.abspath(__file__))
            training_val_dir = os.path.join(script_dir, 'figs', 'training_val_plots')
            new_path = os.path.join(training_val_dir, f'epoch_{epoch+1}')
            os.makedirs(new_path, exist_ok=True)
            evaluate_and_plot(model, val_loader, device, num_plots=20, save_path=new_path, paradigm=paradigm, num_classes=num_classes)

    # 5. Visualize History
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figs_dir = os.path.join(script_dir, 'figs')
    os.makedirs(figs_dir, exist_ok=True)
    plot_training_history(history, paradigm=paradigm, num_classes=num_classes, save_dir=figs_dir)

if __name__ == "__main__":
    import sys
    import argparse

    # If launched without CLI flags (e.g. F5 in VS Code debugger), default to mouse training
    if len(sys.argv) == 1:
        print("No CLI arguments provided. Running default mouse paradigm (verbose=True)...")
        main(paradigm="mouse", verbose=True, 
            pretrained_weights_path=r"C:\Users\bcary\Documents\Work\CODE\CricketCode\LSTM_hunt_label_mouse\models\mouse_models\HuntLSTM_mouse_feat45_dim256_lay3_ep45_valf1=0.7169.pth")
        # main(paradigm="mouse", verbose=True)
    
    else:
        parser = argparse.ArgumentParser(description="Train LSTM hunting behavior classifier")
        parser.add_argument("--paradigm", type=str, default="mouse", choices=["rat", "mouse"], help="Paradigm: rat or mouse")
        parser.add_argument("--target_fps", type=float, default=30.0, help="Canonical target FPS to standardize all sequences to (default: 30.0)")
        parser.add_argument("--excel", type=str, default=None, help="Path to Diane master Excel sheet")
        parser.add_argument("--dlc_dir", type=str, default=None, help="Path to directory of mouse DLC pose CSVs")
        parser.add_argument("--video_dir", type=str, default=None, help="Path to directory of video files (if different)")
        parser.add_argument("--strategy", type=str, default="engagement", choices=["engagement", "continuous", "final_only", "split"], help="Tentative approach labeling strategy")
        parser.add_argument("--diane_format", type=str, default="diane", choices=["diane", "diane_parts"], help="Diane format variant: 'diane' (continuous/stitched) or 'diane_parts' (split videos with resetting frame counts)")
        parser.add_argument("--recursive", action="store_true", default=None, help="Force recursive search across DLC subdirectories")
        parser.add_argument("--no-recursive", action="store_false", dest="recursive", help="Disable recursive search across subdirectories")
        parser.add_argument("-v", "--verbose", action="store_true", default=True, help="Enable verbose diagnostic logging")
        parser.add_argument("-q", "--quiet", action="store_false", dest="verbose", help="Disable verbose diagnostic logging")
        parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained model checkpoint")
        parser.add_argument("--save_dir", type=str, default=None, help="Directory to save model weights")
        parser.add_argument("--continuous_cricket", "--continuous-cricket", action="store_true", default=False, help="Use continuous DLC cricket confidence [0, 1] instead of hard binary cutoff")
        args = parser.parse_args()

        main(
            paradigm=args.paradigm,
            target_fps=args.target_fps,
            diane_excel_path=args.excel,
            diane_dlc_dir=args.dlc_dir,
            diane_video_dir=args.video_dir,
            diane_format=args.diane_format,
            tentative_approach_strategy=args.strategy,
            recursive=args.recursive,
            verbose=args.verbose,
            pretrained_weights_path=args.pretrained,
            model_save_dir=args.save_dir,
            continuous_cricket_confidence=args.continuous_cricket
        )

    
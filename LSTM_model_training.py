import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
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

def create_dataset_sequences(paired_files):
    all_sequences = []
    all_targets = []
    
    if VERBOSE: print(f"Processing {len(paired_files)} files (Balanced Mode)...")
    iterator = tqdm(paired_files, desc="Processing Videos", unit="vid", disable=not VERBOSE)
    
    for pair in iterator:
        try:
            video_width, _, video_fps = dataset_utils.get_video_params(pair['dlc_path'])
            
            processed_data = dataset_utils.load_and_process_dlc(pair['dlc_path'], video_fps)
            if processed_data is None: continue
            
            dlc_df = processed_data
            
            rat_head, rat_mid, rat_butt, rat_tail, cricket_pos = dlc_df
            full_features = dataset_utils.generate_features(rat_head, rat_mid, rat_butt, rat_tail, cricket_pos, video_width, video_fps)
            total_frames = len(full_features)
            
            # Decide if this video is fully chunked
            if random.random() < dataset_utils.FULL_VIDEO_CHUNK_PROPORTION:
                for _ in range(dataset_utils.FULL_VIDEO_CHUNK_ITERATIONS):
                    current_idx = 0
                    while current_idx <= total_frames - dataset_utils.FULL_VIDEO_CHUNK_MIN:
                        chunk_size = random.randint(dataset_utils.FULL_VIDEO_CHUNK_MIN, dataset_utils.FULL_VIDEO_CHUNK_MAX)
                        end_idx = min(total_frames, current_idx + chunk_size)
                        
                        if (end_idx - current_idx) >= dataset_utils.FULL_VIDEO_CHUNK_MIN:
                            seq_features = full_features[current_idx:end_idx]
                            seq_len = len(seq_features)
                            
                            seq_targets = np.zeros(seq_len, dtype=int)
                            
                            for hunt_data in pair['hunts']:
                                enter = hunt_data['enter']
                                capture = hunt_data['capture']
                                pursuit_start = hunt_data['pursuit_start']
                                cons_end = hunt_data['consumption_end']
                                
                                if enter >= capture or pursuit_start > capture: continue
                                if cons_end is not None:
                                    eating_end_frame = cons_end
                                else:
                                    eating_end_frame = capture + dataset_utils.EATING_BUFFER
                                    
                                local_enter = enter - current_idx
                                local_pursuit = pursuit_start - current_idx
                                local_capture = capture - current_idx
                                local_eating_end = eating_end_frame - current_idx
                                
                                if local_pursuit > max(0, local_enter) and local_enter < seq_len:
                                    seq_targets[max(0, local_enter):min(seq_len, local_pursuit)] = 1
                                if local_capture > max(0, local_pursuit) and local_pursuit < seq_len:
                                    seq_targets[max(0, local_pursuit):min(seq_len, local_capture)] = 2
                                if local_eating_end > max(0, local_capture) and local_capture < seq_len:
                                    seq_targets[max(0, local_capture):min(seq_len, local_eating_end)] = 3
                                    
                            all_sequences.append(seq_features)
                            all_targets.append(seq_targets)
                            
                        current_idx = end_idx
                continue
            
            # --- BG CHUNK LIMITER ---
            # We want background, but not infinite background.
            bg_chunks_generated = 0
            
            last_event_end = 0 
            
            for hunt_data in pair['hunts']:
                enter = hunt_data['enter']
                capture = hunt_data['capture']
                pursuit_start = hunt_data['pursuit_start']
                cons_end = hunt_data['consumption_end']
                
                # 1. GAP HANDLING (With Limits)
                gap_start = last_event_end
                gap_end = max(0, enter - dataset_utils.CONTEXT_BUFFER)
                
                # Only add background if we haven't hit the limit for this video
                if (gap_end - gap_start) > dataset_utils.MIN_BG_LEN and bg_chunks_generated < dataset_utils.MAX_BG_CHUNKS_PER_VIDEO:
                    
                    # Instead of slicing the whole gap, take random chunks from it
                    # This ensures we get variety without bloating the dataset
                    
                    # Attempt to grab up to 2 chunks from this specific gap
                    gap_attempts = 0
                    current_idx = gap_start
                    
                    while current_idx < (gap_end - dataset_utils.MIN_BG_LEN) and gap_attempts < 2:
                        chunk_size = random.randint(dataset_utils.MIN_BG_LEN, dataset_utils.MAX_BG_LEN)
                        end_idx = min(gap_end, current_idx + chunk_size)
                        
                        if (end_idx - current_idx) >= dataset_utils.MIN_BG_LEN:
                            all_sequences.append(full_features[current_idx:end_idx])
                            
                            targ_arr = np.zeros((end_idx - current_idx,), dtype=int)
                            all_targets.append(targ_arr)
                            bg_chunks_generated += 1
                            gap_attempts += 1
                            
                        current_idx = end_idx + random.randint(100, 300) # Jump forward

                # 2. HUNT HANDLING (Same as before)
                if enter >= capture or pursuit_start > capture: continue
                
                rand_buffer = random.randint(50, 400) # Reduced buffer variation slightly
                
                if cons_end is not None:
                    eating_end_frame = cons_end
                else:
                    eating_end_frame = capture + dataset_utils.EATING_BUFFER
                
                seq_start = max(0, enter - rand_buffer)
                seq_end = min(total_frames, eating_end_frame + rand_buffer)
                
                seq_features = full_features[seq_start:seq_end]
                seq_len = len(seq_features)
                if seq_len < 10: continue 
                
                seq_targets = np.zeros(seq_len, dtype=int)
                local_enter = enter - seq_start
                local_pursuit = pursuit_start - seq_start
                local_capture = capture - seq_start
                local_eating_end = eating_end_frame - seq_start
                
                if local_pursuit > local_enter:
                    seq_targets[max(0, local_enter):min(seq_len, local_pursuit)] = 1
                if local_capture > local_pursuit:
                    seq_targets[max(0, local_pursuit):min(seq_len, local_capture)] = 2
                if local_eating_end > local_capture:
                    seq_targets[max(0, local_capture):min(seq_len, local_eating_end)] = 3
                    
                all_sequences.append(seq_features)
                all_targets.append(seq_targets)
                
                last_event_end = seq_end
                
        except Exception as e:
            if VERBOSE: tqdm.write(f"Error: {e}")
            continue
            
    return all_sequences, all_targets

# ==========================================
# 4. Model & Training Utils
# ==========================================

class HuntingBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=0.4)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        logits = self.fc(output)
        return logits

def validate_epoch(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    all_preds = []
    all_targs = []
    
    with torch.no_grad():
        for seqs, targs, lens in loader:
            seqs = seqs.to(device)
            targs = targs.to(device)
            
            logits = model(seqs, lens)
            
            # Calc Loss
            loss = criterion(logits.view(-1, 4), targs.view(-1))
            val_loss += loss.item()
            
            # Calc Metrics (Mask padding out)
            probs = torch.softmax(logits, dim=2)
            preds = torch.argmax(probs, dim=2)
            
            # Iterate batch to mask padding correctly
            for i in range(len(lens)):
                length = int(lens[i])
                all_preds.extend(preds[i, :length].cpu().numpy())
                all_targs.extend(targs[i, :length].cpu().numpy())
                
    avg_loss = val_loss / len(loader)
    
    # Calculate F1 Score (Macro avg handles class imbalance)
    f1 = f1_score(all_targs, all_preds, average='macro', zero_division=0)
    
    # Calculate per-class accuracy/recall roughly using confusion matrix
    # or just rely on f1_score with average=None
    per_class_f1 = f1_score(all_targs, all_preds, average=None, zero_division=0)
    
    return avg_loss, f1, per_class_f1


def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(20, 5)) # Wider figure for 4 plots
    
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
    classes = ['Background', 'Stalking', 'Pursuit', 'Eating']
    for i in range(4):
        if i < per_class.shape[1]:
            plt.plot(epochs, per_class[:, i], label=classes[i])
    plt.title('Per-Class F1')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Gradient Norms (NEW)
    plt.subplot(1, 4, 4)
    plt.plot(epochs, history['grad_norm_avg'], 'purple', label='Avg Grad Norm')
    plt.plot(epochs, history['grad_norm_max'], 'orange', linestyle='--', label='Max Grad Norm')
    plt.title('Gradient Norms')
    plt.xlabel('Epochs')
    plt.ylabel('Norm')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def evaluate_and_plot(model, loader, device, num_plots=5, save_path=None):
    """
    Plots model predictions vs ground truth for 4-Class Rat Hunting Model.
    Classes: 0=Background, 1=Stalking, 2=Pursuit, 3=Eating
    """
    model.eval()
    
    found_plots = 0
    
    # Visualization Configuration
    class_names = ['Background', 'Stalking', 'Pursuit', 'Eating']
    class_colors = ['gray', 'orange', 'red', 'green']
    
    print(f"Generating {num_plots} validation plots...")
    
    with torch.no_grad():
        for seqs, targs, lens in loader:
            if found_plots >= num_plots:
                break
            
            # Move to GPU
            seqs = seqs.to(device)
            targs = targs.to(device)
            
            # Inference
            logits = model(seqs, lens) # Shape: [Batch, SeqLen, 4]
            probs = torch.softmax(logits, dim=2) # Probabilities (0-1)
            preds = torch.argmax(probs, dim=2)   # Hard Class Predictions (0,1,2,3)
            
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
                for cls_idx in range(4):
                    ax1.plot(frames, curr_probs[:, cls_idx], 
                             label=class_names[cls_idx], 
                             color=class_colors[cls_idx], 
                             linewidth=2, alpha=0.8)
                
                ax1.set_title(f"Example {found_plots + 1}: Model Confidence")
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
                ax2.set_yticks([0, 1, 2, 3])
                ax2.set_yticklabels(class_names)
                ax2.set_xlabel("Frame Index (Includes Pre/Post Buffers)")
                ax2.set_ylabel("Behavior Class")
                ax2.legend(loc='upper left', framealpha=0.9)
                ax2.grid(True, alpha=0.3)
                
                # Highlight "Pursuit" regions in Truth for visual anchoring
                pursuit_mask = (curr_targs == 2)
                if np.any(pursuit_mask):
                     ax2.fill_between(frames, -0.5, 3.5, where=pursuit_mask, 
                                      color='red', alpha=0.1, label='True Pursuit Zone')

                plt.tight_layout()
                if save_path:
                    os.makedirs(save_path, exist_ok=True)
                    plot_filename = os.path.join(save_path, f"eval_plot_{found_plots + 1}.png")
                    plt.savefig(plot_filename, dpi=200, bbox_inches='tight')
                    plt.close(fig)
                else:
                    plt.show()
                
                found_plots += 1


# ==========================================
# 5. Main Execution
# ==========================================

def main():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cap VRAM usage to ~80% to allow for video playback/other tasks without throttling
    if device.type == 'cuda':
        try:
            torch.cuda.set_per_process_memory_fraction(0.8, device=0)
        except Exception as e:
            if VERBOSE: print(f"Could not cap VRAM: {e}")
            
    if VERBOSE: print(f"Training on: {device}")
    
    pairs = dataset_utils.pair_dataset_files(verbose=VERBOSE)
    if not pairs: return

    # 2. Data
    X, y = create_dataset_sequences(pairs)
    if len(X) == 0: return

    # Print Stats
    lengths = [len(seq) for seq in X]
    print(f"\n--- Sequence Length Statistics ---")
    print(f"Total Sequences: {len(lengths)}")
    print(f"Min: {min(lengths)}, Max: {max(lengths)}")
    print(f"Avg: {np.mean(lengths):.1f}")
    print(f"----------------------------------\n")

    # # plot a histogram of the lengths
    # plt.hist(lengths, bins=50)
    # plt.xlabel("Sequence Length")
    # plt.ylabel("Frequency")
    # plt.title("Histogram of Sequence Lengths")
    # plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    bg_ct = 0
    for i in y_train:
        if np.sum(i) == 0:
            bg_ct += 1
    print(f'BG X prop: {bg_ct/len(y_train)}')

    class RatDataset(Dataset):
        def __init__(self, seqs, targs): self.seqs, self.targs = seqs, targs
        def __len__(self): return len(self.seqs)
        def __getitem__(self, i): return torch.tensor(self.seqs[i], dtype=torch.float32), torch.tensor(self.targs[i], dtype=torch.long)

    def collate(batch):
        seqs, targs = zip(*batch)
        lens = torch.tensor([len(s) for s in seqs])
        seqs_pad = pad_sequence(seqs, batch_first=True, padding_value=0)
        targs_pad = pad_sequence(targs, batch_first=True, padding_value=-100) 
        return seqs_pad, targs_pad, lens

    train_loader = DataLoader(RatDataset(X_train, y_train), batch_size=12, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(RatDataset(X_test, y_test), batch_size=12, shuffle=False, collate_fn=collate)

    # 3. Model
    feat_dim = 22
    h_dim = 256
    num_layers = 3
    model = HuntingBiLSTM(input_dim=feat_dim, hidden_dim=h_dim, num_layers=num_layers, num_classes=4).to(device) 
    
    # load prev model
    model.load_state_dict(torch.load(r"C:\Users\bcary\Documents\Work\CODE\CricketCode\LSTM_hunt_label\models\dan+lucas_0.15chunk2iters\BGadd_feat22_dim256_lay3_ep38_valf1=0.8270.pth", map_location=device))

    # Weights: [Background, Stalking, Pursuit, Eating]
    weights = torch.tensor([0.25, 1.5, 2, 0.4], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-100) 
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    
    # Scheduler: If validation F1 doesn't improve for n epochs, cut LR in half.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=4, verbose=True
    )

    # 4. Training Loop with History
    history = {
        'train_loss': [], 'val_loss': [], 'val_f1': [], 
        'val_per_class_f1': [], 'lr': [],
        'grad_norm_avg': [], 'grad_norm_max': [] # <--- New Tracking
    }
    best_val_f1 = 0.0

    num_epochs = 200
    if VERBOSE: print("Starting training...")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        epoch_grad_norms = [] # Store all batch norms for this epoch

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not VERBOSE)
        
        for seqs, targs, lens in pbar:
            seqs, targs = seqs.to(device), targs.to(device)
            optimizer.zero_grad()
            logits = model(seqs, lens)
            loss = criterion(logits.view(-1, 4), targs.view(-1))
            loss.backward()

            # --- TRACKING GRADIENTS ---
            # clip_grad_norm_ returns the Total Norm BEFORE clipping.
            # We use a high max_norm (e.g. 10.0) or keep 1.0.
            # Even if we clip at 1.0, it returns what the norm WAS.
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            epoch_grad_norms.append(grad_norm.item())

            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        # End of Epoch Validation
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss, val_f1, val_per_class = validate_epoch(model, val_loader, criterion, device)
        
        # scheduler.step(avg_val_loss)
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # torch.save(model.state_dict(), 'best_model_f1.pth')
            best_model = copy.deepcopy(model)

            # Save Model
            model_save_dir = r'C:\Users\bcary\Documents\Work\CODE\CricketCode\LSTM_hunt_label\models\dan+lucas_0.15chunk2iters'
            model_save_name = f'BGadd_feat{feat_dim}_dim{h_dim}_lay{num_layers}_ep{epoch}_valf1={best_val_f1:.4f}.pth'
            torch.save(best_model.state_dict(), os.path.join(model_save_dir, model_save_name))
            tqdm.write(f"--> New Best Model Saved! (F1: {best_val_f1:.4f})")

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

            training_val_dir = r'C:\Users\bcary\Documents\Work\CODE\CricketCode\LSTM_hunt_label\figs\temp_eval_plots'
            new_path = os.path.join(training_val_dir, f'epoch_{epoch+1}')
            os.makedirs(new_path, exist_ok=True)
            evaluate_and_plot(model, val_loader, device, num_plots=10, save_path=new_path)

    # 5. Visualize History
    plot_training_history(history)
    
    # # 6. Save Model
    # model_save_dir = r'C:\Users\bcary\Documents\Work\CODE\CricketCode\LSTM_hunt_label\models\dan_pv-cre'
    # model_save_name = f'BGadd_feat{feat_dim}_dim{h_dim}_lay{num_layers}_ep{epoch}_valf1={best_val_f1:.4f}.pth'
    # torch.save(best_model.state_dict(), model_save_name)

    # evaluate_and_plot(best_model, val_loader, device, num_plots=20)

    # # load prev model
    # model.load_state_dict(torch.load(r"C:\Users\bcary\Documents\Work\CODE\CricketCode\LSTM_hunt_label\models\dan_pv-cre\BGadd_feat22_dim256_lay3_ep23_valf1=0.9132.pth", map_location=device))

if __name__ == "__main__":
    main()

    
import os
import matplotlib.pyplot as plt
import numpy as np
import dataset_utils
import random

def generate_chunked_iteration(full_features, total_frames, pair):
    chunked_targets_reconstructed = np.zeros(total_frames, dtype=float)
    chunked_targets_reconstructed[:] = np.nan # Use nan so gaps are clear

    current_idx = 0
    chunks_collected = []

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
                    
            # Insert chunk back into reconstructed full array
            chunked_targets_reconstructed[current_idx:end_idx] = seq_targets
            
            chunks_collected.append((current_idx, end_idx))
            
        current_idx = end_idx
        
    return chunked_targets_reconstructed, chunks_collected

def main():
    # 1. Grab first available dataset pair
    pairs = dataset_utils.pair_dataset_files(verbose=False)
    if not pairs:
        print("No paired datasets found.")
        return
        
    pair = None
    for p in pairs:
        if len(p['hunts']) > 0:
            pair = p
            break
            
    if pair is None:
        print("No video with hunts found.")
        return
        
    print(f"Using video: {pair['dlc_path']}")

    # 2. Get basic params
    video_width, _, video_fps = dataset_utils.get_video_params(pair['dlc_path'])
    dlc_df = dataset_utils.load_and_process_dlc(pair['dlc_path'], video_fps)
    if dlc_df is None:
        print("Failed to load DLC for test video.")
        return
        
    rat_head, rat_mid, rat_butt, rat_tail, cricket_pos = dlc_df
    full_features = dataset_utils.generate_features(rat_head, rat_mid, rat_butt, rat_tail, cricket_pos, video_width, video_fps)
    total_frames = len(full_features)

    # 3. Create the 'Full Video Original Labels'
    original_targets = np.zeros(total_frames, dtype=int)
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
            
        if pursuit_start > enter:
            original_targets[max(0, enter):min(total_frames, pursuit_start)] = 1
        if capture > pursuit_start:
            original_targets[max(0, pursuit_start):min(total_frames, capture)] = 2
        if eating_end_frame > capture:
            original_targets[max(0, capture):min(total_frames, eating_end_frame)] = 3


    # 4. Run the chunking logic twice for two iterations
    chunked_targets_1, chunks_collected_1 = generate_chunked_iteration(full_features, total_frames, pair)
    chunked_targets_2, chunks_collected_2 = generate_chunked_iteration(full_features, total_frames, pair)


    # 5. Plot the result
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    ax1.plot(original_targets, label='Original Global Labels', color='blue', linewidth=2)
    ax1.set_title("Global Video Labels")
    ax1.set_ylabel("Class (0-3)")
    ax1.set_yticks([0, 1, 2, 3])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(chunked_targets_1, label='Iteration 1 Chunk Labels', color='orange', linewidth=2, linestyle='--')
    for idx, (start, end) in enumerate(chunks_collected_1):
        ax2.axvline(x=start, color='black', alpha=0.5, linestyle=':')
        if idx % 2 == 0:
            ax2.axvspan(start, end, color='gray', alpha=0.1)
    ax2.set_title("Chunked Video Labels - Iteration 1")
    ax2.set_ylabel("Class (0-3)")
    ax2.set_yticks([0, 1, 2, 3])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.plot(chunked_targets_2, label='Iteration 2 Chunk Labels', color='green', linewidth=2, linestyle='--')
    for idx, (start, end) in enumerate(chunks_collected_2):
        ax3.axvline(x=start, color='black', alpha=0.5, linestyle=':')
        if idx % 2 == 0:
            ax3.axvspan(start, end, color='gray', alpha=0.1)
    ax3.set_title("Chunked Video Labels - Iteration 2")
    ax3.set_ylabel("Class (0-3)")
    ax3.set_yticks([0, 1, 2, 3])
    ax3.set_xlabel("Frame Index")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('chunk_label_verification_iters.png', dpi=150)
    print("Saved plot to chunk_label_verification_iters.png")

    # Verify both iterations
    for it_num, chunked_targets in enumerate([chunked_targets_1, chunked_targets_2], start=1):
        diff = chunked_targets - original_targets
        diff_valid = diff[~np.isnan(chunked_targets)]
        if np.all(diff_valid == 0):
            print(f"SUCCESS: Iteration {it_num} perfectly matches the original global labels.")
        else:
            print(f"ERROR: Mismatches found in Iteration {it_num}.")
            print(f"Number of mismatching frames: {np.count_nonzero(diff_valid)}")

if __name__ == '__main__':
    main()

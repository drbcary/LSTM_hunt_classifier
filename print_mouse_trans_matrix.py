import os
import sys
import argparse
import numpy as np
import pandas as pd

import dataset_utils
from dataset.transition_matrix import calculate_empirical_transition_matrix as modular_transition_matrix
from labels.base_parser import HuntEvent

CLASS_NAMES_MOUSE = [
    "Background (0)",
    "Cricket Present (1)",
    "First Approach (2)",
    "Final Approach (3)",
    "Eating (4)"
]

def format_matrix(mat: np.ndarray, title: str = "Transition Probability Matrix"):
    print(f"\n{'='*75}")
    print(f"  {title}")
    print(f"{'='*75}")
    
    header = f"{'From \\ To':<22}" + "".join([f"{name[:10]:>11}" for name in CLASS_NAMES_MOUSE]) + f"{'Stay Sec (30fps)':>18}"
    print(header)
    print("-" * len(header))
    
    for i, row in enumerate(mat):
        row_str = f"{CLASS_NAMES_MOUSE[i]:<22}"
        for p in row:
            if p >= 0.0001:
                row_str += f"{p:>11.4f}"
            else:
                row_str += f"{'0.0000':>11}"
                
        p_stay = row[i]
        if p_stay < 1.0:
            expected_frames = 1.0 / (1.0 - p_stay)
            expected_sec = expected_frames / 30.0
            row_str += f"{expected_sec:>15.2f}s"
        else:
            row_str += f"{'inf':>16}"
            
        print(row_str)
    print("=" * len(header) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Calculate and inspect empirical transition matrix for mouse dataset.")
    parser.add_argument("--excel", type=str, default=None, help="Path to a specific master Excel file (optional override).")
    parser.add_argument("--dlc_dir", type=str, default=None, help="Path to DLC tracking folder matching the Excel (optional override).")
    parser.add_argument("--strategy", type=str, default="engagement", choices=["engagement", "final_only", "split"], help="Diane parsing strategy.")
    args = parser.parse_args()

    print("\n[1/3] Gathering mouse hunting events from dataset...")
    
    if args.excel and os.path.exists(args.excel):
        print(f"Using specified master Excel: {args.excel}")
        dlc_dir = args.dlc_dir or os.path.dirname(args.excel)
        pairs = dataset_utils.pair_diane_dataset(
            master_excel_path=args.excel,
            dlc_dir=dlc_dir,
            tentative_approach_strategy=args.strategy,
            verbose=True
        )
    else:
        print("Using standard MOUSE_DATASETS configuration from dataset_utils...")
        try:
            pairs = dataset_utils.pair_dataset_files(datasets_config=dataset_utils.MOUSE_DATASETS, verbose=True)
        except Exception as e:
            print(f"Error accessing MOUSE_DATASETS: {e}")
            pairs = []

    if not pairs:
        print("\n[!] No paired mouse video sessions could be loaded.")
        print("    If you are running offline or the research server is disconnected, you can run:")
        print("    cmd /c ..\\LSTM_hunt_label\\.venv\\Scripts\\python.exe print_mouse_trans_matrix.py --excel <path_to_master.xlsx> --dlc_dir <path_to_dlc_csvs>")
        sys.exit(1)

    print(f"\n[2/3] Successfully gathered {len(pairs)} paired session(s). Extracting hunt event objects...")
    all_events = []
    total_hunts = 0
    for p in pairs:
        hunt_events = []
        for h in p.get('hunts', []):
            if isinstance(h, HuntEvent):
                hunt_events.append(h)
            elif isinstance(h, dict):
                enter_val = h.get('enter')
                capture_val = h.get('capture')
                if enter_val is not None and capture_val is not None:
                    app_val = h.get('pursuit_start') or h.get('approach_start') or enter_val
                    hunt_events.append(HuntEvent(
                        enter=int(enter_val),
                        approach_start=int(app_val),
                        capture=int(capture_val),
                        consumption_end=int(h['consumption_end']) if h.get('consumption_end') is not None else None,
                        first_approach=int(h['first_approach']) if h.get('first_approach') is not None else None
                    ))
        all_events.append(hunt_events)
        total_hunts += len(hunt_events)

    print(f"Total labeled hunts across all sessions: {total_hunts}")

    print("\n[3/3] Computing empirical transition matrix...")
    trans_mat = modular_transition_matrix(
        all_events=all_events,
        fps=30.0,
        num_classes=5,
        paradigm="mouse"
    )

    format_matrix(trans_mat, title="Empirical Mouse 5-Class Transition Matrix (With Persistence Floor)")
    
    # Also show what it produces with transition_weight = 5.0 (as used during inference)
    from MouseHunt_Inference import viterbi_decode
    tw = 5.0
    weighted_mat = trans_mat.copy()
    for i in range(5):
        non_diag = [j for j in range(5) if j != i]
        weighted_mat[i, non_diag] /= tw
    row_sums = weighted_mat.sum(axis=1, keepdims=True)
    weighted_mat /= row_sums
    
    format_matrix(weighted_mat, title=f"Inference Transition Matrix (transition_weight={tw} applied)")

if __name__ == "__main__":
    main()

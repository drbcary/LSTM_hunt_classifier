import numpy as np
from typing import List, Dict, Any, Optional
from labels.base_parser import HuntEvent
from config.base_config import sec_to_frames

def calculate_empirical_transition_matrix(
    all_events: List[List[HuntEvent]],
    fps: float = 30.0,
    bg_dur_sec: float = 33.33,
    eating_buf_sec: float = 10.0,
    num_classes: int = 4,
    paradigm: str = "rat"
) -> np.ndarray:
    """
    Computes a transition probability matrix from manually labeled hunting events.
    Supports:
      - 4-class (rat): 0: Background, 1: Stalking, 2: Pursuit, 3: Eating
      - 5-class (mouse): 0: Background, 1: Cricket Present, 2: First Approach, 3: Final Approach, 4: Eating
    Temporal gaps and buffers are scaled dynamically by FPS.
    """
    is_mouse = (num_classes == 5 or paradigm == "mouse")
    k = 5 if is_mouse else 4
    counts = np.zeros((k, k), dtype=float)
    eating_buffer_frames = sec_to_frames(eating_buf_sec, fps)
    bg_dur_frames = sec_to_frames(bg_dur_sec, fps)

    for hunt_list in all_events:
        for hunt in hunt_list:
            enter = hunt.enter
            capture = hunt.capture
            eating_end = hunt.get_eating_end(eating_buffer_frames)

            if is_mouse:
                final = hunt.approach_start
                if enter >= capture or final > capture:
                    continue

                # Hunt onset from Background (0) is ALWAYS Cricket Present (1)
                counts[0, 0] += bg_dur_frames - 1
                counts[0, 1] += 1

                first = hunt.first_approach
                has_first_approach = (first is not None and first > enter and first < final)

                if has_first_approach:
                    cp_dur = max(0, first - enter)
                    fa_dur = max(0, final - first)
                else:
                    cp_dur = max(0, final - enter)
                    fa_dur = 0

                final_dur = max(0, capture - final)
                eating_dur = max(0, eating_end - capture)

                if cp_dur > 0:
                    counts[1, 1] += cp_dur - 1

                if has_first_approach and fa_dur > 0:
                    counts[1, 2] += 1
                    counts[2, 2] += fa_dur - 1
                    if final_dur > 0:
                        counts[2, 3] += 1
                        counts[3, 3] += final_dur - 1
                        if eating_dur > 0:
                            counts[3, 4] += 1
                            counts[4, 4] += eating_dur - 1
                            counts[4, 0] += 1
                        else:
                            counts[3, 0] += 1
                    else:
                        if eating_dur > 0:
                            counts[2, 3] += 1
                            counts[3, 4] += 1
                            counts[4, 4] += eating_dur - 1
                            counts[4, 0] += 1
                        else:
                            counts[2, 0] += 1
                else:
                    # Direct transition from Cricket Present to Final Approach / Strike
                    if final_dur > 0:
                        counts[1, 3] += 1
                        counts[3, 3] += final_dur - 1
                        if eating_dur > 0:
                            counts[3, 4] += 1
                            counts[4, 4] += eating_dur - 1
                            counts[4, 0] += 1
                        else:
                            counts[3, 0] += 1
                    else:
                        if eating_dur > 0:
                            counts[1, 3] += 1
                            counts[3, 4] += 1
                            counts[4, 4] += eating_dur - 1
                            counts[4, 0] += 1
                        else:
                            counts[1, 0] += 1
            else:
                pursuit = hunt.approach_start
                if enter >= capture or pursuit > capture:
                    continue

                stalk_dur = max(0, pursuit - enter)
                pursuit_dur = max(0, capture - pursuit)
                eating_dur = max(0, eating_end - capture)

                counts[0, 0] += bg_dur_frames - 1

                if stalk_dur > 0:
                    counts[0, 1] += 1
                    counts[1, 1] += stalk_dur - 1
                    if pursuit_dur > 0:
                        counts[1, 2] += 1
                        counts[2, 2] += pursuit_dur - 1
                        if eating_dur > 0:
                            counts[2, 3] += 1
                            counts[3, 3] += eating_dur - 1
                            counts[3, 0] += 1
                        else:
                            counts[2, 0] += 1
                    else:
                        if eating_dur > 0:
                            counts[1, 3] += 1
                            counts[3, 3] += eating_dur - 1
                            counts[3, 0] += 1
                        else:
                            counts[1, 0] += 1
                else:
                    counts[0, 2] += 1
                    if pursuit_dur > 0:
                        counts[2, 2] += pursuit_dur - 1
                        if eating_dur > 0:
                            counts[2, 3] += 1
                            counts[3, 3] += eating_dur - 1
                            counts[3, 0] += 1
                        else:
                            counts[2, 0] += 1

    total_hunts = sum(len(h) for h in all_events)
    if is_mouse:
        counts[1, 0] += max(1.0, total_hunts * 0.08)  # Abort cricket present
        counts[2, 0] += max(1.0, total_hunts * 0.05)  # Abort first approach
        counts[3, 0] += max(1.0, total_hunts * 0.02)  # Abort final approach
        counts[2, 1] += max(1.0, total_hunts * 0.05)  # Disengage first approach back to cricket present
        counts[3, 2] += max(1.0, total_hunts * 0.03)  # Missed strike resets to first approach
        counts[4, 1] += max(1.0, total_hunts * 0.02)  # Back-to-back hunt after eating

        forbidden = [
            (0, 2),  # Cannot jump Background -> First Approach
            (0, 3),  # Cannot jump Background -> Final Approach
            (0, 4),  # Cannot jump Background -> Eating
            (1, 4),  # Cannot jump Cricket Present -> Eating
            (2, 4),  # Cannot jump First Approach -> Eating
            (4, 2),  # Cannot transition Eating -> First Approach
            (4, 3),  # Cannot transition Eating -> Final Approach
        ]
        for i in range(5):
            for j in range(5):
                if (i, j) in forbidden:
                    counts[i, j] = 0.0
                elif counts[i, j] == 0.0:
                    counts[i, j] = 1e-5
    else:
        counts[1, 0] += max(1.0, total_hunts * 0.10)  # 10% chance to abort stalk
        counts[2, 0] += max(1.0, total_hunts * 0.05)  # 5% chance to abort pursuit
        counts[3, 1] += max(1.0, total_hunts * 0.02)  # Direct transition eating -> stalk
        counts[3, 2] += max(1.0, total_hunts * 0.01)  # Direct transition eating -> pursuit

        forbidden = [(2, 1)]
        for i in range(4):
            for j in range(4):
                if (i, j) in forbidden:
                    counts[i, j] = 0.0
                elif counts[i, j] == 0.0:
                    counts[i, j] = 1e-5

    # Ensure every behavioral state maintains temporal persistence (self-transition prior).
    # If a state (e.g. State 2 First Approach) was unannotated or rare in ground truth,
    # its self-transition count must be scaled relative to its outgoing transitions so that
    # P(i -> i) >= 0.95 (i.e. counts[i, i] >= 19 * outgoing). This prevents high-frequency
    # 1-frame oscillation / barcode flickering in the Viterbi decoder.
    for i in range(k):
        outgoing = sum(counts[i, j] for j in range(k) if j != i)
        if outgoing > 0:
            min_self = outgoing * 19.0  # Guarantees >= 0.95 self-transition probability
            if counts[i, i] < min_self:
                counts[i, i] = min_self

    row_sums = counts.sum(axis=1, keepdims=True)
    return counts / np.maximum(row_sums, 1e-6)

import random
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from config.base_config import BaseParadigmConfig, sec_to_frames
from labels.base_parser import HuntEvent

def _safe_frame_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    try:
        f = float(val)
        if f != f:  # NaN check
            return None
        return int(f)
    except (ValueError, TypeError):
        return None

def events_to_label_sequence(events: List[HuntEvent], num_frames: int, eating_buffer_frames: int = 300) -> np.ndarray:
    """
    Converts a list of HuntEvents into a 1D frame label sequence.
    If events have first_approach defined, creates a 5-class sequence [0, 1, 2, 3, 4]:
      0: Background, 1: Cricket Present, 2: First Approach, 3: Final Approach, 4: Eating
    Otherwise, creates legacy 4-class sequence [0, 1, 2, 3]:
      0: Background, 1: Stalking/Cricket Present, 2: Pursuit, 3: Eating
    """
    seq = np.zeros(num_frames, dtype=int)
    for ev in events:
        enter_val = _safe_frame_int(ev.enter)
        if enter_val is None:
            continue
        enter = max(0, min(num_frames, enter_val))

        cap_val = _safe_frame_int(ev.capture)
        capture = max(enter, min(num_frames, cap_val if cap_val is not None else enter))

        eat_end_val = _safe_frame_int(ev.get_eating_end(eating_buffer_frames))
        eating_end = max(capture, min(num_frames, eat_end_val if eat_end_val is not None else capture + eating_buffer_frames))

        first_app_val = _safe_frame_int(ev.first_approach)
        app_start_val = _safe_frame_int(ev.approach_start)

        if first_app_val is not None:
            first_app = max(enter, min(num_frames, first_app_val))
            final_app = max(first_app, min(num_frames, app_start_val if app_start_val is not None else first_app))
            if first_app > enter:
                seq[enter:first_app] = 1
            if final_app > first_app:
                seq[first_app:final_app] = 2
            if capture > final_app:
                seq[final_app:capture] = 3
            if eating_end > capture:
                seq[capture:eating_end] = 4
        else:
            pursuit = max(enter, min(num_frames, app_start_val if app_start_val is not None else enter))
            if pursuit > enter:
                seq[enter:pursuit] = 1
            if capture > pursuit:
                seq[pursuit:capture] = 2
            if eating_end > capture:
                seq[capture:eating_end] = 3
    return seq

class SequenceBuilder:
    """
    Creates balanced training chunks from full continuous feature sequences and hunt annotations.
    Dynamically scales all temporal buffers by each video's true FPS.
    """
    def __init__(self, config: Optional[BaseParadigmConfig] = None):
        from config.base_config import BaseParadigmConfig
        self.config = config or BaseParadigmConfig()

    def build_video_sequences(
        self,
        features: np.ndarray,
        events: List[HuntEvent],
        fps: float
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        total_frames = len(features)
        time_cfg = self.config.time

        # Convert temporal constants to frame counts dynamically based on true FPS
        context_buf = sec_to_frames(time_cfg.context_buffer_sec, fps)
        eating_buf = sec_to_frames(time_cfg.eating_buffer_sec, fps)
        min_bg_len = sec_to_frames(time_cfg.min_bg_len_sec, fps)
        max_bg_len = sec_to_frames(time_cfg.max_bg_len_sec, fps)
        chunk_min = sec_to_frames(time_cfg.full_video_chunk_min_sec, fps)
        chunk_max = sec_to_frames(time_cfg.full_video_chunk_max_sec, fps)

        sequences = []
        targets = []

        # 1. Full video random chunking mode
        if random.random() < time_cfg.full_video_chunk_proportion:
            for _ in range(time_cfg.full_video_chunk_iterations):
                current_idx = 0
                while current_idx <= total_frames - chunk_min:
                    chunk_size = random.randint(chunk_min, chunk_max)
                    end_idx = min(total_frames, current_idx + chunk_size)

                    if (end_idx - current_idx) >= chunk_min:
                        seq_feat = features[current_idx:end_idx]
                        seq_len = len(seq_feat)
                        seq_targ = np.zeros(seq_len, dtype=int)

                        for ev in events:
                            enter = ev.enter
                            capture = ev.capture
                            pursuit = ev.approach_start
                            eating_end = ev.get_eating_end(eating_buf)

                            if enter >= capture or pursuit > capture:
                                continue

                            local_enter = enter - current_idx
                            local_pursuit = pursuit - current_idx
                            local_capture = capture - current_idx
                            local_eating = eating_end - current_idx

                            if ev.first_approach is not None:
                                local_first = ev.first_approach - current_idx
                                if local_first > max(0, local_enter) and local_enter < seq_len:
                                    seq_targ[max(0, local_enter):min(seq_len, local_first)] = 1
                                if local_pursuit > max(0, local_first) and local_first < seq_len:
                                    seq_targ[max(0, local_first):min(seq_len, local_pursuit)] = 2
                                if local_capture > max(0, local_pursuit) and local_pursuit < seq_len:
                                    seq_targ[max(0, local_pursuit):min(seq_len, local_capture)] = 3
                                if local_eating > max(0, local_capture) and local_capture < seq_len:
                                    seq_targ[max(0, local_capture):min(seq_len, local_eating)] = 4
                            else:
                                if local_pursuit > max(0, local_enter) and local_enter < seq_len:
                                    seq_targ[max(0, local_enter):min(seq_len, local_pursuit)] = 1
                                if local_capture > max(0, local_pursuit) and local_pursuit < seq_len:
                                    seq_targ[max(0, local_pursuit):min(seq_len, local_capture)] = 2
                                if local_eating > max(0, local_capture) and local_capture < seq_len:
                                    seq_targ[max(0, local_capture):min(seq_len, local_eating)] = 3

                        sequences.append(seq_feat)
                        targets.append(seq_targ)

                    current_idx = end_idx
            return sequences, targets

        # 2. Balanced background & hunt event sampling
        bg_chunks = 0
        last_event_end = 0

        for ev in events:
            enter = ev.enter
            capture = ev.capture
            pursuit = ev.approach_start
            eating_end = ev.get_eating_end(eating_buf)

            # Sample background gaps between hunts
            gap_start = min(total_frames, max(0, last_event_end))
            gap_end = min(total_frames, max(0, enter - context_buf))

            if (gap_end - gap_start) > min_bg_len and bg_chunks < self.config.max_bg_chunks_per_video:
                gap_attempts = 0
                curr_idx = gap_start
                while curr_idx < (gap_end - min_bg_len) and gap_attempts < 2:
                    chunk_sz = random.randint(min_bg_len, max_bg_len)
                    end_idx = min(gap_end, curr_idx + chunk_sz)

                    chunk_feat = features[curr_idx:end_idx]
                    if len(chunk_feat) >= min_bg_len:
                        sequences.append(chunk_feat)
                        targets.append(np.zeros(len(chunk_feat), dtype=int))
                        bg_chunks += 1
                        gap_attempts += 1
                    curr_idx = end_idx + random.randint(sec_to_frames(3.0, fps), sec_to_frames(10.0, fps))

            # Sample hunt event with jittered buffer
            if enter >= capture or pursuit > capture:
                continue

            # Cap maximum eating duration to prevent massive sequence lengths (e.g. 60s max consumption window)
            max_eating_frames = int(60 * fps)
            if ev.consumption_end is not None:
                capped_eating_end = min(ev.consumption_end, capture + max_eating_frames)
            else:
                capped_eating_end = capture + min(eating_buf, max_eating_frames)

            rand_buf = random.randint(sec_to_frames(1.5, fps), sec_to_frames(13.0, fps))
            seq_start = max(0, min(total_frames, enter - rand_buf))
            seq_end = max(0, min(total_frames, capped_eating_end + rand_buf))
            if seq_end <= seq_start:
                continue

            seq_feat = features[seq_start:seq_end]
            seq_len = len(seq_feat)
            if seq_len < 10:
                continue

            seq_targ = np.zeros(seq_len, dtype=int)
            local_enter = enter - seq_start
            local_pursuit = pursuit - seq_start
            local_capture = capture - seq_start
            local_eating = capped_eating_end - seq_start

            if ev.first_approach is not None:
                local_first = ev.first_approach - seq_start
                if local_first > local_enter:
                    seq_targ[max(0, local_enter):min(seq_len, local_first)] = 1
                if local_pursuit > local_first:
                    seq_targ[max(0, local_first):min(seq_len, local_pursuit)] = 2
                if local_capture > local_pursuit:
                    seq_targ[max(0, local_pursuit):min(seq_len, local_capture)] = 3
                if local_eating > local_capture:
                    seq_targ[max(0, local_capture):min(seq_len, local_eating)] = 4
            else:
                if local_pursuit > local_enter:
                    seq_targ[max(0, local_enter):min(seq_len, local_pursuit)] = 1
                if local_capture > local_pursuit:
                    seq_targ[max(0, local_pursuit):min(seq_len, local_capture)] = 2
                if local_eating > local_capture:
                    seq_targ[max(0, local_capture):min(seq_len, local_eating)] = 3

            # If a hunt sequence is exceptionally long (> 5000 frames), chunk it with overlap
            MAX_HUNT_SEQ_LEN = 5000
            if seq_len > MAX_HUNT_SEQ_LEN:
                stride = 3500
                for sub_start in range(0, seq_len, stride):
                    sub_end = min(seq_len, sub_start + MAX_HUNT_SEQ_LEN)
                    sub_feat = seq_feat[sub_start:sub_end]
                    sub_targ = seq_targ[sub_start:sub_end]
                    if len(sub_feat) >= min_bg_len:
                        sequences.append(sub_feat)
                        targets.append(sub_targ)
                    if sub_end == seq_len:
                        break
            else:
                sequences.append(seq_feat)
                targets.append(seq_targ)
            last_event_end = seq_end

        # Strict validation filter: eliminate any 0-length or mismatched sequences
        clean_sequences = []
        clean_targets = []
        for s, t in zip(sequences, targets):
            if len(s) >= 10 and len(s) == len(t):
                clean_sequences.append(s)
                clean_targets.append(t)

        return clean_sequences, clean_targets

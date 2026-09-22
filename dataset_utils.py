"""
dataset_utils.py
Modular backward-compatibility facade for the LSTM hunt classification pipeline.
Delegates to submodules in `config`, `pose`, `features`, `labels`, and `dataset`.
"""

import os
import glob
import re
import cv2
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union

# -------------------------------------------------------------
# Modular Subsystem Imports
# -------------------------------------------------------------
import config
from config import (
    BaseParadigmConfig,
    TimeConfig,
    MousePoseSchema,
    MouseParadigmConfig,
    RatPoseSchema,
    RatParadigmConfig,
    sec_to_frames,
    frames_to_sec
)
from pose import (
    PoseData,
    DLCAdapter,
    calculate_angle_3pts,
    calculate_bearing,
    angular_diff,
    dist,
    apply_poop_penalty,
    extract_validated_poop_clusters,
    extract_filename_timestamp
)
from features import feature_registry, FeatureManifest, FeatureItem, generate_features as modular_generate_features
from labels import (
    HuntEvent,
    BaseLabelParser,
    GenericExcelLabelParser,
    MultiSessionSheetParser,
    BorisLabelParser,
    DianeExcelLabelParser
)
from dataset import VideoSessionMetadata, DatasetMatcher, SequenceBuilder, events_to_label_sequence, calculate_empirical_transition_matrix as modular_transition_matrix, get_video_properties

# -------------------------------------------------------------
# Global Configurations & Paths (Legacy Compatibility)
# -------------------------------------------------------------
BASE_SERVER_PATH = r"\\research.brandeis.edu\turrigiano-lab"

DATASETS: Dict[str, Dict[str, Any]] = {
    'PV_Cre-': {
        'label_dir': os.path.join(BASE_SERVER_PATH, r'\DPL\DATA\PV-Cre\CricketHuntTimes_PV'), 
        'dlc_dir': os.path.join(BASE_SERVER_PATH, r'\DPL\DATA\PV-Cre\Cre-'),
        'animal_basename': 'PV',
        'file_filter': None,
        'format': 'dan',
    },
    'PV_Cre+': {
        'label_dir': os.path.join(BASE_SERVER_PATH, r'\DPL\DATA\PV-Cre\CricketHuntTimes_PV'), 
        'dlc_dir': os.path.join(BASE_SERVER_PATH, r'\DPL\DATA\PV-Cre\Cre+'),
        'animal_basename': 'PV',
        'file_filter': None,
        'format': 'dan',
    },
    'XPRO': {
        'label_dir': os.path.join(BASE_SERVER_PATH, r'\DPL\DATA\XPro\XPRO_BEHAVIOR\ALL'), 
        'dlc_dir': os.path.join(BASE_SERVER_PATH, r'\DPL\DATA\XPro\XPRO'),
        'animal_basename': 'XP',
        'file_filter': None,
        'format': 'dan',
    },
    'Lucas_Exp4': {
        'label_dir': os.path.join(BASE_SERVER_PATH, r'LucasWahl\Excel files\Exp 4\WT observer'),
        'dlc_dir': os.path.join(BASE_SERVER_PATH, r'LucasWahl\Exp 4 - WT naive + WT observer data\WT observer data\hunt_crop'),
        'animal_basename': 'LW',
        'format': 'lucas',
    },
    'Lucas_Exp5': {
        'label_dir': os.path.join(BASE_SERVER_PATH, r'LucasWahl\Excel files\Exp 5\Scn2a Observer'),
        'dlc_dir': os.path.join(BASE_SERVER_PATH, r'LucasWahl\Exp 5 - Scn2a mut observer data\Scn2a observer\hunt_crop'),
        'animal_basename': 'LW',
        'format': 'lucas',
    },    
}

DAN_DATASETS = {k: v for k, v in DATASETS.items() if v['format'] == 'dan'}

# -------------------------------------------------------------
# Mouse Datasets Configuration (Diane Format)
# -------------------------------------------------------------
MOUSE_DATASETS: Dict[str, Dict[str, Any]] = {
    'Craniotomy': {
        'excel_path': os.path.join(BASE_SERVER_PATH, r'\DianeBissen\Cricket hunting videos\Cricket_hunting_all_hunting_videos_labeling_Diane_BCFixed_MostUpdated.xlsx'),
        'dlc_dir': os.path.join(BASE_SERVER_PATH, r'\DianeBissen\Cricket hunting videos\Craniotomy hunting videos_DLC'),
        'video_dir': None,          # Optional: None if videos are in the same dir as DLC
        'format': 'diane',
        'strategy': 'engagement',   # 'engagement' (default continuous hunt), 'final_only', or 'split'
        'file_filter': None,
        'recursive': False,         # Set to True to recursively search subdirectories for DLC files
    },
    'Non-Craniotomy': {
        'excel_path': os.path.join(BASE_SERVER_PATH, r'\DianeBissen\Cricket hunting videos\Cricket_hunting_all_hunting_videos_labeling_Diane_NonCran_BCEdit.xlsx'),
        'dlc_dir': os.path.join(BASE_SERVER_PATH, r'\DianeBissen\Cricket hunting videos\Non-craniotomy videos'),
        'video_dir': None,          # Optional: None if videos are in the same dir as DLC
        'format': 'diane',
        'strategy': 'engagement',   # 'engagement' (default continuous hunt), 'final_only', or 'split'
        'file_filter': None,
        'recursive': False,         # Set to True to recursively search subdirectories for DLC files
    },
    # Add additional experimental cohorts as needed:
    # 'Diane_Exp19': {
    #     'excel_path': os.path.join(BASE_SERVER_PATH, r'path\to\Exp19_master_labels.xlsx'),
    #     'dlc_dir': os.path.join(BASE_SERVER_PATH, r'path\to\Exp19_DLC_CSVs'),
    #     'video_dir': None,
    #     'format': 'diane',
    #     'strategy': 'engagement',
    #     'file_filter': None,
    # },
}

VERBOSE = True
DEFAULT_WIDTH = 514
DEFAULT_HEIGHT = 544
DEFAULT_FPS = 30.0

# Legacy Frame Buffers (30 FPS default)
CONTEXT_BUFFER = 300
EATING_BUFFER = 300
MIN_BG_LEN = 500
MAX_BG_LEN = 2000
MAX_BG_CHUNKS_PER_VIDEO = 5

FULL_VIDEO_CHUNK_MIN = 1000
FULL_VIDEO_CHUNK_MAX = 5000
FULL_VIDEO_CHUNK_PROPORTION = 0.15
FULL_VIDEO_CHUNK_ITERATIONS = 2

RAT_LIKELIHOOD_THRESH = 0.9
CRICKET_LIKELIHOOD_THRESH = 0.85
MAX_INTERP_GAP = 15
CRICKET_MOVEMENT_THRESH = 0.3
CRICKET_DEAD_TIME_THRESH = 30
MAX_SPEED_THRESH = 3.0
MAX_ACCEL_THRESH = 25.0

# -------------------------------------------------------------
# Feature Generation Functions
# -------------------------------------------------------------
def generate_features(rat_head, rat_mid, rat_butt, rat_tail, cricket_pos, video_width, video_fps):
    """
    Legacy 22-feature generator function preserving exact feature order and normalization.
    """
    def _dist(p1, p2): return np.linalg.norm(p1 - p2, axis=1)
    def _get_derivs(pos_array):
        vel_vec = np.diff(pos_array, axis=0, prepend=pos_array[0:1]) * video_fps
        speed = np.linalg.norm(vel_vec, axis=1) / video_width
        acc_vec = np.diff(vel_vec, axis=0, prepend=vel_vec[0:1]) * video_fps
        accel = np.linalg.norm(acc_vec, axis=1) / video_width
        return speed, accel

    d_head_cricket = _dist(rat_head, cricket_pos) / video_width
    d_head_mid = _dist(rat_head, rat_mid) / video_width
    d_mid_tail = _dist(rat_mid, rat_tail) / video_width
    d_head_tail = _dist(rat_head, rat_tail) / video_width
    
    angle_spine = calculate_angle_3pts(rat_head, rat_mid, rat_tail) / 180.0
    rat_orientation = calculate_bearing(rat_mid, rat_head)
    target_vector = calculate_bearing(rat_head, cricket_pos)
    angle_diff = target_vector - rat_orientation
    angle_diff_raw = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    angle_diff = angle_diff_raw / np.pi
    
    rat_speed, rat_accel = _get_derivs(rat_head)
    cricket_speed, cricket_accel = _get_derivs(cricket_pos)
    cricket_visible = np.where(np.isnan(d_head_cricket), 0.0, 1.0)
    
    closing_velocity = -np.diff(d_head_cricket, prepend=d_head_cricket[0:1]) * video_fps
    inverse_ttc = np.clip(closing_velocity / (d_head_cricket + 1e-6), -100, 100) 

    def _ang_diff(angles):
        d_angles = np.diff(angles, prepend=angles[0:1])
        return (d_angles + np.pi) % (2 * np.pi) - np.pi

    rat_turn_rate = np.abs(_ang_diff(rat_orientation)) * video_fps / np.pi
    aim_correction_rate = np.abs(_ang_diff(angle_diff_raw)) * video_fps / np.pi

    vec_hc = cricket_pos - rat_head
    dist_hc = np.linalg.norm(vec_hc, axis=1, keepdims=True)
    unit_hc = vec_hc / (dist_hc + 1e-6)

    rat_vel = np.diff(rat_head, axis=0, prepend=rat_head[0:1]) * video_fps
    lunge_speed = np.sum(rat_vel * unit_hc, axis=1) / video_width

    theta = rat_orientation
    dx = vec_hc[:, 0]
    dy = vec_hc[:, 1]
    ego_forward = (dx * np.cos(-theta) - dy * np.sin(-theta)) / video_width
    ego_lateral = (dx * np.sin(-theta) + dy * np.cos(-theta)) / video_width

    body_compression = d_head_tail / (d_head_mid + d_mid_tail + 1e-6)
    compression_rate = np.diff(body_compression, prepend=body_compression[0:1]) * video_fps

    angle_tail = calculate_angle_3pts(rat_mid, rat_butt, rat_tail) / 180.0
    tail_angular_vel = np.abs(np.diff(angle_tail, prepend=angle_tail[0:1])) * video_fps

    c_vel = np.diff(cricket_pos, axis=0, prepend=cricket_pos[0:1])
    c_bearing = np.arctan2(c_vel[:, 1], c_vel[:, 0])
    c_jitter = np.abs(_ang_diff(c_bearing)) / np.pi
    
    # Robust Normalization
    rat_speed = np.clip(rat_speed, 0, MAX_SPEED_THRESH) / MAX_SPEED_THRESH
    cricket_speed = np.clip(cricket_speed, 0, MAX_SPEED_THRESH) / MAX_SPEED_THRESH
    rat_accel = np.clip(rat_accel, 0, MAX_ACCEL_THRESH) / MAX_ACCEL_THRESH
    cricket_accel = np.clip(cricket_accel, 0, MAX_ACCEL_THRESH) / MAX_ACCEL_THRESH
    closing_velocity = np.clip(closing_velocity, -MAX_SPEED_THRESH, MAX_SPEED_THRESH) / MAX_SPEED_THRESH
    lunge_speed = np.clip(lunge_speed, -MAX_SPEED_THRESH, MAX_SPEED_THRESH) / MAX_SPEED_THRESH
    
    inverse_ttc = np.tanh(inverse_ttc / 10.0)
    compression_rate = np.tanh(compression_rate / 10.0)
    tail_angular_vel = np.tanh(tail_angular_vel / 5.0)
    rat_turn_rate = np.tanh(rat_turn_rate / 10.0)
    aim_correction_rate = np.tanh(aim_correction_rate / 10.0)
    
    feature_stack = np.column_stack([
        d_head_cricket, d_head_mid, d_mid_tail, d_head_tail,
        angle_spine, angle_diff, rat_speed, rat_accel,
        cricket_speed, cricket_accel, cricket_visible,
        closing_velocity, inverse_ttc, rat_turn_rate, aim_correction_rate,
        lunge_speed, ego_forward, ego_lateral, body_compression,
        compression_rate, tail_angular_vel, c_jitter
    ])
    return np.nan_to_num(feature_stack, nan=0.0)

def generate_mouse_features(
    pose_data: PoseData,
    video_width: float,
    video_fps: float,
    enabled_blocks: Optional[List[str]] = None,
    config_obj: Optional[MouseParadigmConfig] = None
) -> Tuple[np.ndarray, FeatureManifest]:
    """Generates features for mouse model using the modular FeatureRegistry."""
    cfg = config_obj or MouseParadigmConfig()
    blocks = enabled_blocks or cfg.enabled_feature_blocks
    return modular_generate_features(pose_data, video_width, video_fps, blocks, cfg)

# -------------------------------------------------------------
# DLC Loading and Video Utilities
# -------------------------------------------------------------
def get_video_params(dlc_csv_path, video_path=None, verbose=False):
    """
    Retrieves width, height, and FPS directly from video file via OpenCV.
    Uses video_path if provided, else auto-discovers video matching dlc_csv_path.
    Safely handles dlc_csv_path and video_path being lists of multi-part files.
    """
    first_dlc = dlc_csv_path[0] if isinstance(dlc_csv_path, (list, tuple)) else dlc_csv_path
    first_video = video_path[0] if isinstance(video_path, (list, tuple)) else video_path
    matcher = DatasetMatcher(extract_video_fps=True, default_fps=DEFAULT_FPS, default_width=DEFAULT_WIDTH, default_height=DEFAULT_HEIGHT)
    v_path = first_video or (matcher.find_video_for_dlc(first_dlc) if first_dlc else None)
    if v_path and os.path.exists(v_path):
        w, h, fps = get_video_properties(v_path)
        if w > 0 and h > 0 and fps > 0:
            if verbose: print(f"Read video properties for {os.path.basename(v_path)}: {w}x{h} @ {fps:.1f} FPS")
            return w, h, fps
    if verbose: print(f"Warning: Could not read video properties for {first_dlc}. Using defaults ({DEFAULT_WIDTH}x{DEFAULT_HEIGHT} @ {DEFAULT_FPS} FPS).")
    return DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS

def resample_feature_matrix(features: np.ndarray, orig_fps: float, target_fps: float = 30.0) -> np.ndarray:
    """
    Resamples a (N, D) feature matrix from orig_fps to target_fps using 1D linear interpolation.
    Preserves continuous feature values across time.
    """
    if abs(orig_fps - target_fps) < 1e-2 or len(features) < 2:
        return features
    
    num_orig = len(features)
    total_duration = (num_orig - 1) / orig_fps
    num_target = max(2, round(num_orig * (target_fps / orig_fps)))
    
    t_orig = np.linspace(0, total_duration, num_orig)
    t_target = np.linspace(0, total_duration, num_target)
    
    resampled = np.empty((num_target, features.shape[1]), dtype=features.dtype)
    for d in range(features.shape[1]):
        resampled[:, d] = np.interp(t_target, t_orig, features[:, d])
    return resampled

def scale_hunt_events_fps(hunts: List[Dict[str, Any]], orig_fps: float, target_fps: float = 30.0) -> List[Dict[str, Any]]:
    """
    Scales hunt frame boundaries from orig_fps to target_fps (e.g. 20 FPS -> 30 FPS).
    """
    if abs(orig_fps - target_fps) < 1e-2:
        return hunts
    
    scale = target_fps / orig_fps
    scaled = []
    for h in hunts:
        new_h = dict(h)
        for k in ['enter', 'first_approach', 'pursuit_start', 'capture', 'consumption_end']:
            if h.get(k) is not None:
                new_h[k] = int(round(h[k] * scale))
        scaled.append(new_h)
    return scaled

def extract_dlc_tracking_data(
    filepath,
    rat_thresh=RAT_LIKELIHOOD_THRESH,
    cricket_thresh=CRICKET_LIKELIHOOD_THRESH,
    interp_limit=MAX_INTERP_GAP,
    include_blurry_cricket=False,
    filter_poops=True,
    poop_thresh=0.9,
    poop_dist_thresh=6.0
):
    """Legacy extraction helper returning raw coordinates tuple without dead cricket filtering."""
    adapter = DLCAdapter(RatParadigmConfig())
    pose = adapter.load(
        filepath,
        pred_thresh=rat_thresh,
        cricket_thresh=cricket_thresh,
        interp_limit=interp_limit,
        include_blurry_cricket=include_blurry_cricket,
        filter_poops=filter_poops,
        poop_dist_thresh=poop_dist_thresh,
        poop_thresh=poop_thresh,
        filter_dead_crickets=False
    )
    if pose is None:
        return None
    return pose.head, pose.center, pose.rear, pose.primary_tail_end, pose.cricket

def load_and_process_dlc(
    filepath,
    video_fps,
    include_blurry_cricket=False,
    filter_poops=True,
    poop_thresh=0.9,
    poop_dist_thresh=6.0
):
    """Loads and filters DLC tracking data for a full video, including dead cricket filtering."""
    adapter = DLCAdapter(RatParadigmConfig())
    pose = adapter.load(
        filepath,
        video_fps=video_fps,
        pred_thresh=RAT_LIKELIHOOD_THRESH,
        cricket_thresh=CRICKET_LIKELIHOOD_THRESH,
        interp_limit=MAX_INTERP_GAP,
        include_blurry_cricket=include_blurry_cricket,
        filter_poops=filter_poops,
        poop_dist_thresh=poop_dist_thresh,
        poop_thresh=poop_thresh,
        filter_dead_crickets=True
    )
    if pose is None:
        return None
    return pose.head, pose.center, pose.rear, pose.primary_tail_end, pose.cricket

def load_mouse_dlc(
    filepath: Union[str, List[str]],
    video_fps: float,
    mouse_cfg: Optional[MouseParadigmConfig] = None,
    include_blurry_cricket: Optional[bool] = None,
    include_cricket_immo: Optional[bool] = None,
    **kwargs
) -> Optional[PoseData]:
    """Loads mouse DLC data with multi-segment tail points, ear orientation, and multi-candidate cricket support (regular, blurry, immobilized)."""
    cfg = mouse_cfg or MouseParadigmConfig()
    adapter = DLCAdapter(cfg)
    return adapter.load(
        filepath,
        video_fps=video_fps,
        include_blurry_cricket=include_blurry_cricket,
        include_cricket_immo=include_cricket_immo,
        **kwargs
    )

# -------------------------------------------------------------
# Label Parsers & Transition Matrix
# -------------------------------------------------------------
def hunts_to_sequence(hunts, num_frames):
    """Converts a list of hunt dictionaries to 1D label array."""
    events = [
        HuntEvent(
            enter=h['enter'],
            approach_start=h['pursuit_start'],
            capture=h['capture'],
            consumption_end=h.get('consumption_end')
        )
        for h in hunts
    ]
    return events_to_label_sequence(events, num_frames, EATING_BUFFER)

def parse_lucas_labels(label_path):
    parser = GenericExcelLabelParser()
    events = parser.parse_file(label_path)
    return [
        {
            'enter': ev.enter,
            'capture': ev.capture,
            'pursuit_start': ev.approach_start,
            'consumption_end': ev.consumption_end
        }
        for ev in events
    ]

def parse_dan_label_file(file_path):
    parser = MultiSessionSheetParser()
    h_map = parser.parse_file(file_path)
    out = {}
    for (animal, day, session), events in h_map.items():
        event_dicts = [
            {
                'enter': ev.enter,
                'pursuit_start': ev.approach_start,
                'capture': ev.capture,
                'consumption_end': ev.consumption_end
            }
            for ev in events
        ]
        animal_clean = animal.strip().upper()
        # Support full (Animal, Day, Session) matching
        out[(animal_clean, int(day), int(session))] = event_dicts
        # Normalized animal without leading zeros (e.g. XP3 vs XP03)
        norm_animal = re.sub(r'([A-Z]+)0*(\d+)', r'\1\2', animal_clean)
        out[(norm_animal, int(day), int(session))] = event_dicts
        # Fallback 2-tuple (Animal, Session)
        if (animal_clean, int(session)) not in out:
            out[(animal_clean, int(session))] = event_dicts
        if (norm_animal, int(session)) not in out:
            out[(norm_animal, int(session))] = event_dicts
    return out

def load_dan_labels(label_dir):
    dan_hunts = {}
    if os.path.isfile(label_dir):
        label_files = [label_dir]
    elif os.path.isdir(label_dir):
        label_files = glob.glob(os.path.join(label_dir, '*.xlsx')) + glob.glob(os.path.join(label_dir, '*.csv'))
    else:
        label_files = []
    for lf in label_files:
        try:
            fh = parse_dan_label_file(lf)
            for k, v in fh.items():
                if k not in dan_hunts:
                    dan_hunts[k] = []
                dan_hunts[k].extend(v)
        except Exception as e:
            if VERBOSE: print(f"Error reading {lf}: {e}")
    return dan_hunts

def pair_dataset_files(datasets_config=DATASETS, verbose=True):
    """Legacy dataset pairing wrapper."""
    paired_data = []
    for ds_name, cfg in datasets_config.items():
        fmt = cfg['format']
        label_dir = cfg.get('label_dir')
        dlc_dir = cfg.get('dlc_dir')

        if label_dir and not os.path.exists(label_dir) and 'fallback_label_dir' in cfg:
            label_dir = cfg['fallback_label_dir']

        if not dlc_dir or not os.path.exists(dlc_dir):
            if verbose: print(f"Warning: DLC directory not found for {ds_name} ({dlc_dir})")
            continue

        is_rec = bool(cfg.get('recursive', False))
        if is_rec:
            dlc_files = glob.glob(os.path.join(dlc_dir, '*DLC*.csv')) + glob.glob(os.path.join(dlc_dir, '**', '*DLC*.csv'), recursive=True)
            dlc_files = sorted(list(set(dlc_files)))
        else:
            dlc_files = glob.glob(os.path.join(dlc_dir, '*DLC*.csv'))
        if verbose: print(f"Scanning {ds_name} (recursive={is_rec})...")

        if fmt == 'lucas':
            if not label_dir:
                continue
            animal_basename = cfg.get('animal_basename', 'LW')
            label_files = glob.glob(os.path.join(label_dir, '*analyzed*.xlsx'))
            dlc_map = {}
            for dlc_f in dlc_files:
                match = re.search(fr'({animal_basename}\d+)[-_](hunt[-_]?\d+)', os.path.basename(dlc_f), re.IGNORECASE)
                if match:
                    dlc_map[(match.group(1), match.group(2).replace('_', '-'))] = dlc_f

            unmatched_lucas = []
            for lf in label_files:
                match = re.search(fr'({animal_basename}\d+)[-_](hunt[-_]?\d+)', os.path.basename(lf), re.IGNORECASE)
                if match:
                    key = (match.group(1), match.group(2).replace('_', '-'))
                    if key in dlc_map:
                        paired_data.append({
                            'id': f"{key[0]}_{key[1]}",
                            'dlc_path': dlc_map[key],
                            'format': fmt,
                            'hunts': parse_lucas_labels(lf)
                        })
                    else:
                        unmatched_lucas.append(key)
            if verbose and unmatched_lucas:
                print(f"  [MISSING DLC] {len(unmatched_lucas)} labeled Lucas sessions had no matching DLC file: {sorted(unmatched_lucas)[:10]}")
        elif fmt == 'dan':
            animal_basename = cfg.get('animal_basename', 'PV')
            dan_labels = load_dan_labels(label_dir)
            matched_dan_keys = set()
            for dlc in dlc_files:
                base = os.path.basename(dlc)
                match = re.search(fr'({animal_basename}\d+)_D(\d+)_HS(\d+)', base, re.IGNORECASE)
                if match:
                    animal = match.group(1).upper()
                    day = int(match.group(2))
                    session = int(match.group(3))
                    key = (animal, session)
                    if key in dan_labels:
                        matched_dan_keys.add(key)
                        paired_data.append({
                            'id': f"{animal}_D{day}_HS{session}",
                            'dlc_path': dlc,
                            'format': fmt,
                            'hunts': dan_labels[key]
                        })
            if verbose:
                unmatched_dan = [k for k in dan_labels.keys() if k not in matched_dan_keys]
                if unmatched_dan:
                    print(f"  [MISSING DLC] {len(unmatched_dan)} labeled Dan sessions had no matching DLC file: {sorted(unmatched_dan)[:10]}")
                unmatched_dan_dlc = []
                for dlc in dlc_files:
                    base = os.path.basename(dlc)
                    match = re.search(fr'({animal_basename}\d+)_D(\d+)_HS(\d+)', base, re.IGNORECASE)
                    if match:
                        animal = match.group(1).upper()
                        session = int(match.group(3))
                        if (animal, session) not in dan_labels:
                            unmatched_dan_dlc.append(f"{animal}_HS{session} ({base})")
                if unmatched_dan_dlc:
                    print(f"  [MISSING IN LABELS] {len(unmatched_dan_dlc)} DLC files had no match in Dan labels: {unmatched_dan_dlc[:10]}")
        elif fmt in ('diane', 'diane_parts'):
            excel_path = cfg.get('excel_path') or cfg.get('label_file')
            ld = cfg.get('label_dir')
            if not excel_path and ld and os.path.exists(ld):
                excel_candidates = glob.glob(os.path.join(ld, '*.xlsx')) + glob.glob(os.path.join(ld, '*.csv'))
                if excel_candidates:
                    excel_path = excel_candidates[0]

            if not excel_path or not os.path.exists(excel_path):
                if verbose: print(f"Warning: Master Excel file not found for {ds_name} ({excel_path})")
                continue

            strat = str(cfg.get('strategy', 'engagement') or 'engagement')
            rec = bool(cfg.get('recursive', False))
            if fmt == 'diane_parts':
                diane_pairs = pair_diane_parts_dataset(
                    master_excel_path=excel_path,
                    dlc_dir=dlc_dir,
                    video_dir=cfg.get('video_dir'),
                    tentative_approach_strategy=strat,
                    recursive=rec,
                    verbose=verbose
                )
            else:
                diane_pairs = pair_diane_dataset(
                    master_excel_path=excel_path,
                    dlc_dir=dlc_dir,
                    video_dir=cfg.get('video_dir'),
                    tentative_approach_strategy=strat,
                    recursive=rec,
                    verbose=verbose
                )
            paired_data.extend(diane_pairs)
    if verbose: print(f"Total paired videos: {len(paired_data)}")
    return paired_data

def pair_diane_dataset(
    master_excel_path: str,
    dlc_dir: str,
    video_dir: Optional[str] = None,
    tentative_approach_strategy: str = "engagement",
    tentative_approach_max_sec: float = 2.0,
    tentative_label_state: int = 1,
    recursive: bool = False,
    split_parts: bool = False,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Pairs Diane format master behavioral manual labels with DLC pose .csv files.
    
    If split_parts=True, delegates to pair_diane_parts_dataset to match split video parts
    individually without stitching when frame counts reset back to zero.
    """
    if split_parts:
        return pair_diane_parts_dataset(
            master_excel_path=master_excel_path,
            dlc_dir=dlc_dir,
            video_dir=video_dir,
            tentative_approach_strategy=tentative_approach_strategy,
            tentative_approach_max_sec=tentative_approach_max_sec,
            tentative_label_state=tentative_label_state,
            recursive=recursive,
            verbose=verbose
        )
    """
    Pairs Diane format master behavioral manual labels with DLC pose .csv files.
    
    Parameters:
      master_excel_path: Path to master Excel file with columns:
                         [Animal, Day, Cricket, Released, First approach, Final approach, Capture, Eaten]
      dlc_dir: Directory containing DLC pose tracking CSV files (one per hunting session/day).
      video_dir: Optional directory containing matching video files (if different from dlc_dir).
      tentative_approach_strategy: 'engagement' (default continuous hunt from First approach to Final approach into Capture),
                                   'final_only' (only train on final attack), or 'split' (experimental fixed-duration split).
      tentative_approach_max_sec: Duration in seconds to assign if using experimental split strategy.
      tentative_label_state: State index for tentative approach if using experimental split strategy.
      recursive: Whether to search subdirectories recursively for DLC tracking files (default: False).
      verbose: Print scanning info.
    """
    parser = DianeExcelLabelParser(
        tentative_approach_max_sec=tentative_approach_max_sec,
        tentative_label_state=tentative_label_state
    )
    if verbose: print(f"Parsing Diane master labels: {master_excel_path}")
    if not os.path.exists(master_excel_path):
        if verbose: print(f"  [ERROR] Master Excel file not found: {master_excel_path}")
        return []

    session_hunts = parser.parse_file(master_excel_path)
    if verbose:
        total_hunts = sum(len(v) for v in session_hunts.values())
        print(f"  Parsed {len(session_hunts)} unique (Animal, Day) sessions ({total_hunts} total hunt events) from Excel.")
        sample_keys = list(session_hunts.keys())[:5]
        print(f"  Sample Excel keys: {sample_keys}")

    if not os.path.exists(dlc_dir):
        if verbose: print(f"  [ERROR] DLC directory does not exist or is not accessible: {dlc_dir}")
        return []

    # Search for both CSV and H5 files (recursive if specified)
    if recursive:
        csv_files = glob.glob(os.path.join(dlc_dir, "*.csv")) + glob.glob(os.path.join(dlc_dir, "**", "*.csv"), recursive=True)
        h5_files = glob.glob(os.path.join(dlc_dir, "*.h5")) + glob.glob(os.path.join(dlc_dir, "**", "*.h5"), recursive=True)
    else:
        csv_files = glob.glob(os.path.join(dlc_dir, "*.csv"))
        h5_files = glob.glob(os.path.join(dlc_dir, "*.h5"))
    all_found = csv_files + h5_files

    def is_tracking_file(f: str) -> bool:
        base = os.path.basename(f).lower()
        if base.startswith('~$'):
            return False
        if 'labeledinds' in base or 'labeledbouts' in base:
            return False
        if 'dlc' not in base:
            return False
        return base.endswith('.csv') or base.endswith('.h5')

    all_tracking = [f for f in all_found if is_tracking_file(f)]

    # Deduplicate between .csv and .h5: if both stem.csv and stem.h5 exist, prefer .csv
    file_map = {}
    for f in all_tracking:
        stem = os.path.splitext(f)[0]
        ext = os.path.splitext(f)[1].lower()
        if stem not in file_map or ext == '.csv':
            file_map[stem] = f
    dlc_files = sorted(list(file_map.values()))
    
    if verbose:
        print(f"  Found {len(dlc_files)} unique DLC tracking files in {dlc_dir} (recursive={recursive})")
        if dlc_files:
            print(f"  Sample DLC files found: {[os.path.basename(f) for f in dlc_files[:3]]}")
        else:
            print(f"  [WARNING] No valid DLC tracking files found in {dlc_dir}.")

    matcher = DatasetMatcher(extract_video_fps=True, default_fps=DEFAULT_FPS)
    paired_data = []
    seen_sessions = set()
    
    # Pre-index video files strictly from video_dir (if provided) or dlc_dir
    video_dirs_to_check = []
    if video_dir and os.path.exists(video_dir):
        video_dirs_to_check.append(video_dir)
    elif dlc_dir and os.path.exists(dlc_dir):
        video_dirs_to_check.append(dlc_dir)

    all_raw_videos = []
    video_files = {}
    v_exts = ('*.mp4', '*.avi', '*.mkv', '*.mov')
    for vd in video_dirs_to_check:
        for ext in v_exts:
            if recursive:
                all_raw_videos.extend(glob.glob(os.path.join(vd, ext)) + glob.glob(os.path.join(vd, "**", ext), recursive=True))
            else:
                all_raw_videos.extend(glob.glob(os.path.join(vd, ext)))

    # Deduplicate video files
    all_raw_videos = list(dict.fromkeys(all_raw_videos))
    for v in all_raw_videos:
        base = os.path.splitext(os.path.basename(v))[0]
        video_files[base] = v

    def find_excel_key(animal_str: str, day_val: int) -> Optional[Tuple]:
        clean_num = animal_str.replace("YFP-", "").replace("YFP_", "").replace("YFP", "").replace("YB-", "").replace("YB_", "").replace("YB", "").strip()
        candidates = [
            animal_str,
            clean_num,
            f"YFP-{clean_num}",
            f"YFP_{clean_num}",
            f"YB-{clean_num}",
            f"YB_{clean_num}",
        ]
        for cand in candidates:
            if (cand, day_val) in session_hunts:
                return (cand, day_val)
        for (sh_animal, sh_day) in session_hunts.keys():
            if sh_day == day_val:
                sh_clean = sh_animal.upper().replace("YFP-", "").replace("YFP_", "").replace("YFP", "").replace("YB-", "").replace("YB_", "").replace("YB", "").strip()
                if sh_clean == clean_num or sh_animal.upper() == animal_str:
                    return (sh_animal, sh_day)
        return None

    unparsed_dlc = []
    unmatched_dlc = []
    session_dlc_map: Dict[Tuple, List[str]] = {}

    for dlc_f in dlc_files:
        info = matcher.parse_filename(dlc_f)
        if not info:
            unparsed_dlc.append(os.path.basename(dlc_f))
            continue
        
        animal = str(info['animal_id']).upper()
        day = info['day']
        
        matched_key = find_excel_key(animal, day)
        if matched_key:
            if matched_key not in session_dlc_map:
                session_dlc_map[matched_key] = []
            session_dlc_map[matched_key].append(dlc_f)
        else:
            unmatched_dlc.append((animal, day, os.path.basename(dlc_f)))

    stitched_sessions = []

    for matched_key, matched_files in session_dlc_map.items():
        animal, day = matched_key
        # Sort files chronologically based on filename timestamp
        sorted_files = sorted(matched_files, key=extract_filename_timestamp)
        if len(sorted_files) > 1:
            stitched_sessions.append((animal, day, sorted_files))

        seen_sessions.add(matched_key)

        hunt_entries = session_hunts[matched_key]
        events = parser.hunts_to_events(hunt_entries, tentative_approach_strategy=tentative_approach_strategy)
        
        # Find video(s)
        part_videos = []
        for f in sorted_files:
            v_path = matcher.find_video_for_dlc(f)
            if not v_path and video_files:
                base = os.path.basename(f).split('DLC')[0].rstrip('_-')
                for v_base, v_p in video_files.items():
                    if base in v_base or v_base in base:
                        v_path = v_p
                        break
            part_videos.append(v_path)

        primary_video = part_videos[0] if part_videos else None
        primary_dlc = sorted_files if len(sorted_files) > 1 else sorted_files[0]

        paired_data.append({
            'id': f"{animal}_Day{day}",
            'animal': animal,
            'day': day,
            'dlc_path': primary_dlc,
            'dlc_paths': sorted_files,
            'video_path': primary_video,
            'video_paths': part_videos,
            'format': 'diane',
            'is_stitched': (len(sorted_files) > 1),
            'hunts': [
                {
                    'enter': ev.enter,
                    'first_approach': ev.first_approach if ev.first_approach is not None else ev.approach_start,
                    'pursuit_start': ev.approach_start,
                    'capture': ev.capture,
                    'consumption_end': ev.consumption_end,
                    'metadata': ev.metadata
                }
                for ev in events
            ]
        })

    # Check video files directly for missing Excel matches
    unmatched_videos = []
    unparsed_videos = []
    for v_f in all_raw_videos:
        v_base = os.path.basename(v_f).lower()
        if 'labeled' in v_base or 'annotated' in v_base:
            continue
        v_info = matcher.parse_filename(v_f)
        if not v_info:
            unparsed_videos.append(os.path.basename(v_f))
            continue
        v_animal = str(v_info['animal_id']).upper()
        v_day = v_info['day']
        v_match = find_excel_key(v_animal, v_day)
        if not v_match:
            unmatched_videos.append((v_animal, v_day, os.path.basename(v_f)))

    if verbose:
        # 1. Sessions in master Excel that have no DLC tracking file
        missing_dlc = [k for k in session_hunts.keys() if k not in seen_sessions]
        if missing_dlc:
            sorted_misses = sorted(missing_dlc, key=lambda x: (x[0], int(x[1]) if str(x[1]).isdigit() else 0))
            if len(sorted_misses) <= 10:
                print(f"  [MISSING DLC] {len(sorted_misses)} sessions in master Excel had no matching DLC tracking file: {sorted_misses}")
            else:
                print(f"  [MISSING DLC] {len(sorted_misses)} sessions in master Excel had no matching DLC tracking file: {sorted_misses[:10]} ... (+{len(sorted_misses)-10} more)")

        # 2. DLC tracking files on disk that have no match in master Excel
        if unmatched_dlc:
            sorted_unmatched_dlc = sorted(unmatched_dlc, key=lambda x: (x[0], int(x[1]) if str(x[1]).isdigit() else 0, x[2]))
            sample_dlc_str = [f"({u[0]}, Day {u[1]}): {u[2]}" for u in sorted_unmatched_dlc]
            if len(sample_dlc_str) <= 10:
                print(f"  [MISSING IN EXCEL] {len(sorted_unmatched_dlc)} DLC tracking files had no matching session in master Excel:\n    " + "\n    ".join(sample_dlc_str))
            else:
                print(f"  [MISSING IN EXCEL] {len(sorted_unmatched_dlc)} DLC tracking files had no matching session in master Excel:\n    " + "\n    ".join(sample_dlc_str[:10]) + f"\n    ... (+{len(sample_dlc_str)-10} more)")

        # 2b. Multi-part stitched DLC tracking files
        if stitched_sessions:
            print(f"  [STITCHED SESSIONS] {len(stitched_sessions)} sessions with camera freezes stitched in chronological order:")
            for s_animal, s_day, s_files in stitched_sessions:
                parts_str = " -> ".join([os.path.basename(sf) for sf in s_files])
                print(f"    - ({s_animal}, Day {s_day}): {len(s_files)} parts:\n      {parts_str}")

        # 3. Video files on disk that have no match in master Excel
        if unmatched_videos:
            sorted_unmatched_vid = sorted(unmatched_videos, key=lambda x: (x[0], int(x[1]) if str(x[1]).isdigit() else 0, x[2]))
            sample_vid_str = [f"({u[0]}, Day {u[1]}): {u[2]}" for u in sorted_unmatched_vid]
            if len(sample_vid_str) <= 10:
                print(f"  [MISSING IN EXCEL] {len(sorted_unmatched_vid)} video files had no matching session in master Excel:\n    " + "\n    ".join(sample_vid_str))
            else:
                print(f"  [MISSING IN EXCEL] {len(sorted_unmatched_vid)} video files had no matching session in master Excel:\n    " + "\n    ".join(sample_vid_str[:10]) + f"\n    ... (+{len(sample_vid_str)-10} more)")

        # 4. Paired DLC sessions with no matching raw video file
        missing_videos = [p['id'] for p in paired_data if not p.get('video_path')]
        if missing_videos:
            if len(missing_videos) <= 10:
                print(f"  [MISSING VIDEO] {len(missing_videos)} paired DLC sessions had no matching video file: {missing_videos}")
            else:
                print(f"  [MISSING VIDEO] {len(missing_videos)} paired DLC sessions had no matching video file: {missing_videos[:10]} ... (+{len(missing_videos)-10} more)")

        if unparsed_dlc:
            print(f"  [UNPARSED DLC] {len(unparsed_dlc)} files could not be parsed by filename regex. Examples: {unparsed_dlc[:3]}")
        if unparsed_videos:
            print(f"  [UNPARSED VIDEO] {len(unparsed_videos)} video files could not be parsed by filename regex. Examples: {unparsed_videos[:3]}")
        print(f"Total paired Diane sessions: {len(paired_data)}")
    return paired_data

def pair_diane_parts_dataset(
    master_excel_path: str,
    dlc_dir: str,
    video_dir: Optional[str] = None,
    tentative_approach_strategy: str = "engagement",
    tentative_approach_max_sec: float = 2.0,
    tentative_label_state: int = 1,
    recursive: bool = False,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Pairs Diane format master behavioral manual labels with DLC pose files in 'diane_parts' format.
    In this format, multi-part split videos within a single session have frame numbers that reset
    back to 0 for each split video part (instead of continuous concatenated frames).
    
    Each part is matched 1-to-1 with its corresponding DLC file (ordered chronologically by timestamp)
    and treated as an independent session without stitching.
    """
    parser = DianeExcelLabelParser(
        tentative_approach_max_sec=tentative_approach_max_sec,
        tentative_label_state=tentative_label_state
    )
    if verbose: print(f"Parsing Diane parts master labels: {master_excel_path}")
    if not os.path.exists(master_excel_path):
        if verbose: print(f"  [ERROR] Master Excel file not found: {master_excel_path}")
        return []

    session_parts = parser.parse_file_parts(master_excel_path)
    if verbose:
        total_parts = sum(len(parts) for parts in session_parts.values())
        total_hunts = sum(sum(len(h) for h in parts) for parts in session_parts.values())
        multi_part_count = sum(1 for parts in session_parts.values() if len(parts) > 1)
        print(f"  Parsed {len(session_parts)} unique (Animal, Day) sessions ({total_parts} video parts, {total_hunts} total hunt events).")
        if multi_part_count > 0:
            print(f"  Found {multi_part_count} multi-part session(s) with frame counter resets in Excel.")

    if not os.path.exists(dlc_dir):
        if verbose: print(f"  [ERROR] DLC directory does not exist or is not accessible: {dlc_dir}")
        return []

    # Search for tracking files
    if recursive:
        csv_files = glob.glob(os.path.join(dlc_dir, "*.csv")) + glob.glob(os.path.join(dlc_dir, "**", "*.csv"), recursive=True)
        h5_files = glob.glob(os.path.join(dlc_dir, "*.h5")) + glob.glob(os.path.join(dlc_dir, "**", "*.h5"), recursive=True)
    else:
        csv_files = glob.glob(os.path.join(dlc_dir, "*.csv"))
        h5_files = glob.glob(os.path.join(dlc_dir, "*.h5"))
    all_found = csv_files + h5_files

    def is_tracking_file(f: str) -> bool:
        base = os.path.basename(f).lower()
        if base.startswith('~$') or 'labeledinds' in base or 'labeledbouts' in base or 'dlc' not in base:
            return False
        return base.endswith('.csv') or base.endswith('.h5')

    all_tracking = [f for f in all_found if is_tracking_file(f)]
    file_map = {}
    for f in all_tracking:
        stem = os.path.splitext(f)[0]
        ext = os.path.splitext(f)[1].lower()
        if stem not in file_map or ext == '.csv':
            file_map[stem] = f
    dlc_files = sorted(list(file_map.values()))

    matcher = DatasetMatcher(extract_video_fps=True, default_fps=DEFAULT_FPS)
    paired_data = []

    # Pre-index video files
    video_dirs_to_check = []
    if video_dir and os.path.exists(video_dir):
        video_dirs_to_check.append(video_dir)
    elif dlc_dir and os.path.exists(dlc_dir):
        video_dirs_to_check.append(dlc_dir)

    all_raw_videos = []
    video_files = {}
    v_exts = ('*.mp4', '*.avi', '*.mkv', '*.mov')
    for vd in video_dirs_to_check:
        for ext in v_exts:
            if recursive:
                all_raw_videos.extend(glob.glob(os.path.join(vd, ext)) + glob.glob(os.path.join(vd, "**", ext), recursive=True))
            else:
                all_raw_videos.extend(glob.glob(os.path.join(vd, ext)))
    all_raw_videos = list(dict.fromkeys(all_raw_videos))
    for v in all_raw_videos:
        base = os.path.splitext(os.path.basename(v))[0]
        video_files[base] = v

    def find_excel_key(animal_str: str, day_val: int) -> Optional[Tuple]:
        clean_num = animal_str.replace("YFP-", "").replace("YFP_", "").replace("YFP", "").replace("YB-", "").replace("YB_", "").replace("YB", "").strip()
        candidates = [
            animal_str,
            clean_num,
            f"YFP-{clean_num}",
            f"YFP_{clean_num}",
            f"YB-{clean_num}",
            f"YB_{clean_num}",
        ]
        for cand in candidates:
            if (cand, day_val) in session_parts:
                return (cand, day_val)
        for (sh_animal, sh_day) in session_parts.keys():
            if sh_day == day_val:
                sh_clean = sh_animal.upper().replace("YFP-", "").replace("YFP_", "").replace("YFP", "").replace("YB-", "").replace("YB_", "").replace("YB", "").strip()
                if sh_clean == clean_num or sh_animal.upper() == animal_str:
                    return (sh_animal, sh_day)
        return None

    session_dlc_map: Dict[Tuple, List[str]] = {}
    for dlc_f in dlc_files:
        info = matcher.parse_filename(dlc_f)
        if not info:
            continue
        animal = str(info['animal_id']).upper()
        day = info['day']
        matched_key = find_excel_key(animal, day)
        if matched_key:
            if matched_key not in session_dlc_map:
                session_dlc_map[matched_key] = []
            session_dlc_map[matched_key].append(dlc_f)

    for matched_key, matched_files in session_dlc_map.items():
        animal, day = matched_key
        sorted_files = sorted(matched_files, key=extract_filename_timestamp)
        parts_list = session_parts[matched_key]

        is_multi_file = len(sorted_files) > 1
        is_multi_part = len(parts_list) > 1

        if (is_multi_file or is_multi_part) and verbose:
            print(f"  Session {animal} Day {day}: multi-part detected ({len(sorted_files)} DLC file(s), {len(parts_list)} Excel part(s)).")

        for p_idx, part_hunts in enumerate(parts_list):
            if p_idx >= len(sorted_files):
                if verbose:
                    print(f"  [WARNING] {animal} Day {day} Part {p_idx+1} has Excel labels but only {len(sorted_files)} DLC file(s) found. Skipping.")
                continue

            dlc_f = sorted_files[p_idx]
            events = parser.hunts_to_events(part_hunts, tentative_approach_strategy=tentative_approach_strategy)

            v_path = matcher.find_video_for_dlc(dlc_f)
            if not v_path and video_files:
                base = os.path.basename(dlc_f).split('DLC')[0].rstrip('_-')
                for v_base, v_p in video_files.items():
                    if base in v_base or v_base in base:
                        v_path = v_p
                        break

            if is_multi_file or is_multi_part:
                sess_id = f"{animal}_Day{day}_Part{p_idx+1}"
            else:
                sess_id = f"{animal}_Day{day}"

            paired_data.append({
                'id': sess_id,
                'animal': animal,
                'day': day,
                'part': p_idx + 1,
                'dlc_path': dlc_f,
                'dlc_paths': [dlc_f],
                'video_path': v_path,
                'video_paths': [v_path] if v_path else [],
                'format': 'diane_parts',
                'is_stitched': False,
                'hunts': [
                    {
                        'enter': ev.enter,
                        'first_approach': ev.first_approach if ev.first_approach is not None else ev.approach_start,
                        'pursuit_start': ev.approach_start,
                        'capture': ev.capture,
                        'consumption_end': ev.consumption_end,
                        'metadata': ev.metadata
                    }
                    for ev in events
                ]
            })

        if len(sorted_files) > len(parts_list) and verbose:
            unlabeled_files = [os.path.basename(f) for f in sorted_files[len(parts_list):]]
            print(f"  Note: {animal} Day {day} has {len(sorted_files)} DLC files but only {len(parts_list)} parts in Excel labels. Skipping unlabelled file(s): {unlabeled_files}")

    if verbose:
        print(f"Total paired 'diane_parts' sessions/parts: {len(paired_data)}")
    return paired_data

def calculate_empirical_transition_matrix(pairs=None, paradigm="rat"):
    if pairs is None:
        if paradigm == "mouse":
            pairs = pair_dataset_files(datasets_config=MOUSE_DATASETS, verbose=True)
        else:
            pairs = pair_dataset_files(datasets_config=DAN_DATASETS, verbose=True)

    all_events = []
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

    num_classes = 5 if paradigm == "mouse" else 4
    return modular_transition_matrix(all_events, fps=DEFAULT_FPS, num_classes=num_classes, paradigm=paradigm)

def verify_split_handling(*args, **kwargs):
    """Generates verification test plots for multi-part video split handling."""
    from verify_split_handling import verify_split_handling as _verify
    return _verify(*args, **kwargs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dataset utilities & multi-part split verification.")
    parser.add_argument(
        "--format", type=str, default="diane", choices=["diane_parts", "diane", "both"],
        help="Format to verify: 'diane_parts' (resetting frames), 'diane' (stitched), or 'both' (default: diane_parts)"
    )
    parser.add_argument(
        "--dlc_path", type=str,
        default=os.path.join(BASE_SERVER_PATH, r"DianeBissen\Cricket hunting videos\Craniotomy hunting videos_DLC"),
        help="Path to DLC tracking CSV file or directory of DLC files"
    )
    parser.add_argument(
        "--excel_path", type=str,
        default=os.path.join(BASE_SERVER_PATH, r"DianeBissen\Cricket hunting videos\Cricket_hunting_all_hunting_videos_labeling_Diane_BCFixed_MostUpdated.xlsx"),
        help="Path to Diane master Excel file"
    )
    parser.add_argument(
        "--save_dir", type=str,
        default="figs/split_verification",
        help="Output directory to save verification figures (default: figs/split_verification)"
    )
    parser.add_argument(
        "--num_examples", type=int, default=2,
        help="Number of split examples to find and plot (default: 2)"
    )
    parser.add_argument(
        "--feature", type=str, default="both", choices=["speed", "tail_angle", "tail", "both"],
        help="Feature to overlay: 'speed', 'tail_angle', or 'both' (default: both)"
    )
    parser.add_argument(
        "--show", action="store_true", default=False,
        help="Show interactive plot window in addition to saving"
    )

    args = parser.parse_args()

    verify_split_handling(
        dlc_path=args.dlc_path,
        excel_path=args.excel_path,
        save_dir=args.save_dir,
        format_type=args.format,
        num_examples=args.num_examples,
        feature=args.feature,
        show_plot=args.show,
        verbose=True
    )


import os
import glob
import re
import cv2
import pandas as pd
import numpy as np

BASE_SERVER_PATH = r"\\research.brandeis.edu\turrigiano-lab"

# DATASETS = {
#     'Lucas_Exp4': {
#         'label_dir': os.path.join(BASE_SERVER_PATH, r'LucasWahl\Excel files\Exp 4\WT observer'),
#         'dlc_dir': os.path.join(BASE_SERVER_PATH, r'LucasWahl\Exp 4 - WT naive + WT observer data\WT observer data\hunt_crop'),
#         'animal_basename': 'LW',
#         'format': 'lucas',
#     },
#     'Lucas_Exp5': {
#         'label_dir': os.path.join(BASE_SERVER_PATH, r'LucasWahl\Excel files\Exp 5\Scn2a Observer'),
#         'dlc_dir': os.path.join(BASE_SERVER_PATH, r'LucasWahl\Exp 5 - Scn2a mut observer data\Scn2a observer\hunt_crop'),
#         'animal_basename': 'LW',
#         'format': 'lucas',
#     },
#     'Dan_Dataset': {
#         'label_dir': os.path.join(BASE_SERVER_PATH, r'Path\To\Dan\Labels'), 
#         'dlc_dir': os.path.join(BASE_SERVER_PATH, r'Path\To\Dan\DLC'),
#         'animal_basename': 'PV',
#         'file_filter': None,
#         'format': 'dan',
#     }
# }
DATASETS = {
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

DAN_DATASETS = {
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
    }
}



VERBOSE = True
DEFAULT_WIDTH = 514
DEFAULT_HEIGHT = 544
DEFAULT_FPS = 30.0

# ==========================================
# Common Pipeline Configuration
# ==========================================
# Context Settings
CONTEXT_BUFFER = 300  # Frames of background to include BEFORE Enter
EATING_BUFFER = 300   # Updated to 300 as discussed for better generalization

# Sequence Length Randomization for Background
MIN_BG_LEN = 500   # ~16 seconds
MAX_BG_LEN = 2000  # ~66 seconds

# Max number of background chunks to generate per video 
MAX_BG_CHUNKS_PER_VIDEO = 5

# Full Video Sequential Chunking
FULL_VIDEO_CHUNK_MIN = 1000
FULL_VIDEO_CHUNK_MAX = 5000
FULL_VIDEO_CHUNK_PROPORTION = 0.15
FULL_VIDEO_CHUNK_ITERATIONS = 2
# ==========================================
# Interpolation Settings
RAT_LIKELIHOOD_THRESH = 0.9
CRICKET_LIKELIHOOD_THRESH = 0.85
MAX_INTERP_GAP = 15 

# Dead Cricket Filtering Settings
CRICKET_MOVEMENT_THRESH = 0.3
CRICKET_DEAD_TIME_THRESH = 30

# Feature Normalization Bounds
MAX_SPEED_THRESH = 3.0  # max expected speed in video-widths per sec
MAX_ACCEL_THRESH = 25.0  # max expected accel in video-widths per sec^2

# ==========================================
# Common DLC Data Processing Functions
# ==========================================

def calculate_angle_3pts(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.sum(ba*bc, axis=1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_bearing(start_pt, end_pt):
    return np.arctan2(end_pt[:, 1] - start_pt[:, 1], end_pt[:, 0] - start_pt[:, 0])

def apply_poop_penalty(df, scorer, c_x, c_y, c_conf, poop_thresh=0.9, poop_dist_thresh=4.0, cricket_thresh=CRICKET_LIKELIHOOD_THRESH):
    """
    Finds all 'poop' detections in the video. If a cricket detection is within 
    dist_thresh pixels of ANY valid poop detection, lowers the cricket confidence.
    """
    bodyparts = df[scorer].columns.get_level_values(0).unique()
    poop_parts = [bp for bp in bodyparts if isinstance(bp, str) and bp.startswith('poop')]
    
    if not poop_parts:
        print("POOP FILTER DEBUG: No bodyparts starting with 'poop' found in the DLC file!")
        return c_conf

    all_px, all_py = [], []
    for poop_part in poop_parts:
        p_x = df[scorer][poop_part]['x'].values
        p_y = df[scorer][poop_part]['y'].values
        p_conf = df[scorer][poop_part]['likelihood'].values
        
        valid = p_conf > poop_thresh
        if np.any(valid):
            all_px.extend(p_x[valid])
            all_py.extend(p_y[valid])
    
    if not all_px:
        print(f"POOP FILTER DEBUG: No poop detected with confidence > {poop_thresh} across {len(poop_parts)} labels.")
        return c_conf
        
    poop_coords = np.column_stack((all_px, all_py))
    print(f"POOP FILTER DEBUG: Found {len(poop_coords)} total poop coordinate instances across {len(poop_parts)} labels.")
    
    from scipy.spatial import cKDTree
    tree = cKDTree(poop_coords)
    
    # Use .values to ensure we are working with raw numpy arrays, preventing pandas index misalignment
    c_x_vals = c_x.values if hasattr(c_x, 'values') else c_x
    c_y_vals = c_y.values if hasattr(c_y, 'values') else c_y
    c_conf_vals = c_conf.values if hasattr(c_conf, 'values') else c_conf
    
    c_coords = np.column_stack((c_x_vals, c_y_vals))
    valid_c = ~np.isnan(c_x_vals) & ~np.isnan(c_y_vals)
    
    new_c_conf = c_conf_vals.copy()
    
    if np.any(valid_c):
        distances, _ = tree.query(c_coords[valid_c])
        too_close_mask = distances < poop_dist_thresh
        
        # We must subtract from the difference so that a perfect 1.0 confidence gets dragged just above the threshold
        penalty = (1.0 - cricket_thresh) - 0.02
        
        valid_indices = np.where(valid_c)[0]
        too_close_indices = valid_indices[too_close_mask]
        
        new_c_conf[too_close_indices] -= penalty
        print(f"POOP FILTER DEBUG: Penalized {len(too_close_indices)} cricket points that were too close to poop.")
        
    # Return as Pandas Series if the original was a Pandas Series to maintain downstream index-based assignment
    if hasattr(c_conf, 'index'):
        import pandas as pd
        return pd.Series(new_c_conf, index=c_conf.index)
        
    return new_c_conf

def extract_dlc_tracking_data(filepath, rat_thresh=RAT_LIKELIHOOD_THRESH, cricket_thresh=CRICKET_LIKELIHOOD_THRESH,
                              interp_limit=MAX_INTERP_GAP, include_blurry_cricket=False, filter_poops=False, poop_thresh=0.75, poop_dist_thresh=4.0):
    try:
        df = pd.read_csv(filepath, header=[0, 1, 2], index_col=0)
    except Exception as e:
        if VERBOSE: print(f"Error loading CSV {filepath}: {e}")
        return None
        
    scorer = df.columns.get_level_values(0)[0]
    
    def get_part(part_name, thresh, interp_limit):
        if part_name not in df[scorer]:
            return np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))
            
        x = df[scorer][part_name]['x'].copy()
        y = df[scorer][part_name]['y'].copy()
        conf = df[scorer][part_name]['likelihood'].copy()
        
        mask = conf < thresh
        x[mask] = np.nan
        y[mask] = np.nan
        
        if interp_limit > 0:
            x = x.interpolate(method='linear', limit=interp_limit, limit_direction='both')
            y = y.interpolate(method='linear', limit=interp_limit, limit_direction='both')
        
        return x.values, y.values, conf.values

    # Process Rat
    nose_x, nose_y, _ = get_part('nose', rat_thresh, interp_limit)
    head_x, head_y, _ = get_part('head', rat_thresh, interp_limit)
    mid_x, mid_y, _   = get_part('mid_body', rat_thresh, interp_limit)
    butt_x, butt_y, _ = get_part('butt', rat_thresh, interp_limit)
    tail_x, tail_y, _ = get_part('endtail', rat_thresh, interp_limit)

    rat_head = np.column_stack((head_x, head_y))
    rat_nose = np.column_stack((nose_x, nose_y))
    mask_head = np.isnan(rat_head[:,0])
    rat_head[mask_head] = rat_nose[mask_head]
    
    rat_mid = np.column_stack((mid_x, mid_y))
    rat_butt = np.column_stack((butt_x, butt_y))
    rat_tail = np.column_stack((tail_x, tail_y))

    # Process Cricket
    c_x, c_y, c_conf = get_part('cricket', 0.0, interp_limit)
    
    if filter_poops:
        c_conf = apply_poop_penalty(df, scorer, c_x, c_y, c_conf, poop_thresh, poop_dist_thresh, cricket_thresh)
        
    final_cx = np.full_like(c_x, np.nan)
    final_cy = np.full_like(c_y, np.nan)
    
    c_valid = c_conf > cricket_thresh
    
    if include_blurry_cricket:
        bc_x, bc_y, bc_conf = get_part('blurry_cricket', 0.0, interp_limit)
        
        if filter_poops:
            bc_conf = apply_poop_penalty(df, scorer, bc_x, bc_y, bc_conf, poop_thresh, poop_dist_thresh, cricket_thresh)
            
        bc_valid = bc_conf > cricket_thresh
        
        final_cx[bc_valid] = bc_x[bc_valid]
        final_cy[bc_valid] = bc_y[bc_valid]
        
    final_cx[c_valid] = c_x[c_valid]
    final_cy[c_valid] = c_y[c_valid]
    
    if interp_limit > 0:
        final_cx = pd.Series(final_cx).interpolate(limit=interp_limit).values
        final_cy = pd.Series(final_cy).interpolate(limit=interp_limit).values
        
    cricket_pos = np.column_stack((final_cx, final_cy))
    return rat_head, rat_mid, rat_butt, rat_tail, cricket_pos

def load_and_process_dlc(filepath, video_fps, include_blurry_cricket=False, filter_poops=False, poop_thresh=0.75, poop_dist_thresh=4.0):
    res = extract_dlc_tracking_data(filepath, RAT_LIKELIHOOD_THRESH, CRICKET_LIKELIHOOD_THRESH, MAX_INTERP_GAP, include_blurry_cricket, filter_poops, poop_thresh, poop_dist_thresh)
    if res is None: return None
    rat_head, rat_mid, rat_butt, rat_tail, cricket_pos = res
    
    final_cx = cricket_pos[:, 0].copy()
    final_cy = cricket_pos[:, 1].copy()
    
    # Filter fake/dead crickets
    cx_diff = np.diff(final_cx, prepend=final_cx[0])
    cy_diff = np.diff(final_cy, prepend=final_cy[0])
    movement = np.sqrt(cx_diff**2 + cy_diff**2)
    
    window_size = max(1, int(0.5 * video_fps))
    smoothed_movement = pd.Series(movement).rolling(window=window_size, min_periods=1).mean().values
    
    is_dead = smoothed_movement < CRICKET_MOVEMENT_THRESH
    is_dead_clean = np.nan_to_num(is_dead, nan=0).astype(bool)
    
    padded = np.insert(is_dead_clean, [0, len(is_dead_clean)], [False, False])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    dead_frame_thresh = int(CRICKET_DEAD_TIME_THRESH * video_fps)
    
    for s, e in zip(starts, ends):
        if (e - s) >= dead_frame_thresh:
            final_cx[s:e] = np.nan
            final_cy[s:e] = np.nan
    
    cricket_pos = np.column_stack((final_cx, final_cy))

    return rat_head, rat_mid, rat_butt, rat_tail, cricket_pos

def generate_features(rat_head, rat_mid, rat_butt, rat_tail, cricket_pos, video_width, video_fps):
    def dist(p1, p2): return np.linalg.norm(p1 - p2, axis=1)
    def get_derivs(pos_array):
        vel_vec = np.diff(pos_array, axis=0, prepend=pos_array[0:1]) * video_fps
        speed = np.linalg.norm(vel_vec, axis=1) / video_width
        acc_vec = np.diff(vel_vec, axis=0, prepend=vel_vec[0:1]) * video_fps
        accel = np.linalg.norm(acc_vec, axis=1) / video_width
        return speed, accel

    d_head_cricket = dist(rat_head, cricket_pos) / video_width
    d_head_mid = dist(rat_head, rat_mid) / video_width
    d_mid_tail = dist(rat_mid, rat_tail) / video_width
    d_head_tail = dist(rat_head, rat_tail) / video_width
    
    angle_spine = calculate_angle_3pts(rat_head, rat_mid, rat_tail) / 180.0
    rat_orientation = calculate_bearing(rat_mid, rat_head)
    target_vector = calculate_bearing(rat_head, cricket_pos)
    angle_diff = target_vector - rat_orientation
    angle_diff_raw = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    angle_diff = angle_diff_raw / np.pi
    
    rat_speed, rat_accel = get_derivs(rat_head)
    cricket_speed, cricket_accel = get_derivs(cricket_pos)
    cricket_visible = np.where(np.isnan(d_head_cricket), 0.0, 1.0)
    
    closing_velocity = -np.diff(d_head_cricket, prepend=d_head_cricket[0:1]) * video_fps
    inverse_ttc = np.clip(closing_velocity / (d_head_cricket + 1e-6), -100, 100) 

    def angular_diff(angles):
        d_angles = np.diff(angles, prepend=angles[0:1])
        return (d_angles + np.pi) % (2 * np.pi) - np.pi

    rat_turn_rate = np.abs(angular_diff(rat_orientation)) * video_fps / np.pi
    aim_correction_rate = np.abs(angular_diff(angle_diff_raw)) * video_fps / np.pi

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
    c_jitter = np.abs(angular_diff(c_bearing)) / np.pi
    
    # === ROBUST NORMALIZATION ===
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
    feature_stack = np.nan_to_num(feature_stack, nan=0.0)

    return feature_stack

def get_video_params(dlc_csv_path, verbose=False):
    """Find the associated video using the DLC base name and extract its properties via OpenCV."""
    dir_path = os.path.dirname(dlc_csv_path)
    base_name = os.path.basename(dlc_csv_path).split('DLC')[0]
    
    for ext in ['.mp4', '.avi', '.mkv']:
        vid_path = os.path.join(dir_path, base_name + ext)
        if os.path.exists(vid_path):
            cap = cv2.VideoCapture(vid_path)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if width > 0 and height > 0 and fps > 0:
                    if verbose: print(f"Found video {base_name}{ext}: {width}x{height} @ {fps} FPS")
                    return width, height, fps
                    
    if verbose: print(f"Warning: Could not read video properties for {base_name}. Using defaults.")
    return DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS

def hunts_to_sequence(hunts, num_frames):
    """
    Converts a list of hunt dictionaries to a 1D numpy array of frame labels:
    0: Background, 1: Stalking, 2: Pursuit, 3: Eating
    """
    seq = np.zeros(num_frames, dtype=int)
    for row in hunts:
        enter = row['enter']
        capture = row['capture']
        pursuit_start = row['pursuit_start']
        cons_end = row.get('consumption_end')
        
        eating_end = cons_end if cons_end is not None else capture + EATING_BUFFER
        
        enter = max(0, min(num_frames, enter))
        pursuit_start = max(enter, min(num_frames, pursuit_start))
        capture = max(pursuit_start, min(num_frames, capture))
        eating_end = max(capture, min(num_frames, eating_end))
        
        if pursuit_start > enter: seq[enter:pursuit_start] = 1
        if capture > pursuit_start: seq[pursuit_start:capture] = 2
        if eating_end > capture: seq[capture:eating_end] = 3
    return seq

def parse_lucas_labels(label_path):
    """
    Parses a Lucas/Standard analyzed label spreadsheet (.xlsx or .csv).
    Returns a list of hunt dictionaries.
    """
    if label_path.endswith('.csv'):
        df = pd.read_csv(label_path)
    else:
        df = pd.read_excel(label_path)
        
    df.columns = df.columns.astype(str).str.strip()
    if 'Include?' in df.columns:
        valid_hunts = df[df['Include?'] != 'Exclude'].sort_values('Enter_Frame')
    else:
        valid_hunts = df.sort_values('Enter_Frame')
        
    hunts = []
    for _, row in valid_hunts.iterrows():
        if pd.isna(row.get('Enter_Frame')) or pd.isna(row.get('Capture_Frame')):
            continue
        enter = int(row['Enter_Frame'])
        capture = int(row['Capture_Frame'])
        
        if 'Pursuit_Frame' in row and pd.notna(row['Pursuit_Frame']):
            pursuit_start = int(row['Pursuit_Frame'])
        elif 'Pursuit_Start_Frame' in row and pd.notna(row['Pursuit_Start_Frame']):
            pursuit_start = int(row['Pursuit_Start_Frame'])
        elif 'Latency_to_Attack_Frames' in row and pd.notna(row['Latency_to_Attack_Frames']):
            pursuit_start = enter + int(row['Latency_to_Attack_Frames'])
        else:
            pursuit_start = enter
            
        cons_end = None
        if 'Consumption_End_Frame' in row and pd.notna(row['Consumption_End_Frame']):
            cons_end = int(row['Consumption_End_Frame'])
            
        hunts.append({
            'enter': enter,
            'capture': capture,
            'pursuit_start': pursuit_start,
            'consumption_end': cons_end
        })
    return hunts

def parse_dan_label_file(file_path):
    """
    Parses a single Dan label file (.xlsx or .csv).
    Animal name appears once per file — grabs it from the first non-NaN entry.
    Returns:
        dict: mapping (animal_upper, session_int) -> list of hunt dictionaries
    """
    dan_hunts = {}
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
        
    df.columns = df.columns.astype(str).str.strip()
    if not all(col in df.columns for col in ['Animal', 'Day', 'Session']):
        return dan_hunts
        
    animal_series = df['Animal'].dropna()
    if animal_series.empty:
        return dan_hunts
    animal = str(animal_series.iloc[0]).strip().upper()
    
    for _, row in df.iterrows():
        if pd.isna(row.get('On')) or pd.isna(row.get('Capture')):
            continue
            
        try:
            day = int(row['Day'])
            session = int(row['Session'])
        except (ValueError, TypeError):
            continue
            
        key = (animal, session)
        if key not in dan_hunts:
            dan_hunts[key] = []
            
        hunts = {
            'enter': int(row['On']),
            'pursuit_start': int(row['Approach Start']) if pd.notna(row.get('Approach Start')) else int(row['On']),
            'capture': int(row['Capture']),
            'consumption_end': int(row['Consumption']) if pd.notna(row.get('Consumption')) else None,
        }
        dan_hunts[key].append(hunts)
        
    return dan_hunts

def load_dan_labels(label_dir):
    """
    Scans a directory for Dan label files (.xlsx, .csv) and aggregates all hunts.
    Returns dict mapping (animal, session) -> list of hunt dictionaries.
    """
    dan_hunts = {}
    label_files = glob.glob(os.path.join(label_dir, '*.xlsx')) + glob.glob(os.path.join(label_dir, '*.csv'))
    
    for lf in label_files:
        try:
            file_hunts = parse_dan_label_file(lf)
            for key, val in file_hunts.items():
                if key not in dan_hunts:
                    dan_hunts[key] = []
                dan_hunts[key].extend(val)
        except Exception as e:
            if VERBOSE: print(f"Error reading {lf}: {e}")
            
    return dan_hunts

def pair_dataset_files(datasets_config=DATASETS, verbose=True):
    paired_data = []
    
    for ds_name, config in datasets_config.items():
        fmt = config['format']
        label_dir = config['label_dir']
        dlc_dir = config['dlc_dir']
        
        if not os.path.exists(label_dir) and 'fallback_label_dir' in config:
            label_dir = config['fallback_label_dir']
            
        dlc_files = glob.glob(os.path.join(dlc_dir, '*DLC*.csv'))
        
        if verbose: print(f"Scanning {ds_name}...")
        
        if fmt == 'lucas':
            animal_basename = config.get('animal_basename', 'LW')
            label_files = glob.glob(os.path.join(label_dir, '*analyzed*.xlsx'))
            dlc_map = {}
            for dlc_f in dlc_files:
                match = re.search(fr'({animal_basename}\d+)[-_](hunt[-_]?\d+)', os.path.basename(dlc_f), re.IGNORECASE)
                if match:
                    dlc_map[(match.group(1), match.group(2).replace('_', '-'))] = dlc_f
                else:
                    if verbose: print(f"WARNING ({ds_name}): Could not parse ID from DLC -> {os.path.basename(dlc_f)}")
            
            matched_keys = set()
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
                        matched_keys.add(key)
            if verbose:
                for key, dlc_path in dlc_map.items():
                    if key not in matched_keys:
                        print(f"WARNING ({ds_name}): DLC file without matching label file -> {os.path.basename(dlc_path)}")
                        
        elif fmt == 'dan':
            animal_basename = config.get('animal_basename', 'PV')
            dan_labels = load_dan_labels(label_dir)
            
            for dlc in dlc_files:
                base = os.path.basename(dlc)
                match = re.search(fr'({animal_basename}\d+)_D(\d+)_HS(\d+)', base, re.IGNORECASE)
                if match:
                    animal = match.group(1).upper()
                    day = int(match.group(2))
                    session = int(match.group(3))
                    
                    key = (animal, session)
                    if key in dan_labels:
                        paired_data.append({
                            'id': f"{animal}_D{day}_HS{session}",
                            'dlc_path': dlc,
                            'format': fmt,
                            'hunts': dan_labels[key]
                        })
                    else:
                        if verbose: print(f"WARNING ({ds_name}): DLC file without matching label entry -> {base}")
                else:
                    if verbose: print(f"WARNING ({ds_name}): Could not parse animal/day/session from DLC -> {base}")
                        
    if verbose: print(f"Total paired videos: {len(paired_data)}")
    return paired_data

def calculate_empirical_transition_matrix(pairs=None):
    """
    Computes a 4x4 transition probability matrix from the manually labeled training data durations.
    State 0 = Background, State 1 = Stalking, State 2 = Pursuit, State 3 = Eating
    """
    if pairs is None:
        pairs = pair_dataset_files(datasets_config=DAN_DATASETS, verbose=True)
        
    counts = np.zeros((4, 4), dtype=float)
    
    for pair in pairs:
        for hunt in pair['hunts']:
            enter = hunt['enter']
            capture = hunt['capture']
            pursuit_start = hunt['pursuit_start']
            cons_end = hunt['consumption_end']
            
            if enter >= capture or pursuit_start > capture: continue
            
            eating_end_frame = cons_end if cons_end is not None else capture + EATING_BUFFER
            
            stalk_dur = max(0, pursuit_start - enter)
            pursuit_dur = max(0, capture - pursuit_start)
            eating_dur = max(0, eating_end_frame - capture)
            
            # Assume ~1000 frames of background between events
            bg_dur = 1000
            counts[0, 0] += bg_dur - 1
            
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
                        
    # Add priors for "aborting" a hunt or immediately initiating a back-to-back hunt
    num_hunts = sum(len(p['hunts']) for p in pairs)
    counts[1, 0] += max(1, num_hunts * 0.1)  # 10% chance to abort stalk to background
    counts[2, 0] += max(1, num_hunts * 0.05) # 5% chance to abort pursuit to background
    counts[3, 1] += max(1, num_hunts * 0.02) # 2% chance to transition directly from eating to next stalk
    counts[3, 2] += max(1, num_hunts * 0.01) # 1% chance to transition directly from eating to next pursuit
    
    # Strictly forbid chronological violations within an active hunt (pursuit back to stalking)
    forbidden = [(2, 1)]
    for i in range(4):
        for j in range(4):
            if (i, j) in forbidden:
                counts[i, j] = 0.0
            elif counts[i, j] == 0:
                counts[i, j] = 1e-5 # Minor Laplace smoothing for unobserved but possible transitions
                
    row_sums = counts.sum(axis=1, keepdims=True)
    return counts / row_sums

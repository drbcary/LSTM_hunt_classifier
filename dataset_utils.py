import os
import glob
import re
import cv2
import pandas as pd
import numpy as np

BASE_SERVER_PATH = r"\\research.brandeis.edu\turrigiano-lab"

DATASETS = {
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
    'Dan_Dataset': {
        'label_dir': os.path.join(BASE_SERVER_PATH, r'Path\To\Dan\Labels'), 
        'dlc_dir': os.path.join(BASE_SERVER_PATH, r'Path\To\Dan\DLC'),
        'animal_basename': 'PV',
        'file_filter': None,
        'format': 'dan',
    }
}

VERBOSE = True
DEFAULT_WIDTH = 514
DEFAULT_HEIGHT = 544
DEFAULT_FPS = 30.0

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

def parse_lucas_labels(label_path):
    df = pd.read_excel(label_path)
    valid_hunts = df[df['Include?'] == 'Include'].sort_values('Enter_Frame')
    hunts = []
    for _, row in valid_hunts.iterrows():
        enter = int(row['Enter_Frame'])
        capture = int(row['Capture_Frame'])
        latency = int(row['Latency_to_Attack_Frames'])
        hunts.append({
            'enter': enter,
            'capture': capture,
            'pursuit_start': enter + latency,
            'consumption_end': None
        })
    return hunts

def load_dan_labels(label_dir):
    dan_hunts = {}
    label_files = glob.glob(os.path.join(label_dir, '*.xlsx'))
    
    for lf in label_files:
        try:
            df = pd.read_excel(lf)
            if not all(col in df.columns for col in ['Animal', 'Day', 'Session']): continue
            
            # Animal name appears once per file — grab it from the first non-NaN entry
            animal_series = df['Animal'].dropna()
            if animal_series.empty: continue
            animal = str(animal_series.iloc[0]).strip()
            
            for _, row in df.iterrows():
                # Skip rows without frame data
                if pd.isna(row.get('On')) or pd.isna(row.get('Capture')):
                    continue
                
                day = int(row['Day'])
                session = int(row['Session'])
                
                key = (animal, day, session)
                if key not in dan_hunts:
                    dan_hunts[key] = []
                    
                hunts = {
                    'enter': int(row['On']),
                    'pursuit_start': int(row['Approach Start']) if pd.notna(row.get('Approach Start')) else int(row['On']),
                    'capture': int(row['Capture']),
                    'consumption_end': int(row['Consumption']) if pd.notna(row.get('Consumption')) else None,
                }
                
                dan_hunts[key].append(hunts)
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
        
        # Check local example_files too for testing if they exist
        if os.path.exists('example_files'):
            dlc_files.extend(glob.glob(os.path.join('example_files', '*DLC*.csv')))
            
        if verbose: print(f"Scanning {ds_name}...")
        
        if fmt == 'lucas':
            animal_basename = config.get('animal_basename', 'LW')
            label_files = glob.glob(os.path.join(label_dir, '*analyzed*.xlsx'))
            dlc_map = {}
            for dlc_f in dlc_files:
                match = re.search(fr'({animal_basename}\d+)[-_](hunt[-_]?\d+)', os.path.basename(dlc_f), re.IGNORECASE)
                if match:
                    dlc_map[(match.group(1), match.group(2).replace('_', '-'))] = dlc_f
            
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
                    
                    key = (animal, day, session)
                    if key in dan_labels:
                        paired_data.append({
                            'id': f"{animal}_D{day}_HS{session}",
                            'dlc_path': dlc,
                            'format': fmt,
                            'hunts': dan_labels[key]
                        })
                        
    if verbose: print(f"Total paired videos: {len(paired_data)}")
    return paired_data

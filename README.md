# Mouse Hunting Behavior LSTM Classifier (`mouse-model` branch)

This repository contains the modular DeepLabCut (DLC) + BiLSTM behavioral classification pipeline for hunting paradigms in rodents, refactored specifically for mouse experiments.

---

## Key Refactoring Highlights

### 1. Unified Time & Video FPS Normalization
- All temporal constants (context buffer, eating buffer, minimum/maximum background chunks, rolling velocity filters, dead cricket filters, sliding windows) are specified in **seconds** (`TimeConfig`) and converted dynamically per video based on its true FPS (`sec_to_frames(sec, fps)`).
- Eliminates temporal distortion across 30 FPS, 60 FPS, or 120 FPS video recordings.

### 2. Multi-Segment Tail Tracking (`MousePoseSchema`)
- Mouse models track multiple tail points (e.g. `tail_base`, `tail_mid1`, `tail_mid2`, `tail_tip`).
- The `multi_tail` feature block computes:
  - Inter-segment angles at each joint along the body-to-tail chain.
  - Joint angular velocities ($\tanh$ scaled).
  - Cumulative tail curvature ($\sum |\Delta \theta|$) and curvature rate of change.
  - Distal tail tip speed.

### 3. Automatic Video FPS & Resolution Extraction
- `DatasetMatcher(extract_video_fps=True)` locates the corresponding video file (`.mp4`, `.avi`, `.mkv`), extracts the exact frame rate and resolution via OpenCV, and caches the metadata in a JSON file to avoid repeated disk reads.
- Falls back to configured defaults if video files are offline.

### 4. Variable Sessions Across Days
- Mice paradigms often have variable numbers of hunting sessions per day (e.g. Day 1 has 3 sessions, Day 2 has 5).
- `DatasetMatcher` utilizes flexible named regex patterns to extract `animal`, `day`, `session`, and optional `trial` indices from filenames or folder structures.

### 5. Modular Feature Registry (`features/`)
- Replaces monolithic hardcoded feature arrays with composable feature extractors:
  - `kinematics`: Predator and prey speed, acceleration, and predator turn rate.
  - `geometry`: Head-prey distances, line-of-sight angle difference, closing velocity, inverse TTC, lunge speed, egocentric relative forward/lateral coordinates.
  - `posture`: Spine angle, body compression ratio, compression rate.
  - `multi_tail`: Multi-segment tail angles, cumulative curvature, tail-whip dynamics.
  - `mouse_specific`: Cricket bearing jitter (erratic hopping), ear-based head roll/tilt, cranial gaze alignment.
- Each feature computation outputs a `FeatureManifest`, which dynamically informs the model's `input_dim`.

### 6. Pluggable Manual Label Parsers (`labels/`)
- Unified `HuntEvent` representation (`enter`, `approach_start`, `capture`, `consumption_end`).
- Parsers for:
  - Single-session Excel/CSV sheets (`GenericExcelLabelParser`).
  - Multi-session/multi-day spreadsheets (`MultiSessionSheetParser`).
  - Boris event logs (`BorisLabelParser`).

---

## Directory Structure

```
LSTM_hunt_label_mouse/
├── config/
│   ├── base_config.py           # Base configuration & TimeConfig (in seconds)
│   ├── mouse_config.py          # MousePoseSchema (multi-tail) & MouseParadigmConfig
│   └── rat_config.py            # Legacy rat schema for backward compatibility
│
├── pose/
│   ├── keypoints.py             # PoseData container, angular math, geometry
│   └── dlc_adapter.py           # DLC CSV/H5 loader, poop penalty, dead cricket filter
│
├── features/
│   ├── registry.py              # FeatureRegistry & FeatureManifest
│   ├── kinematics.py            # Speed, acceleration, turn rates
│   ├── geometry.py              # Distance, line-of-sight, closing speed, egocentric
│   ├── posture.py               # Spine angle, compression, multi-segment tail
│   └── mouse_features.py        # Ear tilt, gaze alignment, cricket jitter
│
├── labels/
│   ├── base_parser.py           # HuntEvent dataclass and BaseLabelParser interface
│   ├── excel_parser.py          # Excel parsers (single & multi-session)
│   └── boris_parser.py          # Boris behavioral event parser
│
├── dataset/
│   ├── matcher.py               # Variable session matcher & video FPS extractor
│   ├── sequence_builder.py      # Balanced chunking & training sequence creator (FPS-scaled)
│   └── transition_matrix.py     # Time-normalized empirical HMM transition matrix
│
├── dataset_utils.py             # Backward-compatibility facade pointing to new modules
├── LSTM_model_training.py       # Model training script
├── MouseHunt_Inference.py       # Modular mouse inference pipeline
└── test_features.py             # Unit tests for legacy & mouse feature extractors
```

---

## Usage Examples

### 1. Customizing Mouse Keypoint Schema
```python
from config.mouse_config import MouseParadigmConfig, MousePoseSchema

config = MouseParadigmConfig()
config.schema = MousePoseSchema(
    head="snout",
    center="body_center",
    rear="tail_base",
    tail_points=["tail_base", "tail_1", "tail_2", "tail_tip"],
    left_ear="left_ear",
    right_ear="right_ear",
    cricket="cricket"
)
```

### 2. Loading DLC Tracking Data
```python
from pose.dlc_adapter import DLCAdapter

adapter = DLCAdapter(config)
pose_data = adapter.load("path/to/mouse_D1_HS1DLC_resnet50.csv", video_fps=60.0)
```

### 3. Extracting Features with Dynamic Manifest
```python
from features import generate_features

features, manifest = generate_features(
    pose_data=pose_data,
    video_width=1024,
    video_fps=60.0,
    config=config
)

print(f"Computed {manifest.num_features} features across {len(features)} frames.")
for item in manifest.items:
    print(f"[{item.group}] {item.name}: {item.description}")
```

### 4. Running Inference on Mouse Data
```python
from MouseHunt_Inference import MouseInferencePipeline

pipeline = MouseInferencePipeline(
    model_weights_path="models/mouse_bilstm_best.pth",
    config=config
)

results = pipeline.run_on_session("path/to/mouse_D1_HS1DLC_resnet50.csv")
probabilities = results['probabilities']  # (N, 4)
predictions = results['raw_predictions']  # (N,)
```

### 5. Verifying Feature Implementation
Run `test_features.py` to verify both legacy 22-feature consistency and extended mouse multi-segment tail extraction:
```bash
python test_features.py
```

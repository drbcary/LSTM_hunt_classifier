import math
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional

def sec_to_frames(sec: float, fps: float) -> int:
    """Convert a duration in seconds to integer frame count for a specific FPS."""
    return max(1, round(sec * fps))

def frames_to_sec(frames: int, fps: float) -> float:
    """Convert a frame count to duration in seconds."""
    return float(frames) / max(fps, 1e-6)

@dataclass
class TimeConfig:
    """All temporal buffers and durations specified strictly in seconds."""
    context_buffer_sec: float = 10.0       # ~300 frames @ 30fps
    eating_buffer_sec: float = 10.0        # ~300 frames @ 30fps
    min_bg_len_sec: float = 16.67          # ~500 frames @ 30fps
    max_bg_len_sec: float = 66.67          # ~2000 frames @ 30fps
    dead_cricket_time_sec: float = 30.0    # Movement stillness window in seconds (~900 frames @ 30fps)
    cricket_dead_window_sec: float = 0.5   # Rolling window for cricket velocity smoothing (~15 frames @ 30fps)
    
    # Sequential full video chunking
    full_video_chunk_min_sec: float = 33.33  # ~1000 frames @ 30fps
    full_video_chunk_max_sec: float = 166.67 # ~5000 frames @ 30fps
    full_video_chunk_proportion: float = 0.15
    full_video_chunk_iterations: int = 2
    
    # Inference sliding window
    inference_window_sec: float = 300.0     # Sliding window size in seconds
    inference_step_sec: float = 50.0       # Step size in seconds (e.g. 50 frames @ 30fps)
    
    # Viterbi transition assumed background duration between events
    viterbi_bg_dur_sec: float = 60.0      # ~1000 frames @ 30fps

@dataclass
class ThresholdConfig:
    """Likelihood and filtering thresholds."""
    predator_likelihood_thresh: float = 0.9
    cricket_likelihood_thresh: float = 0.85
    cricket_coord_thresh: float = 0.30     # Minimum noise-floor likelihood to retain cricket coordinates
    max_interp_gap_sec: float = 0.5        # Maximum interpolation gap in seconds (0.5s = 15 frames @ 30fps)
    cricket_movement_thresh: float = 0.3   # In arena-normalized or pixel units
    
    # Normalization thresholds
    max_speed_thresh: float = 3.0          # In arena-widths per sec
    max_accel_thresh: float = 25.0         # In arena-widths per sec^2
    
    # Poop / artifact filtering
    filter_poops: bool = True
    poop_thresh: float = 0.9
    poop_dist_thresh: float = 6.0          # In pixels
    min_poop_frames: int = 15              # Minimum persistent frames to qualify as stationary poop

@dataclass
class BaseParadigmConfig:
    """Base configuration class inherited by mouse and rat paradigms."""
    name: str = "base"
    default_fps: float = 30.0
    default_width: int = 514
    default_height: int = 544
    time: TimeConfig = field(default_factory=TimeConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    max_bg_chunks_per_video: int = 5
    extract_video_fps: bool = True
    continuous_cricket_confidence: bool = False   # True for continuous [0, 1] confidence; False for legacy cutoff boolean

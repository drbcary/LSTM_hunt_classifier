import numpy as np
import pandas as pd
from typing import Tuple, List, Any
from pose.keypoints import PoseData, calculate_bearing, angular_diff, dist
from .registry import feature_registry, FeatureItem

@feature_registry.register("mouse_specific")
def extract_mouse_features(
    pose_data: PoseData,
    video_width: float,
    video_fps: float,
    config: Any = None
) -> Tuple[np.ndarray, List[FeatureItem]]:
    """
    Features tailored specifically for mouse hunting behavior:
    - Cricket bearing jitter (erratic hopping)
    - Ear-based head tilt / roll dynamics & cranial gaze alignment
    - Approach directness (vector projection of mouse velocity onto prey bearing)
    - Meander index / path tortuosity (distinguishes direct approach from meandering)
    - Sensitive closing velocity (captures subtle slow-speed approaches)
    - Compact body length (snout-to-butt distance)
    """
    num_frames = pose_data.num_frames

    # 1. Cricket jitter / erratic jump
    c_vel = np.diff(pose_data.cricket, axis=0, prepend=pose_data.cricket[0:1])
    c_bearing = np.arctan2(c_vel[:, 1], c_vel[:, 0])
    c_jitter = np.abs(angular_diff(c_bearing)) / np.pi

    # 2. Ear & Head orientation features (if available)
    if pose_data.has_ears and pose_data.left_ear is not None and pose_data.right_ear is not None:
        l_ear = pose_data.left_ear
        r_ear = pose_data.right_ear
        inter_ear_dist = dist(l_ear, r_ear) / max(video_width, 1.0)
        ear_bearing = calculate_bearing(l_ear, r_ear)
        ear_roll_rate = np.abs(angular_diff(ear_bearing)) * video_fps / np.pi
        ear_roll_rate = np.tanh(ear_roll_rate / 10.0)

        # Gaze axis (forward from midpoint of ears towards snout)
        ear_mid = (l_ear + r_ear) / 2.0
        head_forward_vec = calculate_bearing(ear_mid, pose_data.head)
        cricket_vec = calculate_bearing(pose_data.head, pose_data.cricket)
        gaze_diff = (cricket_vec - head_forward_vec + np.pi) % (2 * np.pi) - np.pi
        gaze_alignment = gaze_diff / np.pi
    else:
        inter_ear_dist = np.zeros(num_frames)
        ear_roll_rate = np.zeros(num_frames)
        gaze_alignment = np.zeros(num_frames)

    # 3. Approach Directness (Cosine of velocity direction vs line-of-sight vector)
    pred_vel = np.diff(pose_data.head, axis=0, prepend=pose_data.head[0:1])
    pred_speed_px = np.linalg.norm(pred_vel, axis=1, keepdims=True)
    unit_vel = np.divide(pred_vel, pred_speed_px + 1e-6)

    vec_to_cricket = pose_data.cricket - pose_data.head
    dist_to_cricket = np.linalg.norm(vec_to_cricket, axis=1, keepdims=True)
    unit_to_cricket = np.divide(vec_to_cricket, dist_to_cricket + 1e-6)

    # Dot product: +1.0 = moving straight at cricket, 0.0 = moving sideways, -1.0 = moving away
    approach_directness = np.sum(unit_vel * unit_to_cricket, axis=1)
    # Zero out when mouse is virtually still (< 1 px/frame)
    approach_directness[pred_speed_px[:, 0] < 1.0] = 0.0

    # 4. Path Tortuosity / Meander Index over ~1.0 second rolling window
    window_sz = max(3, round(1.0 * video_fps))
    step_dists = np.linalg.norm(pred_vel, axis=1)
    rolling_path_len = pd.Series(step_dists).rolling(window=window_sz, min_periods=1).sum().values

    # Net displacement between frame (t - window) and frame t
    lag_head = np.roll(pose_data.head, shift=window_sz, axis=0)
    # Fix the initial boundary window
    for i in range(min(window_sz, num_frames)):
        lag_head[i] = pose_data.head[0]
    net_disp = np.linalg.norm(pose_data.head - lag_head, axis=1)

    tortuosity = rolling_path_len / (net_disp + 1e-5)
    # Straight line = 1.0 -> 0.0; meandering/turning >> 1.0 -> approaches 1.0
    meander_index = np.tanh(np.clip(tortuosity - 1.0, 0.0, 10.0) / 2.0)

    # 5. Sensitive Closing Velocity (Scaled for slower mouse speeds)
    d_norm = dist_to_cricket[:, 0] / max(video_width, 1.0)
    closing_vel = -np.diff(d_norm, prepend=d_norm[0:1]) * video_fps
    sensitive_closing_rate = np.tanh(closing_vel * 5.0)

    # 6. Body Compactness (Snout-to-Butt distance)
    body_length = dist(pose_data.head, pose_data.rear) / max(video_width, 1.0)

    matrix = np.column_stack([
        c_jitter,
        inter_ear_dist,
        ear_roll_rate,
        gaze_alignment,
        approach_directness,
        meander_index,
        sensitive_closing_rate,
        body_length
    ])

    items = [
        FeatureItem("cricket_jitter", "mouse_specific", "Angular jitter in cricket heading (erratic jumping)"),
        FeatureItem("inter_ear_distance", "mouse_specific", "Distance between ears (head tilt proxy)"),
        FeatureItem("head_roll_rate", "mouse_specific", "Rate of head roll/tilt from ear axis (tanh scaled)"),
        FeatureItem("gaze_alignment", "mouse_specific", "Alignment between cranial axis and cricket in [-1, 1]"),
        FeatureItem("approach_directness", "mouse_specific", "Cosine of velocity direction vs prey line-of-sight in [-1, 1]"),
        FeatureItem("meander_index", "mouse_specific", "Path tortuosity over 1s window (0=straight attack, 1=meandering)"),
        FeatureItem("sensitive_closing_rate", "mouse_specific", "Amplified closing velocity for slow mouse movements"),
        FeatureItem("body_compactness", "mouse_specific", "Snout-to-butt length normalized to arena width"),
    ]

    return matrix, items

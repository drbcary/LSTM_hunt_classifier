import numpy as np
from typing import Tuple, List, Any
from pose.keypoints import PoseData, calculate_bearing, angular_diff
from .registry import feature_registry, FeatureItem

def get_derivatives(pos_array: np.ndarray, video_fps: float, video_width: float) -> Tuple[np.ndarray, np.ndarray]:
    """Computes speed and acceleration normalized by video width."""
    vel_vec = np.diff(pos_array, axis=0, prepend=pos_array[0:1]) * video_fps
    speed = np.linalg.norm(vel_vec, axis=1) / max(video_width, 1.0)
    acc_vec = np.diff(vel_vec, axis=0, prepend=vel_vec[0:1]) * video_fps
    accel = np.linalg.norm(acc_vec, axis=1) / max(video_width, 1.0)
    return speed, accel

@feature_registry.register("kinematics")
def extract_kinematics(
    pose_data: PoseData,
    video_width: float,
    video_fps: float,
    config: Any = None
) -> Tuple[np.ndarray, List[FeatureItem]]:
    max_speed_th = getattr(config.thresholds, 'max_speed_thresh', 3.0) if config else 3.0
    max_accel_th = getattr(config.thresholds, 'max_accel_thresh', 25.0) if config else 25.0

    # Predator kinematics (head)
    pred_speed, pred_accel = get_derivatives(pose_data.head, video_fps, video_width)
    pred_speed = np.clip(pred_speed, 0, max_speed_th) / max_speed_th
    pred_accel = np.clip(pred_accel, 0, max_accel_th) / max_accel_th

    # Prey kinematics (cricket)
    cricket_speed, cricket_accel = get_derivatives(pose_data.cricket, video_fps, video_width)
    cricket_speed = np.clip(cricket_speed, 0, max_speed_th) / max_speed_th
    cricket_accel = np.clip(cricket_accel, 0, max_accel_th) / max_accel_th

    # Turn rate
    pred_orientation = calculate_bearing(pose_data.center, pose_data.head)
    pred_turn_rate = np.abs(angular_diff(pred_orientation)) * video_fps / np.pi
    pred_turn_rate = np.tanh(pred_turn_rate / 10.0)

    matrix = np.column_stack([
        pred_speed,
        pred_accel,
        cricket_speed,
        cricket_accel,
        pred_turn_rate
    ])

    items = [
        FeatureItem("predator_speed", "kinematics", "Predator head speed normalized to [0, 1]"),
        FeatureItem("predator_accel", "kinematics", "Predator head acceleration normalized to [0, 1]"),
        FeatureItem("cricket_speed", "kinematics", "Prey speed normalized to [0, 1]"),
        FeatureItem("cricket_accel", "kinematics", "Prey acceleration normalized to [0, 1]"),
        FeatureItem("predator_turn_rate", "kinematics", "Predator body angular turn rate"),
    ]

    return matrix, items

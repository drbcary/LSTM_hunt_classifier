import numpy as np
from typing import Tuple, List, Any
from pose.keypoints import PoseData, calculate_bearing, angular_diff, dist
from .registry import feature_registry, FeatureItem

@feature_registry.register("geometry")
def extract_geometry(
    pose_data: PoseData,
    video_width: float,
    video_fps: float,
    config: Any = None
) -> Tuple[np.ndarray, List[FeatureItem]]:
    max_speed_th = getattr(config.thresholds, 'max_speed_thresh', 3.0) if config else 3.0

    d_head_cricket = dist(pose_data.head, pose_data.cricket) / max(video_width, 1.0)
    d_head_center = dist(pose_data.head, pose_data.center) / max(video_width, 1.0)
    
    primary_tail = pose_data.primary_tail_end
    d_center_tail = dist(pose_data.center, primary_tail) / max(video_width, 1.0)
    d_head_tail = dist(pose_data.head, primary_tail) / max(video_width, 1.0)

    # Predator-to-prey orientation geometry
    pred_orientation = calculate_bearing(pose_data.center, pose_data.head)
    target_vector = calculate_bearing(pose_data.head, pose_data.cricket)
    angle_diff = target_vector - pred_orientation
    angle_diff_raw = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    norm_angle_diff = angle_diff_raw / np.pi

    cricket_visible = np.where(np.isnan(d_head_cricket), 0.0, 1.0)

    # Closing velocity and inverse time-to-collision
    closing_velocity = -np.diff(d_head_cricket, prepend=d_head_cricket[0:1]) * video_fps
    inverse_ttc = np.clip(closing_velocity / (d_head_cricket + 1e-6), -100.0, 100.0)

    # Aim correction rate
    aim_correction_rate = np.abs(angular_diff(angle_diff_raw)) * video_fps / np.pi

    # Lunge speed along line of sight
    vec_hc = pose_data.cricket - pose_data.head
    dist_hc = np.linalg.norm(vec_hc, axis=1, keepdims=True)
    unit_hc = vec_hc / (dist_hc + 1e-6)
    pred_vel = np.diff(pose_data.head, axis=0, prepend=pose_data.head[0:1]) * video_fps
    lunge_speed = np.sum(pred_vel * unit_hc, axis=1) / max(video_width, 1.0)

    # Egocentric coordinates
    theta = pred_orientation
    dx = vec_hc[:, 0]
    dy = vec_hc[:, 1]
    ego_forward = (dx * np.cos(-theta) - dy * np.sin(-theta)) / max(video_width, 1.0)
    ego_lateral = (dx * np.sin(-theta) + dy * np.cos(-theta)) / max(video_width, 1.0)

    # Normalization
    closing_velocity = np.clip(closing_velocity, -max_speed_th, max_speed_th) / max_speed_th
    lunge_speed = np.clip(lunge_speed, -max_speed_th, max_speed_th) / max_speed_th
    inverse_ttc = np.tanh(inverse_ttc / 10.0)
    aim_correction_rate = np.tanh(aim_correction_rate / 10.0)

    matrix = np.column_stack([
        d_head_cricket,
        d_head_center,
        d_center_tail,
        d_head_tail,
        norm_angle_diff,
        cricket_visible,
        closing_velocity,
        inverse_ttc,
        aim_correction_rate,
        lunge_speed,
        ego_forward,
        ego_lateral
    ])

    items = [
        FeatureItem("d_head_cricket", "geometry", "Normalized distance from head to cricket"),
        FeatureItem("d_head_center", "geometry", "Normalized distance from head to body center"),
        FeatureItem("d_center_tail", "geometry", "Normalized distance from body center to tail"),
        FeatureItem("d_head_tail", "geometry", "Normalized distance from head to tail"),
        FeatureItem("angle_diff", "geometry", "Line of sight angle difference in [-1, 1]"),
        FeatureItem("cricket_visible", "geometry", "Binary indicator if cricket is detected"),
        FeatureItem("closing_velocity", "geometry", "Rate of approach toward cricket"),
        FeatureItem("inverse_ttc", "geometry", "Inverse time-to-collision (tanh scaled)"),
        FeatureItem("aim_correction_rate", "geometry", "Rate of heading correction (tanh scaled)"),
        FeatureItem("lunge_speed", "geometry", "Predator velocity projected along line of sight"),
        FeatureItem("ego_forward", "geometry", "Egocentric forward displacement to cricket"),
        FeatureItem("ego_lateral", "geometry", "Egocentric lateral displacement to cricket"),
    ]

    return matrix, items

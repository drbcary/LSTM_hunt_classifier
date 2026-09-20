import numpy as np
from typing import Tuple, List, Any
from pose.keypoints import PoseData, calculate_angle_3pts, dist
from .registry import feature_registry, FeatureItem

@feature_registry.register("posture")
def extract_posture(
    pose_data: PoseData,
    video_width: float,
    video_fps: float,
    config: Any = None
) -> Tuple[np.ndarray, List[FeatureItem]]:
    """Extracts fundamental body posture (spine curvature, compression)."""
    primary_tail = pose_data.primary_tail_end
    
    # Spine angle
    angle_spine = calculate_angle_3pts(pose_data.head, pose_data.center, primary_tail) / 180.0

    # Body compression ratio and compression rate
    d_head_tail = dist(pose_data.head, primary_tail)
    d_head_mid = dist(pose_data.head, pose_data.center)
    d_mid_tail = dist(pose_data.center, primary_tail)
    
    body_compression = d_head_tail / (d_head_mid + d_mid_tail + 1e-6)
    compression_rate = np.diff(body_compression, prepend=body_compression[0:1]) * video_fps
    compression_rate = np.tanh(compression_rate / 10.0)

    # Tail angle (between center, rear, and primary tail point)
    angle_tail = calculate_angle_3pts(pose_data.center, pose_data.rear, primary_tail) / 180.0
    tail_angular_vel = np.abs(np.diff(angle_tail, prepend=angle_tail[0:1])) * video_fps
    tail_angular_vel = np.tanh(tail_angular_vel / 5.0)

    matrix = np.column_stack([
        angle_spine,
        body_compression,
        compression_rate,
        angle_tail,
        tail_angular_vel
    ])

    items = [
        FeatureItem("angle_spine", "posture", "Spine curvature angle normalized to [0, 1]"),
        FeatureItem("body_compression", "posture", "Ratio of head-tail distance to path length"),
        FeatureItem("compression_rate", "posture", "Rate of change of body compression (tanh scaled)"),
        FeatureItem("angle_tail_base", "posture", "Angle at pelvis between center and tail base"),
        FeatureItem("tail_base_angular_vel", "posture", "Angular velocity at tail base (tanh scaled)"),
    ]

    return matrix, items

@feature_registry.register("multi_tail")
def extract_multi_tail_dynamics(
    pose_data: PoseData,
    video_width: float,
    video_fps: float,
    config: Any = None
) -> Tuple[np.ndarray, List[FeatureItem]]:
    """
    Computes multi-segment tail dynamics for mice tracking 2 or more sequential tail keypoints.
    Extracts inter-segment angles, cumulative curvature, and tail whipping velocity.
    """
    tail_pts = pose_data.tail_points
    num_frames = pose_data.num_frames
    
    # Form a chain of points from body center -> rear -> tail_0 -> tail_1 ... -> tail_k
    # Avoid appending duplicate contiguous point if rear and tail_pts[0] refer to the same keypoint
    chain = [pose_data.center, pose_data.rear]
    for pt in tail_pts:
        if np.array_equal(pt, chain[-1]):
            continue
        chain.append(pt)
    
    segment_angles = []
    angle_vels = []
    items = []

    # Calculate angles at each internal joint in the chain
    for i in range(1, len(chain) - 1):
        pt_prev = chain[i - 1]
        pt_curr = chain[i]
        pt_next = chain[i + 1]
        
        joint_angle = calculate_angle_3pts(pt_prev, pt_curr, pt_next) / 180.0
        joint_vel = np.abs(np.diff(joint_angle, prepend=joint_angle[0:1])) * video_fps
        joint_vel = np.tanh(joint_vel / 5.0)
        
        segment_angles.append(joint_angle)
        angle_vels.append(joint_vel)
        
        joint_name = f"tail_joint_{i}"
        items.append(FeatureItem(f"{joint_name}_angle", "multi_tail", f"Angle at joint {i} normalized to [0, 1]"))
        items.append(FeatureItem(f"{joint_name}_angular_vel", "multi_tail", f"Angular velocity at joint {i} (tanh scaled)"))

    # Cumulative tail curvature: sum of deviations from 180 degrees (straight line)
    # Deviation = 1.0 - joint_angle (where 1.0 is 180 deg)
    deviations = [np.abs(1.0 - ang) for ang in segment_angles]
    cumulative_curvature = np.sum(deviations, axis=0)
    curvature_rate = np.diff(cumulative_curvature, prepend=cumulative_curvature[0:1]) * video_fps
    curvature_rate = np.tanh(curvature_rate / 10.0)

    items.append(FeatureItem("cumulative_tail_curvature", "multi_tail", "Total angular deviation along entire tail"))
    items.append(FeatureItem("tail_curvature_rate", "multi_tail", "Rate of change of total tail curvature (tanh scaled)"))

    # Explicit 3-point angle of the last 3 tail points (e.g. tail3, near_endtail, endtail)
    if len(tail_pts) >= 3:
        distal_3pt_angle = calculate_angle_3pts(tail_pts[-3], tail_pts[-2], tail_pts[-1]) / 180.0
        distal_3pt_vel = np.abs(np.diff(distal_3pt_angle, prepend=distal_3pt_angle[0:1])) * video_fps
        distal_3pt_vel = np.tanh(distal_3pt_vel / 5.0)
    elif len(chain) >= 3:
        distal_3pt_angle = calculate_angle_3pts(chain[-3], chain[-2], chain[-1]) / 180.0
        distal_3pt_vel = np.abs(np.diff(distal_3pt_angle, prepend=distal_3pt_angle[0:1])) * video_fps
        distal_3pt_vel = np.tanh(distal_3pt_vel / 5.0)
    else:
        distal_3pt_angle = np.ones(num_frames)
        distal_3pt_vel = np.zeros(num_frames)

    items.append(FeatureItem("distal_3pt_tail_angle", "multi_tail", "3-point angle formed by last 3 tail points (tail3, near_endtail, endtail) in [0, 1]"))
    items.append(FeatureItem("distal_3pt_tail_vel", "multi_tail", "Angular velocity of the distal 3-point tail tip (tanh scaled)"))

    # Distal tail tip speed
    tip_pt = chain[-1]
    tip_vel = np.diff(tip_pt, axis=0, prepend=tip_pt[0:1]) * video_fps
    tip_speed = np.linalg.norm(tip_vel, axis=1) / max(video_width, 1.0)
    tip_speed = np.clip(tip_speed, 0.0, 5.0) / 5.0
    items.append(FeatureItem("tail_tip_speed", "multi_tail", "Speed of the distal tail tip"))

    all_cols = []
    for ang, vel in zip(segment_angles, angle_vels):
        all_cols.append(ang)
        all_cols.append(vel)
    all_cols.extend([cumulative_curvature, curvature_rate, distal_3pt_angle, distal_3pt_vel, tip_speed])

    matrix = np.column_stack(all_cols)
    return matrix, items

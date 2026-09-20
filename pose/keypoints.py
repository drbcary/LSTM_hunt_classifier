import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

def calculate_angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Calculates angle at vertex b formed by points a, b, c in degrees [0, 180].
    a, b, c are (N, 2) coordinate arrays.
    """
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba, axis=1)
    norm_bc = np.linalg.norm(bc, axis=1)
    cosine_angle = np.sum(ba * bc, axis=1) / (norm_ba * norm_bc + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_bearing(start_pt: np.ndarray, end_pt: np.ndarray) -> np.ndarray:
    """Calculates directional angle (radians in [-pi, pi]) from start_pt to end_pt."""
    return np.arctan2(end_pt[:, 1] - start_pt[:, 1], end_pt[:, 0] - start_pt[:, 0])

def angular_diff(angles: np.ndarray) -> np.ndarray:
    """Computes frame-to-frame change in angle, wrapped to [-pi, pi]."""
    d_angles = np.diff(angles, prepend=angles[0:1])
    return (d_angles + np.pi) % (2 * np.pi) - np.pi

def dist(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Euclidean distance between two (N, 2) arrays."""
    return np.linalg.norm(p1 - p2, axis=1)

@dataclass
class PoseData:
    """
    Container for extracted animal and prey keypoints across all video frames.
    Provides uniform accessors regardless of underlying DLC column naming conventions.
    """
    num_frames: int
    head: np.ndarray                 # (N, 2) coordinates of head/snout
    center: np.ndarray               # (N, 2) coordinates of body center
    rear: np.ndarray                 # (N, 2) coordinates of pelvis/butt/tail_base
    tail_points: List[np.ndarray]    # List of (N, 2) arrays for sequential tail landmarks
    cricket: np.ndarray              # (N, 2) coordinates of cricket/prey
    
    # Optional keypoints
    left_ear: Optional[np.ndarray] = None
    right_ear: Optional[np.ndarray] = None
    extra_parts: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Likelihood / confidence arrays (N,)
    confidences: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def primary_tail_end(self) -> np.ndarray:
        """Returns the distal-most tail point available."""
        if self.tail_points:
            return self.tail_points[-1]
        return self.rear

    @property
    def has_ears(self) -> bool:
        return (self.left_ear is not None and self.right_ear is not None)

    def get_ear_midpoint(self) -> Optional[np.ndarray]:
        if self.left_ear is not None and self.right_ear is not None:
            return (self.left_ear + self.right_ear) / 2.0
        return None

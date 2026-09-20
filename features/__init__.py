import numpy as np
from typing import Tuple, List, Optional, Any

from .registry import feature_registry, FeatureManifest, FeatureItem
from . import kinematics
from . import geometry
from . import posture
from . import mouse_features

def generate_features(
    pose_data: Any,
    video_width: float,
    video_fps: float,
    enabled_blocks: Optional[List[str]] = None,
    config: Any = None
) -> Tuple[np.ndarray, FeatureManifest]:
    """
    Computes all requested feature blocks for the given PoseData.
    Returns:
        matrix: (N, num_features) float numpy array
        manifest: FeatureManifest describing each feature column
    """
    if enabled_blocks is None:
        cfg_blocks = getattr(config, 'enabled_feature_blocks', None) if config else None
        blocks = list(cfg_blocks) if cfg_blocks is not None else ["kinematics", "geometry", "posture", "mouse_specific"]
    else:
        blocks = enabled_blocks

    return feature_registry.compute(
        pose_data=pose_data,
        video_width=video_width,
        video_fps=video_fps,
        enabled_blocks=blocks,
        config=config
    )

__all__ = [
    "feature_registry",
    "FeatureManifest",
    "FeatureItem",
    "generate_features",
]

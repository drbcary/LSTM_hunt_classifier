from dataclasses import dataclass, field
from typing import List, Optional
from .base_config import BaseParadigmConfig, TimeConfig, ThresholdConfig

@dataclass
class RatPoseSchema:
    """Legacy rat pose schema."""
    head: str = "head"
    head_alt: Optional[str] = "nose"
    center: str = "mid_body"
    center_alt: Optional[str] = None
    rear: str = "butt"
    rear_alt: Optional[str] = None
    tail_points: List[str] = field(default_factory=lambda: ["endtail"])
    cricket: str = "cricket"
    blurry_cricket: Optional[str] = "blurry_cricket"
    include_blurry_cricket: bool = False
    left_ear: Optional[str] = None
    right_ear: Optional[str] = None
    extra_parts: List[str] = field(default_factory=list)
    poop_prefix: str = "poop"

@dataclass
class RatParadigmConfig(BaseParadigmConfig):
    """Configuration reproducing legacy rat training and inference exactly."""
    name: str = "rat"
    schema: RatPoseSchema = field(default_factory=RatPoseSchema)
    enabled_feature_blocks: List[str] = field(default_factory=lambda: [
        "kinematics",
        "geometry",
        "posture"
    ])

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .base_config import BaseParadigmConfig, TimeConfig, ThresholdConfig

@dataclass
class MousePoseSchema:
    """
    Keypoint schema for mouse DLC models matching Diane's mouse tracking dataset:
    ['nose', 'leftear', 'rightear', 'mid_body', 'butt', 'tail1', 'tail2', 'tail3', 'near_endtail', 'endtail', 'cricket', 'poop*']
    """
    head: str = "nose"
    head_alt: Optional[str] = "snout"
    center: str = "mid_body"
    center_alt: Optional[str] = "body_center"
    rear: str = "butt"
    rear_alt: Optional[str] = "tail_base"
    
    # 5 sequential tail points from base to tip
    tail_points: List[str] = field(default_factory=lambda: [
        "tail1",
        "tail2",
        "tail3",
        "near_endtail",
        "endtail"
    ])
    
    # Cricket landmarks (standard, blurry, and immobilized)
    cricket: str = "cricket"
    blurry_cricket: Optional[str] = "blurry_cricket"
    cricket_immo: Optional[str] = "cricket_immo"
    include_blurry_cricket: bool = True
    include_cricket_immo: bool = True
    
    # Ear landmarks for head tilt and gaze calculation
    left_ear: Optional[str] = "leftear"
    right_ear: Optional[str] = "rightear"
    extra_parts: List[str] = field(default_factory=list)
    
    # Poop / artifact label prefix
    poop_prefix: str = "poop"

@dataclass
class MouseParadigmConfig(BaseParadigmConfig):
    """
    Paradigm profile for mouse hunting experiments.
    Configured for Diane format datasets (variable hunts per day, master Excel labeling,
    multi-segment tail dynamics, and automatic video FPS extraction).
    """
    name: str = "mouse"
    schema: MousePoseSchema = field(default_factory=MousePoseSchema)
    
    # Video & dataset parameters
    extract_video_fps: bool = True          # Auto-extract true FPS and resolution from video
    video_extensions: List[str] = field(default_factory=lambda: [".avi", ".mp4", ".mkv"])
    
    # Regex patterns for matching video and DLC filenames to Diane's (animal, day) identifiers:
    # Example: 20221019_DB_Exp18_day3-221019-094338_YFP-158-221019-094347_Cam1.avi
    session_regex_patterns: List[str] = field(default_factory=lambda: [
        # Diane DB format with YFP or YB prefix: e.g. 20220615_DB_Exp16_day1-220615-093956_YFP-131-220615-094028_Cam1
        r"day(?P<day>\d+).*?[_-](?P<animal>(?:YFP|YB|[A-Za-z]+)[-_]?[0-9]+)",
        r"(?P<animal>(?:YFP|YB|[A-Za-z]+)[-_]?[0-9]+).*?day(?P<day>\d+)",
        r"day(?P<day>\d+)(?:[-_]\d{6}[-_]\d{6})?[_-](?P<animal>[A-Za-z0-9]+(?:[-_][0-9]+)?)",
        r"(?P<animal>[A-Za-z0-9]+)_D(?P<day>\d+)_HS(?P<session>\d+)",  # Standard: M1_D1_HS2
        r"(?P<animal>[A-Za-z0-9]+)[-_]d(?P<day>\d+)[-_]s(?P<session>\d+)", # Alt: M1-d1-s2
        r"(?P<animal>[A-Za-z0-9]+)[-_]day(?P<day>\d+)",                # M1_day1
    ])
    
    # Active feature blocks to compute
    enabled_feature_blocks: List[str] = field(default_factory=lambda: [
        "kinematics",
        "geometry",
        "posture",
        "multi_tail",
        "mouse_specific"
    ])
    
    # Approach labeling strategy:
    # 'split' (mark tentative approach + return to background + final approach)
    # 'final_only' (mark only final approach to capture)
    # 'continuous' (mark entire window from first approach to capture)
    tentative_approach_strategy: str = "split"
    tentative_approach_max_sec: float = 2.0

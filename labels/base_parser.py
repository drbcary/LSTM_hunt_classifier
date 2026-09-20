from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union

@dataclass
class HuntEvent:
    """
    Standard representation of an individual hunting sequence event.
    Frames are 0-indexed integers.
    """
    enter: int                            # Cricket release / entrance (Cricket Present start)
    approach_start: int                   # Final approach / pursuit start
    capture: int                          # Capture frame
    consumption_end: Optional[int] = None # Eating end frame
    first_approach: Optional[int] = None  # First approach frame (for paradigms distinguishing first vs final approach)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Validates temporal consistency."""
        if self.enter >= self.capture:
            return False
        if self.approach_start > self.capture:
            return False
        if self.approach_start < self.enter:
            return False
        if self.first_approach is not None:
            if self.first_approach < self.enter or self.first_approach > self.approach_start:
                return False
        return True

    def get_eating_end(self, eating_buffer_frames: int = 300) -> int:
        if self.consumption_end is not None:
            try:
                f = float(self.consumption_end)
                if not (f != f):  # check not NaN
                    return int(f)
            except (ValueError, TypeError):
                pass
        try:
            return int(float(self.capture)) + eating_buffer_frames
        except (ValueError, TypeError):
            return 300 + eating_buffer_frames

class BaseLabelParser(ABC):
    """Abstract base class for manual annotation file parsers."""
    @abstractmethod
    def parse_file(self, filepath: str, **kwargs: Any) -> Any:
        """Parses a single label file and returns hunt events."""
        pass

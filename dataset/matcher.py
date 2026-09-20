import os
import glob
import re
import json
import cv2
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

@dataclass
class VideoSessionMetadata:
    """Standard metadata container for an experimental hunting video session."""
    animal_id: str
    day: int
    session: int
    trial: Optional[int] = None
    video_path: Optional[str] = None
    dlc_path: Optional[str] = None
    label_path: Optional[str] = None
    fps: float = 30.0
    width: int = 512
    height: int = 512
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def session_key(self) -> Tuple[str, int, int]:
        return (self.animal_id.upper(), self.day, self.session)

    @property
    def session_id(self) -> str:
        if self.trial is not None:
            return f"{self.animal_id}_D{self.day}_HS{self.session}_T{self.trial}"
        return f"{self.animal_id}_D{self.day}_HS{self.session}"

def get_video_properties(video_path: str) -> Tuple[int, int, float]:
    """Reads width, height, and FPS directly from video file via OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0, 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return width, height, fps

class DatasetMatcher:
    """
    Scans directories and pairs DLC pose files with video files and ground truth labels.
    Handles variable hunting sessions per day, different trial lengths, and dynamic video FPS extraction.
    """
    def __init__(
        self,
        extract_video_fps: bool = True,
        cache_file: Optional[str] = None,
        default_fps: float = 30.0,
        default_width: int = 514,
        default_height: int = 544
    ):
        self.extract_video_fps = extract_video_fps
        self.cache_file = cache_file
        self.default_fps = default_fps
        self.default_width = default_width
        self.default_height = default_height
        self._cache: Dict[str, Dict[str, Any]] = self._load_cache()

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if self.cache_file and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        if self.cache_file:
            try:
                with open(self.cache_file, 'w') as f:
                    json.dump(self._cache, f, indent=2)
            except Exception:
                pass

    def parse_filename(self, filename: str, regex_patterns: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        patterns = regex_patterns or [
            # Diane DB format with YFP or YB prefix: e.g. 20220615_DB_Exp16_day1-220615-093956_YFP-131-220615-094028_Cam1
            r"day(?P<day>\d+).*?[_-](?P<animal>(?:YFP|YB|[A-Za-z]+)[-_]?[0-9]+)",
            # Alternate Diane format where animal is before day: e.g. YFP-131_day1
            r"(?P<animal>(?:YFP|YB|[A-Za-z]+)[-_]?[0-9]+).*?day(?P<day>\d+)",
            # Generic format skipping 6-digit timestamp if present: e.g. day1-220615-093956_131
            r"day(?P<day>\d+)(?:[-_]\d{6}[-_]\d{6})?[_-](?P<animal>[A-Za-z0-9]+(?:[-_][0-9]+)?)",
            r"(?P<animal>[A-Za-z0-9]+)_D(?P<day>\d+)_HS(?P<session>\d+)",
            r"(?P<animal>[A-Za-z0-9]+)[-_]d(?P<day>\d+)[-_]s(?P<session>\d+)",
            r"(?P<animal>[A-Za-z0-9]+)[-_]day(?P<day>\d+)[-_]session(?P<session>\d+)",
            r"(?P<animal>[A-Za-z0-9]+)[-_]hunt[-_]?(?P<session>\d+)",
        ]

        base = os.path.basename(filename).split('DLC')[0]
        for pat in patterns:
            match = re.search(pat, base, re.IGNORECASE)
            if match:
                gd = match.groupdict()
                raw_animal = gd.get('animal', 'ANIMAL').upper()
                # Normalize YFP-158 -> 158 if it has prefix
                if raw_animal.startswith('YFP-') or raw_animal.startswith('YFP_'):
                    animal = raw_animal[4:]
                elif raw_animal.startswith('YFP') and len(raw_animal) > 3 and raw_animal[3:].isdigit():
                    animal = raw_animal[3:]
                else:
                    animal = raw_animal

                day = int(gd.get('day', 1))
                sess_raw = gd.get('session', 1)
                try:
                    session = int(sess_raw)
                except ValueError:
                    num_match = re.search(r'\d+', str(sess_raw))
                    session = int(num_match.group(0)) if num_match else 1

                trial = int(gd['trial']) if 'trial' in gd and gd['trial'] is not None else None
                return {'animal_id': animal, 'day': day, 'session': session, 'trial': trial}
        return None

    def find_video_for_dlc(self, dlc_path: str, video_extensions: Optional[List[str]] = None) -> Optional[str]:
        extensions = video_extensions or ['.mp4', '.avi', '.mkv', '.mov']
        dir_path = os.path.dirname(dlc_path)
        base_name = os.path.basename(dlc_path).split('DLC')[0]

        for ext in extensions:
            candidate = os.path.join(dir_path, base_name + ext)
            if os.path.exists(candidate):
                return candidate
        return None

    def match_session(
        self,
        dlc_path: str,
        video_path: Optional[str] = None,
        label_path: Optional[str] = None,
        regex_patterns: Optional[List[str]] = None
    ) -> Optional[VideoSessionMetadata]:
        info = self.parse_filename(dlc_path, regex_patterns)
        if not info:
            return None

        # Auto-find video if not provided
        v_path = video_path or self.find_video_for_dlc(dlc_path)
        width, height, fps = self.default_width, self.default_height, self.default_fps

        if v_path and self.extract_video_fps:
            if v_path in self._cache:
                c_info = self._cache[v_path]
                width, height, fps = c_info['width'], c_info['height'], c_info['fps']
            else:
                vw, vh, vfps = get_video_properties(v_path)
                if vw > 0 and vh > 0 and vfps > 0:
                    width, height, fps = vw, vh, vfps
                    self._cache[v_path] = {'width': width, 'height': height, 'fps': fps}
                    self._save_cache()

        return VideoSessionMetadata(
            animal_id=info['animal_id'],
            day=info['day'],
            session=info['session'],
            trial=info.get('trial'),
            video_path=v_path,
            dlc_path=dlc_path,
            label_path=label_path,
            fps=fps,
            width=width,
            height=height
        )

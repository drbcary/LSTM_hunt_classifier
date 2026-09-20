import pandas as pd
from typing import Dict, List, Optional, Any
from .base_parser import BaseLabelParser, HuntEvent

class BorisLabelParser(BaseLabelParser):
    """
    Parses observation events exported from BORIS (.csv or .tsv).
    Supports mapping behavior event names ('Enter', 'Approach', 'Capture', 'Eating')
    to HuntEvent structures using either frame numbers or timestamps with video FPS.
    """
    def __init__(
        self,
        time_col: str = "Time",
        frame_col: str = "Image index",
        behavior_col: str = "Behavior",
        status_col: str = "Status",
        fps: float = 30.0
    ):
        self.time_col = time_col
        self.frame_col = frame_col
        self.behavior_col = behavior_col
        self.status_col = status_col
        self.fps = fps

    def parse_file(self, filepath: str, **kwargs: Any) -> List[HuntEvent]:
        sep = '\t' if filepath.endswith('.tsv') else ','
        try:
            df = pd.read_csv(filepath, sep=sep, comment='#')
        except Exception:
            return []

        df.columns = df.columns.astype(str).str.strip()
        if self.behavior_col not in df.columns:
            return []

        # Convert timestamps to frames if frame_col missing
        frames = None
        if self.frame_col in df.columns:
            frame_s = pd.Series(pd.to_numeric(df[self.frame_col], errors='coerce'))
            if frame_s.notna().any():
                frames = frame_s.fillna(0).astype(int).values

        if frames is None:
            if self.time_col in df.columns:
                time_s = pd.Series(pd.to_numeric(df[self.time_col], errors='coerce'))
                frames = (time_s.fillna(0.0) * self.fps).round().astype(int).values
            else:
                return []

        behaviors = df[self.behavior_col].str.lower().str.strip().values
        statuses = df[self.status_col].str.lower().str.strip().values if self.status_col in df.columns else ['point'] * len(df)

        events: List[HuntEvent] = []
        current_enter = None
        current_approach = None

        for f, b, s in zip(frames, behaviors, statuses):
            if 'enter' in b or 'on' in b or 'start' in b:
                current_enter = f
                current_approach = f
            elif 'approach' in b or 'pursuit' in b or 'stalk' in b:
                if current_enter is None:
                    current_enter = f
                current_approach = f
            elif 'capture' in b or 'strike' in b:
                if current_enter is not None:
                    events.append(HuntEvent(
                        enter=current_enter,
                        approach_start=current_approach or current_enter,
                        capture=f,
                        consumption_end=None
                    ))
                    current_enter = None
                    current_approach = None
            elif 'eat' in b or 'consum' in b:
                if events and events[-1].consumption_end is None:
                    events[-1].consumption_end = f

        return events

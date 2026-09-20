import os
import re
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, overload, Literal
from .base_parser import BaseLabelParser, HuntEvent

class DianeExcelLabelParser(BaseLabelParser):
    """
    Parses master manual labeling sheets in Diane's format.
    Columns:
      Animal | Day | Cricket | Released | First approach | Final approach | Capture | Eaten

    Handles:
      - Animal code listed only once on the first row of an animal block.
      - Numeric animal codes (e.g. 158 -> 'YFP-158' or '158').
      - Optional prefix animals (e.g. 'YB-12').
      - Variable hunting sessions (crickets) across days.
      - First approach vs Final approach behavioral segmenting.
    """
    def __init__(
        self,
        tentative_approach_max_sec: float = 2.0,  # Max duration in seconds to assign to an aborted first approach
        default_fps: float = 30.0,
        tentative_label_state: int = 1  # 1 = Stalking, 2 = Pursuit
    ):
        self.tentative_approach_max_sec = tentative_approach_max_sec
        self.default_fps = default_fps
        self.tentative_label_state = tentative_label_state

    def normalize_animal_id(self, raw_val: Any) -> str:
        """Converts raw animal cell entry to normalized string (e.g. 158 -> '158', or 'YFP-158')."""
        if pd.isna(raw_val):
            return ""
        # If integer or float represented as int
        if isinstance(raw_val, (int, float)):
            try:
                return str(int(raw_val)).strip()
            except (ValueError, OverflowError):
                pass
        val_str = str(raw_val).strip()
        # If float string like '158.0'
        if val_str.endswith('.0'):
            val_str = val_str[:-2]
        return val_str

    @overload
    def parse_file(
        self,
        filepath: str,
        fps: Optional[float] = ...,
        split_parts: Literal[False] = ...,
        **kwargs: Any
    ) -> Dict[Tuple[str, int], List[Dict[str, Any]]]: ...

    @overload
    def parse_file(
        self,
        filepath: str,
        fps: Optional[float] = ...,
        split_parts: Literal[True] = ...,
        **kwargs: Any
    ) -> Dict[Tuple[str, int], List[List[Dict[str, Any]]]]: ...

    @overload
    def parse_file(
        self,
        filepath: str,
        fps: Optional[float] = ...,
        split_parts: bool = ...,
        **kwargs: Any
    ) -> Union[Dict[Tuple[str, int], List[Dict[str, Any]]], Dict[Tuple[str, int], List[List[Dict[str, Any]]]]]: ...

    def parse_file(
        self,
        filepath: str,
        fps: Optional[float] = None,
        split_parts: bool = False,
        **kwargs: Any
    ) -> Union[Dict[Tuple[str, int], List[Dict[str, Any]]], Dict[Tuple[str, int], List[List[Dict[str, Any]]]]]:
        """
        Parses master Excel/CSV file and returns a mapping:
          If split_parts=False:
            (normalized_animal_id, day_int) -> List of detailed hunt dictionaries.
          If split_parts=True ('diane_parts' format):
            (normalized_animal_id, day_int) -> List of video parts, where each part is a List of hunt dictionaries.
            A new part is started whenever released frame count resets (released < previous_released).
        """
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        df.columns = df.columns.astype(str).str.strip()

        # Check required columns
        req_cols = ['Day', 'Released', 'Capture']
        for rc in req_cols:
            if rc not in df.columns:
                raise ValueError(f"Master label sheet missing required column '{rc}' in {filepath}")

        # Forward-fill animal column
        animal_col = 'Animal' if 'Animal' in df.columns else str(df.columns[0])
        df[animal_col] = df[animal_col].ffill()

        target_fps = fps or self.default_fps
        tentative_max_frames = round(self.tentative_approach_max_sec * target_fps)

        session_hunts: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        session_parts: Dict[Tuple[str, int], List[List[Dict[str, Any]]]] = {}
        last_released: Dict[Tuple[str, int], int] = {}

        for _, row in df.iterrows():
            if pd.isna(row.get('Released')) or pd.isna(row.get('Capture')):
                continue

            raw_animal = row.get(animal_col)
            animal_id = self.normalize_animal_id(raw_animal)
            if not animal_id:
                continue

            try:
                day_val = row.get('Day')
                if pd.isna(day_val):
                    continue
                if isinstance(day_val, (int, float)):
                    day = int(day_val)
                else:
                    m = re.search(r'\d+', str(day_val))
                    day = int(m.group(0)) if m else None
                if day is None:
                    continue

                released = int(float(row['Released']))
                capture = int(float(row['Capture']))
            except (ValueError, TypeError):
                continue

            first_app_val = row.get('First approach')
            final_app_val = row.get('Final approach')
            eaten_val = row.get('Eaten')

            first_app = None
            if first_app_val is not None and pd.notna(first_app_val):
                try:
                    first_app = int(float(first_app_val))
                except (ValueError, TypeError):
                    first_app = None

            final_app = first_app or released
            if final_app_val is not None and pd.notna(final_app_val):
                try:
                    final_app = int(float(final_app_val))
                except (ValueError, TypeError):
                    final_app = first_app or released

            eaten = None
            if eaten_val is not None and pd.notna(eaten_val):
                try:
                    eaten = int(float(eaten_val))
                except (ValueError, TypeError):
                    eaten = None

            cricket_num = row.get('Cricket')
            try:
                cricket_idx = int(float(cricket_num)) if (cricket_num is not None and pd.notna(cricket_num)) else 1
            except (ValueError, TypeError):
                cricket_idx = 1

            hunt_entry = {
                'animal': animal_id,
                'day': day,
                'cricket': cricket_idx,
                'released': released,
                'first_approach': first_app,
                'final_approach': final_app,
                'capture': capture,
                'eaten': eaten,
                'has_aborted_approach': (first_app is not None and final_app is not None and first_app < final_app),
                'tentative_max_frames': tentative_max_frames
            }

            key = (animal_id.upper(), day)
            if key not in session_hunts:
                session_hunts[key] = []
            session_hunts[key].append(hunt_entry)

            # Track parts for diane_parts format (frame counter resets)
            if key not in session_parts:
                session_parts[key] = [[hunt_entry]]
            else:
                if released < last_released.get(key, -1):
                    session_parts[key].append([hunt_entry])
                else:
                    session_parts[key][-1].append(hunt_entry)
            last_released[key] = released

        if split_parts:
            return session_parts
        return session_hunts

    def parse_file_parts(
        self,
        filepath: str,
        fps: Optional[float] = None
    ) -> Dict[Tuple[str, int], List[List[Dict[str, Any]]]]:
        """
        Parses master Excel/CSV file and splits hunts into discrete video parts whenever
        a frame counter reset is detected (i.e. released < previous_released within the same animal and day).
        
        Returns:
            (normalized_animal_id, day_int) -> List of parts, where each part is a List of hunt dictionaries.
        """
        return self.parse_file(filepath, fps=fps, split_parts=True)

    def hunts_to_events(
        self,
        hunt_entries: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        tentative_approach_strategy: str = "engagement"
    ) -> List[HuntEvent]:
        """
        Converts parsed Diane hunt entries into standard HuntEvent objects.
        
        5-Class Behavioral States:
          - State 0: Background (prior to Released, and after Eaten)
          - State 1: Cricket Present [Released, First approach)
          - State 2: First Approach / Stalking [First approach, Final approach)
          - State 3: Final Approach / Pursuit & Strike [Final approach, Capture)
          - State 4: Eating [Capture, Eaten)
        """
        events: List[HuntEvent] = []
        if not hunt_entries:
            return events

        # Flatten if nested parts list was passed
        entries_list: List[Dict[str, Any]] = []
        if isinstance(hunt_entries[0], list):
            for part in hunt_entries:
                if isinstance(part, list):
                    entries_list.extend(part)
        else:
            entries_list = hunt_entries  # type: ignore

        for h in entries_list:
            released = h['released']
            first_app = h['first_approach']
            final_app = h['final_approach']
            capture = h['capture']
            eaten = h['eaten']

            if first_app is None or first_app == final_app or tentative_approach_strategy in ('engagement', 'continuous'):
                events.append(HuntEvent(
                    enter=released,
                    approach_start=final_app,
                    capture=capture,
                    consumption_end=eaten,
                    first_approach=first_app,
                    metadata={'cricket': h['cricket'], 'released': released}
                ))
            elif tentative_approach_strategy == 'final_only':
                events.append(HuntEvent(
                    enter=released,
                    approach_start=final_app,
                    capture=capture,
                    consumption_end=eaten,
                    first_approach=final_app,
                    metadata={'cricket': h['cricket'], 'released': released, 'aborted_at': first_app}
                ))
            elif tentative_approach_strategy == 'split':
                # Event 1: Tentative aborted approach
                tentative_len = min(final_app - first_app, h['tentative_max_frames'])
                tentative_end = first_app + tentative_len
                if tentative_end > first_app:
                    if self.tentative_label_state == 1:
                        # Mark as State 1 (Stalking): enter=first_app, approach_start=tentative_end
                        events.append(HuntEvent(
                            enter=first_app,
                            approach_start=tentative_end,
                            capture=tentative_end,
                            consumption_end=tentative_end,
                            metadata={'cricket': h['cricket'], 'type': 'aborted_stalk'}
                        ))
                    else:
                        # Mark as State 2 (Pursuit): enter=first_app, approach_start=first_app, capture=tentative_end
                        events.append(HuntEvent(
                            enter=first_app,
                            approach_start=first_app,
                            capture=tentative_end,
                            consumption_end=tentative_end,
                            metadata={'cricket': h['cricket'], 'type': 'aborted_pursuit'}
                        ))

                # Event 2: Final approach and capture
                events.append(HuntEvent(
                    enter=final_app,
                    approach_start=final_app,
                    capture=capture,
                    consumption_end=eaten,
                    metadata={'cricket': h['cricket'], 'type': 'final_capture'}
                ))

        return events

import os
import glob
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, cast
from .base_parser import BaseLabelParser, HuntEvent

class GenericExcelLabelParser(BaseLabelParser):
    """
    Parses spreadsheets (.xlsx or .csv) containing hunt event timestamps for a single video.
    Supports flexible column aliases and exclusions.
    """
    def __init__(
        self,
        enter_cols: Optional[List[str]] = None,
        approach_cols: Optional[List[str]] = None,
        capture_cols: Optional[List[str]] = None,
        consumption_cols: Optional[List[str]] = None,
        include_col: str = "Include?",
        exclude_val: str = "Exclude"
    ):
        self.enter_cols = enter_cols or ["Enter_Frame", "Enter", "On", "Start_Frame"]
        self.approach_cols = approach_cols or ["Pursuit_Frame", "Pursuit_Start_Frame", "Approach Start", "Approach", "Attack_Frame"]
        self.capture_cols = capture_cols or ["Capture_Frame", "Capture", "Strike_Frame"]
        self.consumption_cols = consumption_cols or ["Consumption_End_Frame", "Consumption", "Eating_End_Frame"]
        self.include_col = include_col
        self.exclude_val = exclude_val

    def _find_col(self, df_cols: List[str], candidates: List[str]) -> Optional[str]:
        col_map = {c.strip().lower(): c for c in df_cols}
        for cand in candidates:
            if cand.lower() in col_map:
                return col_map[cand.lower()]
        return None

    def parse_file(self, filepath: str, **kwargs: Any) -> List[HuntEvent]:
        if filepath.endswith('.csv'):
            raw_df = pd.read_csv(filepath)
        else:
            raw_df = pd.read_excel(filepath)

        if not isinstance(raw_df, pd.DataFrame):
            return []
        df: pd.DataFrame = raw_df

        df.columns = df.columns.astype(str).str.strip()

        # Handle row exclusions
        if self.include_col in df.columns:
            df = cast(pd.DataFrame, df.loc[df[self.include_col] != self.exclude_val])

        cols_list = list(df.columns)
        enter_c = self._find_col(cols_list, self.enter_cols)
        approach_c = self._find_col(cols_list, self.approach_cols)
        capture_c = self._find_col(cols_list, self.capture_cols)
        cons_c = self._find_col(cols_list, self.consumption_cols)

        if not enter_c or not capture_c:
            return []

        # Sort chronologically by enter frame
        df = df.sort_values(by=enter_c)
        events: List[HuntEvent] = []

        for _, row in df.iterrows():
            if pd.isna(row.get(enter_c)) or pd.isna(row.get(capture_c)):
                continue

            try:
                enter = int(row[enter_c])
                capture = int(row[capture_c])
            except (ValueError, TypeError):
                continue

            approach = None
            if approach_c and pd.notna(row.get(approach_c)):
                try:
                    val = int(float(row[approach_c]))
                    if val > enter:
                        approach = val
                except (ValueError, TypeError):
                    pass

            # Check latency / duration offset if approach was not in explicit frame column
            if approach is None:
                fps = float(kwargs.get('fps', 30.0))
                for lat_cand in ['latency_to_attack_frames', 'latency_to_attack', 'latency', 'attack_latency_frames']:
                    lat_c = self._find_col(cols_list, [lat_cand])
                    if lat_c and pd.notna(row.get(lat_c)):
                        try:
                            lat_val = float(row[lat_c])
                            if lat_val > 0:
                                if 'frame' in lat_c.lower() or lat_val >= 20:
                                    approach = enter + int(round(lat_val))
                                else:
                                    approach = enter + int(round(lat_val * fps))
                                break
                        except (ValueError, TypeError):
                            pass

                if approach is None and capture is not None:
                    for dur_cand in ['pursuit_time_frames', 'attack_duration_frames', 'attack duration', 'pursuit_time']:
                        dur_c = self._find_col(cols_list, [dur_cand])
                        if dur_c and pd.notna(row.get(dur_c)):
                            try:
                                dur_val = float(row[dur_c])
                                if dur_val > 0:
                                    dur_f = int(round(dur_val)) if ('frame' in dur_c.lower() or dur_val >= 20) else int(round(dur_val * fps))
                                    if capture > dur_f:
                                        approach = capture - dur_f
                                        break
                            except (ValueError, TypeError):
                                pass

            cons_end = None
            if cons_c and pd.notna(row.get(cons_c)):
                try:
                    cons_end = int(row[cons_c])
                except (ValueError, TypeError):
                    cons_end = None

            event = HuntEvent(
                enter=enter,
                approach_start=approach,
                capture=capture,
                consumption_end=cons_end
            )
            events.append(event)

        return events

class MultiSessionSheetParser(BaseLabelParser):
    """
    Parses spreadsheets where one file aggregates all hunting sessions across days and animals.
    Returns mapping: (animal_id, day_int, session_int) -> List[HuntEvent]
    """
    def __init__(
        self,
        animal_col: str = "Animal",
        day_col: str = "Day",
        session_col: str = "Session",
        enter_col: str = "On",
        approach_col: str = "Approach Start",
        capture_col: str = "Capture",
        consumption_col: str = "Consumption"
    ):
        self.animal_col = animal_col
        self.day_col = day_col
        self.session_col = session_col
        self.enter_col = enter_col
        self.approach_col = approach_col
        self.capture_col = capture_col
        self.consumption_col = consumption_col

    def parse_file(self, filepath: str, **kwargs: Any) -> Dict[Tuple[str, int, int], List[HuntEvent]]:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        df.columns = df.columns.astype(str).str.strip()
        hunts_map: Dict[Tuple[str, int, int], List[HuntEvent]] = {}

        if not all(c in df.columns for c in [self.animal_col, self.day_col, self.session_col]):
            return hunts_map

        # Forward fill animal column if only listed once at the top of the block
        df[self.animal_col] = df[self.animal_col].ffill()

        fps = 30.0
        if 'FPS' in df.columns and pd.notna(df['FPS'].dropna().iloc[0] if len(df['FPS'].dropna()) > 0 else None):
            try: fps = float(df['FPS'].dropna().iloc[0])
            except (ValueError, TypeError): fps = 30.0
        elif kwargs.get('fps'):
            fps = float(kwargs['fps'])

        for _, row in df.iterrows():
            if pd.isna(row.get(self.enter_col)) or pd.isna(row.get(self.capture_col)):
                continue

            try:
                animal = str(row[self.animal_col]).strip().upper()
                day = int(row[self.day_col])
                session = int(row[self.session_col])
                enter = int(row[self.enter_col])
                capture = int(row[self.capture_col])
            except (ValueError, TypeError):
                continue

            approach = None
            app_cols = [self.approach_col, 'Approach Start', 'Approach_Start_Frame', 'Pursuit_Frame', 'Pursuit_Start_Frame', 'Attack_Frame', 'Approach', 'Pursuit']
            for ac in app_cols:
                if ac in df.columns and pd.notna(row.get(ac)):
                    try:
                        v = int(float(row[ac]))
                        if v > enter:
                            approach = v
                            break
                    except (ValueError, TypeError):
                        pass

            if approach is None:
                for lat_cand in ['Latency_to_Attack_Frames', 'Latency_to_Attack', 'Latency', 'Attack_Latency_Frames']:
                    if lat_cand in df.columns and pd.notna(row.get(lat_cand)):
                        try:
                            lat_val = float(row[lat_cand])
                            if lat_val > 0:
                                lat_f = int(round(lat_val)) if ('frame' in lat_cand.lower() or lat_val >= 20) else int(round(lat_val * fps))
                                approach = enter + lat_f
                                break
                        except (ValueError, TypeError):
                            pass

                if approach is None and capture is not None:
                    for dur_cand in ['Pursuit_Time_Frames', 'Attack Duration', 'Attack_Duration_Frames', 'Pursuit_Time']:
                        if dur_cand in df.columns and pd.notna(row.get(dur_cand)):
                            try:
                                dur_val = float(row[dur_cand])
                                if dur_val > 0:
                                    dur_f = int(round(dur_val)) if ('frame' in dur_cand.lower() or dur_val >= 20) else int(round(dur_val * fps))
                                    if capture > dur_f:
                                        approach = capture - dur_f
                                        break
                            except (ValueError, TypeError):
                                pass

            cons_end = None
            if pd.notna(row.get(self.consumption_col)):
                try:
                    cons_end = int(float(row[self.consumption_col]))
                except (ValueError, TypeError):
                    cons_end = None

            key = (animal, day, session)
            if key not in hunts_map:
                hunts_map[key] = []

            hunts_map[key].append(HuntEvent(
                enter=enter,
                approach_start=approach,
                capture=capture,
                consumption_end=cons_end
            ))

        return hunts_map

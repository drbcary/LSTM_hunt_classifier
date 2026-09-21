import os
import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict, Any, Tuple
import re
try:
    from scipy.spatial import KDTree
except ImportError:
    from scipy.spatial import cKDTree as KDTree  # type: ignore
from scipy.sparse.csgraph import connected_components

from config.base_config import BaseParadigmConfig, sec_to_frames
from config.mouse_config import MousePoseSchema, MouseParadigmConfig
from .keypoints import PoseData

def extract_filename_timestamp(filepath: str) -> Tuple[int, int, str]:
    """
    Extracts chronological sort key from filename for multi-part video stitching.
    Matches date and time patterns like:
      - 220615-094028 (YYMMDD-HHMMSS)
      - 20220615_...
    Returns a comparable tuple (date_int, time_int, filename).
    """
    base = os.path.basename(filepath)
    dt_match = re.findall(r'(\d{6,8})[-_](\d{6})', base)
    if dt_match:
        date_str, time_str = dt_match[-1]
        return (int(date_str), int(time_str), base)
    digits = re.findall(r'\d{6,8}', base)
    if digits:
        return (int(digits[0]), int(digits[1]) if len(digits) > 1 else 0, base)
    return (0, 0, base)

def extract_validated_poop_clusters(
    df: pd.DataFrame,
    scorer: str,
    poop_prefix: str = "poop",
    poop_thresh: float = 0.9,
    poop_dist_thresh: float = 6.0,
    min_persistence_frames: int = 15
) -> List[Dict[str, Any]]:
    """
    Identifies stationary poop clusters in the video while filtering out transient
    detections (such as moving cricket body parts erroneously labeled as poop).
    
    Returns a list of dicts for each validated physical poop cluster:
        {
            'center': (mean_x, mean_y),
            'birth_frame': int (minimum frame index when this cluster was first detected),
            'frame_count': int (number of frames this cluster was detected),
            'points': np.ndarray of shape (K, 2)
        }
    """
    bodyparts = df[scorer].columns.get_level_values(0).unique()
    poop_parts = [bp for bp in bodyparts if isinstance(bp, str) and bp.startswith(poop_prefix)]
    if not poop_parts:
        return []

    all_t: List[int] = []
    all_px: List[float] = []
    all_py: List[float] = []

    for poop_part in poop_parts:
        p_x = df[scorer][poop_part]['x'].values
        p_y = df[scorer][poop_part]['y'].values
        p_conf = df[scorer][poop_part]['likelihood'].values

        valid = (p_conf > poop_thresh) & ~np.isnan(p_x) & ~np.isnan(p_y)
        if np.any(valid):
            v_idx = np.where(valid)[0]
            all_t.extend(v_idx)
            all_px.extend(p_x[valid])
            all_py.extend(p_y[valid])

    if not all_px:
        return []

    t_arr = np.array(all_t, dtype=np.int64)
    px_arr = np.array(all_px, dtype=np.float32)
    py_arr = np.array(all_py, dtype=np.float32)

    # 1. Quantize coordinates to integer pixels for blistering-fast deduplication
    pts = np.column_stack([np.round(px_arr), np.round(py_arr)]).astype(np.int32)
    unique_pts, inverse_indices = np.unique(pts, axis=0, return_inverse=True)

    # 2. Build graph of unique spatial points within poop_dist_thresh
    tree = KDTree(unique_pts)
    adj = tree.sparse_distance_matrix(tree, max_distance=max(poop_dist_thresh, 1.0))
    n_clusters, cluster_labels = connected_components(adj, directed=False)

    # 3. Map cluster labels back to all detections
    det_clusters = cluster_labels[inverse_indices]

    validated_clusters: List[Dict[str, Any]] = []
    for c_id in range(n_clusters):
        c_mask = (det_clusters == c_id)
        count = int(np.sum(c_mask))

        # Discard transient flickers / moving cricket parts
        if count < min_persistence_frames:
            continue

        c_frames = t_arr[c_mask]
        birth = int(np.min(c_frames))
        cx = float(np.mean(px_arr[c_mask]))
        cy = float(np.mean(py_arr[c_mask]))

        validated_clusters.append({
            'center': (cx, cy),
            'birth_frame': birth,
            'frame_count': count,
            'points': np.column_stack([px_arr[c_mask], py_arr[c_mask]])
        })

    return validated_clusters

def apply_poop_penalty(
    df: pd.DataFrame,
    scorer: str,
    c_x: Union[np.ndarray, pd.Series],
    c_y: Union[np.ndarray, pd.Series],
    c_conf: Union[np.ndarray, pd.Series],
    poop_prefix: str = "poop",
    poop_thresh: float = 0.9,
    poop_dist_thresh: float = 6.0,
    cricket_thresh: float = 0.85,
    min_persistence_frames: int = 15,
    validated_clusters: Optional[List[Dict[str, Any]]] = None
) -> np.ndarray:
    """
    Finds validated stationary poop clusters in the video. If a cricket detection is within 
    poop_dist_thresh pixels of a poop cluster AND the frame is >= the birth frame of that poop,
    applies the poop penalty.
    """
    c_conf_vals = c_conf.values if hasattr(c_conf, 'values') else np.array(c_conf)

    if validated_clusters is None:
        validated_clusters = extract_validated_poop_clusters(
            df=df,
            scorer=scorer,
            poop_prefix=poop_prefix,
            poop_thresh=poop_thresh,
            poop_dist_thresh=poop_dist_thresh,
            min_persistence_frames=min_persistence_frames
        )

    if not validated_clusters:
        return c_conf_vals.copy()

    poop_centers = np.array([cl['center'] for cl in validated_clusters], dtype=np.float32)
    poop_births = np.array([cl['birth_frame'] for cl in validated_clusters], dtype=np.int64)
    tree = KDTree(poop_centers)

    c_x_vals = c_x.values if hasattr(c_x, 'values') else np.array(c_x)
    c_y_vals = c_y.values if hasattr(c_y, 'values') else np.array(c_y)
    valid_c = ~np.isnan(c_x_vals) & ~np.isnan(c_y_vals)

    new_c_conf = c_conf_vals.copy()
    if np.any(valid_c):
        c_coords = np.column_stack((c_x_vals[valid_c], c_y_vals[valid_c]))
        valid_indices = np.where(valid_c)[0]

        distances, nearest_poop_idx = tree.query(c_coords)
        close_mask = distances < poop_dist_thresh

        if np.any(close_mask):
            candidate_frames = valid_indices[close_mask]
            matched_poops = nearest_poop_idx[close_mask]

            # Causal check: cricket frame must be >= birth frame of that poop!
            is_causal = candidate_frames >= poop_births[matched_poops]

            penalty = (1.0 - cricket_thresh) - 0.02
            penalize_indices = candidate_frames[is_causal]
            new_c_conf[penalize_indices] -= penalty

    return new_c_conf

class DLCAdapter:
    """
    Parses DeepLabCut tracking files into standard PoseData representation.
    Supports arbitrary keypoints, multi-segment tails, and species-agnostic schemas.
    """
    def __init__(self, config: Optional[BaseParadigmConfig] = None):
        self.config = config or MouseParadigmConfig()

    def load(
        self,
        filepath: Union[str, List[str]],
        video_fps: Optional[float] = None,
        pred_thresh: Optional[float] = None,
        cricket_thresh: Optional[float] = None,
        cricket_coord_thresh: Optional[float] = None,
        interp_limit: Optional[int] = None,
        include_blurry_cricket: Optional[bool] = None,
        include_cricket_immo: Optional[bool] = None,
        filter_poops: Optional[bool] = None,
        poop_dist_thresh: Optional[float] = None,
        poop_thresh: Optional[float] = None,
        filter_dead_crickets: bool = True,
        schema: Optional[Any] = None
    ) -> Optional[PoseData]:
        fps = video_fps or self.config.default_fps
        schema = schema or getattr(self.config, 'schema', MousePoseSchema())
        thresholds = self.config.thresholds
        
        pred_th = pred_thresh if pred_thresh is not None else thresholds.predator_likelihood_thresh
        crick_th = cricket_thresh if cricket_thresh is not None else thresholds.cricket_likelihood_thresh
        if cricket_coord_thresh is not None:
            coord_th = cricket_coord_thresh
        elif getattr(self.config, 'continuous_cricket_confidence', False):
            coord_th = getattr(thresholds, 'cricket_coord_thresh', 0.30)
        else:
            coord_th = crick_th
        do_interp_limit = interp_limit if interp_limit is not None else sec_to_frames(thresholds.max_interp_gap_sec, fps)
        do_blurry = include_blurry_cricket if include_blurry_cricket is not None else getattr(schema, 'include_blurry_cricket', True)
        do_immo = include_cricket_immo if include_cricket_immo is not None else getattr(schema, 'include_cricket_immo', True)
        do_filter_poop = thresholds.filter_poops if filter_poops is None else filter_poops
        p_dist_th = thresholds.poop_dist_thresh if poop_dist_thresh is None else poop_dist_thresh
        p_th = thresholds.poop_thresh if poop_thresh is None else poop_thresh

        try:
            if isinstance(filepath, (list, tuple)):
                if len(filepath) == 0:
                    return None
                if len(filepath) == 1:
                    single_path = filepath[0]
                    if single_path.endswith('.h5'):
                        df = pd.read_hdf(single_path)
                    else:
                        df = pd.read_csv(single_path, header=[0, 1, 2], index_col=0)
                else:
                    sorted_files = sorted(filepath, key=extract_filename_timestamp)
                    dfs = []
                    for f in sorted_files:
                        if f.endswith('.h5'):
                            part_df = pd.read_hdf(f)
                        else:
                            part_df = pd.read_csv(f, header=[0, 1, 2], index_col=0)
                        if not isinstance(part_df, pd.DataFrame):
                            part_df = pd.DataFrame(part_df)
                        dfs.append(part_df)
                    df = pd.concat(dfs, ignore_index=True)
            elif isinstance(filepath, str):
                if filepath.endswith('.h5'):
                    df = pd.read_hdf(filepath)
                else:
                    df = pd.read_csv(filepath, header=[0, 1, 2], index_col=0)
            else:
                return None
        except Exception as e:
            print(f"Error loading DLC file(s) {filepath}: {e}")
            return None

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        scorer = str(df.columns.get_level_values(0)[0])
        num_frames = len(df)
        available_parts = df[scorer].columns.get_level_values(0).unique()
        part_lookup = {str(p).lower().replace('_', '').replace(' ', ''): p for p in available_parts}

        def get_part(part_name: Optional[str], thresh: float):
            if not part_name:
                return np.full(num_frames, np.nan), np.full(num_frames, np.nan), np.zeros(num_frames)
            
            resolved_name = part_name
            if resolved_name not in df[scorer]:
                cleaned = part_name.lower().replace('_', '').replace(' ', '')
                if cleaned in part_lookup:
                    resolved_name = part_lookup[cleaned]
                else:
                    return np.full(num_frames, np.nan), np.full(num_frames, np.nan), np.zeros(num_frames)
            
            x = df[scorer][resolved_name]['x'].copy()
            y = df[scorer][resolved_name]['y'].copy()
            conf = df[scorer][resolved_name]['likelihood'].copy()

            mask = conf < thresh
            x[mask] = np.nan
            y[mask] = np.nan

            if do_interp_limit > 0:
                x = x.interpolate(method='linear', limit=do_interp_limit, limit_direction='both')
                y = y.interpolate(method='linear', limit=do_interp_limit, limit_direction='both')

            return x.values, y.values, conf.values

        # 1. Extract Head
        hx, hy, h_conf = get_part(schema.head, pred_th)
        if getattr(schema, 'head_alt', None) and np.all(np.isnan(hx)):
            hx, hy, h_conf = get_part(schema.head_alt, pred_th)
        elif getattr(schema, 'head_alt', None):
            alt_x, alt_y, alt_conf = get_part(schema.head_alt, pred_th)
            mask_head = np.isnan(hx)
            hx[mask_head] = alt_x[mask_head]
            hy[mask_head] = alt_y[mask_head]
            h_conf[mask_head] = alt_conf[mask_head]

        head_pos = np.column_stack((hx, hy))

        # 2. Extract Center
        cx, cy, c_body_conf = get_part(schema.center, pred_th)
        if getattr(schema, 'center_alt', None) and np.all(np.isnan(cx)):
            cx, cy, c_body_conf = get_part(schema.center_alt, pred_th)
        center_pos = np.column_stack((cx, cy))

        # 3. Extract Rear / Pelvis / Tail base
        rx, ry, r_conf = get_part(schema.rear, pred_th)
        if getattr(schema, 'rear_alt', None) and np.all(np.isnan(rx)):
            rx, ry, r_conf = get_part(schema.rear_alt, pred_th)
        rear_pos = np.column_stack((rx, ry))

        # 4. Extract Multi-Segment Tail Points
        tail_points_list: List[np.ndarray] = []
        tail_names = getattr(schema, 'tail_points', [])
        for t_name in tail_names:
            tx, ty, _ = get_part(t_name, pred_th)
            tail_points_list.append(np.column_stack((tx, ty)))
        
        # If no explicit tail points provided, fall back to rear point
        if not tail_points_list:
            tail_points_list.append(rear_pos.copy())

        # Precompute validated poop clusters once for all candidate streams
        validated_poop_clusters = []
        if do_filter_poop:
            validated_poop_clusters = extract_validated_poop_clusters(
                df=df,
                scorer=scorer,
                poop_prefix=getattr(schema, 'poop_prefix', 'poop'),
                poop_thresh=p_th,
                poop_dist_thresh=p_dist_th,
                min_persistence_frames=getattr(thresholds, 'min_poop_frames', 15)
            )

        # 5. Extract Cricket / Prey Candidates (regular, blurry, and immobilized)
        candidates = []
        # Candidate 1: Standard cricket
        c_part = getattr(schema, 'cricket', 'cricket')
        if c_part:
            c_x, c_y, raw_c = get_part(c_part, 0.0)
            pen_c = apply_poop_penalty(
                df, scorer, c_x, c_y, raw_c,
                poop_prefix=getattr(schema, 'poop_prefix', 'poop'),
                poop_thresh=p_th,
                poop_dist_thresh=p_dist_th,
                cricket_thresh=crick_th,
                validated_clusters=validated_poop_clusters
            ) if do_filter_poop else raw_c.copy()
            candidates.append(('regular', c_x, c_y, pen_c, raw_c))

        # Candidate 2: Blurry cricket
        bc_part = getattr(schema, 'blurry_cricket', 'blurry_cricket')
        if do_blurry and bc_part:
            bc_x, bc_y, raw_bc = get_part(bc_part, 0.0)
            if not np.all(np.isnan(bc_x)) or np.any(raw_bc > 0.0):
                pen_bc = apply_poop_penalty(
                    df, scorer, bc_x, bc_y, raw_bc,
                    poop_prefix=getattr(schema, 'poop_prefix', 'poop'),
                    poop_thresh=p_th,
                    poop_dist_thresh=p_dist_th,
                    cricket_thresh=crick_th,
                    validated_clusters=validated_poop_clusters
                ) if do_filter_poop else raw_bc.copy()
                candidates.append(('blurry', bc_x, bc_y, pen_bc, raw_bc))

        # Candidate 3: Immobilized cricket
        immo_part = getattr(schema, 'cricket_immo', 'cricket_immo')
        if do_immo and immo_part:
            immo_x, immo_y, raw_immo = get_part(immo_part, 0.0)
            if not np.all(np.isnan(immo_x)) or np.any(raw_immo > 0.0):
                pen_immo = apply_poop_penalty(
                    df, scorer, immo_x, immo_y, raw_immo,
                    poop_prefix=getattr(schema, 'poop_prefix', 'poop'),
                    poop_thresh=p_th,
                    poop_dist_thresh=p_dist_th,
                    cricket_thresh=crick_th,
                    validated_clusters=validated_poop_clusters
                ) if do_filter_poop else raw_immo.copy()
                candidates.append(('immo', immo_x, immo_y, pen_immo, raw_immo))

        if candidates:
            cand_names = [c[0] for c in candidates]
            cand_xs = np.column_stack([c[1] for c in candidates])
            cand_ys = np.column_stack([c[2] for c in candidates])
            cand_confs = np.column_stack([c[3] for c in candidates])
            cand_raws = np.column_stack([c[4] for c in candidates])

            # For each frame, select candidate with the highest confidence
            best_idx = np.argmax(cand_confs, axis=1)
            row_idx = np.arange(num_frames)

            best_conf = cand_confs[row_idx, best_idx]
            best_raw_conf = cand_raws[row_idx, best_idx]
            best_x = cand_xs[row_idx, best_idx]
            best_y = cand_ys[row_idx, best_idx]
            best_types = np.array(cand_names)[best_idx]

            valid_mask = best_conf > coord_th
            final_cx = np.full(num_frames, np.nan)
            final_cy = np.full(num_frames, np.nan)
            final_cx[valid_mask] = best_x[valid_mask]
            final_cy[valid_mask] = best_y[valid_mask]

            final_c_conf = np.where(valid_mask, best_conf, 0.0)
            raw_final_c_conf = np.where(valid_mask, best_raw_conf, 0.0)
            is_blurry = (best_types == 'blurry') & (best_conf > crick_th)
            is_immo = (best_types == 'immo') & (best_conf > crick_th)
        else:
            final_cx = np.full(num_frames, np.nan)
            final_cy = np.full(num_frames, np.nan)
            final_c_conf = np.zeros(num_frames)
            raw_final_c_conf = np.zeros(num_frames)
            is_blurry = np.zeros(num_frames, dtype=bool)
            is_immo = np.zeros(num_frames, dtype=bool)

        if do_interp_limit > 0:
            final_cx = pd.Series(final_cx).interpolate(limit=do_interp_limit).values
            final_cy = pd.Series(final_cy).interpolate(limit=do_interp_limit).values
            final_c_conf = pd.Series(final_c_conf).interpolate(limit=do_interp_limit).values

        # Filter fake/dead crickets using rolling window scaled by FPS if enabled
        if filter_dead_crickets:
            cx_diff = np.diff(final_cx, prepend=final_cx[0] if len(final_cx) > 0 else 0)
            cy_diff = np.diff(final_cy, prepend=final_cy[0] if len(final_cy) > 0 else 0)
            movement = np.sqrt(cx_diff**2 + cy_diff**2)

            window_size = max(1, sec_to_frames(self.config.time.cricket_dead_window_sec, fps))
            smoothed_movement = pd.Series(movement).rolling(window=window_size, min_periods=1).mean().values
            is_dead = smoothed_movement < thresholds.cricket_movement_thresh
            is_dead_clean = np.nan_to_num(is_dead, nan=0).astype(bool)

            padded = np.insert(is_dead_clean, [0, len(is_dead_clean)], [False, False])
            diff = np.diff(padded.astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]

            dead_frame_thresh = sec_to_frames(self.config.time.dead_cricket_time_sec, fps)
            for s, e in zip(starts, ends):
                if (e - s) >= dead_frame_thresh:
                    final_cx[s:e] = np.nan
                    final_cy[s:e] = np.nan
                    final_c_conf[s:e] = 0.0
                    raw_final_c_conf[s:e] = 0.0

        cricket_pos = np.column_stack((final_cx, final_cy))

        # 6. Extract Optional Ears
        left_ear_pos = None
        right_ear_pos = None
        if getattr(schema, 'left_ear', None):
            lx, ly, _ = get_part(schema.left_ear, pred_th)
            left_ear_pos = np.column_stack((lx, ly))
        if getattr(schema, 'right_ear', None):
            rx, ry, _ = get_part(schema.right_ear, pred_th)
            right_ear_pos = np.column_stack((rx, ry))

        return PoseData(
            num_frames=num_frames,
            head=head_pos,
            center=center_pos,
            rear=rear_pos,
            tail_points=tail_points_list,
            cricket=cricket_pos,
            left_ear=left_ear_pos,
            right_ear=right_ear_pos,
            confidences={
                'head': h_conf,
                'center': c_body_conf,
                'rear': r_conf,
                'cricket': final_c_conf,
                'raw_cricket': raw_final_c_conf,
                'is_blurry': is_blurry,
                'is_immo': is_immo
            }
        )

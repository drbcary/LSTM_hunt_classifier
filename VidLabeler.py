import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
from tqdm import tqdm

# Pull in DLC thresholds from dataset_utils
import dataset_utils

rat_thresh = dataset_utils.RAT_LIKELIHOOD_THRESH
cricket_thresh = dataset_utils.CRICKET_LIKELIHOOD_THRESH

# ==========================================
# Configuration
# ==========================================
# Rat Configuration (4 classes)
CLASS_NAMES_RAT = ['Background', 'Stalking', 'Pursuit', 'Eating']
CLASS_COLORS_RAT_CV2 = [
    (211, 211, 211), # LightGray
    (0, 165, 255),   # Orange
    (0, 0, 255),     # Red
    (0, 128, 0)      # Green
]

# Mouse Configuration (5 classes)
CLASS_NAMES_MOUSE = ['Background', 'Cricket Present', 'First Approach', 'Final Approach', 'Eating']
CLASS_COLORS_MOUSE_CV2 = [
    (140, 140, 140), # Background (Gray)
    (207, 190, 23),  # Cricket Present (Teal / Cyan in BGR)
    (180, 119, 31),  # First Approach (Steel Blue in BGR)
    (40, 39, 214),   # Final Approach (Crimson Red in BGR)
    (44, 160, 44),   # Eating (Forest Green in BGR)
]

# Aliases for backwards compatibility
CLASS_NAMES = CLASS_NAMES_RAT
CLASS_COLORS_CV2 = CLASS_COLORS_RAT_CV2
CLASS_COLORS_PLT = ['lightgray', 'orange', 'red', 'green']

def render_plot_frame_transparent(probs, current_frame_idx, trailing_frames, width_px, height_px,
                                  plot_poops=False, poop_dist_thresh=None,
                                  class_names=None, class_colors_cv2=None):
    """
    Renders a transparent overlay plot using pure OpenCV for extreme performance.
    Dynamically supports 4-class (rat) and 5-class (mouse) probability displays.
    """
    if class_names is None or class_colors_cv2 is None:
        if probs.shape[1] == 5:
            class_names = CLASS_NAMES_MOUSE
            class_colors_cv2 = CLASS_COLORS_MOUSE_CV2
        else:
            class_names = CLASS_NAMES_RAT
            class_colors_cv2 = CLASS_COLORS_RAT_CV2

    img = np.zeros((height_px, width_px, 4), dtype=np.uint8)
    
    start = max(0, current_frame_idx - trailing_frames)
    end = current_frame_idx + 10
    total_frames = end - start
    
    if total_frames <= 1:
        return img
        
    data_slice = probs[start:end]
    
    # Define plotting area [left, bottom, width, height]
    pad_left = int(width_px * 0.08)
    pad_bottom = int(height_px * 0.25)
    pad_right = int(width_px * 0.02)
    pad_top = int(height_px * 0.10)
    
    plot_w = width_px - pad_left - pad_right
    plot_h = height_px - pad_bottom - pad_top
    
    # Draw grid/axes
    cv2.rectangle(img, (pad_left, pad_top), (pad_left + plot_w, pad_top + plot_h), (255, 255, 255, 50), 1)
    
    # Draw mid grid line for 0.5 prob
    mid_y = int(pad_top + plot_h / 2)
    cv2.line(img, (pad_left, mid_y), (pad_left + plot_w, mid_y), (255, 255, 255, 30), 1, cv2.LINE_AA)
    
    # Current frame vertical line
    curr_x = int(pad_left + ((current_frame_idx - start) / (total_frames - 1)) * plot_w)
    cv2.line(img, (curr_x, pad_top), (curr_x, pad_top + plot_h), (255, 255, 255, 200), 2, cv2.LINE_AA)
    
    # Draw probability lines
    for i, cls_name in enumerate(class_names):
        if i >= data_slice.shape[1]:
            break
        color_bgr = class_colors_cv2[i]
        # overlay_transparent expects RGBA buffer, so we pass (R, G, B, A) here
        color_rgba = (color_bgr[2], color_bgr[1], color_bgr[0], 230)
        
        pts = []
        for f_idx, prob in enumerate(data_slice[:, i]):
            px = int(pad_left + (f_idx / (total_frames - 1)) * plot_w)
            py = int(pad_top + plot_h - (prob * plot_h))
            pts.append((px, py))
            
        if len(pts) > 1:
            pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts_arr], False, color_rgba, 2, cv2.LINE_AA)
            
    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    font_color = (255, 255, 255, 200) # White RGBA
    
    # Y-axis
    cv2.putText(img, "1.0", (5, pad_top + 5), font, font_scale, font_color, 1, cv2.LINE_AA)
    cv2.putText(img, "0.5", (5, mid_y + 5), font, font_scale, font_color, 1, cv2.LINE_AA)
    cv2.putText(img, "0.0", (5, pad_top + plot_h + 5), font, font_scale, font_color, 1, cv2.LINE_AA)
    
    # X-axis
    cv2.putText(img, str(start), (pad_left, pad_top + plot_h + 15), font, font_scale, font_color, 1, cv2.LINE_AA)
    cv2.putText(img, str(end), (pad_left + plot_w - 25, pad_top + plot_h + 15), font, font_scale, font_color, 1, cv2.LINE_AA)
    cv2.putText(img, "Frame Number", (pad_left + plot_w//2 - 40, pad_top + plot_h + 25), font, font_scale, font_color, 1, cv2.LINE_AA)
    
    # Legend
    leg_x = pad_left + 10
    leg_y = pad_top + 13
    for i, cls_name in enumerate(class_names):
        color_bgr = class_colors_cv2[i]
        color_rgba = (color_bgr[2], color_bgr[1], color_bgr[0], 230)
        cv2.line(img, (leg_x, leg_y - 4), (leg_x + 15, leg_y - 4), color_rgba, 2, cv2.LINE_AA)
        cv2.putText(img, cls_name, (leg_x + 20, leg_y), font, 0.3, font_color, 1, cv2.LINE_AA)
        leg_y += 11

    if plot_poops:
        cv2.line(img, (leg_x, leg_y - 4), (leg_x + 15, leg_y - 4), (150, 75, 0, 230), 2, cv2.LINE_AA)
        p_label = f"Poop Mask ({poop_dist_thresh}px)" if poop_dist_thresh is not None else "Poop Mask"
        cv2.putText(img, p_label, (leg_x + 20, leg_y), font, 0.3, font_color, 1, cv2.LINE_AA)

    return img

def overlay_transparent(background, overlay, x, y):
    bg_h, bg_w, _ = background.shape
    fg_h, fg_w, _ = overlay.shape

    if x >= bg_w or y >= bg_h: return background

    w = min(fg_w, bg_w - x)
    h = min(fg_h, bg_h - y)

    if w <= 0 or h <= 0: return background

    # Extract alpha
    alpha = overlay[0:h, 0:w, 3] / 255.0
    overlay_rgb = overlay[0:h, 0:w, 0:3]
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

    # Blend
    for c in range(3):
        background[y:y+h, x:x+w, c] = (alpha * overlay_bgr[:, :, c] + 
                                       (1.0 - alpha) * background[y:y+h, x:x+w, c])
    return background

def create_hunting_video_clip(video_path, probs, preds, start_frame, end_frame, output_path=None,
                              trailing_frames=150, dlc_path=None, circle_radius=4, dlc_thresh=0.7,
                              cricket_thresh=None,
                              show_confidence=True, include_blurry_cricket=True,
                              include_cricket_immo=True, filter_poops=True,
                              poop_dist_thresh=20, poop_thresh=0.9, plot_poops=False,
                              interp_limit=0, paradigm="auto"):
    """
    Renders an annotated hunting behavior video clip.
    Dynamically supports both rat (4-class) and mouse (5-class) paradigms with:
      - Scrolling probability HUD
      - Current state pill box
      - DLC posture skeleton (snout, ears, spine, and 5-segment tail chain for mouse)
      - Cricket landmark and optional poop exclusion mask
    """
    if output_path is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vids")
        os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(out_dir, f"{base_name}_{start_frame}_{end_frame}_analyzed.mp4")

    if paradigm == "auto":
        is_mouse = (probs.shape[1] == 5)
    else:
        is_mouse = (paradigm.lower() == "mouse")

    class_names = CLASS_NAMES_MOUSE if is_mouse else CLASS_NAMES_RAT
    class_colors_cv2 = CLASS_COLORS_MOUSE_CV2 if is_mouse else CLASS_COLORS_RAT_CV2

    print(f"Generating {'Mouse' if is_mouse else 'Rat'} video clip: {output_path}")
    print(f"Frame Range: {start_frame} to {end_frame} (Classes: {len(class_names)})")
    
    dlc_data = None
    mouse_pose = None
    dlc_confs = None
    poop_data = None
    validated_poop_clusters = []

    if dlc_path is not None:
        if is_mouse:
            try:
                from pose.dlc_adapter import DLCAdapter
                adapter = DLCAdapter()
                mouse_pose = adapter.load(
                    filepath=dlc_path,
                    pred_thresh=dlc_thresh,
                    cricket_thresh=cricket_thresh if cricket_thresh is not None else dlc_thresh,
                    interp_limit=interp_limit,
                    include_blurry_cricket=include_blurry_cricket,
                    include_cricket_immo=include_cricket_immo,
                    filter_poops=filter_poops,
                    poop_dist_thresh=poop_dist_thresh,
                    poop_thresh=poop_thresh
                )
            except Exception as e:
                print(f"Could not load mouse DLC via DLCAdapter: {e}")
                mouse_pose = None
        else:
            res = dataset_utils.extract_dlc_tracking_data(
                dlc_path, rat_thresh=rat_thresh, cricket_thresh=cricket_thresh,
                interp_limit=0, include_blurry_cricket=include_blurry_cricket,
                filter_poops=filter_poops, poop_dist_thresh=poop_dist_thresh,
                poop_thresh=poop_thresh
            )
            if res is not None:
                dlc_data = res

        if show_confidence or plot_poops:
            import pandas as pd
            try:
                if isinstance(dlc_path, (list, tuple)):
                    from pose.dlc_adapter import extract_filename_timestamp
                    sorted_files = sorted(dlc_path, key=extract_filename_timestamp)
                    df = pd.concat([pd.read_csv(f, header=[0, 1, 2], index_col=0) for f in sorted_files], ignore_index=True)
                else:
                    df = pd.read_csv(dlc_path, header=[0, 1, 2], index_col=0)
                scorer = str(df.columns.get_level_values(0)[0])
                
                if not is_mouse and show_confidence:
                    head_conf = np.maximum(df[scorer]['head']['likelihood'].values, df[scorer]['nose']['likelihood'].values)
                    if include_blurry_cricket:
                        c_vals = df[scorer]['cricket']['likelihood'].values.copy()
                        bc_vals = df[scorer]['blurry_cricket']['likelihood'].values.copy()
                        if filter_poops:
                            c_vals = dataset_utils.apply_poop_penalty(df, scorer, df[scorer]['cricket']['x'], df[scorer]['cricket']['y'], c_vals, poop_thresh=poop_thresh, poop_dist_thresh=poop_dist_thresh, cricket_thresh=dlc_thresh)
                            bc_vals = dataset_utils.apply_poop_penalty(df, scorer, df[scorer]['blurry_cricket']['x'], df[scorer]['blurry_cricket']['y'], bc_vals, poop_thresh=poop_thresh, poop_dist_thresh=poop_dist_thresh, cricket_thresh=dlc_thresh)
                        cricket_conf = np.maximum(c_vals, bc_vals)
                        is_blurry = bc_vals > c_vals
                    else:
                        cricket_conf = df[scorer]['cricket']['likelihood'].values.copy()
                        if filter_poops:
                            cricket_conf = dataset_utils.apply_poop_penalty(df, scorer, df[scorer]['cricket']['x'], df[scorer]['cricket']['y'], cricket_conf, poop_thresh=poop_thresh, poop_dist_thresh=poop_dist_thresh, cricket_thresh=dlc_thresh)
                        is_blurry = np.zeros_like(cricket_conf, dtype=bool)
                    
                    dlc_confs = [
                        head_conf,
                        df[scorer]['mid_body']['likelihood'].values,
                        df[scorer]['butt']['likelihood'].values,
                        df[scorer]['endtail']['likelihood'].values,
                        cricket_conf,
                        is_blurry
                    ]
                    
                if plot_poops:
                    from pose.dlc_adapter import extract_validated_poop_clusters
                    validated_poop_clusters = extract_validated_poop_clusters(
                        df, scorer,
                        poop_prefix='poop',
                        poop_thresh=poop_thresh,
                        poop_dist_thresh=poop_dist_thresh,
                        min_persistence_frames=15
                    )
                    bodyparts = df[scorer].columns.get_level_values(0).unique()
                    poop_parts = [bp for bp in bodyparts if isinstance(bp, str) and bp.startswith('poop')]
                    if poop_parts:
                        poop_data = []
                        for p_part in poop_parts:
                            px_vals = df[scorer][p_part]['x'].values
                            py_vals = df[scorer][p_part]['y'].values
                            p_conf_vals = df[scorer][p_part]['likelihood'].values
                            poop_data.append({
                                'name': p_part,
                                'x': px_vals,
                                'y': py_vals,
                                'conf': p_conf_vals
                            })
            except Exception as e:
                print(f"Could not load DLC confidences/poops: {e}")
                dlc_confs = None
                poop_data = None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame < 0 or end_frame > total_frames:
        end_frame = total_frames
        print(f"End frame set to {end_frame} (end of video)")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Precompute dynamic poop exclusion cluster renderers across the video if plot_poops is enabled
    poop_cluster_renderers = []
    poop_color_bgr = (0, 75, 150)       # Brown in BGR
    poop_border_color = (0, 110, 220)   # Amber / bright brown border
    
    if plot_poops and validated_poop_clusters:
        mask_radius = max(1, round(poop_dist_thresh))
        for cl in validated_poop_clusters:
            cl_mask_bin = np.zeros((orig_h, orig_w), dtype=np.uint8)
            for pt in cl['points']:
                ix, iy = int(round(pt[0])), int(round(pt[1]))
                if 0 <= ix < orig_w and 0 <= iy < orig_h:
                    cv2.circle(cl_mask_bin, (ix, iy), mask_radius, 255, -1)
            cl_contours, _ = cv2.findContours(cl_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            poop_cluster_renderers.append({
                'birth_frame': cl['birth_frame'],
                'mask_bool': (cl_mask_bin > 0),
                'contours': cl_contours
            })
        print(f"POOP FILTER VISUALIZER: Prepared {len(poop_cluster_renderers)} validated dynamic poop cluster(s) (min 15 frames persistence).")
    
    plot_h = int(orig_h * 0.26) 
    plot_w = orig_w
    
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, orig_fps, (orig_w, orig_h))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    max_data_idx = len(preds) - 1
    current_frame_idx = start_frame
    total_clip_frames = max(0, end_frame - start_frame)
    pbar = tqdm(total=total_clip_frames, desc="Rendering Video Clip", unit="frame")
    
    try:
        while current_frame_idx < end_frame and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            data_idx = min(current_frame_idx, max_data_idx)
            curr_pred_cls = preds[data_idx]
            
            # 1. Generate Transparent Plot
            plot_img_rgba = render_plot_frame_transparent(
                probs, data_idx, trailing_frames, plot_w, plot_h,
                plot_poops=plot_poops, poop_dist_thresh=poop_dist_thresh,
                class_names=class_names, class_colors_cv2=class_colors_cv2
            )
            
            # 2. Overlay Plot on Frame (Top area)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (orig_w, plot_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame) 
            
            frame = overlay_transparent(frame, plot_img_rgba, 0, 0)
            
            # 2.4 Draw Poop Filter Exclusion Mask dynamically if plot_poops is enabled
            current_active_poop_mask = None
            if plot_poops and poop_cluster_renderers:
                active_clusters = [cl for cl in poop_cluster_renderers if current_frame_idx >= cl['birth_frame']]
                if active_clusters:
                    current_active_poop_mask = np.logical_or.reduce([cl['mask_bool'] for cl in active_clusters])
                    if np.any(current_active_poop_mask):
                        frame[current_active_poop_mask] = ((frame[current_active_poop_mask].astype(np.uint16) * 65 + 
                                                            np.array(poop_color_bgr, dtype=np.uint16) * 35) // 100).astype(np.uint8)
                    for cl in active_clusters:
                        if cl['contours']:
                            cv2.drawContours(frame, cl['contours'], -1, poop_border_color, 1, cv2.LINE_AA)
            
            # 2.5 Draw Tracking Data (Mouse or Rat)
            if is_mouse and mouse_pose is not None:
                # --- Mouse Posture & Multi-Segment Tail Chain ---
                hx, hy = mouse_pose.head[data_idx]
                cx, cy = mouse_pose.center[data_idx]
                rx, ry = mouse_pose.rear[data_idx]
                head_valid = not (np.isnan(hx) or np.isnan(hy))
                center_valid = not (np.isnan(cx) or np.isnan(cy))
                rear_valid = not (np.isnan(rx) or np.isnan(ry))

                # Ear lines & dots
                if mouse_pose.left_ear is not None:
                    lx, ly = mouse_pose.left_ear[data_idx]
                    if not (np.isnan(lx) or np.isnan(ly)):
                        cv2.circle(frame, (int(lx), int(ly)), max(2, circle_radius - 1), (0, 215, 255), -1)
                        if head_valid:
                            cv2.line(frame, (int(hx), int(hy)), (int(lx), int(ly)), (180, 220, 255), 1, cv2.LINE_AA)
                
                if mouse_pose.right_ear is not None:
                    rx_e, ry_e = mouse_pose.right_ear[data_idx]
                    if not (np.isnan(rx_e) or np.isnan(ry_e)):
                        cv2.circle(frame, (int(rx_e), int(ry_e)), max(2, circle_radius - 1), (0, 215, 255), -1)
                        if head_valid:
                            cv2.line(frame, (int(hx), int(hy)), (int(rx_e), int(ry_e)), (180, 220, 255), 1, cv2.LINE_AA)

                # Spine connection lines
                if head_valid and center_valid:
                    cv2.line(frame, (int(hx), int(hy)), (int(cx), int(cy)), (230, 216, 173), 2, cv2.LINE_AA)
                if center_valid and rear_valid:
                    cv2.line(frame, (int(cx), int(cy)), (int(rx), int(ry)), (230, 216, 173), 2, cv2.LINE_AA)

                # Multi-segment tail polyline
                tail_chain = []
                if rear_valid:
                    tail_chain.append((int(rx), int(ry)))
                for tp in mouse_pose.tail_points:
                    tx, ty = tp[data_idx]
                    if not (np.isnan(tx) or np.isnan(ty)):
                        tail_chain.append((int(tx), int(ty)))
                        cv2.circle(frame, (int(tx), int(ty)), max(2, circle_radius - 1), (255, 0, 255), -1)
                
                if len(tail_chain) >= 2:
                    cv2.polylines(frame, [np.array(tail_chain, np.int32)], False, (211, 0, 148), 2, cv2.LINE_AA)

                # Body keypoint dots
                if head_valid:
                    cv2.circle(frame, (int(hx), int(hy)), circle_radius, (255, 255, 0), -1) # Cyan
                    if show_confidence and 'head' in mouse_pose.confidences:
                        h_c = mouse_pose.confidences['head'][data_idx]
                        cv2.putText(frame, f"{h_c:.2f}", (int(hx) + 5, int(hy) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                if center_valid:
                    cv2.circle(frame, (int(cx), int(cy)), circle_radius, (0, 165, 255), -1) # Orange
                if rear_valid:
                    cv2.circle(frame, (int(rx), int(ry)), circle_radius, (0, 255, 255), -1) # Yellow

                # Cricket Landmark
                ckx, cky = mouse_pose.cricket[data_idx]
                ck_c = mouse_pose.confidences['cricket'][data_idx] if ('cricket' in mouse_pose.confidences and data_idx < len(mouse_pose.confidences['cricket'])) else 0.0

                # Only draw if coordinates are valid and confidence is greater than 0
                if not (np.isnan(ckx) or np.isnan(cky)) and ck_c > 0.0:
                    text_suffix = ""
                    c_col = (0, 0, 255) # Red for regular cricket

                    if 'is_immo' in mouse_pose.confidences and mouse_pose.confidences['is_immo'] is not None:
                        if data_idx < len(mouse_pose.confidences['is_immo']) and mouse_pose.confidences['is_immo'][data_idx]:
                            c_col = (0, 215, 255) # Amber / Gold for immobilized cricket
                            text_suffix = " (immo)"
                    elif 'is_blurry' in mouse_pose.confidences and mouse_pose.confidences['is_blurry'] is not None:
                        if data_idx < len(mouse_pose.confidences['is_blurry']) and mouse_pose.confidences['is_blurry'][data_idx]:
                            c_col = (255, 0, 255) # Magenta for blurry cricket
                            text_suffix = " (blurry)"

                    if current_active_poop_mask is not None:
                        ix_c, iy_c = int(round(ckx)), int(round(cky))
                        if 0 <= ix_c < orig_w and 0 <= iy_c < orig_h and current_active_poop_mask[iy_c, ix_c]:
                            text_suffix += " [masked]"
                    
                    cv2.circle(frame, (int(ckx), int(cky)), circle_radius + 1, c_col, -1)
                    if show_confidence:
                        cv2.putText(frame, f"{ck_c:.2f}{text_suffix}", (int(ckx) + 5, int(cky) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c_col, 1)

            elif dlc_data is not None:
                # Rat DLC rendering
                colors = [
                    (255, 0, 0),   # rat_head: Blue
                    (0, 165, 255), # rat_mid: Orange
                    (0, 255, 255), # rat_butt: Yellow
                    (0, 255, 0),   # rat_tail: Green
                    (0, 0, 255)    # cricket: Red
                ]
                for p_idx, part in enumerate(dlc_data):
                    if data_idx < len(part):
                        x, y = part[data_idx]
                        if not np.isnan(x) and not np.isnan(y):
                            draw_color = colors[p_idx]
                            text_suffix = ""
                            if p_idx == 4 and dlc_confs is not None and len(dlc_confs) > 5 and dlc_confs[5][data_idx]:
                                draw_color = (255, 0, 255)
                                text_suffix = " (blurry)"
                            if p_idx == 4 and current_active_poop_mask is not None:
                                ix_c, iy_c = int(round(x)), int(round(y))
                                if 0 <= ix_c < orig_w and 0 <= iy_c < orig_h and current_active_poop_mask[iy_c, ix_c]:
                                    text_suffix += " [masked]"
                            cv2.circle(frame, (int(x), int(y)), circle_radius, draw_color, -1)
                            if show_confidence and dlc_confs is not None and data_idx < len(dlc_confs[p_idx]):
                                conf_val = dlc_confs[p_idx][data_idx]
                                text = f"{conf_val:.2f}{text_suffix}"
                                cv2.putText(frame, text, (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, draw_color, 1)
                                
            # 2.6 Draw Poops if requested
            if plot_poops and poop_data is not None:
                poop_color = (0, 75, 150) # Brown in BGR
                for pdic in poop_data:
                    if data_idx < len(pdic['conf']):
                        conf = pdic['conf'][data_idx]
                        if float(conf) > poop_thresh:
                            x, y = pdic['x'][data_idx], pdic['y'][data_idx]
                            if not np.isnan(x) and not np.isnan(y):
                                cv2.circle(frame, (int(x), int(y)), circle_radius, poop_color, -1)
                                if show_confidence:
                                    text = f"{pdic['name']}:{conf:.2f}"
                                    cv2.putText(frame, text, (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, poop_color, 1)
            
            # 3. Add Text Label (Pill box positioned neatly below plot)
            state_text = class_names[curr_pred_cls].upper()
            state_color = class_colors_cv2[curr_pred_cls]
            
            text_scale = 0.45
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(state_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)
            
            text_x = max(10, orig_w - text_w - 20)
            text_y = plot_h + text_h + 10 # 10px buffer below plot
            
            padding = 5
            cv2.rectangle(frame, (text_x - padding, text_y - text_h - padding), 
                          (text_x + text_w + padding, text_y + padding), (255, 255, 255), -1)
            cv2.rectangle(frame, (text_x - padding, text_y - text_h - padding), 
                          (text_x + text_w + padding, text_y + padding), state_color, 2)
            
            cv2.putText(frame, state_text, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, state_color, thickness, cv2.LINE_AA)
            
            out.write(frame)
            current_frame_idx += 1
            pbar.update(1)
                
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        pbar.close()
        cap.release()
        out.release()
        print(f"\nVideo generation complete: {output_path}")
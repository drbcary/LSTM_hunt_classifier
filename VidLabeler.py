import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os

# ==========================================
# Configuration
# ==========================================
CLASS_NAMES = ['Background', 'Stalking', 'Pursuit', 'Eating']
CLASS_COLORS_PLT = ['lightgray', 'orange', 'red', 'green']
# BGR Colors for OpenCV text
CLASS_COLORS_CV2 = [
    (211, 211, 211), # LightGray
    (0, 165, 255),   # Orange
    (0, 0, 255),     # Red
    (0, 128, 0)      # Green
]

def render_plot_frame_transparent(probs, current_frame_idx, trailing_frames, width_px, height_px):
    """
    Renders a transparent overlay plot using pure OpenCV for extreme performance.
    """
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
    
    # Current frame vertical dashed line (simulated with a solid translucent line)
    curr_x = int(pad_left + ((current_frame_idx - start) / (total_frames - 1)) * plot_w)
    cv2.line(img, (curr_x, pad_top), (curr_x, pad_top + plot_h), (255, 255, 255, 200), 2, cv2.LINE_AA)
    
    # Draw probability lines
    for i, cls_name in enumerate(CLASS_NAMES):
        color_bgr = CLASS_COLORS_CV2[i]
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
    leg_y = pad_top + 15
    for i, cls_name in enumerate(CLASS_NAMES):
        color_bgr = CLASS_COLORS_CV2[i]
        color_rgba = (color_bgr[2], color_bgr[1], color_bgr[0], 230)
        cv2.line(img, (leg_x, leg_y - 4), (leg_x + 15, leg_y - 4), color_rgba, 2, cv2.LINE_AA)
        cv2.putText(img, cls_name, (leg_x + 20, leg_y), font, 0.3, font_color, 1, cv2.LINE_AA)
        leg_y += 12

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

def create_hunting_video_clip(video_path, probs, preds, start_frame, end_frame, output_path,
                              trailing_frames=150, dlc_path=None, circle_radius=4, dlc_thresh=0.7,
                              show_confidence=True, include_blurry_cricket=False, filter_poops=True,
                              poop_dist_thresh=20, plot_poops=False):

    print(f"Generating video clip: {output_path}")
    print(f"Frame Range: {start_frame} to {end_frame}")
    
    dlc_data = None
    dlc_confs = None
    if dlc_path is not None:
        import dataset_utils
        res = dataset_utils.extract_dlc_tracking_data(dlc_path, rat_thresh=dlc_thresh, cricket_thresh=dlc_thresh,
                                                      interp_limit=0, include_blurry_cricket=include_blurry_cricket,
                                                      filter_poops=filter_poops, poop_dist_thresh=poop_dist_thresh)
        if res is not None:
            dlc_data = res
            
        poop_data = None
        if show_confidence or plot_poops:
            import pandas as pd
            try:
                df = pd.read_csv(dlc_path, header=[0, 1, 2], index_col=0)
                scorer = df.columns.get_level_values(0)[0]
                
                if show_confidence:
                    # If head was missing, dataset_utils falls back to nose. We grab both and take the max for the text display.
                    head_conf = np.maximum(df[scorer]['head']['likelihood'].values, df[scorer]['nose']['likelihood'].values)
                    
                    if include_blurry_cricket:
                        c_vals = df[scorer]['cricket']['likelihood'].values.copy()
                        bc_vals = df[scorer]['blurry_cricket']['likelihood'].values.copy()
                        
                        if filter_poops:
                            # Use poop_thresh=0.75 to match the extraction logic
                            c_vals = dataset_utils.apply_poop_penalty(df, scorer, df[scorer]['cricket']['x'], df[scorer]['cricket']['y'], c_vals, poop_thresh=0.75, poop_dist_thresh=poop_dist_thresh, cricket_thresh=dlc_thresh)
                            bc_vals = dataset_utils.apply_poop_penalty(df, scorer, df[scorer]['blurry_cricket']['x'], df[scorer]['blurry_cricket']['y'], bc_vals, poop_thresh=0.75, poop_dist_thresh=poop_dist_thresh, cricket_thresh=dlc_thresh)
                            
                        cricket_conf = np.maximum(c_vals, bc_vals)
                        is_blurry = bc_vals > c_vals
                    else:
                        cricket_conf = df[scorer]['cricket']['likelihood'].values.copy()
                        
                        if filter_poops:
                            cricket_conf = dataset_utils.apply_poop_penalty(df, scorer, df[scorer]['cricket']['x'], df[scorer]['cricket']['y'], cricket_conf, poop_thresh=0.75, poop_dist_thresh=poop_dist_thresh, cricket_thresh=dlc_thresh)
                            
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
                    bodyparts = df[scorer].columns.get_level_values(0).unique()
                    poop_parts = [bp for bp in bodyparts if isinstance(bp, str) and bp.startswith('poop')]
                    if poop_parts:
                        poop_data = []
                        for p_part in poop_parts:
                            poop_data.append({
                                'name': p_part,
                                'x': df[scorer][p_part]['x'].values,
                                'y': df[scorer][p_part]['y'].values,
                                'conf': df[scorer][p_part]['likelihood'].values
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
    
    # Height Calculation
    # Original logic was 1/3. You requested a 20% decrease.
    # 1/3 = 0.33. Decreasing by 20% -> 0.33 * 0.8 = 0.26
    plot_h = int(orig_h * 0.26) 
    plot_w = orig_w
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, orig_fps, (orig_w, orig_h))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    max_data_idx = len(preds) - 1
    current_frame_idx = start_frame
    
    try:
        while current_frame_idx < end_frame and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            data_idx = min(current_frame_idx, max_data_idx)
            curr_pred_cls = preds[data_idx]
            
            # 1. Generate Transparent Plot
            plot_img_rgba = render_plot_frame_transparent(probs, data_idx, trailing_frames, plot_w, plot_h)
            
            # 2. Overlay Plot on Frame (Top area)
            # Darken the background behind the plot slightly for readability
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (orig_w, plot_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame) 
            
            frame = overlay_transparent(frame, plot_img_rgba, 0, 0)
            
            # 2.5 Draw DLC data if provided
            if dlc_data is not None:
                # Colors in BGR for OpenCV
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
                            
                            # If it's the cricket (p_idx == 4) and blurry was used, change color and label
                            if p_idx == 4 and dlc_confs is not None and len(dlc_confs) > 5 and dlc_confs[5][data_idx]:
                                draw_color = (255, 0, 255) # Magenta for blurry cricket
                                text_suffix = " (blurry)"
                                
                            cv2.circle(frame, (int(x), int(y)), circle_radius, draw_color, -1)
                            
                            # Draw confidence text if enabled
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
                        if conf > 0.5:
                            x, y = pdic['x'][data_idx], pdic['y'][data_idx]
                            if not np.isnan(x) and not np.isnan(y):
                                cv2.circle(frame, (int(x), int(y)), circle_radius, poop_color, -1)
                                if show_confidence:
                                    text = f"{pdic['name']}:{conf:.2f}"
                                    cv2.putText(frame, text, (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, poop_color, 1)
            
            # 3. Add Text Label (Just below the plot)
            state_text = CLASS_NAMES[curr_pred_cls].upper()
            state_color = CLASS_COLORS_CV2[curr_pred_cls]
            
            # SCALED DOWN TEXT: 1.5 -> 0.4 (approx 75% reduction)
            text_scale = 0.4
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(state_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)
            
            # Position text below plot
            text_x = orig_w - 100
            text_y = plot_h + text_h + 10 # 10px buffer
            
            # Text Background Box (Smaller now)
            padding = 4
            cv2.rectangle(frame, (text_x - padding, text_y - text_h - padding), 
                          (text_x + text_w + padding, text_y + padding), (255, 255, 255), -1)
            cv2.rectangle(frame, (text_x - padding, text_y - text_h - padding), 
                          (text_x + text_w + padding, text_y + padding), state_color, 1)
            
            cv2.putText(frame, state_text, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, state_color, thickness)
            
            out.write(frame)
            current_frame_idx += 1
            
            if (current_frame_idx - start_frame) % 50 == 0:
                print(f"  Rendered {current_frame_idx - start_frame}/{end_frame - start_frame} frames...", end='\r')
                
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        cap.release()
        out.release()
        print("\nVideo generation complete.")
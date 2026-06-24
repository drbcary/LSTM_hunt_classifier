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
    Renders a transparent overlay plot with FIXED axes to prevent jitter.
    """
    # Define window
    start = max(0, current_frame_idx - trailing_frames)
    end = current_frame_idx + 10
    
    # Slice data
    data_slice = probs[start:end]
    x_axis = np.arange(start, end)[:len(data_slice)]

    # Setup Figure
    dpi = 100
    fig_w = width_px / dpi
    fig_h = height_px / dpi
    
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_alpha(0.0) # Transparent background
    
    # --- KEY FIX FOR JITTER ---
    # Instead of tight_layout(), we manually place the axes.
    # [left, bottom, width, height] in fractions of figure size.
    # This ensures the plot area (and thus the dotted line) never moves pixel-wise.
    ax = fig.add_axes([0.08, 0.25, 0.90, 0.65]) 
    ax.set_facecolor("none") 
    
    # Plot lines
    for i, cls_name in enumerate(CLASS_NAMES):
        ax.plot(x_axis, data_slice[:, i], 
                label=cls_name, color=CLASS_COLORS_PLT[i], linewidth=2.0, alpha=0.9)
    
    # Vertical Line
    ax.axvline(x=current_frame_idx, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
    
    # Formatting
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(max(0, current_frame_idx - trailing_frames), current_frame_idx + 10)
    
    # Axis Styling (White for visibility on video)
    ax.tick_params(axis='x', colors='white', labelsize=8)
    ax.tick_params(axis='y', colors='white', labelsize=8)
    
    # Labels
    ax.set_xlabel("Frame Number", color='white', fontsize=9, labelpad=2)
    ax.set_ylabel("Probability", color='white', fontsize=9)
    
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
        spine.set_alpha(0.5)
        
    # Legend
    legend = ax.legend(loc='upper left', fontsize=7, framealpha=0.2)
    plt.setp(legend.get_texts(), color='white')
    
    ax.grid(True, alpha=0.2, color='white', linestyle=':')
    
    # Render
    canvas = FigureCanvas(fig)
    canvas.draw()
    
    w, h = canvas.get_width_height()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)
    
    if img.shape[0] != height_px or img.shape[1] != width_px:
        img = cv2.resize(img, (width_px, height_px))
        
    plt.close(fig)
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

def create_hunting_video_clip(video_path, probs, preds, start_frame, end_frame, output_path, trailing_frames=150):
    print(f"Generating video clip: {output_path}")
    print(f"Frame Range: {start_frame} to {end_frame}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

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
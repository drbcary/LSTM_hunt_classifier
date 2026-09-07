import sys
import os
import glob
import re
import cv2
import numpy as np
import pandas as pd
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox, 
                               QDoubleSpinBox)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QShortcut, QKeySequence

import dataset_utils
import RatHunt_Inference

class TimelineWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.labels = None
        self.probs = None
        self.total_frames = 1
        self.current_frame = 0
        
        self.selection_start = -1
        self.selection_end = -1
        
        self.zoom = 1.0
        self.offset_x = 0.0 # Pixels
        self.is_panning = False
        self.last_mouse_x = 0
        
        # 0=Background, 1=Stalking, 2=Pursuit, 3=Eating
        self.colors = {
            0: QColor(200, 200, 200), # Gray
            1: QColor(255, 165, 0),   # Orange
            2: QColor(255, 0, 0),     # Red
            3: QColor(0, 255, 0)      # Green
        }
        
    def set_data(self, labels, probs, current_frame):
        self.labels = labels
        self.probs = probs
        self.total_frames = len(labels) if labels is not None and len(labels) > 0 else 1
        self.current_frame = current_frame
        
        # Reset zoom
        self.zoom = 1.0
        self.offset_x = 0.0
        self.update()
        
    def set_current_frame(self, frame):
        self.current_frame = frame
        self.update()
        
    def get_x_for_frame(self, frame):
        w = self.rect().width() * self.zoom
        x = (frame / self.total_frames) * w + self.offset_x
        return x
        
    def get_frame_for_x(self, x):
        w = self.rect().width() * self.zoom
        f = ((x - self.offset_x) / w) * self.total_frames
        return int(f)
        
    def center_on_frame(self, frame):
        w = self.rect().width() * self.zoom
        mid = self.rect().width() / 2
        self.offset_x = mid - (frame / self.total_frames) * w
        
        min_offset = -(self.rect().width() * self.zoom - self.rect().width())
        self.offset_x = max(min_offset, min(0.0, self.offset_x))
        self.update()
        
    def ensure_cursor_visible(self, frame):
        x = self.get_x_for_frame(frame)
        mid = self.rect().width() / 2
        if x > mid or x < 0:
            self.center_on_frame(frame)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()
        painter.fillRect(rect, QColor(240, 240, 240))
        
        if self.labels is None: return
            
        w = rect.width() * self.zoom
        h = rect.height()
        
        # Split into upper half (probs) and lower half (labels)
        prob_h = int(h * 0.6)
        lbl_h = h - prob_h
        
        # Draw labels
        if len(self.labels) > 0:
            changes = np.where(self.labels[:-1] != self.labels[1:])[0] + 1
            starts = np.concatenate(([0], changes))
            ends = np.concatenate((changes, [len(self.labels)]))
            
            for s, e in zip(starts, ends):
                val = self.labels[s]
                x1 = int((s / self.total_frames) * w + self.offset_x)
                x2 = int((e / self.total_frames) * w + self.offset_x)
                
                # Only draw if visible
                if x2 > 0 and x1 < rect.width():
                    painter.fillRect(x1, prob_h, x2 - x1, lbl_h, self.colors.get(val, QColor(0,0,0)))
        
        # Draw probs
        if self.probs is not None and len(self.probs) == self.total_frames:
            # Downsample drawing if heavily zoomed out to save CPU
            step = max(1, int(self.total_frames / w))
            
            for c in range(4):
                painter.setPen(QPen(self.colors[c], 3))
                pts = []
                for f in range(0, self.total_frames, step):
                    x = int((f / self.total_frames) * w + self.offset_x)
                    if 0 <= x <= rect.width() + 10:
                        y = prob_h - int(self.probs[f, c] * prob_h)
                        pts.append((x, y))
                
                # Draw lines between pts
                for i in range(len(pts)-1):
                    painter.drawLine(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])
                    
        # Draw Selection
        if self.selection_start != -1 and self.selection_end != -1:
            s = min(self.selection_start, self.selection_end)
            e = max(self.selection_start, self.selection_end)
            x1 = int((s / self.total_frames) * w + self.offset_x)
            x2 = int((e / self.total_frames) * w + self.offset_x)
            painter.fillRect(x1, 0, x2 - x1, h, QColor(0, 0, 255, 80)) 
            
        # Draw Playhead
        x = int(self.get_x_for_frame(self.current_frame))
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        painter.drawLine(x, 0, x, h)
        
    def wheelEvent(self, event):
        zoom_factor = 1.2 if event.angleDelta().y() > 0 else 1.0 / 1.2
        mx = event.position().x()
        frame_at_mouse = self.get_frame_for_x(mx)
        
        self.zoom = max(1.0, min(100.0, self.zoom * zoom_factor))
        
        new_x_for_frame = (frame_at_mouse / self.total_frames) * (self.rect().width() * self.zoom)
        self.offset_x = mx - new_x_for_frame
        
        min_offset = -(self.rect().width() * self.zoom - self.rect().width())
        self.offset_x = max(min_offset, min(0.0, self.offset_x))
        self.update()
        
    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton or event.button() == Qt.MiddleButton:
            self.is_panning = True
            self.last_mouse_x = event.position().x()
        elif event.button() == Qt.LeftButton:
            if self.labels is not None:
                frame = max(0, min(self.total_frames - 1, self.get_frame_for_x(event.position().x())))
                self.selection_start = frame
                self.selection_end = frame
                self.window().set_frame(frame)
                self.update()
            
    def mouseMoveEvent(self, event):
        if self.is_panning:
            dx = event.position().x() - self.last_mouse_x
            self.offset_x += dx
            min_offset = -(self.rect().width() * self.zoom - self.rect().width())
            self.offset_x = max(min_offset, min(0.0, self.offset_x))
            self.last_mouse_x = event.position().x()
            self.update()
        elif self.selection_start != -1 and event.buttons() & Qt.LeftButton:
            if self.labels is not None:
                frame = max(0, min(self.total_frames - 1, self.get_frame_for_x(event.position().x())))
                self.selection_end = frame
                self.window().set_frame(self.selection_end)
                self.update()
                
    def mouseReleaseEvent(self, event):
        self.is_panning = False


class ReviewGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rat Hunt Label Review")
        self.resize(1200, 800)
        
        self.video_path = None
        self.label_path = None
        self.dlc_path = None
        
        self.cap = None
        self.total_frames = 0
        self.current_frame = 0
        self.video_fps = 30
        
        self.labels = None
        self.probs = None
        self.dlc_data = None
        self.dlc_confs = None
        
        self.active_paint_label = None
        self.last_painted_frame = -1
        
        self.is_playing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        
        self.init_ui()
        QApplication.instance().installEventFilter(self)
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Top Bar
        top_layout = QHBoxLayout()
        self.btn_load_vid = QPushButton("Load Video")
        self.btn_load_vid.clicked.connect(self.load_video)
        # Prevent buttons from stealing focus from shortcuts
        self.btn_load_vid.setFocusPolicy(Qt.NoFocus)
        
        self.btn_load_label = QPushButton("Load Label Override")
        self.btn_load_label.clicked.connect(self.load_label)
        self.btn_load_label.setFocusPolicy(Qt.NoFocus)
        
        self.lbl_info = QLabel("No video loaded.")
        
        top_layout.addWidget(self.btn_load_vid)
        top_layout.addWidget(self.btn_load_label)
        top_layout.addWidget(self.lbl_info)
        top_layout.addStretch()
        layout.addLayout(top_layout)
        
        # Video Display
        self.video_lbl = QLabel()
        self.video_lbl.setAlignment(Qt.AlignCenter)
        self.video_lbl.setStyleSheet("background-color: black;")
        self.video_lbl.setMinimumSize(640, 480)
        layout.addWidget(self.video_lbl, stretch=1)
        
        # Controls
        controls_layout = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setFocusPolicy(Qt.NoFocus)
        
        self.lbl_frame = QLabel("Frame: 0 / 0")
        
        self.lbl_skip = QLabel("Skip (sec):")
        self.spin_skip = QDoubleSpinBox()
        self.spin_skip.setRange(0.1, 60.0)
        self.spin_skip.setValue(1.0)
        self.spin_skip.setSingleStep(0.5)
        self.spin_skip.setFocusPolicy(Qt.NoFocus) # No focus means no keyboard traps!
        
        self.btn_save = QPushButton("Save Refined Labels")
        self.btn_save.clicked.connect(self.save_labels)
        self.btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")
        self.btn_save.setFocusPolicy(Qt.NoFocus)
        
        controls_layout.addWidget(self.btn_play)
        controls_layout.addWidget(self.lbl_frame)
        controls_layout.addWidget(self.lbl_skip)
        controls_layout.addWidget(self.spin_skip)
        controls_layout.addStretch()
        controls_layout.addWidget(self.btn_save)
        layout.addLayout(controls_layout)
        
        # Timeline
        self.timeline = TimelineWidget(self)
        layout.addWidget(self.timeline)
        
        # Instructions
        inst = QLabel("<b>Controls:</b> Space=Play/Pause, Left/Right Arrow=Skip by seconds (Hold to repeat). <br><b>Timeline:</b> Scroll wheel to zoom, Middle/Right-click drag to pan. <br><b>Live Editing:</b> Tap <b>0</b>=Bg, <b>1</b>=Stalk, <b>2</b>=Pursuit, <b>3</b>=Eat to toggle continuous painting. Tap again to revert to No Change.")
        inst.setStyleSheet("font-size: 14px;")
        layout.addWidget(inst)
        
    def eventFilter(self, obj, event):
        if event.type() == event.Type.KeyPress:
            key = event.key()
            
            # Allow AutoRepeat for Left/Right
            if key == Qt.Key_Right:
                self.skip(1)
                return True
            elif key == Qt.Key_Left:
                self.skip(-1)
                return True
                
            if not event.isAutoRepeat():
                if key == Qt.Key_Space:
                    self.toggle_play()
                    return True
                elif key in [Qt.Key_0, Qt.Key_QuoteLeft]:
                    self.toggle_painting(0)
                    return True
                elif key == Qt.Key_1:
                    self.toggle_painting(1)
                    return True
                elif key == Qt.Key_2:
                    self.toggle_painting(2)
                    return True
                elif key in [Qt.Key_3, Qt.Key_4]:
                    self.toggle_painting(3)
                    return True
        return super().eventFilter(obj, event)
        
    def toggle_painting(self, val):
        if self.active_paint_label == val:
            self.active_paint_label = None # Re-clicking same label stops painting entirely (No Change)
        else:
            self.active_paint_label = val
            
        self.last_painted_frame = self.current_frame
        if self.active_paint_label is not None and self.labels is not None:
            self.labels[self.current_frame] = self.active_paint_label
            self.timeline.update()
            
        if not self.is_playing:
            self.set_frame(self.current_frame)
        
    def skip(self, direction):
        if not self.cap: return
        skip_frames = int(self.spin_skip.value() * self.video_fps)
        if skip_frames < 1: skip_frames = 1
        self.set_frame(self.current_frame + direction * skip_frames)
        
    def apply_label(self, val):
        if self.labels is not None and self.timeline.selection_start != -1:
            s = min(self.timeline.selection_start, self.timeline.selection_end)
            e = max(self.timeline.selection_start, self.timeline.selection_end)
            self.labels[s:e+1] = val
            
            self.timeline.selection_start = -1
            self.timeline.selection_end = -1
            self.timeline.update()
            self.set_frame(self.current_frame)
        
    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.avi *.mp4 *.mov)")
        if not path: return
        
        self.video_path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open video.")
            return
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        folder = os.path.dirname(path)
        base = os.path.basename(path)
        clean_base = re.sub(r'DLC_resnet50.*\.csv$', '', base)
        if clean_base == base:
            clean_base = base[:base.rfind('.')] if '.' in base else base
        
        # Find DLC
        dlc_files = glob.glob(os.path.join(folder, f"{clean_base}DLC*.csv"))
        if dlc_files:
            self.dlc_path = dlc_files[0]
            print(f"Auto-found DLC: {os.path.basename(self.dlc_path)}")
            res = dataset_utils.extract_dlc_tracking_data(self.dlc_path, filter_poops=True)
            if res:
                self.dlc_data = res
                
            # Grab confidences for drawing
            try:
                df = pd.read_csv(self.dlc_path, header=[0, 1, 2], index_col=0)
                scorer = df.columns.get_level_values(0)[0]
                self.dlc_confs = {
                    0: df[scorer]['head']['likelihood'].values if 'head' in df[scorer] else None,
                    1: df[scorer]['mid_body']['likelihood'].values if 'mid_body' in df[scorer] else None,
                    2: df[scorer]['butt']['likelihood'].values if 'butt' in df[scorer] else None,
                    3: df[scorer]['endtail']['likelihood'].values if 'endtail' in df[scorer] else None,
                    4: df[scorer]['cricket']['likelihood'].values if 'cricket' in df[scorer] else None
                }
            except Exception as e:
                print(f"Failed to load confidences: {e}")
                self.dlc_confs = None
        else:
            self.dlc_path = None
            self.dlc_data = None
            self.dlc_confs = None
            print("No DLC found.")
            
        # Find Probs
        self.probs = None
        prob_paths = glob.glob(os.path.join(folder, f"{clean_base}_probs.csv"))
        if not prob_paths:
            prob_paths = glob.glob(os.path.join(folder, "probs", f"{clean_base}_probs.csv"))
            
        if prob_paths:
            try:
                prob_df = pd.read_csv(prob_paths[0])
                self.probs = prob_df[['Background', 'Stalking', 'Pursuit', 'Eating']].values
                print(f"Loaded probabilities from {os.path.basename(prob_paths[0])}")
            except Exception as e:
                print(f"Failed to load probs: {e}")
            
        # Find Labels
        label_paths = glob.glob(os.path.join(folder, f"{clean_base}_analyzed.xlsx"))
        if not label_paths:
            label_paths = glob.glob(os.path.join(folder, "excel_files", f"{clean_base}_analyzed.xlsx"))
            
        if label_paths:
            self.load_label_file(label_paths[0])
        else:
            self.labels = np.zeros(self.total_frames, dtype=int)
            self.timeline.set_data(self.labels, self.probs, 0)
            self.lbl_info.setText(f"Loaded {base} (No labels found)")
            
        self.set_frame(0)
        
    def load_label(self):
        if not self.cap: return
        path, _ = QFileDialog.getOpenFileName(self, "Open Label File", "", "Excel Files (*.xlsx)")
        if path:
            self.load_label_file(path)
            
    def load_label_file(self, path):
        self.label_path = path
        try:
            df = pd.read_excel(path)
            events = df.to_dict('records')
            
            mapped_events = []
            for ev in events:
                mapped_events.append({
                    'enter': ev.get('Enter_Frame', -1),
                    'pursuit': ev.get('Enter_Frame', -1) + ev.get('Latency_to_Attack_Frames', 0) if ev.get('Latency_to_Attack_Frames', 0) > 0 else -1,
                    'capture': ev.get('Capture_Frame', -1),
                    'capture_end': ev.get('Consumption_End_Frame', ev.get('Capture_Frame', 0) + dataset_utils.EATING_BUFFER)
                })
            
            seq = np.zeros(self.total_frames, dtype=int)
            for event in mapped_events:
                enter = int(event['enter'])
                pursuit = int(event['pursuit'])
                capture = int(event['capture'])
                capture_end = int(event['capture_end'])
                
                enter = max(0, min(self.total_frames, enter))
                pursuit = max(0, min(self.total_frames, pursuit))
                capture = max(0, min(self.total_frames, capture))
                eating_end = min(self.total_frames, capture_end)
                
                if pursuit > enter: seq[enter:pursuit] = 1 
                if capture > pursuit: seq[pursuit:capture] = 2 
                if eating_end > capture: seq[capture:eating_end] = 3 
                
            self.labels = seq
            self.timeline.set_data(self.labels, self.probs, self.current_frame)
            self.lbl_info.setText(f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load label file:\n{e}")
            
    def set_frame(self, frame):
        if not self.cap: return
        new_frame = max(0, min(self.total_frames - 1, frame))
        
        # If we are painting, fill the gap between last_painted_frame and new_frame
        if self.active_paint_label is not None and self.labels is not None and self.last_painted_frame != -1:
            s = min(self.last_painted_frame, new_frame)
            e = max(self.last_painted_frame, new_frame)
            if not np.all(self.labels[s:e+1] == self.active_paint_label):
                self.labels[s:e+1] = self.active_paint_label
                self.timeline.update()
            self.last_painted_frame = new_frame
            
        self.current_frame = new_frame
        
        # 100% Precise Seeking Logic
        # OpenCV cap.set can sometimes seek to the nearest keyframe rather than the exact frame.
        # It is also slow. If we are already exactly at the target frame (e.g. during normal playback), we skip seeking entirely.
        internal_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if internal_pos != self.current_frame:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            
            # Verify the seek landed precisely on the target frame.
            actual_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            if actual_pos < self.current_frame:
                # If OpenCV snapped to an earlier keyframe, manually read forward to guarantee 100% precision.
                diff = self.current_frame - actual_pos
                if diff < 1000: # Safe limit to prevent hanging
                    for _ in range(diff):
                        self.cap.read()
                        
        ret, img = self.cap.read()
        if ret:
            if self.dlc_data is not None:
                colors = [(255, 0, 0), (0, 165, 255), (0, 255, 255), (0, 255, 0), (0, 0, 255)]
                for p_idx, part in enumerate(self.dlc_data):
                    if self.current_frame < len(part):
                        x, y = part[self.current_frame]
                        if not np.isnan(x) and not np.isnan(y):
                            cv2.circle(img, (int(x), int(y)), 4, colors[p_idx], -1)
                            # Draw confidence if available
                            if self.dlc_confs and self.dlc_confs.get(p_idx) is not None:
                                conf = self.dlc_confs[p_idx][self.current_frame]
                                if not np.isnan(conf):
                                    cv2.putText(img, f"{conf:.2f}", (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[p_idx], 1)
            
            if self.labels is not None:
                lbl_val = self.labels[self.current_frame]
                cls_names = {0: 'Background', 1: 'Stalking', 2: 'Pursuit', 3: 'Eating'}
                colors_bgr = {0: (200,200,200), 1: (0,165,255), 2: (0,0,255), 3: (0,255,0)}
                cv2.putText(img, f"Class: {cls_names.get(lbl_val, 'Unknown')}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, colors_bgr.get(lbl_val, (255,255,255)), 3)
                
                # Draw Brush state
                brush_names = {None: 'No Change', 0: 'Background', 1: 'Stalking', 2: 'Pursuit', 3: 'Eating'}
                brush_colors = {None: (255, 255, 0), 0: (200,200,200), 1: (0,165,255), 2: (0,0,255), 3: (0,255,0)}
                cv2.putText(img, f"Brush: {brush_names.get(self.active_paint_label, 'No Change')}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, brush_colors.get(self.active_paint_label, (255,255,0)), 3)
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(qimg)
            self.video_lbl.setPixmap(pixmap.scaled(self.video_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
        self.lbl_frame.setText(f"Frame: {self.current_frame} / {self.total_frames}")
        self.timeline.set_current_frame(self.current_frame)
        
    def next_frame(self):
        self.set_frame(self.current_frame + 1)
        self.timeline.ensure_cursor_visible(self.current_frame)
        if self.current_frame >= self.total_frames - 1:
            self.toggle_play()
            
    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.setText("Pause")
            self.timeline.center_on_frame(self.current_frame)
            self.timer.start(int(1000 / self.video_fps) if self.video_fps > 0 else 33)
        else:
            self.btn_play.setText("Play")
            self.timer.stop()
            
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.set_frame(self.current_frame) 
                
    def sequence_to_events(self):
        events = []
        if self.labels is None: return events
        
        lbls = self.labels.copy()
        
        changes = np.where(lbls[:-1] != lbls[1:])[0] + 1
        starts = np.concatenate(([0], changes))
        ends = np.concatenate((changes, [len(lbls)]))
        
        in_hunt = False
        current_event = None
        
        for s, e in zip(starts, ends):
            val = lbls[s]
            if val != 0 and not in_hunt:
                in_hunt = True
                current_event = {'enter': s, 'pursuit': -1, 'capture': -1, 'capture_end': -1}
                
            if in_hunt:
                if val == 1 and current_event['enter'] == -1:
                    current_event['enter'] = s
                elif val == 2 and current_event['pursuit'] == -1:
                    current_event['pursuit'] = s
                elif val == 3 and current_event['capture'] == -1:
                    current_event['capture'] = s
                    
                if val == 0:
                    current_event['capture_end'] = s
                    events.append(current_event)
                    in_hunt = False
                    current_event = None
                    
        if in_hunt and current_event:
            current_event['capture_end'] = len(lbls)
            events.append(current_event)
            
        return events

    def save_labels(self):
        if self.labels is None or not self.video_path: return
        
        events = self.sequence_to_events()
        
        base = os.path.basename(self.video_path)
        clean_base = base[:base.rfind('.')] if '.' in base else base
        
        out_name = f"{clean_base}_refined.xlsx"
        default_path = os.path.join(os.path.dirname(self.video_path), out_name)
        
        path, _ = QFileDialog.getSaveFileName(self, "Save Refined Labels", default_path, "Excel Files (*.xlsx)")
        if path:
            try:
                RatHunt_Inference.generate_analyzed_excel(events, path, self.video_fps)
                QMessageBox.information(self, "Success", f"Saved to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ReviewGUI()
    window.show()
    sys.exit(app.exec())

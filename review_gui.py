import sys
import os
import glob
import re
import cv2
import numpy as np
import pandas as pd
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox, 
                               QDoubleSpinBox, QScrollBar)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QShortcut, QKeySequence

import dataset_utils
import RatHunt_Inference

class TimelineWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self.labels = None
        self.gt_labels = None
        self.probs = None
        self.total_frames = 1
        self.current_frame = 0
        
        self.selection_start = -1
        self.selection_end = -1
        self.press_x = 0
        
        self.zoom = 1.0
        self.offset_x = 0.0 # Pixels
        self.is_panning = False
        self.last_mouse_x = 0
        
        # 0=Background, 1=Cricket Present, 2=Pursuit, 3=Eating
        self.colors = {
            0: QColor(200, 200, 200), # Gray
            1: QColor(255, 165, 0),   # Orange
            2: QColor(255, 0, 0),     # Red
            3: QColor(0, 255, 0)      # Green
        }
        
    def set_data(self, labels, probs, current_frame, gt_labels=None):
        self.labels = labels
        self.gt_labels = gt_labels
        self.probs = probs
        self.total_frames = len(labels) if labels is not None and len(labels) > 0 else 1
        self.current_frame = current_frame
        
        # Reset zoom
        self.zoom = 1.0
        self.offset_x = 0.0
        self.update()
        self.update_scrollbar()
        
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
        
    def has_selection(self):
        return (self.selection_start != -1 and 
                self.selection_end != -1 and 
                self.selection_start != self.selection_end)
                
    def clear_selection(self):
        self.selection_start = -1
        self.selection_end = -1
        self.update()
        
    def center_on_frame(self, frame):
        w = self.rect().width() * self.zoom
        mid = self.rect().width() / 2
        self.offset_x = mid - (frame / self.total_frames) * w
        
        min_offset = -(self.rect().width() * self.zoom - self.rect().width())
        self.offset_x = max(min_offset, min(0.0, self.offset_x))
        self.update()
        self.update_scrollbar()
        
    def ensure_cursor_visible(self, frame):
        x = self.get_x_for_frame(frame)
        mid = self.rect().width() / 2
        if x > mid or x < 0:
            self.center_on_frame(frame)
            
    def zoom_by(self, factor, center_x=None):
        if center_x is None:
            center_x = self.rect().width() / 2
        frame_at_center = self.get_frame_for_x(center_x)
        self.zoom = max(1.0, min(100.0, self.zoom * factor))
        
        w = self.rect().width() * self.zoom
        new_x = (frame_at_center / self.total_frames) * w
        self.offset_x = center_x - new_x
        
        min_offset = -(w - self.rect().width())
        self.offset_x = max(min_offset, min(0.0, self.offset_x))
        self.update()
        self.update_scrollbar()
        
    def scroll_to_frame(self, first_frame):
        w_vis = self.rect().width()
        if w_vis <= 0 or self.total_frames <= 1:
            return
        w_total = w_vis * self.zoom
        new_offset = -(first_frame / self.total_frames) * w_total
        min_offset = -(w_total - w_vis)
        self.offset_x = max(min_offset, min(0.0, new_offset))
        self.update()
        
    def update_scrollbar(self):
        win = self.window()
        if not win or not hasattr(win, 'scrollbar') or win.scrollbar is None:
            return
        if self.total_frames <= 1 or self.zoom <= 1.0001:
            win._syncing_scrollbar = True
            win.scrollbar.setRange(0, 0)
            win.scrollbar.setEnabled(False)
            win._syncing_scrollbar = False
            return
            
        w_vis = self.rect().width()
        if w_vis <= 0:
            return
            
        frames_visible = self.total_frames / self.zoom
        page_step = max(1, int(round(frames_visible)))
        max_scroll = max(0, self.total_frames - page_step)
        
        w_total = w_vis * self.zoom
        current_first_frame = int(round((-self.offset_x / w_total) * self.total_frames))
        current_first_frame = max(0, min(max_scroll, current_first_frame))
        
        win._syncing_scrollbar = True
        win.scrollbar.setEnabled(True)
        win.scrollbar.setRange(0, max_scroll)
        win.scrollbar.setPageStep(page_step)
        win.scrollbar.setSingleStep(max(1, int(page_step / 10)))
        win.scrollbar.setValue(current_first_frame)
        win._syncing_scrollbar = False
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        min_offset = -(self.rect().width() * self.zoom - self.rect().width())
        self.offset_x = max(min_offset, min(0.0, self.offset_x))
        self.update_scrollbar()

    def _draw_bar(self, painter, label_array, w, y_pos, bar_h):
        if label_array is None or len(label_array) == 0:
            return
        changes = np.where(label_array[:-1] != label_array[1:])[0] + 1
        starts = np.concatenate(([0], changes))
        ends = np.concatenate((changes, [len(label_array)]))
        
        rect_w = self.rect().width()
        for s, e in zip(starts, ends):
            val = label_array[s]
            x1 = int((s / self.total_frames) * w + self.offset_x)
            x2 = int((e / self.total_frames) * w + self.offset_x)
            
            if x2 > 0 and x1 < rect_w:
                painter.fillRect(x1, y_pos, max(1, x2 - x1), bar_h, self.colors.get(val, QColor(0, 0, 0)))

    def _draw_badge(self, painter, text, y_pos, h_pos):
        badge_w = 90
        badge_h = min(20, max(14, h_pos - 6))
        badge_y = y_pos + (h_pos - badge_h) // 2
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(15, 15, 15, 190))
        painter.drawRoundedRect(8, badge_y, badge_w, badge_h, 4, 4)
        painter.setPen(QColor(255, 255, 255))
        font = painter.font()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(8, badge_y, badge_w, badge_h, Qt.AlignCenter, text)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()
        painter.fillRect(rect, QColor(240, 240, 240))
        
        if self.labels is None: return
            
        w = rect.width() * self.zoom
        h = rect.height()
        
        # Determine track layout
        has_gt = (self.gt_labels is not None)
        if has_gt:
            prob_h = int(h * 0.48)
            pred_h = int((h - prob_h) * 0.50)
            gt_h = h - prob_h - pred_h
            pred_y = prob_h
            gt_y = prob_h + pred_h
        else:
            prob_h = int(h * 0.65)
            pred_h = h - prob_h
            pred_y = prob_h
            gt_h = 0
            gt_y = None
        
        # 1. Draw Predicted Bar
        self._draw_bar(painter, self.labels, w, pred_y, pred_h)
        
        # 2. Draw Manual Ground Truth Bar (if comparison active)
        if has_gt:
            self._draw_bar(painter, self.gt_labels, w, gt_y, gt_h)
            # Separator line between Predicted and Manual
            painter.setPen(QPen(QColor(80, 80, 80, 180), 1))
            painter.drawLine(0, gt_y, rect.width(), gt_y)
            
        # Draw Badges for clarity
        self._draw_badge(painter, "PREDICTED", pred_y, pred_h)
        if has_gt:
            self._draw_badge(painter, "MANUAL (GT)", gt_y, gt_h)
        
        # 3. Draw Probabilities (upper section)
        if self.probs is not None and len(self.probs) == self.total_frames:
            step = max(1, int(self.total_frames / w))
            for c in range(4):
                painter.setPen(QPen(self.colors[c], 2.5))
                pts = []
                for f in range(0, self.total_frames, step):
                    x = int((f / self.total_frames) * w + self.offset_x)
                    if 0 <= x <= rect.width() + 10:
                        y = prob_h - int(self.probs[f, c] * (prob_h - 4))
                        pts.append((x, y))
                for i in range(len(pts) - 1):
                    painter.drawLine(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])
                    
        # 4. Draw Selection
        if self.has_selection():
            s = min(self.selection_start, self.selection_end)
            e = max(self.selection_start, self.selection_end)
            x1 = int((s / self.total_frames) * w + self.offset_x)
            x2 = int(((e + 1) / self.total_frames) * w + self.offset_x)
            painter.fillRect(x1, 0, max(2, x2 - x1), h, QColor(0, 120, 255, 90))
            painter.setPen(QPen(QColor(0, 160, 255, 220), 1.5))
            painter.drawLine(x1, 0, x1, h)
            painter.drawLine(x2, 0, x2, h) 
            
        # 5. Draw Playhead
        x = int(self.get_x_for_frame(self.current_frame))
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        painter.drawLine(x, 0, x, h)
        
    def wheelEvent(self, event):
        zoom_factor = 1.2 if event.angleDelta().y() > 0 else 1.0 / 1.2
        self.zoom_by(zoom_factor, event.position().x())
        
    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton or event.button() == Qt.MiddleButton:
            self.is_panning = True
            self.last_mouse_x = event.position().x()
        elif event.button() == Qt.LeftButton:
            if self.labels is not None:
                self.press_x = event.position().x()
                frame = max(0, min(self.total_frames - 1, self.get_frame_for_x(self.press_x)))
                self.selection_start = frame
                self.selection_end = frame
                
                # Stop continuous painting brush if clicking to select/seek
                win = self.window()
                if hasattr(win, 'active_paint_label'):
                    win.active_paint_label = None
                    
                win.set_frame(frame)
                self.update()
            
    def mouseMoveEvent(self, event):
        if self.is_panning:
            dx = event.position().x() - self.last_mouse_x
            self.offset_x += dx
            min_offset = -(self.rect().width() * self.zoom - self.rect().width())
            self.offset_x = max(min_offset, min(0.0, self.offset_x))
            self.last_mouse_x = event.position().x()
            self.update()
            self.update_scrollbar()
        elif self.selection_start != -1 and event.buttons() & Qt.LeftButton:
            if self.labels is not None:
                frame = max(0, min(self.total_frames - 1, self.get_frame_for_x(event.position().x())))
                self.selection_end = frame
                self.window().set_frame(self.selection_end)
                self.update()
                
    def mouseReleaseEvent(self, event):
        if event.button() in [Qt.RightButton, Qt.MiddleButton]:
            self.is_panning = False
        elif event.button() == Qt.LeftButton:
            # If user simply clicked without dragging (< 4 pixels movement), clear selection
            if abs(event.position().x() - getattr(self, 'press_x', 0)) < 4:
                self.selection_start = -1
                self.selection_end = -1
                self.update()


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
        self.gt_labels = None
        self.gt_folder = None
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
        
        if '--compare' in sys.argv:
            QTimer.singleShot(200, self.prompt_gt_folder)
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Top Bar
        top_layout = QHBoxLayout()
        self.btn_load_vid = QPushButton("Load Video")
        self.btn_load_vid.clicked.connect(self.load_video)
        self.btn_load_vid.setFocusPolicy(Qt.NoFocus)
        
        self.btn_compare_mode = QPushButton("Comparison Mode: OFF")
        self.btn_compare_mode.clicked.connect(self.toggle_compare_mode)
        self.btn_compare_mode.setFocusPolicy(Qt.NoFocus)
        
        self.btn_load_label = QPushButton("Load Label Override")
        self.btn_load_label.clicked.connect(self.load_label)
        self.btn_load_label.setFocusPolicy(Qt.NoFocus)
        
        self.lbl_info = QLabel("No video loaded.")
        
        top_layout.addWidget(self.btn_load_vid)
        top_layout.addWidget(self.btn_compare_mode)
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
        self.spin_skip.valueChanged.connect(self.update_skip_step)
        
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
        
        # Navigation Scrollbar (below colored label bars)
        self.scrollbar = QScrollBar(Qt.Horizontal)
        self.scrollbar.setFocusPolicy(Qt.NoFocus)
        self.scrollbar.setEnabled(False)
        self.scrollbar.setRange(0, 0)
        self.scrollbar.setTracking(True)
        self.scrollbar.valueChanged.connect(self.on_scrollbar_changed)
        self.scrollbar.setStyleSheet("""
            QScrollBar:horizontal {
                height: 16px;
                background: #2b2b2b;
                margin: 2px 0px 4px 0px;
                border-radius: 4px;
            }
            QScrollBar::handle:horizontal {
                background: #606060;
                min-width: 30px;
                border-radius: 4px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #858585;
            }
            QScrollBar::handle:horizontal:pressed {
                background: #a8a8a8;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
                background: none;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
            }
        """)
        layout.addWidget(self.scrollbar)
        
        # Instructions
        inst = QLabel("<b>Controls:</b> Space=Play/Pause, &lt; / &gt; (or , / .)=Step 1 frame, Left/Right Arrow=Skip by seconds (Hold to repeat). <br><b>Timeline & Window Navigation:</b> Drag bottom scrollbar to scroll window left/right independently of frame. Scroll wheel or +/- to zoom. Middle/Right-click drag to pan. Click on timeline to seek playhead. <br><b>Editing:</b> Highlight range + press <b>0-3</b> to relabel section (selection stays highlighted). With no selection, <b>0-3</b> toggles continuous painting (0=Bg, 1=Cricket, 2=Pursuit, 3=Eat). Escape=Clear selection.")
        inst.setStyleSheet("font-size: 13px;")
        layout.addWidget(inst)
        
    def eventFilter(self, obj, event):
        if event.type() == event.Type.KeyPress:
            key = event.key()
            
            # Allow AutoRepeat for Left/Right and Comma/Period (< and >)
            if key == Qt.Key_Right:
                self.skip(1)
                return True
            elif key == Qt.Key_Left:
                self.skip(-1)
                return True
            elif key in [Qt.Key_Comma, Qt.Key_Less]:
                self.step_frame(-1)
                return True
            elif key in [Qt.Key_Period, Qt.Key_Greater]:
                self.step_frame(1)
                return True
                
            if not event.isAutoRepeat():
                if key == Qt.Key_Escape:
                    if self.timeline.has_selection():
                        self.timeline.clear_selection()
                        return True
                elif key == Qt.Key_Space:
                    self.toggle_play()
                    return True
                elif key in [Qt.Key_0, Qt.Key_QuoteLeft]:
                    self.handle_label_key(0)
                    return True
                elif key == Qt.Key_1:
                    self.handle_label_key(1)
                    return True
                elif key == Qt.Key_2:
                    self.handle_label_key(2)
                    return True
                elif key in [Qt.Key_3, Qt.Key_4]:
                    self.handle_label_key(3)
                    return True
                elif key in [Qt.Key_Plus, Qt.Key_Equal]:
                    self.timeline.zoom_by(1.25)
                    return True
                elif key in [Qt.Key_Minus, Qt.Key_Underscore]:
                    self.timeline.zoom_by(1.0 / 1.25)
                    return True
        return super().eventFilter(obj, event)
        
    def handle_label_key(self, val):
        if self.timeline.has_selection():
            self.apply_label(val)
        else:
            self.toggle_painting(val)
            
    def step_frame(self, delta):
        if not self.cap or self.total_frames <= 0: return
        if self.is_playing:
            self.toggle_play()
        self.set_frame(self.current_frame + delta)
        self.timeline.ensure_cursor_visible(self.current_frame)
        
    def update_skip_step(self, val):
        if hasattr(self, 'scrollbar') and self.video_fps > 0:
            self.scrollbar.setPageStep(max(1, int(self.video_fps * val)))
            
    def on_scrollbar_changed(self, value):
        if getattr(self, '_syncing_scrollbar', False):
            return
        self.timeline.scroll_to_frame(value)
            
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
        if self.labels is not None and self.timeline.has_selection():
            s = min(self.timeline.selection_start, self.timeline.selection_end)
            e = max(self.timeline.selection_start, self.timeline.selection_end)
            s = max(0, min(len(self.labels) - 1, s))
            e = max(0, min(len(self.labels) - 1, e))
            self.labels[s:e+1] = val
            
            # Keep selection highlighted as requested; do not clear selection
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
        
        # Configure navigation scrollbar
        self.timeline.total_frames = max(1, self.total_frames)
        self.timeline.update_scrollbar()
        
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
                stalk_col = 'Cricket Present' if 'Cricket Present' in prob_df.columns else ('Stalking' if 'Stalking' in prob_df.columns else prob_df.columns[1])
                eat_col = 'Eating' if 'Eating' in prob_df.columns else prob_df.columns[3]
                self.probs = prob_df[['Background', stalk_col, 'Pursuit', eat_col]].values
                print(f"Loaded probabilities from {os.path.basename(prob_paths[0])}")
            except Exception as e:
                print(f"Failed to load probs: {e}")
            
        # Find Labels (.xlsx or .csv)
        label_patterns = [
            os.path.join(folder, f"{clean_base}_refined.xlsx"),
            os.path.join(folder, f"{clean_base}_refined.csv"),
            os.path.join(folder, f"{clean_base}_analyzed.xlsx"),
            os.path.join(folder, f"{clean_base}_analyzed.csv"),
            os.path.join(folder, "excel_files", f"{clean_base}_analyzed.xlsx"),
            os.path.join(folder, "excel_files", f"{clean_base}_analyzed.csv"),
        ]
        found_label = None
        for lp in label_patterns:
            matches = glob.glob(lp)
            if matches:
                found_label = matches[0]
                break
            
        if found_label:
            self.load_label_file(found_label)
        else:
            self.labels = np.zeros(self.total_frames, dtype=int)
            self.timeline.set_data(self.labels, self.probs, 0)
            self.lbl_info.setText(f"Loaded {base} (No labels found)")
            
        if self.gt_folder:
            self.match_and_load_gt()
            
        self.set_frame(0)
        
    def prompt_gt_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with Manual Labels (.csv / .xlsx)")
        if folder:
            self.set_gt_folder(folder)
            
    def set_gt_folder(self, folder):
        self.gt_folder = folder
        folder_name = os.path.basename(folder)
        self.btn_compare_mode.setText(f"GT: {folder_name}")
        self.btn_compare_mode.setStyleSheet("background-color: #1976D2; color: white; font-weight: bold; padding: 4px 8px;")
        if self.video_path:
            self.match_and_load_gt()
        else:
            QMessageBox.information(self, "Comparison Mode", f"Manual labels folder set to:\n{folder}\n\nClick 'Load Video' to begin comparison review.")
            
    def toggle_compare_mode(self):
        if self.gt_folder is None:
            self.prompt_gt_folder()
        else:
            reply = QMessageBox.question(
                self, 
                "Comparison Mode", 
                f"Current GT Folder:\n{self.gt_folder}\n\nDo you want to change folder (Yes), disable comparison (No), or cancel?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            if reply == QMessageBox.Yes:
                self.prompt_gt_folder()
            elif reply == QMessageBox.No:
                self.gt_folder = None
                self.gt_labels = None
                self.btn_compare_mode.setText("Comparison Mode: OFF")
                self.btn_compare_mode.setStyleSheet("")
                if self.labels is not None:
                    self.timeline.set_data(self.labels, self.probs, self.current_frame, gt_labels=None)
                if not self.is_playing:
                    self.set_frame(self.current_frame)
                    
    def match_and_load_gt(self):
        if not self.gt_folder or not self.video_path or self.total_frames <= 0:
            self.gt_labels = None
            if self.labels is not None:
                self.timeline.set_data(self.labels, self.probs, self.current_frame, gt_labels=None)
            return

        base = os.path.basename(self.video_path)
        clean_base = re.sub(r'DLC_resnet50.*\.csv$', '', base, flags=re.IGNORECASE)
        if clean_base == base:
            clean_base = base[:base.rfind('.')] if '.' in base else base

        hunts = []
        matched_source = ""

        # 1. Dan Format matching: Animal_D(day)_HS(session)
        match_dan = re.search(r'([a-zA-Z]+\d+)_D(\d+)_HS(\d+)', clean_base, re.IGNORECASE)
        if match_dan:
            animal = match_dan.group(1).upper()
            session = int(match_dan.group(3))
            
            # Look for file with animal in name first (e.g. CricketHuntTimes_XP03.csv)
            cand_files = glob.glob(os.path.join(self.gt_folder, f"*{animal}*.csv")) + \
                         glob.glob(os.path.join(self.gt_folder, f"*{animal}*.xlsx"))
            if not cand_files:
                cand_files = glob.glob(os.path.join(self.gt_folder, "*.csv")) + \
                             glob.glob(os.path.join(self.gt_folder, "*.xlsx"))
                             
            for cf in cand_files:
                dan_dict = dataset_utils.parse_dan_label_file(cf)
                if (animal, session) in dan_dict:
                    hunts = dan_dict[(animal, session)]
                    matched_source = f"{os.path.basename(cf)} ({animal} S{session})"
                    break
                    
        # 2. Lucas Format matching: (LW\d+)[-_]hunt[-_]?(\d+)
        if not hunts:
            match_lucas = re.search(r'(LW\d+)[-_]hunt[-_]?(\d+)', clean_base, re.IGNORECASE)
            if match_lucas:
                animal = match_lucas.group(1).lower()
                hunt_num = match_lucas.group(2)
                
                cand_direct = os.path.join(self.gt_folder, f"{clean_base}_analyzed.xlsx")
                cand_direct_csv = os.path.join(self.gt_folder, f"{clean_base}_analyzed.csv")
                if os.path.exists(cand_direct):
                    hunts = dataset_utils.parse_lucas_labels(cand_direct)
                    matched_source = os.path.basename(cand_direct)
                elif os.path.exists(cand_direct_csv):
                    hunts = dataset_utils.parse_lucas_labels(cand_direct_csv)
                    matched_source = os.path.basename(cand_direct_csv)
                else:
                    all_files = glob.glob(os.path.join(self.gt_folder, "*.xlsx")) + glob.glob(os.path.join(self.gt_folder, "*.csv"))
                    for pf in all_files:
                        pf_lower = os.path.basename(pf).lower().replace('_', '-')
                        if animal in pf_lower and (f"hunt-{hunt_num}" in pf_lower or f"hunt{hunt_num}" in pf_lower):
                            hunts = dataset_utils.parse_lucas_labels(pf)
                            matched_source = os.path.basename(pf)
                            break

        # 3. Direct base file match fallback
        if not hunts:
            direct_cand = glob.glob(os.path.join(self.gt_folder, f"{clean_base}*.csv")) + \
                          glob.glob(os.path.join(self.gt_folder, f"{clean_base}*.xlsx"))
            for dc in direct_cand:
                try_hunts = dataset_utils.parse_lucas_labels(dc)
                if try_hunts:
                    hunts = try_hunts
                    matched_source = os.path.basename(dc)
                    break

        if hunts:
            self.gt_labels = dataset_utils.hunts_to_sequence(hunts, self.total_frames)
            self.timeline.set_data(self.labels, self.probs, self.current_frame, gt_labels=self.gt_labels)
            print(f"Comparison Mode: Matched manual labels from {matched_source} ({len(hunts)} hunts)")
            curr_text = self.lbl_info.text()
            self.lbl_info.setText(f"{curr_text} | GT: {matched_source} ({len(hunts)} hunts)")
        else:
            self.gt_labels = None
            self.timeline.set_data(self.labels, self.probs, self.current_frame, gt_labels=None)
            print(f"Comparison Mode: No matching manual label found for {clean_base} in {self.gt_folder}")
            curr_text = self.lbl_info.text()
            self.lbl_info.setText(f"{curr_text} | GT: No match found")
        
    def load_label(self):
        if not self.cap: return
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Label File", 
            "", 
            "Label Files (*.xlsx *.csv);;Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self.load_label_file(path)
            
    def load_label_file(self, path):
        self.label_path = path
        try:
            seq = None
            loaded_msg = ""
            
            # 1. Dan Format: Use shared dataset_utils.parse_dan_label_file
            dan_dict = dataset_utils.parse_dan_label_file(path)
            if dan_dict:
                hunts = []
                if self.video_path:
                    base = os.path.basename(self.video_path)
                    match = re.search(r'([a-zA-Z]+\d+)_D(\d+)_HS(\d+)', base, re.IGNORECASE)
                    if match:
                        v_animal = match.group(1).upper()
                        v_session = int(match.group(3))
                        hunts = dan_dict.get((v_animal, v_session), [])
                        if not hunts:
                            for (a, s), h_list in dan_dict.items():
                                if s == v_session:
                                    hunts = h_list
                                    break
                    else:
                        match_s = re.search(r'HS(\d+)', base, re.IGNORECASE)
                        if match_s:
                            v_session = int(match_s.group(1))
                            for (a, s), h_list in dan_dict.items():
                                if s == v_session:
                                    hunts = h_list
                                    break
                
                if not hunts and len(dan_dict) == 1:
                    hunts = next(iter(dan_dict.values()))
                    
                if hunts:
                    seq = dataset_utils.hunts_to_sequence(hunts, self.total_frames)
                    loaded_msg = f"Loaded Dan Labels: {os.path.basename(path)} ({len(hunts)} hunts)"
                else:
                    avail = list(dan_dict.keys())
                    print(f"Warning: Could not match video session in Dan file {path}. Available sessions: {avail}")
            
            # 2. Lucas / Analyzed Format: Use shared dataset_utils.parse_lucas_labels
            if seq is None:
                lucas_hunts = dataset_utils.parse_lucas_labels(path)
                if lucas_hunts:
                    seq = dataset_utils.hunts_to_sequence(lucas_hunts, self.total_frames)
                    loaded_msg = f"Loaded Analyzed Labels: {os.path.basename(path)} ({len(lucas_hunts)} hunts)"
                    
            # 3. Raw Lucas Format (Event and Frame columns)
            if seq is None:
                df = pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)
                df.columns = df.columns.astype(str).str.strip()
                if 'Event' in df.columns and 'Frame' in df.columns:
                    current_hunt = {}
                    raw_hunts = []
                    for _, row in df.iterrows():
                        ev_type = str(row['Event']).lower().strip() if pd.notna(row.get('Event')) else ""
                        if pd.isna(row.get('Frame')): continue
                        f_val = int(row['Frame'])
                        if 'enter' in ev_type:
                            if current_hunt and current_hunt.get('enter') is not None and current_hunt.get('capture') is not None:
                                raw_hunts.append(current_hunt)
                            current_hunt = {'enter': f_val, 'pursuit_start': f_val, 'capture': None, 'consumption_end': None}
                        elif 'pursuit' in ev_type:
                            if current_hunt: current_hunt['pursuit_start'] = f_val
                        elif 'capture' in ev_type:
                            if current_hunt: current_hunt['capture'] = f_val
                    if current_hunt and current_hunt.get('enter') is not None and current_hunt.get('capture') is not None:
                        raw_hunts.append(current_hunt)
                    if raw_hunts:
                        seq = dataset_utils.hunts_to_sequence(raw_hunts, self.total_frames)
                        loaded_msg = f"Loaded Raw Lucas Labels: {os.path.basename(path)} ({len(raw_hunts)} hunts)"

            # 4. Frame-by-frame direct predictions/labels
            if seq is None:
                df = pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)
                df.columns = df.columns.astype(str).str.strip()
                if any(c.lower() in ['label', 'labels', 'prediction', 'pred', 'class'] for c in df.columns):
                    lbl_col = next(c for c in df.columns if c.lower() in ['label', 'labels', 'prediction', 'pred', 'class'])
                    vals = df[lbl_col].dropna().values.astype(int)
                    seq = np.zeros(self.total_frames, dtype=int)
                    n = min(len(vals), self.total_frames)
                    seq[:n] = vals[:n]
                    loaded_msg = f"Loaded Frame Labels: {os.path.basename(path)}"

            if seq is not None:
                self.labels = seq
                self.timeline.set_data(self.labels, self.probs, self.current_frame)
                self.lbl_info.setText(loaded_msg if loaded_msg else f"Loaded: {os.path.basename(path)}")
            else:
                QMessageBox.warning(self, "Warning", f"Could not parse hunts from {os.path.basename(path)}.\nPlease check file columns.")
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
                cls_names = {0: 'Background', 1: 'Cricket Present', 2: 'Pursuit', 3: 'Eating'}
                colors_bgr = {0: (200, 200, 200), 1: (0, 165, 255), 2: (0, 0, 255), 3: (0, 255, 0)}
                
                y_pos = 26
                lbl_val = self.labels[self.current_frame]
                text_pred = f"Pred: {cls_names.get(lbl_val, 'Unknown')}"
                # Crisp outline + text for high contrast on any background
                cv2.putText(img, text_pred, (18, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, text_pred, (18, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors_bgr.get(lbl_val, (255, 255, 255)), 1, cv2.LINE_AA)
                
                if self.gt_labels is not None and self.current_frame < len(self.gt_labels):
                    y_pos += 22
                    gt_val = self.gt_labels[self.current_frame]
                    text_gt = f"Manual: {cls_names.get(gt_val, 'Unknown')}"
                    cv2.putText(img, text_gt, (18, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(img, text_gt, (18, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors_bgr.get(gt_val, (255, 255, 255)), 1, cv2.LINE_AA)
                
                y_pos += 22
                brush_names = {None: 'No Change', 0: 'Background', 1: 'Cricket Present', 2: 'Pursuit', 3: 'Eating'}
                brush_colors = {None: (255, 255, 0), 0: (200, 200, 200), 1: (0, 165, 255), 2: (0, 0, 255), 3: (0, 255, 0)}
                text_brush = f"Brush: {brush_names.get(self.active_paint_label, 'No Change')}"
                cv2.putText(img, text_brush, (18, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, text_brush, (18, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, brush_colors.get(self.active_paint_label, (255, 255, 0)), 1, cv2.LINE_AA)
            
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
            
            # Detect back-to-back hunt: transition from Eating (3) directly to Stalking (1) or Pursuit (2)
            if in_hunt and current_event and current_event['capture'] != -1 and val in (1, 2):
                current_event['capture_end'] = s
                if current_event['pursuit'] == -1:
                    current_event['pursuit'] = current_event['enter']
                events.append(current_event)
                # Begin new hunt immediately
                current_event = {'enter': s, 'pursuit': s if val == 2 else -1, 'capture': -1, 'capture_end': -1}
                in_hunt = True
                continue

            if val != 0 and not in_hunt:
                in_hunt = True
                current_event = {'enter': s, 'pursuit': s if val == 2 else -1, 'capture': -1, 'capture_end': -1}
                
            if in_hunt:
                if val == 1 and current_event['enter'] == -1:
                    current_event['enter'] = s
                elif val == 2 and current_event['pursuit'] == -1:
                    current_event['pursuit'] = s
                elif val == 3 and current_event['capture'] == -1:
                    current_event['capture'] = s
                    
                if val == 0:
                    current_event['capture_end'] = s
                    if current_event['pursuit'] == -1:
                        current_event['pursuit'] = current_event['enter']
                    if current_event['capture'] == -1:
                        current_event['capture'] = current_event['capture_end']
                    events.append(current_event)
                    in_hunt = False
                    current_event = None
                    
        if in_hunt and current_event:
            current_event['capture_end'] = len(lbls)
            if current_event['pursuit'] == -1:
                current_event['pursuit'] = current_event['enter']
            if current_event['capture'] == -1:
                current_event['capture'] = current_event['capture_end']
            events.append(current_event)
            
        return events

    def save_labels(self):
        if self.labels is None or not self.video_path: return
        
        events = self.sequence_to_events()
        
        base = os.path.basename(self.video_path)
        clean_base = base[:base.rfind('.')] if '.' in base else base
        
        out_name = f"{clean_base}_refined.xlsx"
        default_path = os.path.join(os.path.dirname(self.video_path), out_name)
        
        path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Refined Labels", 
            default_path, 
            "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)"
        )
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

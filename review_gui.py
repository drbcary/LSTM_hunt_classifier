import sys
import os
import glob
import re
import cv2
import numpy as np
import pandas as pd
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox, 
                               QDoubleSpinBox, QScrollBar, QDialog, QRadioButton, 
                               QButtonGroup, QDialogButtonBox, QMenu, QSizePolicy)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QShortcut, QKeySequence, QFontMetrics
from typing import Any

from dataset.matcher import DatasetMatcher
import dataset_utils
import RatHunt_Inference
import MouseHunt_Inference

# ==========================================
# Class Definitions & Color Schemes
# ==========================================
CLASS_NAMES_MOUSE = {
    0: 'Background',
    1: 'Cricket Present',
    2: 'First Approach',
    3: 'Final Approach',
    4: 'Eating'
}
CLASS_COLORS_MOUSE_BGR = {
    0: (140, 140, 140), # Background (Gray)
    1: (207, 190, 23),  # Cricket Present (Teal / Cyan in BGR)
    2: (180, 119, 31),  # First Approach (Steel Blue in BGR)
    3: (40, 39, 214),   # Final Approach (Crimson Red in BGR)
    4: (44, 160, 44)    # Eating (Forest Green in BGR)
}
CLASS_COLORS_MOUSE_QT = {
    0: QColor(140, 140, 140),
    1: QColor(23, 190, 207),
    2: QColor(31, 119, 180),
    3: QColor(214, 39, 40),
    4: QColor(44, 160, 44)
}

CLASS_NAMES_RAT = {
    0: 'Background',
    1: 'Stalking',
    2: 'Pursuit',
    3: 'Eating'
}
CLASS_COLORS_RAT_BGR = {
    0: (200, 200, 200),
    1: (0, 165, 255),
    2: (0, 0, 255),
    3: (0, 255, 0)
}
CLASS_COLORS_RAT_QT = {
    0: QColor(200, 200, 200),
    1: QColor(255, 165, 0),
    2: QColor(255, 0, 0),
    3: QColor(0, 255, 0)
}

class ElidedLabel(QLabel):
    """A QLabel that elides its text in the middle when space is tight without blocking window resizing."""
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._full_text = text
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        if text:
            self.setToolTip(text)

    def setText(self, text):
        self._full_text = text
        self.setToolTip(text)
        self._update_elided_text()

    def text(self):
        return getattr(self, '_full_text', super().text())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_elided_text()

    def _update_elided_text(self):
        if not getattr(self, '_full_text', None):
            super().setText("")
            return
        fm = self.fontMetrics()
        elided = fm.elidedText(self._full_text, Qt.TextElideMode.ElideMiddle, max(10, self.width() - 4))
        super().setText(elided)


class TimelineWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.setMinimumWidth(0)
        self.labels = None
        self.gt_labels = None
        self.probs = None
        self.total_frames = 1
        self.current_frame = 0
        self.paradigm = "mouse"
        
        self.selection_start = -1
        self.selection_end = -1
        self.press_x = 0
        
        self.zoom = 1.0
        self.offset_x = 0.0 # Pixels
        self.is_panning = False
        self.last_mouse_x = 0
        
        self.colors = CLASS_COLORS_MOUSE_QT.copy()
        
    def set_paradigm(self, paradigm="mouse"):
        self.paradigm = paradigm
        if paradigm == "mouse":
            self.colors = CLASS_COLORS_MOUSE_QT.copy()
        else:
            self.colors = CLASS_COLORS_RAT_QT.copy()
        self.update()
        
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
        win: Any = self.window()
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
        page_step = max(1, round(frames_visible))
        max_scroll = max(0, self.total_frames - page_step)
        
        w_total = w_vis * self.zoom
        current_first_frame = round((-self.offset_x / w_total) * self.total_frames)
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
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(15, 15, 15, 190))
        painter.drawRoundedRect(8, badge_y, badge_w, badge_h, 4, 4)
        painter.setPen(QColor(255, 255, 255))
        font = painter.font()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(8, badge_y, badge_w, badge_h, Qt.AlignmentFlag.AlignCenter, text)
        
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
            gt_y = 0
        
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
        if self.probs is not None and len(self.probs) > 0:
            step = max(1, int(self.total_frames / w))
            num_classes = min(self.probs.shape[1], len(self.colors))
            for c in range(num_classes):
                painter.setPen(QPen(self.colors.get(c, QColor(0, 0, 0)), 2.2))
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
        if event.button() == Qt.MouseButton.RightButton or event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = True
            self.last_mouse_x = event.position().x()
        elif event.button() == Qt.MouseButton.LeftButton:
            if self.labels is not None:
                self.press_x = event.position().x()
                frame = max(0, min(self.total_frames - 1, self.get_frame_for_x(self.press_x)))
                self.selection_start = frame
                self.selection_end = frame
                
                # Stop continuous painting brush if clicking to select/seek
                win: Any = self.window()
                if hasattr(win, 'active_paint_label'):
                    win.active_paint_label = None
                    
                if hasattr(win, 'set_frame'):
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
        elif self.selection_start != -1 and event.buttons() & Qt.MouseButton.LeftButton:
            if self.labels is not None:
                frame = max(0, min(self.total_frames - 1, self.get_frame_for_x(event.position().x())))
                self.selection_end = frame
                win: Any = self.window()
                if hasattr(win, 'set_frame'):
                    win.set_frame(self.selection_end)
                self.update()
                
    def mouseReleaseEvent(self, event):
        if event.button() in [Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton]:
            self.is_panning = False
        elif event.button() == Qt.MouseButton.LeftButton:
            # If user simply clicked without dragging (< 4 pixels movement), clear selection
            if abs(event.position().x() - getattr(self, 'press_x', 0)) < 4:
                self.selection_start = -1
                self.selection_end = -1
                self.update()


class GTFormatDialog(QDialog):
    """
    Interactive dialog prompting the user to pick from supported ground truth master label formats.
    """
    def __init__(self, current_format="diane", current_path=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Ground Truth / Master Label Format")
        self.resize(520, 320)
        
        layout = QVBoxLayout(self)
        
        lbl_title = QLabel("<b>Select Manual Ground Truth Format:</b>")
        lbl_title.setStyleSheet("font-size: 14px; margin-bottom: 6px;")
        layout.addWidget(lbl_title)
        
        self.btn_group = QButtonGroup(self)
        
        options = [
            ("diane", "Diane Master Excel (Whole Day / Single Video)", 
             "Matches (Animal, Day) against master spreadsheet; assumes 1 contiguous session."),
            ("diane_parts", "Diane Master Excel - Split Parts", 
             "Detects and maps sibling parts when frame counters reset (Logitech split cameras)."),
            ("dan", "Dan Format (CricketHuntTimes)", 
             "Matches Animal_D#_HS# against cohort Excel/CSV session tables."),
            ("lucas", "Lucas / Analyzed Excel", 
             "Matches individual _analyzed.xlsx or LW session files."),
            ("auto", "Auto-Detect", 
             "Attempts automatic format detection from file structure / headers."),
        ]
        
        for fmt_id, title, desc in options:
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(4, 2, 4, 2)
            row_layout.setSpacing(10)
            
            rb = QRadioButton()
            rb.setProperty("fmt_id", fmt_id)
            if fmt_id == current_format:
                rb.setChecked(True)
            self.btn_group.addButton(rb)
            
            lbl = QLabel(f"<div style='line-height: 125%;'><b>{title}</b><br><span style='color: #888888; font-size: 11px;'>{desc}</span></div>")
            lbl.setTextFormat(Qt.TextFormat.RichText)
            lbl.mousePressEvent = (lambda b: lambda ev: b.setChecked(True))(rb)
            
            row_layout.addWidget(rb)
            row_layout.addWidget(lbl, stretch=1)
            layout.addWidget(row_widget)
            
        if current_path:
            layout.addSpacing(6)
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(4, 2, 4, 2)
            row_layout.setSpacing(10)
            
            rb = QRadioButton()
            rb.setProperty("fmt_id", "disable")
            self.btn_group.addButton(rb)
            
            lbl = QLabel("<div style='line-height: 125%;'><b style='color: #d9534f;'>Disable Comparison Mode</b><br><span style='color: #888888; font-size: 11px;'>Turn off ground truth overlay</span></div>")
            lbl.setTextFormat(Qt.TextFormat.RichText)
            lbl.mousePressEvent = (lambda b: lambda ev: b.setChecked(True))(rb)
            
            row_layout.addWidget(rb)
            row_layout.addWidget(lbl, stretch=1)
            layout.addWidget(row_widget)
            
        if not self.btn_group.checkedButton() and self.btn_group.buttons():
            self.btn_group.buttons()[0].setChecked(True)
            
        layout.addSpacing(12)
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)
        
    def get_selected_format(self):
        checked = self.btn_group.checkedButton()
        if checked:
            return checked.property("fmt_id")
        return "auto"


class ReviewGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hunt Label Review GUI (Mouse & Rat)")
        self.resize(1240, 840)
        self.setMinimumSize(540, 380)
        
        self.video_path = None
        self.label_path = None
        self.dlc_path = None
        
        self.cap = None
        self.total_frames = 0
        self.current_frame = 0
        self.video_fps = 30.0
        
        self.paradigm = "mouse"  # "mouse" (5 classes) or "rat" (4 classes)
        self.labels = None
        self.gt_labels = None
        self.gt_path = None
        self.label_format = None
        self.probs = None
        self.dlc_data = None
        self.dlc_confs = None
        
        self.active_paint_label = None
        self.last_painted_frame = -1
        
        self.is_playing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        
        self.init_ui()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        
        if '--compare' in sys.argv:
            QTimer.singleShot(200, self.toggle_compare_mode)
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Top Bar
        top_layout = QHBoxLayout()
        self.btn_load_vid = QPushButton("Load Video")
        self.btn_load_vid.clicked.connect(self.load_video)
        self.btn_load_vid.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_load_vid.setMinimumWidth(0)
        
        self.btn_compare_mode = QPushButton("Comparison Mode: OFF")
        self.btn_compare_mode.clicked.connect(self.toggle_compare_mode)
        self.btn_compare_mode.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_compare_mode.setMinimumWidth(0)
        
        self.btn_load_label = QPushButton("Load Label Override")
        self.btn_load_label.clicked.connect(self.load_label)
        self.btn_load_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_load_label.setMinimumWidth(0)
        
        self.btn_paradigm = QPushButton("Paradigm: Mouse (5-Class)")
        self.btn_paradigm.setStyleSheet("background-color: #00838F; color: white; font-weight: bold; padding: 4px 8px;")
        self.btn_paradigm.clicked.connect(self.toggle_paradigm)
        self.btn_paradigm.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_paradigm.setMinimumWidth(0)
        
        self.lbl_info = ElidedLabel("No video loaded.")
        
        top_layout.addWidget(self.btn_load_vid)
        top_layout.addWidget(self.btn_compare_mode)
        top_layout.addWidget(self.btn_load_label)
        top_layout.addWidget(self.btn_paradigm)
        top_layout.addWidget(self.lbl_info, stretch=1)
        layout.addLayout(top_layout)
        
        # Video Display
        self.video_lbl = QLabel()
        self.video_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_lbl.setStyleSheet("background-color: black;")
        self.video_lbl.setMinimumSize(160, 120)
        layout.addWidget(self.video_lbl, stretch=1)
        
        # HUD overlay positioned in the top-left outside of the video frame in the black box
        self.hud_lbl = QLabel(self.video_lbl)
        self.hud_lbl.setStyleSheet("""
            background-color: rgba(18, 18, 18, 185);
            border: 1px solid rgba(255, 255, 255, 35);
            border-radius: 6px;
            padding: 6px 10px;
        """)
        self.hud_lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.hud_lbl.move(14, 14)
        self.hud_lbl.hide()
        
        # Controls
        controls_layout = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_play.setMinimumWidth(0)
        
        self.lbl_frame = QLabel("Frame: 0 / 0 (00:00:00 / 00:00:00)")
        
        self.lbl_skip = QLabel("Skip (sec):")
        self.spin_skip = QDoubleSpinBox()
        self.spin_skip.setRange(0.1, 60.0)
        self.spin_skip.setValue(1.0)
        self.spin_skip.setSingleStep(0.5)
        self.spin_skip.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.spin_skip.valueChanged.connect(self.update_skip_step)
        
        self.btn_save = QPushButton("Save Refined Labels")
        self.btn_save.clicked.connect(self.save_labels)
        self.btn_save.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold; padding: 6px 12px;")
        self.btn_save.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_save.setMinimumWidth(0)
        
        controls_layout.addWidget(self.btn_play)
        controls_layout.addWidget(self.lbl_frame)
        controls_layout.addWidget(self.lbl_skip)
        controls_layout.addWidget(self.spin_skip)
        controls_layout.addStretch()
        controls_layout.addWidget(self.btn_save)
        layout.addLayout(controls_layout)
        
        # Timeline
        self.timeline = TimelineWidget(self)
        self.timeline.set_paradigm(self.paradigm)
        layout.addWidget(self.timeline)
        
        # Navigation Scrollbar (below colored label bars)
        self.scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self.scrollbar.setFocusPolicy(Qt.FocusPolicy.NoFocus)
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
        inst = QLabel(
            "<b>Controls:</b> Space=Play/Pause | &lt; / &gt; (or , / .)=Step 1 frame | Left/Right Arrow=Skip seconds | +/-=Zoom.<br>"
            "<b>Editing:</b> Highlight range + press <b>0-4</b> to relabel section. With no selection, <b>0-4</b> toggles continuous painting brush. Escape=Clear selection.<br>"
            "<b>Mouse (5-Class):</b> 0=Background (Gray) | 1=Cricket Present (Teal) | 2=First Approach (Blue) | 3=Final Approach (Red) | 4=Eating (Green).<br>"
            "<b>Rat (4-Class):</b> 0=Background (Gray) | 1=Stalking (Orange) | 2=Pursuit (Red) | 3=Eating (Green)."
        )
        inst.setWordWrap(True)
        inst.setMinimumWidth(0)
        inst.setStyleSheet("font-size: 11px; color: #E0E0E0; background-color: #222222; padding: 6px 10px; border-radius: 4px; line-height: 1.3;")
        layout.addWidget(inst)
        
    def toggle_paradigm(self):
        new_paradigm = "rat" if self.paradigm == "mouse" else "mouse"
        self.set_paradigm(new_paradigm)
        
    def set_paradigm(self, paradigm="mouse"):
        self.paradigm = paradigm
        if hasattr(self, 'timeline'):
            self.timeline.set_paradigm(paradigm)
        if hasattr(self, 'btn_paradigm'):
            if paradigm == "mouse":
                self.btn_paradigm.setText("Paradigm: Mouse (5-Class)")
                self.btn_paradigm.setStyleSheet("background-color: #00838F; color: white; font-weight: bold; padding: 4px 8px;")
            else:
                self.btn_paradigm.setText("Paradigm: Rat (4-Class)")
                self.btn_paradigm.setStyleSheet("background-color: #455A64; color: white; font-weight: bold; padding: 4px 8px;")
        if not self.is_playing and self.cap:
            self.set_frame(self.current_frame)
        elif hasattr(self, '_update_hud'):
            self._update_hud()
        
    def eventFilter(self, obj, event):
        if event.type() == event.Type.KeyPress:
            key = event.key()
            
            # Allow AutoRepeat for Left/Right and Comma/Period (< and >)
            if key == Qt.Key.Key_Right:
                self.skip(1)
                return True
            elif key == Qt.Key.Key_Left:
                self.skip(-1)
                return True
            elif key in [Qt.Key.Key_Comma, Qt.Key.Key_Less]:
                self.step_frame(-1)
                return True
            elif key in [Qt.Key.Key_Period, Qt.Key.Key_Greater]:
                self.step_frame(1)
                return True
                
            if not event.isAutoRepeat():
                if key == Qt.Key.Key_Escape:
                    if self.timeline.has_selection():
                        self.timeline.clear_selection()
                        return True
                elif key == Qt.Key.Key_Space:
                    self.toggle_play()
                    return True
                elif key in [Qt.Key.Key_0, Qt.Key.Key_QuoteLeft]:
                    self.handle_label_key(0)
                    return True
                elif key == Qt.Key.Key_1:
                    self.handle_label_key(1)
                    return True
                elif key == Qt.Key.Key_2:
                    self.handle_label_key(2)
                    return True
                elif key == Qt.Key.Key_3:
                    self.handle_label_key(3)
                    return True
                elif key == Qt.Key.Key_4:
                    if self.paradigm == "mouse":
                        self.handle_label_key(4)
                    else:
                        self.handle_label_key(3)
                    return True
                elif key in [Qt.Key.Key_Plus, Qt.Key.Key_Equal]:
                    self.timeline.zoom_by(1.25)
                    return True
                elif key in [Qt.Key.Key_Minus, Qt.Key.Key_Underscore]:
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
            self.active_paint_label = None
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
            self.timeline.update()
            self.set_frame(self.current_frame)
            
    def toggle_compare_mode(self):
        current_fmt = self.label_format or ("diane" if self.paradigm == "mouse" else "dan")
        dlg = GTFormatDialog(
            current_format=current_fmt,
            current_path=self.gt_path,
            parent=self
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
            
        selected_fmt = dlg.get_selected_format()
        
        if selected_fmt == "disable":
            self.gt_path = None
            self.gt_labels = None
            self.label_format = None
            self.btn_compare_mode.setText("Comparison Mode: OFF")
            self.btn_compare_mode.setStyleSheet("")
            if self.labels is not None:
                self.timeline.set_data(self.labels, self.probs, self.current_frame, gt_labels=None)
            if not self.is_playing:
                self.set_frame(self.current_frame)
            return
            
        start_dir = ""
        if self.gt_path:
            start_dir = self.gt_path if os.path.isdir(self.gt_path) else os.path.dirname(self.gt_path)
        elif self.video_path:
            start_dir = os.path.dirname(self.video_path)
            
        if selected_fmt in ("diane", "diane_parts"):
            mode_name = "Split Parts" if selected_fmt == "diane_parts" else "Standard"
            path, _ = QFileDialog.getOpenFileName(
                self,
                f"Select Diane Master Excel File ({mode_name})",
                start_dir,
                "Excel / CSV Files (*.xlsx *.csv);;All Files (*)"
            )
            if not path:
                return
            self.set_gt_source(path, selected_fmt)
        elif selected_fmt == "dan":
            folder = QFileDialog.getExistingDirectory(
                self,
                "Select Folder with Dan Labels (CricketHuntTimes_*.csv)",
                start_dir
            )
            if not folder:
                return
            self.set_gt_source(folder, selected_fmt)
        elif selected_fmt == "lucas":
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Lucas Analyzed Label File or Folder",
                start_dir,
                "Excel / CSV Files (*.xlsx *.csv);;All Files (*)"
            )
            if not path:
                return
            self.set_gt_source(path, selected_fmt)
        else:  # auto
            msg = QMessageBox.question(
                self,
                "Ground Truth Source",
                "Do you want to select a single Master Excel file (Yes) or a directory of label files (No)?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
            )
            if msg == QMessageBox.StandardButton.Yes:
                path, _ = QFileDialog.getOpenFileName(self, "Select Ground Truth File", start_dir, "Excel / CSV Files (*.xlsx *.csv);;All Files (*)")
                if not path: return
                self.set_gt_source(path, "auto")
            elif msg == QMessageBox.StandardButton.No:
                folder = QFileDialog.getExistingDirectory(self, "Select Ground Truth Directory", start_dir)
                if not folder: return
                self.set_gt_source(folder, "auto")
                
    def set_gt_source(self, path, label_format):
        self.gt_path = path
        self.label_format = label_format
        name = os.path.basename(path)
        fmt_label = {
            'diane': 'Diane',
            'diane_parts': 'Diane Parts',
            'dan': 'Dan',
            'lucas': 'Lucas',
            'auto': 'Auto'
        }.get(label_format, label_format)
        
        short_name = name if len(name) <= 24 else f"{name[:12]}...{name[-8:]}"
        self.btn_compare_mode.setText(f"GT: {fmt_label} ({short_name})")
        self.btn_compare_mode.setToolTip(f"Ground Truth ({fmt_label}):\n{path}")
        self.btn_compare_mode.setStyleSheet("background-color: #1976D2; color: white; font-weight: bold; padding: 4px 8px;")
        if self.video_path:
            self.match_and_load_gt()
        else:
            QMessageBox.information(self, "Comparison Mode", f"Ground truth source set ({fmt_label}):\n{path}\n\nClick 'Load Video' to align with sessions.")

    def match_and_load_gt(self):
        if not self.gt_path or not self.video_path or self.total_frames <= 0:
            self.gt_labels = None
            if self.labels is not None:
                self.timeline.set_data(self.labels, self.probs, self.current_frame, gt_labels=None)
            return

        base = os.path.basename(self.video_path)
        clean_base = re.sub(r'DLC.*(?:\.csv|\.h5)$', '', base, flags=re.IGNORECASE)
        if clean_base == base:
            clean_base = os.path.splitext(base)[0]

        fmt = None if self.label_format == "auto" else self.label_format
        y_true = None
        true_hunts = None
        matched_source = os.path.basename(self.gt_path)

        try:
            if self.paradigm == "mouse":
                y_true, true_hunts = MouseHunt_Inference.load_ground_truth_sequence(
                    label_path=self.gt_path,
                    dlc_path=self.dlc_path or self.video_path,
                    num_frames=self.total_frames,
                    fps=self.video_fps,
                    label_format=fmt,
                    paradigm="mouse"
                )
            else:
                y_true, true_hunts = RatHunt_Inference.load_ground_truth_sequence(
                    label_path=self.gt_path,
                    dlc_path=self.dlc_path or self.video_path,
                    num_frames=self.total_frames,
                    fps=self.video_fps,
                    label_format=fmt
                )
        except Exception as e:
            print(f"Error loading ground truth sequence: {e}")
            y_true, true_hunts = None, None

        if y_true is not None and true_hunts is not None and len(true_hunts) > 0:
            self.gt_labels = y_true
            self.timeline.set_data(self.labels, self.probs, self.current_frame, gt_labels=self.gt_labels)
            print(f"Comparison Mode: Matched manual labels from {matched_source} ({len(true_hunts)} hunts)")
            curr_text = self.lbl_info.text()
            self.lbl_info.setText(f"{curr_text} | GT: {matched_source} ({len(true_hunts)} hunts)")
        else:
            self.gt_labels = None
            self.timeline.set_data(self.labels, self.probs, self.current_frame, gt_labels=None)
            print(f"Comparison Mode: No matching manual label found for {clean_base} in {self.gt_path}")
            curr_text = self.lbl_info.text()
            self.lbl_info.setText(f"{curr_text} | GT: No match found in {matched_source}")
            
        if not self.is_playing and self.cap:
            self.set_frame(self.current_frame)
        elif hasattr(self, '_update_hud'):
            self._update_hud()
            
    def load_dlc_tracking(self, dlc_path):
        """Loads all bodyparts and likelihoods from DLC CSV or H5."""
        try:
            if dlc_path.endswith('.h5'):
                df = pd.read_hdf(dlc_path)
            else:
                df = pd.read_csv(dlc_path, header=[0, 1, 2], index_col=0)
            scorer = df.columns.get_level_values(0)[0]
            bparts = list(dict.fromkeys(df.columns.get_level_values(1)))
            
            coords_dict = {}
            confs_dict = {}
            for bp in bparts:
                try:
                    x = df[scorer][bp]['x'].values
                    y = df[scorer][bp]['y'].values
                    conf = df[scorer][bp]['likelihood'].values
                    coords_dict[bp] = np.column_stack([x, y])
                    confs_dict[bp] = conf
                except Exception:
                    continue
            self.dlc_data = coords_dict
            self.dlc_confs = confs_dict
            print(f"Loaded {len(coords_dict)} DLC body parts: {list(coords_dict.keys())}")
        except Exception as e:
            print(f"Failed to load DLC tracking from {dlc_path}: {e}")
            self.dlc_data = None
            self.dlc_confs = None

    def _select_display_keypoints(self):
        """
        Filters DLC keypoints to display only:
        - Mouse: head, midbody, butt, and last three tail points.
        - Cricket: cricket landmarks with confidence text.
        """
        if not self.dlc_data:
            return [], []
            
        all_bps = list(self.dlc_data.keys())
        all_bps_lower = {bp: bp.lower().replace(' ', '_').replace('-', '_') for bp in all_bps}
        
        # 1. Head (snout, nose, head)
        head_bp = None
        for cand in ['snout', 'nose', 'head']:
            for bp, bpl in all_bps_lower.items():
                if cand == bpl or cand in bpl:
                    head_bp = bp
                    break
            if head_bp:
                break
                
        # 2. Midbody (mid_body, midbody, center, body_center, spine)
        mid_bp = None
        for cand in ['mid_body', 'midbody', 'center', 'body_center', 'spine']:
            for bp, bpl in all_bps_lower.items():
                if cand == bpl or cand in bpl:
                    mid_bp = bp
                    break
            if mid_bp:
                break
                
        # 3. Butt (butt, rear, tail_base, tailbase)
        butt_bp = None
        for cand in ['butt', 'rear', 'tail_base', 'tailbase']:
            for bp, bpl in all_bps_lower.items():
                if cand == bpl or cand in bpl:
                    butt_bp = bp
                    break
            if butt_bp:
                break
                
        # 4. Tail points (tail1, tail2, tail3, near_endtail, endtail, etc.)
        tail_candidates = []
        for bp, bpl in all_bps_lower.items():
            if 'tail' in bpl and bp != butt_bp and 'base' not in bpl:
                tail_candidates.append(bp)
                
        tail_order = ['tail1', 'tail2', 'tail3', 'tail_mid', 'tail4', 'near_endtail', 'tail5', 'endtail', 'tail_end', 'tailtip', 'tip']
        def _tail_sort_key(bp):
            bpl = all_bps_lower[bp]
            for idx, order_name in enumerate(tail_order):
                if order_name in bpl:
                    return idx
            m = re.search(r'\d+', bpl)
            if m:
                return 10 + int(m.group(0))
            return 99
            
        sorted_tails = sorted(tail_candidates, key=_tail_sort_key)
        last_three_tails = sorted_tails[-3:] if len(sorted_tails) >= 3 else sorted_tails
        
        # 5. Cricket points
        cricket_bps = [bp for bp, bpl in all_bps_lower.items() if 'cricket' in bpl]
        
        # Assemble mouse display list
        mouse_bps = []
        for bp in [head_bp, mid_bp, butt_bp] + last_three_tails:
            if bp and bp not in mouse_bps and bp in self.dlc_data:
                mouse_bps.append(bp)
                
        return mouse_bps, cricket_bps

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.avi *.mp4 *.mov)")
        if not path: return
        
        self.video_path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open video.")
            return
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        # Configure navigation scrollbar
        self.timeline.total_frames = max(1, self.total_frames)
        self.timeline.update_scrollbar()
        
        folder = os.path.dirname(path)
        base = os.path.basename(path)
        
        # Build candidate stems
        raw_stem = os.path.splitext(base)[0]
        stems = [raw_stem]
        
        # Strip trailing _VIDEO / -VIDEO / _video
        stem_no_vid = re.sub(r'[_.-]?video$', '', raw_stem, flags=re.IGNORECASE)
        if stem_no_vid not in stems:
            stems.append(stem_no_vid)
            
        # Strip DLC.* suffix if present
        stem_no_dlc = re.sub(r'DLC.*$', '', raw_stem, flags=re.IGNORECASE)
        if stem_no_dlc not in stems:
            stems.append(stem_no_dlc)
            
        clean_base = stems[1] if len(stems) > 1 else stems[0]
        
        # 1. Find DLC tracking files
        cand_dlc = []
        for s in stems:
            cand_dlc.extend(glob.glob(os.path.join(folder, f"{s}*DLC*.csv")))
            cand_dlc.extend(glob.glob(os.path.join(folder, f"{s}*DLC*.h5")))
            
        if not cand_dlc:
            all_folder_dlc = glob.glob(os.path.join(folder, "*DLC*.csv")) + glob.glob(os.path.join(folder, "*DLC*.h5"))
            matcher = DatasetMatcher(default_fps=self.video_fps)
            v_info = matcher.parse_filename(base)
            if v_info:
                for f in all_folder_dlc:
                    f_info = matcher.parse_filename(os.path.basename(f))
                    if f_info and str(f_info['animal_id']).upper() == str(v_info['animal_id']).upper() and int(f_info['day']) == int(v_info['day']):
                        cand_dlc.append(f)
                        
        dlc_files = [f for f in cand_dlc if 'labeledinds' not in f.lower() and 'labeledbouts' not in f.lower() and not f.lower().endswith('_probs.csv')]
        dlc_files = sorted(dlc_files, key=lambda p: (0 if p.lower().endswith('.csv') else 1))
        if dlc_files:
            self.dlc_path = dlc_files[0]
            dlc_stem = re.sub(r'DLC.*(?:\.csv|\.h5)$', '', os.path.basename(self.dlc_path), flags=re.IGNORECASE)
            if dlc_stem and dlc_stem not in stems:
                stems.append(dlc_stem)
            print(f"Auto-found DLC: {os.path.basename(self.dlc_path)}")
            self.load_dlc_tracking(self.dlc_path)
        else:
            self.dlc_path = None
            self.dlc_data = None
            self.dlc_confs = None
            print("No DLC found.")
            
        # 2. Find Probabilities (_probs.csv)
        self.probs = None
        prob_dirs = [
            folder,
            os.path.join(folder, "probs"),
            os.path.join(folder, "prob"),
            os.path.join(os.path.dirname(folder), "probs"),
            os.path.join(os.path.dirname(folder), "prob"),
        ]
        
        found_prob = None
        for pdir in prob_dirs:
            if not os.path.isdir(pdir):
                continue
            # Try matching candidate stems
            for s in stems:
                for cand in [os.path.join(pdir, f"{s}_probs.csv"), os.path.join(pdir, f"{s}probs.csv")]:
                    if os.path.exists(cand):
                        found_prob = cand
                        break
                if found_prob:
                    break
            if found_prob:
                break
                
            # Case-insensitive stem matching in pdir
            try:
                for fname in os.listdir(pdir):
                    fl = fname.lower()
                    if fl.endswith('_probs.csv') or fl.endswith('probs.csv'):
                        for s in stems:
                            if s.lower() in fl or fl.startswith(s.lower()):
                                found_prob = os.path.join(pdir, fname)
                                break
                        if found_prob:
                            break
            except Exception:
                pass
            if found_prob:
                break
                
            # Session-based matching via DatasetMatcher
            try:
                matcher = DatasetMatcher(default_fps=self.video_fps)
                v_info = matcher.parse_filename(base)
                if v_info:
                    for fname in os.listdir(pdir):
                        fl = fname.lower()
                        if fl.endswith('_probs.csv') or fl.endswith('probs.csv'):
                            p_info = matcher.parse_filename(fname)
                            if p_info and str(p_info['animal_id']).upper() == str(v_info['animal_id']).upper() and int(p_info['day']) == int(v_info['day']):
                                found_prob = os.path.join(pdir, fname)
                                break
            except Exception:
                pass
            if found_prob:
                break
                
        if found_prob:
            try:
                prob_df = pd.read_csv(found_prob)
                if prob_df.shape[1] >= 5 or any('approach' in str(c).lower() for c in prob_df.columns):
                    self.set_paradigm("mouse")
                    cols_needed = ['Background', 'Cricket Present', 'First Approach', 'Final Approach', 'Eating']
                    if all(c in prob_df.columns for c in cols_needed):
                        self.probs = prob_df[cols_needed].values
                    else:
                        self.probs = prob_df.values[:, :5]
                else:
                    self.set_paradigm("rat")
                    stalk_col = 'Cricket Present' if 'Cricket Present' in prob_df.columns else ('Stalking' if 'Stalking' in prob_df.columns else prob_df.columns[1])
                    eat_col = 'Eating' if 'Eating' in prob_df.columns else prob_df.columns[3]
                    self.probs = prob_df[['Background', stalk_col, 'Pursuit', eat_col]].values
                print(f"Loaded probabilities from {os.path.basename(found_prob)} (Paradigm={self.paradigm}, Shape={self.probs.shape if self.probs is not None else 'None'})")
            except Exception as e:
                print(f"Failed to load probs: {e}")
        else:
            print(f"No probability file found for {base} in searched directories: {prob_dirs}")
            
        # 3. Find Labels (.xlsx or .csv)
        excel_dirs = [
            folder,
            os.path.join(folder, "excel_files"),
            os.path.join(os.path.dirname(folder), "excel_files")
        ]
        found_label = None
        for edir in excel_dirs:
            if not os.path.isdir(edir):
                continue
            for s in stems:
                for suffix in ['_refined.xlsx', '_refined.csv', '_analyzed.xlsx', '_analyzed.csv']:
                    cand = os.path.join(edir, f"{s}{suffix}")
                    if os.path.exists(cand):
                        found_label = cand
                        break
                if found_label:
                    break
            if found_label:
                break
            
        if found_label:
            self.load_label_file(found_label)
        else:
            self.labels = np.zeros(self.total_frames, dtype=int)
            self.timeline.set_data(self.labels, self.probs, 0)
            self.lbl_info.setText(f"Loaded {base} (No labels found)")
            
        if self.gt_path:
            self.match_and_load_gt()
            
        self.set_frame(0)
        
    def load_label(self):
        if not self.cap: return
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Label File", 
            os.path.dirname(self.video_path) if self.video_path else "", 
            "Label Files (*.xlsx *.csv);;Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self.load_label_file(path)
            
    def load_label_file(self, path):
        self.label_path = path
        try:
            seq = None
            loaded_msg = ""
            
            # Helper to safely convert numerical frame values
            def _safe_int(val, default=None):
                if val is None or pd.isna(val):
                    return default
                try:
                    f = float(val)
                    if np.isnan(f) or np.isinf(f):
                        return default
                    return int(f)
                except (ValueError, TypeError, OverflowError):
                    return default
            
            df = pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)
            df.columns = df.columns.astype(str).str.strip()
            
            # 1. Check for Diane Master Excel / CSV format (Released, Capture, Day)
            if seq is None and any('released' in c.lower() for c in df.columns) and any('capture' in c.lower() for c in df.columns):
                self.set_paradigm("mouse")
                fmt = None if self.label_format == "auto" else self.label_format
                y_true, true_hunts = MouseHunt_Inference.load_ground_truth_sequence(
                    label_path=path,
                    dlc_path=self.dlc_path or self.video_path,
                    num_frames=self.total_frames,
                    fps=self.video_fps,
                    label_format=fmt,
                    paradigm="mouse"
                )
                if y_true is not None and true_hunts:
                    seq = y_true
                    loaded_msg = f"Loaded Diane Master Labels: {os.path.basename(path)} ({len(true_hunts)} hunts)"
                    self.gt_path = path
                else:
                    QMessageBox.warning(
                        self, 
                        "No Match in Master Sheet", 
                        f"Parsed Diane master label file '{os.path.basename(path)}',\n"
                        f"but found no matching session for video '{os.path.basename(self.video_path or '')}'.\n\n"
                        f"Please check animal ID and day in the video filename."
                    )
                    return
            
            # 2. Check for Mouse Analyzed Excel
            if seq is None and 'Enter_Frame' in df.columns and ('First_Approach_Frame' in df.columns or 'Capture_Frame' in df.columns):
                events = []
                for _, row in df.iterrows():
                    enter = _safe_int(row.get('Enter_Frame'))
                    if enter is None:
                        continue
                    events.append({
                        'enter': enter,
                        'first_approach': _safe_int(row.get('First_Approach_Frame')),
                        'final_approach': _safe_int(row.get('Final_Approach_Frame')),
                        'pursuit_start': _safe_int(row.get('Final_Approach_Frame')),
                        'capture': _safe_int(row.get('Capture_Frame')),
                        'consumption_end': _safe_int(row.get('Consumption_End_Frame'))
                    })
                self.set_paradigm("mouse")
                seq = MouseHunt_Inference.events_to_sequence(events, self.total_frames, paradigm="mouse")
                loaded_msg = f"Loaded Mouse Analyzed Labels: {os.path.basename(path)} ({len(events)} hunts)"

            # 3. Check for Dan Format
            if seq is None:
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
                    if not hunts and len(dan_dict) == 1:
                        hunts = next(iter(dan_dict.values()))
                    if hunts:
                        self.set_paradigm("rat")
                        seq = dataset_utils.hunts_to_sequence(hunts, self.total_frames)
                        loaded_msg = f"Loaded Dan Labels: {os.path.basename(path)} ({len(hunts)} hunts)"
                        self.gt_path = path
            
            # 4. Check for Lucas / Analyzed Format
            if seq is None:
                lucas_hunts = dataset_utils.parse_lucas_labels(path)
                if lucas_hunts:
                    self.set_paradigm("rat")
                    seq = dataset_utils.hunts_to_sequence(lucas_hunts, self.total_frames)
                    loaded_msg = f"Loaded Analyzed Labels: {os.path.basename(path)} ({len(lucas_hunts)} hunts)"
                    
            # 5. Check for Frame-by-frame direct predictions/labels
            if seq is None:
                lbl_cols = [c for c in df.columns if c.lower().strip() in ['label', 'labels', 'prediction', 'pred', 'class']]
                if lbl_cols:
                    lbl_col = lbl_cols[0]
                    try:
                        numeric_s = pd.Series(pd.to_numeric(df[lbl_col], errors='coerce')).dropna()
                        vals = numeric_s.values.astype(int)
                        if len(vals) > 0:
                            seq = np.zeros(self.total_frames, dtype=int)
                            n = min(len(vals), self.total_frames)
                            seq[:n] = vals[:n]
                            if np.max(seq) >= 4:
                                self.set_paradigm("mouse")
                            loaded_msg = f"Loaded Frame Labels: {os.path.basename(path)}"
                    except Exception as e:
                        print(f"Could not convert column {lbl_col} to numeric frames: {e}")

            if seq is not None:
                self.labels = seq
                self.timeline.set_data(self.labels, self.probs, self.current_frame)
                self.lbl_info.setText(loaded_msg if loaded_msg else f"Loaded: {os.path.basename(path)}")
                if not self.is_playing and self.cap:
                    self.set_frame(self.current_frame)
                elif hasattr(self, '_update_hud'):
                    self._update_hud()
            else:
                QMessageBox.warning(self, "Warning", f"Could not parse hunts from {os.path.basename(path)}.\nPlease check file columns.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load label file:\n{e}")
            
    def set_frame(self, frame):
        if not self.cap: return
        new_frame = max(0, min(self.total_frames - 1, frame))
        
        # If painting, fill range
        if self.active_paint_label is not None and self.labels is not None and self.last_painted_frame != -1:
            s = min(self.last_painted_frame, new_frame)
            e = max(self.last_painted_frame, new_frame)
            if not np.all(self.labels[s:e+1] == self.active_paint_label):
                self.labels[s:e+1] = self.active_paint_label
                self.timeline.update()
            self.last_painted_frame = new_frame
            
        self.current_frame = new_frame
        
        # 100% Precise Seeking Logic
        internal_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if internal_pos != self.current_frame:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            actual_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            if actual_pos < self.current_frame:
                diff = self.current_frame - actual_pos
                if diff < 1000:
                    for _ in range(diff):
                        self.cap.read()
                        
        ret, img = self.cap.read()
        if ret:
            # 1. Draw DLC Keypoints
            if self.dlc_data is not None:
                mouse_bps, cricket_bps = self._select_display_keypoints()
                
                # Distinct visible palette for mouse (head, midbody, butt, last 3 tail points)
                mouse_colors = [
                    (0, 255, 255),   # Head: Yellow
                    (0, 165, 255),   # Midbody: Orange
                    (0, 0, 255),     # Butt: Red
                    (218, 112, 214), # Tail 1 (3rd from end): Orchid
                    (255, 105, 180), # Tail 2 (near end): Hot Pink
                    (255, 0, 255),   # Tail 3 (tip): Magenta
                ]
                
                # Draw mouse points (smaller dots with thin black outline for visibility, no text clutter)
                for idx, bp_name in enumerate(mouse_bps):
                    pts = self.dlc_data[bp_name]
                    if self.current_frame < len(pts):
                        x, y = pts[self.current_frame]
                        if not np.isnan(x) and not np.isnan(y):
                            col = mouse_colors[idx % len(mouse_colors)]
                            radius = 3
                            cv2.circle(img, (int(x), int(y)), radius + 1, (0, 0, 0), -1)
                            cv2.circle(img, (int(x), int(y)), radius, col, -1)

                # Draw only the single cricket pose with the highest probability
                best_cricket_bp = None
                best_conf = -1.0
                best_coords = None
                
                for bp_name in cricket_bps:
                    pts = self.dlc_data[bp_name]
                    if self.current_frame < len(pts):
                        x, y = pts[self.current_frame]
                        if not np.isnan(x) and not np.isnan(y):
                            conf = 1.0
                            if self.dlc_confs and bp_name in self.dlc_confs:
                                c_val = self.dlc_confs[bp_name][self.current_frame]
                                if not np.isnan(c_val):
                                    conf = float(c_val)
                            if conf > best_conf:
                                best_conf = conf
                                best_cricket_bp = bp_name
                                best_coords = (int(x), int(y))
                
                # Only show cricket if highest confidence is >= 0.5
                if best_coords is not None and best_conf >= 0.5:
                    cricket_thresh = getattr(dataset_utils, 'CRICKET_LIKELIHOOD_THRESH', 0.85)
                    cx, cy = best_coords
                    radius = 4
                    
                    if best_conf < cricket_thresh:
                        # Above 0.5 but below inference default threshold (0.85)
                        cricket_col = (0, 215, 255) # Gold / Amber
                        label_txt = f"cricket:{best_conf:.2f} (below thresh)"
                    else:
                        cricket_col = (0, 255, 0)   # Bright Green
                        label_txt = f"cricket:{best_conf:.2f}"
                        
                    # 1. Render only the circle transparently so the cricket underneath is not obscured
                    x1 = max(0, cx - radius - 2)
                    y1 = max(0, cy - radius - 2)
                    x2 = min(img.shape[1], cx + radius + 3)
                    y2 = min(img.shape[0], cy + radius + 3)
                    
                    if x2 > x1 and y2 > y1:
                        roi = img[y1:y2, x1:x2]
                        overlay = roi.copy()
                        rcx, rcy = cx - x1, cy - y1
                        cv2.circle(overlay, (rcx, rcy), radius + 1, (0, 0, 0), -1)
                        cv2.circle(overlay, (rcx, rcy), radius, cricket_col, -1)
                        alpha = 0.35
                        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)
                    
                    # 2. Render cricket text label fully opaque next to the circle
                    tx, ty = cx + 6, cy - 6
                    cv2.putText(img, label_txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(img, label_txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, cricket_col, 1, cv2.LINE_AA)
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            pixmap = QPixmap.fromImage(qimg)
            self.video_lbl.setPixmap(pixmap.scaled(self.video_lbl.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self._update_hud()
        else:
            if hasattr(self, 'hud_lbl'):
                self.hud_lbl.hide()
            
        cur_sec = self.current_frame / self.video_fps if self.video_fps > 0 else 0
        tot_sec = self.total_frames / self.video_fps if self.video_fps > 0 else 0
        self.lbl_frame.setText(f"Frame: {self.current_frame} / {self.total_frames} ({self._format_time(cur_sec)} / {self._format_time(tot_sec)})")
        self.timeline.set_current_frame(self.current_frame)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Converts elapsed seconds to HH:MM:SS string."""
        s = max(0, int(seconds))
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:02d}"

    def _update_hud(self):
        """Renders HUD text (Pred, Manual, Brush) cleanly in the top-left black box outside the video frame."""
        if not hasattr(self, 'hud_lbl') or self.labels is None:
            if hasattr(self, 'hud_lbl'):
                self.hud_lbl.hide()
            return
            
        if self.paradigm == "mouse":
            cls_names = CLASS_NAMES_MOUSE
            cls_colors = {
                0: "#B0B0B0", # Background (Gray)
                1: "#00E5FF", # Cricket Present (Teal / Cyan)
                2: "#2196F3", # First Approach (Steel Blue)
                3: "#FF1744", # Final Approach (Crimson Red)
                4: "#00E676"  # Eating (Forest Green)
            }
        else:
            cls_names = CLASS_NAMES_RAT
            cls_colors = {
                0: "#B0B0B0", # Background (Gray)
                1: "#FFA726", # Stalking (Orange)
                2: "#FF1744", # Pursuit (Red)
                3: "#00E676"  # Eating (Green)
            }
            
        lbl_val = self.labels[self.current_frame] if self.current_frame < len(self.labels) else 0
        pred_name = cls_names.get(lbl_val, 'Unknown')
        pred_col = cls_colors.get(lbl_val, '#FFFFFF')
        
        gt_html = ""
        if self.gt_labels is not None and self.current_frame < len(self.gt_labels):
            gt_val = self.gt_labels[self.current_frame]
            gt_name = cls_names.get(gt_val, 'Unknown')
            gt_col = cls_colors.get(gt_val, '#FFFFFF')
            gt_html = f"<div><span style='color: #8E8E93;'>Manual:</span> <span style='color: {gt_col};'>{gt_name}</span></div>"
            
        brush_names = {None: 'No Change', **cls_names}
        brush_colors = {None: '#FFD600', **cls_colors}
        brush_name = brush_names.get(self.active_paint_label, 'No Change')
        brush_col = brush_colors.get(self.active_paint_label, '#FFD600')
        
        html = (
            f"<div style=\"font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 13px; font-weight: bold; line-height: 1.35;\">"
            f"<div><span style='color: #8E8E93;'>Pred:</span> <span style='color: {pred_col};'>{pred_name}</span></div>"
            f"{gt_html}"
            f"<div><span style='color: #8E8E93;'>Brush:</span> <span style='color: {brush_col};'>{brush_name}</span></div>"
            f"</div>"
        )
        self.hud_lbl.setText(html)
        self.hud_lbl.adjustSize()
        self.hud_lbl.show()
        
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
        
        if self.paradigm == "mouse":
            # Mouse 5-Class: 0: BG, 1: Cricket Present, 2: First App, 3: Final App, 4: Eat
            in_hunt = False
            curr = None
            for s, e in zip(starts, ends):
                val = lbls[s]
                
                # Back-to-back hunt: transition from Eating (4) directly to CP (1), FA (2), or Final App (3)
                if in_hunt and curr and curr['capture'] is not None and val in (1, 2, 3):
                    curr['consumption_end'] = s
                    events.append(curr)
                    curr = {
                        'enter': s,
                        'first_approach': s if val == 2 else None,
                        'final_approach': s if val == 3 else None,
                        'pursuit_start': s if val == 3 else None,
                        'capture': None,
                        'consumption_end': None,
                        'remarks': 'Manual Refined (Mouse 5-Class)'
                    }
                    in_hunt = True
                    continue
                    
                if val != 0 and not in_hunt:
                    in_hunt = True
                    curr = {
                        'enter': s,
                        'first_approach': s if val == 2 else None,
                        'final_approach': s if val == 3 else None,
                        'pursuit_start': s if val == 3 else None,
                        'capture': s if val == 4 else None,
                        'consumption_end': None,
                        'remarks': 'Manual Refined (Mouse 5-Class)'
                    }
                    
                if in_hunt and curr is not None:
                    if val == 2 and curr['first_approach'] is None:
                        curr['first_approach'] = s
                    elif val == 3 and curr['final_approach'] is None:
                        curr['final_approach'] = s
                        curr['pursuit_start'] = s
                    elif val == 4 and curr['capture'] is None:
                        curr['capture'] = s
                        
                    if val == 0:
                        curr['consumption_end'] = s
                        events.append(curr)
                        in_hunt = False
                        curr = None
                        
            if in_hunt and curr:
                curr['consumption_end'] = len(lbls)
                events.append(curr)
        else:
            # Rat 4-Class: 0: BG, 1: Stalking, 2: Pursuit, 3: Eat
            in_hunt = False
            current_event = None
            for s, e in zip(starts, ends):
                val = lbls[s]
                if in_hunt and current_event and current_event['capture'] != -1 and val in (1, 2):
                    current_event['capture_end'] = s
                    if current_event['pursuit'] == -1:
                        current_event['pursuit'] = current_event['enter']
                    events.append(current_event)
                    current_event = {'enter': s, 'pursuit': s if val == 2 else -1, 'capture': -1, 'capture_end': -1}
                    in_hunt = True
                    continue

                if val != 0 and not in_hunt:
                    in_hunt = True
                    current_event = {'enter': s, 'pursuit': s if val == 2 else -1, 'capture': -1, 'capture_end': -1}
                    
                if in_hunt and current_event is not None:
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
                if self.paradigm == "mouse":
                    MouseHunt_Inference.generate_mouse_analyzed_excel(events, path, self.video_fps)
                else:
                    RatHunt_Inference.generate_analyzed_excel(events, path, self.video_fps)
                QMessageBox.information(self, "Success", f"Saved {len(events)} hunt event(s) to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ReviewGUI()
    window.show()
    sys.exit(app.exec())

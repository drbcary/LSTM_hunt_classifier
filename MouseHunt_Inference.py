"""
MouseHunt_Inference.py
Modular, paradigm-configurable inference script for mouse hunting behavior analysis.
Supports multi-segment tail tracking, dynamic video FPS extraction, and extensible feature sets.
"""

import os
import re
import glob
from datetime import timedelta
import warnings

# Compatibility patch for debugpy / pydevd Qt event loop integration with PySide6
try:
    import PySide6.QtWidgets, PySide6.QtGui
    if not hasattr(PySide6.QtGui, 'QApplication'):
        setattr(PySide6.QtGui, 'QApplication', PySide6.QtWidgets.QApplication)
except ImportError:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional, Dict, Any, List, Union, Tuple, overload, Literal
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
try:
    import scipy.stats as sp_stats
except ImportError:
    sp_stats = None

from config.mouse_config import MouseParadigmConfig, MousePoseSchema
from config.base_config import sec_to_frames, frames_to_sec
from pose.dlc_adapter import DLCAdapter, extract_filename_timestamp
from features import generate_features
from dataset.matcher import DatasetMatcher
from labels.base_parser import HuntEvent
from LSTM_model_training import HuntingBiLSTM

warnings.filterwarnings('ignore')

# Canonical default 5-class mouse transition matrix
_DEFAULT_MOUSE_TRANS_MAT = np.array([
    #  0:BG    1:CP    2:FA    3:Att   4:Eat
    [0.9990, 0.0010, 0.0000, 0.0000, 0.0000],  # 0: BG (strictly only transitions to 0 or 1)
    [0.0100, 0.9650, 0.0200, 0.0050, 0.0000],  # 1: Cricket Present
    [0.0100, 0.0100, 0.9500, 0.0300, 0.0000],  # 2: First Approach
    [0.0050, 0.0050, 0.0100, 0.9400, 0.0400],  # 3: Final Approach
    [0.0200, 0.0050, 0.0000, 0.0000, 0.9750]   # 4: Eating (strictly transitions to 0, 1, or 4)
], dtype=np.float64)
_DEFAULT_MOUSE_TRANS_MAT /= _DEFAULT_MOUSE_TRANS_MAT.sum(axis=1, keepdims=True)

_EMPIRICAL_MOUSE_TRANS_MAT = None

class MouseInferencePipeline:
    """
    End-to-end inference pipeline for mouse hunting sessions.
    Handles video property extraction, DLC parsing with multi-tail support,
    feature extraction, BiLSTM model prediction, and Viterbi sequence decoding.
    """
    def __init__(
        self,
        model_weights_path: Optional[str] = None,
        config: Optional[MouseParadigmConfig] = None,
        device: Optional[torch.device] = None,
        max_mem: Optional[float] = None,
        continuous_cricket_confidence: Optional[bool] = None
    ):
        self.config = config or MouseParadigmConfig()
        self.continuous_cricket_confidence = continuous_cricket_confidence
        if continuous_cricket_confidence is not None:
            self.config.continuous_cricket_confidence = continuous_cricket_confidence
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_mem = max_mem
        self.adapter = DLCAdapter(self.config)
        self.matcher = DatasetMatcher(
            extract_video_fps=self.config.extract_video_fps,
            default_fps=self.config.default_fps,
            default_width=self.config.default_width,
            default_height=self.config.default_height
        )
        self.model: Optional[HuntingBiLSTM] = None
        self.model_weights_path = model_weights_path
        self.target_fps: float = 30.0
        if self.model_weights_path and os.path.exists(self.model_weights_path):
            self._inspect_checkpoint()

    def _inspect_checkpoint(self):
        if not (self.model_weights_path and os.path.exists(self.model_weights_path)):
            return
        try:
            ckpt = torch.load(self.model_weights_path, map_location='cpu')
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                self.target_fps = float(ckpt.get('target_fps', self.target_fps))
                if 'continuous_cricket_confidence' in ckpt:
                    ckpt_cont = bool(ckpt['continuous_cricket_confidence'])
                    if self.continuous_cricket_confidence is None:
                        self.continuous_cricket_confidence = ckpt_cont
                        self.config.continuous_cricket_confidence = ckpt_cont
                        print(f"Auto-configured continuous_cricket_confidence={ckpt_cont} from model checkpoint.")
                else:
                    if self.continuous_cricket_confidence is None:
                        self.continuous_cricket_confidence = False
                        self.config.continuous_cricket_confidence = False
            self.adapter = DLCAdapter(self.config)
        except Exception:
            pass

    def init_model(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 3, num_classes: int = 5):
        if self.model_weights_path and os.path.exists(self.model_weights_path):
            ckpt = torch.load(self.model_weights_path, map_location=self.device)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                self.target_fps = float(ckpt.get('target_fps', 30.0))
                num_classes = int(ckpt.get('num_classes', num_classes))
                hidden_dim = int(ckpt.get('hidden_dim', hidden_dim))
                num_layers = int(ckpt.get('num_layers', num_layers))
                if 'continuous_cricket_confidence' in ckpt:
                    ckpt_cont = bool(ckpt['continuous_cricket_confidence'])
                    if self.continuous_cricket_confidence is None:
                        self.config.continuous_cricket_confidence = ckpt_cont
                        self.continuous_cricket_confidence = ckpt_cont
                        print(f"Auto-configured continuous_cricket_confidence={ckpt_cont} from model checkpoint.")
                else:
                    if self.continuous_cricket_confidence is None:
                        self.continuous_cricket_confidence = False
                        self.config.continuous_cricket_confidence = False
                self.model = HuntingBiLSTM(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    num_classes=num_classes
                ).to(self.device)
                self.model.load_state_dict(ckpt['model_state_dict'])
                print(f"Loaded rich model checkpoint from {self.model_weights_path} (Trained at target_fps={self.target_fps}, num_classes={num_classes})")
            else:
                self.model = HuntingBiLSTM(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    num_classes=num_classes
                ).to(self.device)
                self.model.load_state_dict(ckpt)
                print(f"Loaded legacy model weights from {self.model_weights_path}")
            self.model.eval()
        else:
            self.model = HuntingBiLSTM(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_classes=num_classes
            ).to(self.device)
            print(f"Initialized untrained model with input_dim={input_dim}, num_classes={num_classes}")

    def run_on_session(
        self,
        dlc_path: Union[str, List[str]],
        video_path: Optional[str] = None,
        label_path: Optional[str] = None,
        label_format: Optional[str] = None,
        window_sec: Optional[float] = None,
        step_sec: Optional[float] = None,
        refine_events: bool = True,
        transition_weight: float = 5.0,
        temperature: float = 1.0,
        max_mem: Optional[float] = None,
        include_blurry_cricket: Optional[bool] = None,
        include_cricket_immo: Optional[bool] = None,
        filter_poops: Optional[bool] = None,
        poop_thresh: Optional[float] = None,
        poop_dist_thresh: Optional[float] = None,
        cricket_thresh: Optional[float] = None,
        cricket_coord_thresh: Optional[float] = None,
        pred_thresh: Optional[float] = None,
        continuous_cricket_confidence: Optional[bool] = None,
        window_size: Optional[int] = None,
        step_size: Optional[int] = None,
        paradigm: str = "mouse",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Runs complete inference pipeline on a single session.
        Returns probabilities, frame predictions, extracted events, and metadata.
        """
        if continuous_cricket_confidence is not None:
            self.config.continuous_cricket_confidence = continuous_cricket_confidence

        # Configure GPU memory limit if requested
        effective_max_mem = max_mem if max_mem is not None else self.max_mem
        if effective_max_mem is not None and torch.cuda.is_available():
            try:
                dev_idx = self.device.index if (self.device.type == "cuda" and self.device.index is not None) else 0
                torch.cuda.set_per_process_memory_fraction(float(effective_max_mem), device=dev_idx)
                print(f"Set CUDA per-process memory fraction to {effective_max_mem:.1%} on device={dev_idx}")
            except Exception as e:
                warnings.warn(f"Could not set CUDA per-process memory fraction: {e}")

        first_dlc = dlc_path[0] if isinstance(dlc_path, (list, tuple)) else dlc_path
        session_meta = self.matcher.match_session(first_dlc, video_path=video_path)
        fps = session_meta.fps if session_meta else self.config.default_fps
        width = session_meta.width if session_meta else self.config.default_width

        if window_size is not None and window_sec is None:
            window_sec = float(window_size) / float(fps)
        if step_size is not None and step_sec is None:
            step_sec = float(step_size) / float(fps)

        print(f"Processing session: {first_dlc}")
        print(f"Video properties: {width}px width @ {fps:.2f} FPS")

        # 1. Load Pose Data
        pose_data = self.adapter.load(
            dlc_path,
            video_fps=fps,
            include_blurry_cricket=include_blurry_cricket,
            include_cricket_immo=include_cricket_immo,
            filter_poops=filter_poops,
            poop_thresh=poop_thresh,
            poop_dist_thresh=poop_dist_thresh,
            cricket_thresh=cricket_thresh,
            cricket_coord_thresh=cricket_coord_thresh,
            pred_thresh=pred_thresh
        )
        if pose_data is None:
            raise ValueError(f"Could not load tracking data from {dlc_path}")

        # Normalize by the DLC crop box (approximately the arena) rather than the full camera frame
        if pose_data.crop_offset.width:
            print(f"Using DLC crop width {pose_data.crop_offset.width:g}px (from {pose_data.crop_offset.source}) instead of video width {width}px for feature scaling")
            width = pose_data.crop_offset.width

        # 2. Extract Features
        features, manifest = generate_features(
            pose_data=pose_data,
            video_width=width,
            video_fps=fps,
            enabled_blocks=self.config.enabled_feature_blocks,
            config=self.config
        )
        num_frames, feat_dim = features.shape

        # Initialize model if not yet initialized
        if self.model is None:
            self.init_model(input_dim=feat_dim)

        # 3. Canonical Target FPS Resampling (if test video FPS differs from model's trained target_fps)
        from dataset_utils import resample_feature_matrix
        needs_resample = abs(fps - self.target_fps) > 0.1
        if needs_resample:
            eval_features = resample_feature_matrix(features, fps, self.target_fps)
            eval_fps = self.target_fps
        else:
            eval_features = features
            eval_fps = fps

        eval_num_frames = len(eval_features)

        # 4. Sliding Window Inference
        w_sec = window_sec or self.config.time.inference_window_sec
        s_sec = step_sec or self.config.time.inference_step_sec
        w_frames = sec_to_frames(w_sec, eval_fps)
        s_frames = sec_to_frames(s_sec, eval_fps)

        num_classes = getattr(self.model, 'num_classes', 5)
        features_t = torch.tensor(eval_features, dtype=torch.float32).to(self.device)
        total_probs = np.zeros((eval_num_frames, num_classes), dtype=np.float32)
        counts = np.zeros((eval_num_frames, 1), dtype=np.float32)

        assert self.model is not None
        starts = list(range(0, eval_num_frames, s_frames))
        with torch.no_grad():
            for st in tqdm(starts, desc="Model Inference", unit="window"):
                en = min(st + w_frames, eval_num_frames)
                if (en - st) < 10:
                    continue

                chunk = features_t[st:en].unsqueeze(0)
                lens = torch.tensor([en - st]).cpu()
                logits = self.model(chunk, lens)
                logits_scaled = logits / temperature if temperature != 1.0 else logits
                probs = torch.softmax(logits_scaled, dim=2)[0].cpu().numpy()

                chunk_len = en - st
                # Gaussian tapering weights
                t = np.linspace(-1, 1, chunk_len)
                weights = np.exp(-0.5 * (t / 0.5) ** 2)[:, np.newaxis]

                total_probs[st:en] += probs * weights
                counts[st:en] += weights

        counts[counts == 0] = 1.0
        norm_probs_eval = total_probs / counts

        # Map back to original native video frames if resampled
        if needs_resample:
            t_eval = np.linspace(0, 1, eval_num_frames)
            t_orig = np.linspace(0, 1, num_frames)
            norm_probs = np.empty((num_frames, num_classes), dtype=np.float32)
            for c in range(num_classes):
                norm_probs[:, c] = np.interp(t_orig, t_eval, norm_probs_eval[:, c])
        else:
            norm_probs = norm_probs_eval

        raw_preds = np.argmax(norm_probs, axis=1)

        # 5. Optional HMM Viterbi Refinement
        refined_events = None
        refined_path = None
        if refine_events:
            refined_events, refined_path = refine_predictions(
                norm_probs, raw_preds, transition_weight=transition_weight, paradigm="mouse", return_path=True
            )

        # 6. Optional Ground Truth Loading
        gt_events = None
        gt_seq = None
        if label_path and os.path.exists(label_path):
            gt_seq, gt_events = load_ground_truth_sequence(
                label_path=label_path,
                dlc_path=first_dlc,
                num_frames=num_frames,
                fps=fps,
                label_format=label_format,
                paradigm="mouse"
            )

        return {
            'metadata': session_meta,
            'pose_data': pose_data,
            'features': features,
            'manifest': manifest,
            'probabilities': norm_probs,
            'raw_predictions': raw_preds,
            'refined_events': refined_events,
            'refined_path': refined_path,
            'ground_truth_events': gt_events,
            'ground_truth_sequence': gt_seq,
            'fps': fps,
            'target_fps': self.target_fps,
            'num_frames': num_frames
        }

    def create_video_clip(
        self,
        video_path: str,
        probs: np.ndarray,
        preds: np.ndarray,
        start_frame: int,
        end_frame: int,
        output_path: str,
        dlc_path: Optional[Union[str, List[str]]] = None,
        **kwargs
    ):
        """Creates an annotated hunting video clip with mouse 5-class HUD and posture overlays."""
        return create_mouse_hunting_video_clip(
            video_path=video_path,
            probs=probs,
            preds=preds,
            start_frame=start_frame,
            end_frame=end_frame,
            output_path=output_path,
            dlc_path=dlc_path,
            **kwargs
        )

def viterbi_decode(
    probs: np.ndarray,
    transition_matrix: Optional[np.ndarray] = None,
    transition_weight: float = 1.0
) -> np.ndarray:
    """
    Decodes the most likely state sequence using Viterbi algorithm in log space.
    transition_weight > 1.0 penalizes off-diagonal transitions, making states more persistent.
    Supports arbitrary number of states (e.g. 5 for mouse, 4 for rat).
    """
    num_frames, num_states = probs.shape
    if transition_matrix is None:
        if num_states == 5:
            transition_matrix = _DEFAULT_MOUSE_TRANS_MAT
        else:
            transition_matrix = np.eye(num_states) * 0.98 + 0.02 / num_states
    assert transition_matrix is not None

    if transition_weight != 1.0:
        trans_mat = transition_matrix.copy()
        for i in range(num_states):
            non_diag = [j for j in range(num_states) if j != i]
            trans_mat[i, non_diag] /= transition_weight
        row_sums = trans_mat.sum(axis=1, keepdims=True)
        trans_mat = trans_mat / np.maximum(row_sums, 1e-12)
    else:
        trans_mat = transition_matrix

    log_probs = np.log(np.clip(probs, 1e-10, 1.0))
    # Strict forbidden mask: zero-probability transitions are assigned -1e9 so they are never traversed
    log_trans = np.where(trans_mat <= 0.0, -1e9, np.log(np.clip(trans_mat, 1e-10, 1.0)))

    dp = np.zeros((num_frames, num_states))
    backpointers = np.zeros((num_frames, num_states), dtype=int)
    dp[0] = log_probs[0]

    for t in range(1, num_frames):
        score = dp[t - 1, :, None] + log_trans
        dp[t] = np.max(score, axis=0) + log_probs[t]
        backpointers[t] = np.argmax(score, axis=0)

    path = np.zeros(num_frames, dtype=int)
    path[-1] = np.argmax(dp[-1])
    for t in range(num_frames - 2, -1, -1):
        path[t] = backpointers[t + 1, path[t + 1]]

    return path

def extract_events_from_path(path: np.ndarray, paradigm: str = "mouse") -> List[Dict[str, Any]]:
    """
    Extracts structured hunting event dictionaries from a decoded state sequence path.
    For Mouse (5 classes):
      - enter: cricket entrance / hunt start
      - first_approach: first approach phase start
      - pursuit_start: final approach / strike start
      - capture: eating start
      - consumption_end: eating finish
    """
    events = []
    num_frames = len(path)
    if num_frames == 0:
        return events

    is_mouse = (paradigm == "mouse" or np.max(path) == 4)
    eat_class = 4 if is_mouse else 3
    pursuit_class = 3 if is_mouse else 2
    first_app_class = 2 if is_mouse else None

    # Detect all hunt starts:
    # 1. Entering from Background (0 -> >0)
    # 2. Back-to-back hunt: direct transition from Eating to approach
    starts = []
    for t in range(num_frames):
        if path[t] > 0:
            if t == 0 or path[t - 1] == 0:
                starts.append(t)
            elif path[t - 1] == eat_class and path[t] in ([1, 2, 3] if is_mouse else [1, 2]):
                starts.append(t)

    # For each hunt start, find its termination frame
    for i, s in enumerate(starts):
        if i + 1 < len(starts):
            next_s = starts[i + 1]
            sub = path[s:next_s]
            zeros = np.where(sub == 0)[0]
            e = s + zeros[0] if len(zeros) > 0 else next_s
        else:
            sub = path[s:]
            zeros = np.where(sub == 0)[0]
            e = s + zeros[0] if len(zeros) > 0 else num_frames

        block = path[s:e]

        if is_mouse:
            first_idx = np.where(block == first_app_class)[0]
            first_approach = s + first_idx[0] if len(first_idx) > 0 else None

            pursuit_idx = np.where(block == pursuit_class)[0]
            pursuit_start = s + pursuit_idx[0] if len(pursuit_idx) > 0 else None

            eat_idx = np.where(block == eat_class)[0]
            capture = s + eat_idx[0] if len(eat_idx) > 0 else None

            events.append({
                'enter': s,
                'first_approach': first_approach,
                'pursuit_start': pursuit_start,
                'capture': capture,
                'consumption_end': e,
                'remarks': 'HMM Decoded (Mouse 5-Class)'
            })
        else:
            pursuit_idx = np.where(block == pursuit_class)[0]
            if len(pursuit_idx) == 0:
                continue
            pursuit_start = s + pursuit_idx[0]

            eat_idx = np.where(block == eat_class)[0]
            capture = s + eat_idx[0] if len(eat_idx) > 0 else e

            events.append({
                'enter': s,
                'pursuit': pursuit_start,
                'capture': capture,
                'capture_end': e,
                'remarks': 'HMM Decoded (Rat 4-Class)'
            })

    return events

@overload
def refine_predictions(
    probs: np.ndarray,
    preds: Optional[np.ndarray] = ...,
    transition_weight: float = ...,
    transition_matrix: Optional[np.ndarray] = ...,
    paradigm: str = ...,
    return_path: Literal[False] = ...
) -> List[Dict[str, Any]]: ...

@overload
def refine_predictions(
    probs: np.ndarray,
    preds: Optional[np.ndarray] = ...,
    transition_weight: float = ...,
    transition_matrix: Optional[np.ndarray] = ...,
    paradigm: str = ...,
    return_path: Literal[True] = ...
) -> Tuple[List[Dict[str, Any]], np.ndarray]: ...

def refine_predictions(
    probs: np.ndarray,
    preds: Optional[np.ndarray] = None,
    transition_weight: float = 5.0,
    transition_matrix: Optional[np.ndarray] = None,
    paradigm: str = "mouse",
    return_path: bool = False
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], np.ndarray]]:
    """
    Refines probability predictions using a strict chronological HMM Viterbi Decoder.
    Supports mouse 5-class and rat 4-class models.
    """
    global _EMPIRICAL_MOUSE_TRANS_MAT
    num_classes = probs.shape[1]
    is_mouse = (paradigm == "mouse" or num_classes == 5)

    if transition_matrix is None:
        if is_mouse:
            if _EMPIRICAL_MOUSE_TRANS_MAT is None:
                try:
                    import dataset_utils
                    _EMPIRICAL_MOUSE_TRANS_MAT = dataset_utils.calculate_empirical_transition_matrix(paradigm="mouse")
                    print("Loaded empirical mouse transition matrix from training data.")
                except Exception as e:
                    print(f"Using default mouse transition matrix ({e})")
                    _EMPIRICAL_MOUSE_TRANS_MAT = _DEFAULT_MOUSE_TRANS_MAT
            trans_mat = _EMPIRICAL_MOUSE_TRANS_MAT
        else:
            try:
                import dataset_utils
                trans_mat = dataset_utils.calculate_empirical_transition_matrix(paradigm="rat")
            except Exception:
                trans_mat = np.array([
                    [0.999, 0.001, 0.0,   0.0  ],
                    [0.05,  0.94,  0.01,  0.0  ],
                    [0.05,  0.0,   0.94,  0.01 ],
                    [0.05,  0.005, 0.001, 0.944]
                ])
    else:
        trans_mat = transition_matrix

    path = viterbi_decode(probs, trans_mat, transition_weight=transition_weight)
    events = extract_events_from_path(path, paradigm="mouse" if is_mouse else "rat")

    if return_path:
        return events, path
    return events

def _safe_frame_int(val: Any) -> Optional[int]:
    """Safely converts a value to integer, returning None if NaN, None, or invalid."""
    if val is None or pd.isna(val):
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None

def events_to_sequence(events: List[Any], num_frames: int, paradigm: str = "mouse") -> np.ndarray:
    """Converts structured hunt event list (HuntEvent objects or dicts) into an integer behavioral label sequence."""
    seq = np.zeros(num_frames, dtype=int)
    is_mouse = (paradigm == "mouse")
    for event in events:
        if isinstance(event, HuntEvent):
            raw_enter = event.enter
            consumption_end = event.consumption_end
            raw_first = event.first_approach
            raw_pursuit = event.approach_start
            raw_capture = event.capture
        elif isinstance(event, dict):
            raw_enter = event.get('enter', event.get('released', 0))
            consumption_end = event.get('consumption_end', event.get('capture_end', event.get('eaten')))
            raw_first = event.get('first_approach')
            raw_pursuit = event.get('pursuit_start', event.get('pursuit', event.get('approach_start', event.get('final_approach'))))
            raw_capture = event.get('capture')
        else:
            continue

        enter_int = _safe_frame_int(raw_enter)
        enter = max(0, min(num_frames, enter_int if enter_int is not None else 0))

        cons_int = _safe_frame_int(consumption_end)
        eating_end = max(enter, min(num_frames, cons_int)) if cons_int is not None else None

        first_app = _safe_frame_int(raw_first)
        pursuit = _safe_frame_int(raw_pursuit)
        capture = _safe_frame_int(raw_capture)

        if is_mouse:
            # 1: Cricket Present (from enter until first approach, pursuit, or capture)
            cp_end = first_app if first_app is not None else (pursuit if pursuit is not None else (capture if capture is not None else (eating_end if eating_end is not None else enter)))
            cp_end = max(enter, min(num_frames, cp_end))
            if cp_end > enter:
                seq[enter:cp_end] = 1

            # 2: First Approach
            if first_app is not None:
                first_app = max(enter, min(num_frames, first_app))
                fa_end = pursuit if pursuit is not None else (capture if capture is not None else (eating_end if eating_end is not None else first_app))
                fa_end = max(first_app, min(num_frames, fa_end))
                if fa_end > first_app:
                    seq[first_app:fa_end] = 2

            # 3: Final Approach / Strike
            if pursuit is not None:
                pursuit = max(enter, min(num_frames, pursuit))
                strike_end = capture if capture is not None else (eating_end if eating_end is not None else pursuit)
                strike_end = max(pursuit, min(num_frames, strike_end))
                if strike_end > pursuit:
                    seq[pursuit:strike_end] = 3

            # 4: Eating
            if capture is not None:
                capture = max(enter, min(num_frames, capture))
                end_eat = eating_end if eating_end is not None else min(num_frames, capture + 300)
                end_eat = max(capture, min(num_frames, end_eat))
                if end_eat > capture:
                    seq[capture:end_eat] = 4
        else:
            capture = capture if capture is not None else num_frames
            capture = max(enter, min(num_frames, capture))
            end_eat = eating_end if eating_end is not None else min(num_frames, capture + 300)
            end_eat = max(capture, min(num_frames, end_eat))

            if pursuit is not None and pursuit > enter:
                pursuit = min(capture, max(enter, pursuit))
                seq[enter:pursuit] = 1           # Stalking / Cricket Present (Orange)
                seq[pursuit:capture] = 2         # Pursuit / Attack (Red)
            else:
                seq[enter:capture] = 1           # Entire pre-capture duration is Stalking (Orange)

            if end_eat > capture:
                seq[capture:end_eat] = 3         # Eating (Green)

    return seq

def load_ground_truth_sequence(
    label_path: str,
    dlc_path: Optional[str] = None,
    num_frames: Optional[int] = None,
    fps: float = 30.0,
    label_format: Optional[str] = None,
    paradigm: str = "mouse"
) -> Tuple[Optional[np.ndarray], Optional[List[Dict[str, Any]]]]:
    """
    Parses a ground truth label file (Diane master Excel/CSV, Lucas, or Dan format)
    and maps it to the session specified by dlc_path.
    Returns:
        (true_seq, valid_hunts)
    """
    if not os.path.exists(label_path):
        print(f"Warning: Label file does not exist: {label_path}")
        return None, None

    is_mouse = (paradigm == "mouse")

    # 1. Mouse / Diane Master Excel/CSV format
    if is_mouse and (label_format in (None, 'diane', 'diane_parts') or label_path.endswith('.xlsx') or label_path.endswith('.csv')):
        try:
            from labels.diane_parser import DianeExcelLabelParser
            parser = DianeExcelLabelParser(default_fps=fps)
            is_parts_fmt = (label_format == 'diane_parts')

            valid_hunts = []
            if dlc_path:
                matcher = DatasetMatcher(default_fps=fps)
                parsed = matcher.parse_filename(dlc_path)
                if parsed:
                    clean_target = str(parsed['animal_id']).upper().replace("YFP-", "").replace("YFP_", "").replace("YFP", "").replace("YB-", "").replace("YB_", "").replace("YB", "").strip()
                    try:
                        target_day = int(float(parsed['day']))
                    except (ValueError, TypeError):
                        target_day = 1

                    if is_parts_fmt:
                        session_parts = parser.parse_file_parts(label_path, fps=fps)
                        target_parts = None
                        for k, parts_list in session_parts.items():
                            k_animal = k[0].upper().replace("YFP-", "").replace("YFP_", "").replace("YFP", "").replace("YB-", "").replace("YB_", "").replace("YB", "").strip()
                            try:
                                k_day = int(float(k[1]))
                            except (ValueError, TypeError):
                                k_day = 1
                            if k_animal == clean_target and k_day == target_day:
                                target_parts = parts_list
                                break

                        if target_parts:
                            if len(target_parts) == 1:
                                valid_hunts = target_parts[0]
                                print(f"Matched manual Diane parts labels for Animal '{parsed['animal_id']}', Day {target_day}: {len(valid_hunts)} hunts found.")
                            else:
                                # Determine unique video recordings among siblings in the same folder
                                parent_dir = os.path.dirname(dlc_path)
                                candidate_files = glob.glob(os.path.join(parent_dir, "*DLC*.csv")) + glob.glob(os.path.join(parent_dir, "*DLC*.h5"))
                                candidate_files = [f for f in candidate_files if "labeledinds" not in f.lower() and "labeledbouts" not in f.lower()]
                                candidate_files = sorted(candidate_files, key=lambda p: (0 if p.lower().endswith('.csv') else 1))
                                
                                # Deduplicate by recording stem (so .csv and .h5 for the same video are not treated as separate parts)
                                seen_stems = set()
                                sibling_files = []
                                for cand in candidate_files:
                                    cand_info = matcher.parse_filename(cand)
                                    if cand_info:
                                        c_animal = str(cand_info['animal_id']).upper().replace("YFP-", "").replace("YFP_", "").replace("YFP", "").replace("YB-", "").replace("YB_", "").replace("YB", "").strip()
                                        if c_animal == clean_target and int(float(cand_info['day'])) == target_day:
                                            stem = re.sub(r'DLC.*(?:\.csv|\.h5)$', '', os.path.basename(cand), flags=re.IGNORECASE)
                                            if stem not in seen_stems:
                                                seen_stems.add(stem)
                                                sibling_files.append(cand)
                                                
                                sibling_files = sorted(sibling_files, key=extract_filename_timestamp)
                                
                                if len(sibling_files) <= 1:
                                    all_parsed = parser.parse_file(label_path, fps=fps)
                                    all_session_hunts = None
                                    for k, h_list in all_parsed.items():
                                        k_animal = k[0].upper().replace("YFP-", "").replace("YFP_", "").replace("YFP", "").replace("YB-", "").replace("YB_", "").replace("YB", "").strip()
                                        try:
                                            k_day = int(float(k[1]))
                                        except (ValueError, TypeError):
                                            k_day = 1
                                        if k_animal == clean_target and k_day == target_day:
                                            all_session_hunts = h_list
                                            break
                                    if all_session_hunts:
                                        valid_hunts = all_session_hunts
                                        print(f"Matched manual Diane labels for Animal '{parsed['animal_id']}', Day {target_day} (Single continuous recording, {len(valid_hunts)} hunts found).")
                                    else:
                                        valid_hunts = [h for part in target_parts for h in part]
                                else:
                                    dlc_stem = re.sub(r'DLC.*(?:\.csv|\.h5)$', '', os.path.basename(dlc_path), flags=re.IGNORECASE)
                                    sibling_stems = [re.sub(r'DLC.*(?:\.csv|\.h5)$', '', os.path.basename(f), flags=re.IGNORECASE) for f in sibling_files]
                                    try:
                                        part_idx = sibling_stems.index(dlc_stem)
                                    except ValueError:
                                        part_idx = 0

                                    if part_idx < len(target_parts):
                                        valid_hunts = target_parts[part_idx]
                                        print(f"Matched manual Diane parts labels for Animal '{parsed['animal_id']}', Day {target_day}, Part {part_idx+1}: {len(valid_hunts)} hunts found.")
                                    else:
                                        valid_hunts = target_parts[0]
                                        print(f"Warning: Part index {part_idx+1} exceeds parsed parts ({len(target_parts)}) for Animal '{parsed['animal_id']}', Day {target_day}. Using Part 1.")
                        else:
                            print(f"Notice: Could not find matching session (Animal={parsed['animal_id']}, Day={target_day}) in {label_path}.")
                    else:
                        session_hunts = parser.parse_file(label_path, fps=fps)
                        for k, hunts in session_hunts.items():
                            k_animal = k[0].upper().replace("YFP-", "").replace("YFP_", "").replace("YFP", "").replace("YB-", "").replace("YB_", "").replace("YB", "").strip()
                            try:
                                k_day = int(float(k[1]))
                            except (ValueError, TypeError):
                                k_day = 1
                            if k_animal == clean_target and k_day == target_day:
                                valid_hunts = hunts
                                print(f"Matched manual Diane labels for Animal '{parsed['animal_id']}', Day {target_day}: {len(valid_hunts)} hunts found.")
                                break
                        if not valid_hunts:
                            print(f"Notice: Could not find matching session (Animal={parsed['animal_id']}, Day={target_day}) in {label_path}.")
                else:
                    print(f"Notice: Could not parse animal/day from {dlc_path}")
            elif not is_parts_fmt:
                session_hunts = parser.parse_file(label_path, fps=fps)
                if len(session_hunts) == 1:
                    valid_hunts = next(iter(session_hunts.values()))

            if valid_hunts:
                events = parser.hunts_to_events(valid_hunts)
                true_seq = events_to_sequence(events, num_frames or 100000, paradigm="mouse")
                return true_seq, list(valid_hunts) if isinstance(valid_hunts, list) else None
            else:
                return None, None
        except Exception as e:
            print(f"Warning: Failed to parse manual labels via DianeExcelLabelParser: {e}")
            return None, None

    if is_mouse:
        return None, None

    # 2. Rat Formats (Lucas or Dan)
    try:
        import dataset_utils
        is_dan_file = False
        if os.path.isfile(label_path):
            try:
                hdr = pd.read_csv(label_path, nrows=2) if label_path.endswith('.csv') else pd.read_excel(label_path, nrows=2)
                hdr_cols = [str(c).strip().lower() for c in hdr.columns]
                if 'animal' in hdr_cols and any(k in hdr_cols for k in ('session', 'cricket', 'day', 'on')):
                    is_dan_file = True
            except Exception:
                pass

        if label_format == 'dan' or is_dan_file or (dlc_path and 'dan' in str(label_format).lower()) or 'dan' in os.path.basename(label_path).lower() or 'crickethunttimes' in os.path.basename(label_path).lower():
            if dlc_path is None:
                print("Warning: dlc_path required to match session for Dan format.")
                return None, None
            dan_hunts = dataset_utils.load_dan_labels(label_path)
            base = os.path.basename(dlc_path)
            match = re.search(r'([a-zA-Z]+\d+)_D(\d+)_HS(\d+)', base, re.IGNORECASE)
            if match:
                animal = match.group(1).upper()
                day = int(match.group(2))
                session = int(match.group(3))
                norm_animal = re.sub(r'([A-Z]+)0*(\d+)', r'\1\2', animal)
                valid_hunts = (
                    dan_hunts.get((animal, day, session)) or 
                    dan_hunts.get((norm_animal, day, session)) or 
                    dan_hunts.get((animal, session)) or 
                    dan_hunts.get((norm_animal, session), [])
                )
                true_seq = events_to_sequence(valid_hunts, num_frames or 100000, paradigm="rat")
                return true_seq, valid_hunts
        else:
            valid_hunts = dataset_utils.parse_lucas_labels(label_path)
            true_seq = events_to_sequence(valid_hunts, num_frames or 100000, paradigm="rat")
            return true_seq, valid_hunts
    except Exception as e:
        print(f"Warning: Could not load ground truth labels: {e}")

    return None, None

def plot_inference_results(
    probs: Union[np.ndarray, Dict[str, Any]],
    preds: np.ndarray,
    refined_events: Optional[List[Dict[str, Any]]] = None,
    refined_path: Optional[np.ndarray] = None,
    label_path: Optional[str] = None,
    label_format: Optional[str] = None,
    dlc_path: Optional[str] = None,
    ground_truth_hunts: Optional[List[Dict[str, Any]]] = None,
    save_path: Optional[str] = None,
    title: str = "Mouse Hunting Inference Results",
    paradigm: str = "mouse"
):
    """
    Plots multi-class model confidence traces, raw step predictions,
    and HMM Viterbi refined state sequences.
    """
    num_frames = len(preds)
    frames = np.arange(num_frames)

    if isinstance(probs, dict):
        final_probs = probs.get('final', next(iter(probs.values()))) if probs else np.empty((0, 0))
    else:
        final_probs = probs
    if not isinstance(final_probs, np.ndarray):
        final_probs = np.asarray(final_probs)
    num_classes = final_probs.shape[1]

    if num_classes == 5 or paradigm == "mouse":
        classes = ['Background', 'Cricket Present', 'First Approach', 'Final Approach', 'Eating']
        colors = ['#808080', '#17becf', '#1f77b4', '#d62728', '#2ca02c']
    else:
        classes = ['Background', 'Stalking', 'Pursuit', 'Eating']
        colors = ['lightgray', 'orange', 'red', 'green']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

    for i, cls_name in enumerate(classes):
        ax1.plot(frames, final_probs[:, i], label=cls_name, color=colors[i], linewidth=1.8, alpha=0.85)

    ax1.set_ylabel("Probability", fontsize=10)
    ax1.set_title(f"{title} - Model Class Probabilities", fontsize=12, fontweight='bold')
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.25)
    ax1.set_ylim(-0.05, 1.05)

    ax2.step(frames, preds, where='post', label='Raw Model Prediction',
             color='blue', linewidth=1.5, alpha=0.5, linestyle='--')

    if refined_path is not None:
        ax2.step(frames, refined_path, where='post', label='HMM Viterbi Refined',
                 color='magenta', linewidth=1.8, alpha=0.9)
    elif refined_events:
        refined_seq = events_to_sequence(refined_events, num_frames, paradigm=paradigm)
        ax2.step(frames, refined_seq, where='post', label='HMM Viterbi Refined',
                 color='magenta', linewidth=1.8, alpha=0.9)

    # Ground Truth manual labels overlay
    true_seq = None
    if label_path and os.path.exists(label_path):
        true_seq, gt_parsed = load_ground_truth_sequence(
            label_path=label_path,
            dlc_path=dlc_path,
            num_frames=num_frames,
            label_format=label_format,
            paradigm=paradigm
        )
        if ground_truth_hunts is None and gt_parsed:
            ground_truth_hunts = gt_parsed
    elif ground_truth_hunts:
        true_seq = events_to_sequence(ground_truth_hunts, num_frames, paradigm=paradigm)

    if true_seq is not None:
        ax2.step(frames, true_seq, where='post', label='Manual Labels',
                 color='black', linewidth=3.5, alpha=0.35, zorder=0)

    ax2.set_yticks(range(len(classes)))
    ax2.set_yticklabels(classes, fontsize=9)
    ax2.set_ylabel("Predicted State", fontsize=10)
    ax2.set_xlabel("Frame Number", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.25)
    ax2.set_ylim(-0.5, len(classes) - 0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved inference plot to: {save_path}")
    return fig

def create_mouse_hunting_video_clip(
    video_path: str,
    probs: np.ndarray,
    preds: np.ndarray,
    start_frame: int,
    end_frame: int,
    output_path: Optional[str] = None,
    trailing_frames: int = 300,
    dlc_path: Optional[Union[str, List[str]]] = None,
    circle_radius: int = 4,
    dlc_thresh: float = 0.7,
    cricket_thresh: Optional[float] = None,
    show_confidence: bool = True,
    include_blurry_cricket: bool = True,
    include_cricket_immo: bool = True,
    filter_poops: bool = True,
    poop_dist_thresh: float = 6.0,
    poop_thresh: float = 0.9,
    plot_poops: bool = True,
    interp_limit: int = 0
):
    """
    Creates an annotated hunting video clip for mouse sessions with 5-class behavioral HUD,
    multi-segment tail tracking, mouse posture skeleton, and cricket landmarks.
    """
    from VidLabeler import create_hunting_video_clip
    return create_hunting_video_clip(
        video_path=video_path,
        probs=probs,
        preds=preds,
        start_frame=start_frame,
        end_frame=end_frame,
        output_path=output_path,
        trailing_frames=trailing_frames,
        dlc_path=dlc_path,
        circle_radius=circle_radius,
        dlc_thresh=dlc_thresh,
        cricket_thresh=cricket_thresh,
        show_confidence=show_confidence,
        include_blurry_cricket=include_blurry_cricket,
        include_cricket_immo=include_cricket_immo,
        filter_poops=filter_poops,
        poop_dist_thresh=poop_dist_thresh,
        poop_thresh=poop_thresh,
        plot_poops=plot_poops,
        interp_limit=interp_limit,
        paradigm="mouse"
    )

# Alias matching rat inference import style
create_hunting_video_clip = create_mouse_hunting_video_clip

# ==========================================
# 8. Model Accuracy Evaluation & Metrics (Mouse 5-Class)
# ==========================================

EVAL_CLASS_NAMES = ['Background', 'Cricket Present', 'First Approach', 'Final Approach', 'Eating']
EVAL_CLASS_COLORS = ['#7f8c8d', '#f39c12', '#2980b9', '#e74c3c', '#27ae60']

def frames_to_time(frames, video_fps):
    """Converts frame count to MM:SS.SS time string."""
    if frames is None or (isinstance(frames, float) and np.isnan(frames)):
        return ""
    total_seconds = float(frames) / float(video_fps)
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:05.2f}"

def calc_pearson_correlation(x, y):
    """Calculates Pearson r, R^2, and p-value safely."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 2:
        return 0.0, 0.0, 1.0
    x = x[valid]
    y = y[valid]
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 0.0, 1.0
    if sp_stats is not None:
        try:
            r, p = sp_stats.pearsonr(x, y)
            return float(r), float(r**2), float(p)
        except Exception:
            pass
    r = float(np.corrcoef(x, y)[0, 1])
    n = len(x)
    if abs(r) >= 1.0:
        p = 0.0
    else:
        t_stat = r * np.sqrt((n - 2) / max(1e-8, 1 - r**2))
        p = float(2.0 * (1.0 - 0.5 * (1.0 + np.tanh(abs(t_stat) / 1.5))))
    return r, float(r**2), p

def generate_mouse_analyzed_excel(events, output_path, video_fps):
    """
    Generates a structured behavioral summary Excel spreadsheet for detected mouse hunt events.
    Includes Enter, First Approach, Final Approach, Capture, and Consumption timings.
    """
    data_rows = []
    
    for i, event in enumerate(events):
        enter = event.get('enter')
        first_app = event.get('first_approach')
        final_app = event.get('pursuit_start', event.get('final_approach', event.get('pursuit')))
        capture = event.get('capture')
        consumption_end = event.get('consumption_end', event.get('capture_end'))
        remarks = event.get('remarks', '')
        
        latency_first_frames = (first_app - enter) if (first_app is not None and enter is not None) else None
        attack_dur_frames = (capture - final_app) if (capture is not None and final_app is not None) else None
        total_frames = (capture - enter) if (capture is not None and enter is not None) else None
        
        include_status = 'Include'
        if remarks and 'HMM' not in remarks:
            include_status = 'Review'
        
        row = {
            'Cricket_Number': i + 1,
            'Enter_Frame': enter,
            'Enter_Time': frames_to_time(enter, video_fps),
            'First_Approach_Frame': first_app,
            'First_Approach_Time': frames_to_time(first_app, video_fps),
            'Final_Approach_Frame': final_app,
            'Final_Approach_Time': frames_to_time(final_app, video_fps),
            'Capture_Frame': capture,
            'Capture_Time': frames_to_time(capture, video_fps),
            'Consumption_End_Frame': consumption_end,
            'Consumption_Time': frames_to_time(consumption_end, video_fps),
            'Time_to_Capture_Frames': total_frames,
            'Time_to_Capture': frames_to_time(total_frames, video_fps),
            'Latency_to_First_Approach_Frames': latency_first_frames,
            'Latency_to_First_Approach': frames_to_time(latency_first_frames, video_fps),
            'Attack_Duration_Frames': attack_dur_frames,
            'Attack_Duration': frames_to_time(attack_dur_frames, video_fps),
            'Remarks': remarks,
            'Include?': include_status
        }
        data_rows.append(row)
        
    df = pd.DataFrame(data_rows)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if output_path.endswith('.csv'):
            df.to_csv(output_path, index=False)
        else:
            df.to_excel(output_path, index=False)
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Error saving analyzed excel: {e}")

generate_analyzed_excel = generate_mouse_analyzed_excel

def _get_event_field(ev, *field_names, default=None):
    """Helper to safely extract fields from dictionaries, objects, or series."""
    for f in field_names:
        if isinstance(ev, dict):
            if f in ev and ev[f] is not None and not (isinstance(ev[f], float) and np.isnan(ev[f])):
                return ev[f]
        elif hasattr(ev, f):
            val = getattr(ev, f)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                return val
        elif isinstance(ev, pd.Series):
            if f in ev and pd.notna(ev[f]):
                return ev[f]
    return default

def match_hunt_events(true_hunts, pred_events, fps=30.0, video_name=""):
    """
    Matches ground truth hunts with predicted hunt events using temporal interval overlap.
    Adapts for 5-class mouse hunting: Enter, First Approach, Final Approach / Strike, Capture.
    Returns:
        matched_records: list of dicts with timing details and errors for true positives
        unmatched_true: list of missed manual hunts (false negatives)
        unmatched_pred: list of spurious predicted hunts (false positives)
    """
    matched_records = []
    
    if len(true_hunts) == 0 and len(pred_events) == 0:
        return matched_records, [], []
    if len(true_hunts) == 0:
        unmatched_pred = [{**p, 'video': video_name, 'fps': fps} for p in pred_events]
        return matched_records, [], unmatched_pred
    if len(pred_events) == 0:
        unmatched_true = [{**t, 'video': video_name, 'fps': fps} for t in true_hunts]
        return matched_records, unmatched_true, []
        
    candidates = []
    for t_idx, t_hunt in enumerate(true_hunts):
        t_enter = _get_event_field(t_hunt, 'enter', 'released', 'Enter_Frame')
        t_capture = _get_event_field(t_hunt, 'capture', 'Capture_Frame')
        if t_enter is None:
            continue
        if t_capture is None:
            t_capture = _get_event_field(t_hunt, 'consumption_end', 'eaten', 'capture_end', default=t_enter + int(30 * fps))
        if t_capture is None:
            continue
            
        for p_idx, p_event in enumerate(pred_events):
            p_enter = _get_event_field(p_event, 'enter', 'Enter_Frame')
            p_capture = _get_event_field(p_event, 'capture', 'Capture_Frame')
            if p_enter is None:
                continue
            if p_capture is None:
                p_capture = _get_event_field(p_event, 'consumption_end', 'capture_end', default=p_enter + int(30 * fps))
            if p_capture is None:
                continue
                
            overlap = max(0, min(t_capture, p_capture) - max(t_enter, p_enter))
            union = max(t_capture, p_capture) - min(t_enter, p_enter) + 1e-6
            iou = overlap / union
            
            capture_diff_frames = abs(p_capture - t_capture)
            is_valid_match = (iou > 0.05) or (overlap > 0) or (capture_diff_frames <= (3.0 * fps) and p_enter <= (t_capture + fps))
            
            if is_valid_match:
                score = iou + max(0, 1.0 - (capture_diff_frames / (5.0 * fps)))
                candidates.append((score, t_idx, p_idx, iou))
                
    candidates.sort(key=lambda x: x[0], reverse=True)
    matched_t = set()
    matched_p = set()
    
    for score, t_idx, p_idx, iou in candidates:
        if t_idx not in matched_t and p_idx not in matched_p:
            matched_t.add(t_idx)
            matched_p.add(p_idx)
            
            t_hunt = true_hunts[t_idx]
            p_event = pred_events[p_idx]
            
            t_enter = _get_event_field(t_hunt, 'enter', 'released', 'Enter_Frame')
            t_first = _get_event_field(t_hunt, 'first_approach', 'First_Approach_Frame')
            t_final = _get_event_field(t_hunt, 'final_approach', 'approach_start', 'pursuit_start', 'pursuit', 'Final_Approach_Frame')
            t_capture = _get_event_field(t_hunt, 'capture', 'Capture_Frame')
            
            p_enter = _get_event_field(p_event, 'enter', 'Enter_Frame')
            p_first = _get_event_field(p_event, 'first_approach', 'First_Approach_Frame')
            p_final = _get_event_field(p_event, 'final_approach', 'pursuit_start', 'pursuit', 'Final_Approach_Frame')
            p_capture = _get_event_field(p_event, 'capture', 'Capture_Frame')
            
            # Errors in seconds
            enter_err = (p_enter - t_enter) / fps if (p_enter is not None and t_enter is not None) else np.nan
            first_err = (p_first - t_first) / fps if (p_first is not None and t_first is not None) else np.nan
            final_err = (p_final - t_final) / fps if (p_final is not None and t_final is not None) else np.nan
            cap_err = (p_capture - t_capture) / fps if (p_capture is not None and t_capture is not None) else np.nan
            
            # Behavioral durations
            true_dur = max(0, t_capture - t_enter) / fps if (t_capture is not None and t_enter is not None) else np.nan
            pred_dur = max(0, p_capture - p_enter) / fps if (p_capture is not None and p_enter is not None) else np.nan
            
            true_att = max(0, t_capture - t_final) / fps if (t_capture is not None and t_final is not None) else np.nan
            pred_att = max(0, p_capture - p_final) / fps if (p_capture is not None and p_final is not None) else np.nan
            
            t_app_ref = t_first if t_first is not None else t_final
            p_app_ref = p_first if p_first is not None else p_final
            true_lat = max(0, t_app_ref - t_enter) / fps if (t_app_ref is not None and t_enter is not None) else np.nan
            pred_lat = max(0, p_app_ref - p_enter) / fps if (p_app_ref is not None and p_enter is not None) else np.nan
            
            rec = {
                'video': video_name,
                'iou': iou,
                'true_enter_frame': t_enter,
                'pred_enter_frame': p_enter,
                'enter_error_sec': enter_err,
                'true_first_approach_frame': t_first,
                'pred_first_approach_frame': p_first,
                'first_approach_error_sec': first_err,
                'true_final_approach_frame': t_final,
                'pred_final_approach_frame': p_final,
                'final_approach_error_sec': final_err,
                'pursuit_error_sec': final_err, # Backward compatibility
                'true_capture_frame': t_capture,
                'pred_capture_frame': p_capture,
                'capture_error_sec': cap_err,
                'true_duration_sec': true_dur,
                'pred_duration_sec': pred_dur,
                'true_attack_sec': true_att,
                'pred_attack_sec': pred_att,
                'true_pursuit_sec': true_att,   # Backward compatibility
                'pred_pursuit_sec': pred_att,   # Backward compatibility
                'true_latency_sec': true_lat,
                'pred_latency_sec': pred_lat
            }
            matched_records.append(rec)
            
    unmatched_true = [{**true_hunts[i], 'video': video_name, 'fps': fps} for i in range(len(true_hunts)) if i not in matched_t]
    unmatched_pred = [{**pred_events[j], 'video': video_name, 'fps': fps} for j in range(len(pred_events)) if j not in matched_p]
    
    return matched_records, unmatched_true, unmatched_pred

def plot_confusion_matrices_and_f1(all_true, all_raw, all_refined, save_path=None):
    """
    Plots side-by-side normalized Confusion Matrices (Raw vs HMM Refined)
    and a grouped bar chart of Precision, Recall, and F1 across all 5 mouse classes.
    """
    fig = plt.figure(figsize=(19, 5.8))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1.25], figure=fig, wspace=0.32)
    
    classes = EVAL_CLASS_NAMES
    n_cls = len(classes)
    
    # 1. Raw Model Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    cm_raw = confusion_matrix(all_true, all_raw, labels=list(range(n_cls)))
    cm_raw_norm = cm_raw.astype(float) / np.maximum(cm_raw.sum(axis=1, keepdims=True), 1e-10) * 100.0
    
    im1 = ax1.imshow(cm_raw_norm, cmap='Blues', vmin=0, vmax=100)
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Recall (% of True Frames)', fontsize=9.5)
    
    ax1.set_xticks(range(n_cls))
    ax1.set_yticks(range(n_cls))
    ax1.set_xticklabels(classes, rotation=25, ha='right', fontsize=9)
    ax1.set_yticklabels(classes, fontsize=9)
    ax1.set_xlabel('Predicted Class', fontsize=10.5, fontweight='bold')
    ax1.set_ylabel('Ground Truth Class', fontsize=10.5, fontweight='bold')
    ax1.set_title('Raw Predictions Confusion Matrix', fontsize=12, fontweight='bold', pad=10)
    
    for i in range(n_cls):
        for j in range(n_cls):
            val = cm_raw_norm[i, j]
            count = cm_raw[i, j]
            color = "white" if val > 50 else "black"
            ax1.text(j, i, f"{val:.1f}%\n({count:,})", ha='center', va='center', color=color, fontsize=8, fontweight='bold')
            
    # 2. Refined HMM Confusion Matrix
    ax2 = fig.add_subplot(gs[0, 1])
    cm_ref = confusion_matrix(all_true, all_refined, labels=list(range(n_cls)))
    cm_ref_norm = cm_ref.astype(float) / np.maximum(cm_ref.sum(axis=1, keepdims=True), 1e-10) * 100.0
    
    im2 = ax2.imshow(cm_ref_norm, cmap='YlGn', vmin=0, vmax=100)
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Recall (% of True Frames)', fontsize=9.5)
    
    ax2.set_xticks(range(n_cls))
    ax2.set_yticks(range(n_cls))
    ax2.set_xticklabels(classes, rotation=25, ha='right', fontsize=9)
    ax2.set_yticklabels(classes, fontsize=9)
    ax2.set_xlabel('Predicted Class', fontsize=10.5, fontweight='bold')
    ax2.set_ylabel('Ground Truth Class', fontsize=10.5, fontweight='bold')
    ax2.set_title('HMM Refined Confusion Matrix', fontsize=12, fontweight='bold', pad=10)
    
    for i in range(n_cls):
        for j in range(n_cls):
            val = cm_ref_norm[i, j]
            count = cm_ref[i, j]
            color = "white" if val > 50 else "black"
            ax2.text(j, i, f"{val:.1f}%\n({count:,})", ha='center', va='center', color=color, fontsize=8, fontweight='bold')
            
    # 3. Per-Class Precision / Recall / F1 Bar Chart
    ax3 = fig.add_subplot(gs[0, 2])
    p_raw, r_raw, f1_raw, _ = precision_recall_fscore_support(all_true, all_raw, labels=list(range(n_cls)), zero_division=0)
    p_ref, r_ref, f1_ref, _ = precision_recall_fscore_support(all_true, all_refined, labels=list(range(n_cls)), zero_division=0)
    
    macro_raw_f1 = f1_score(all_true, all_raw, average='macro', zero_division=0)
    macro_ref_f1 = f1_score(all_true, all_refined, average='macro', zero_division=0)
    acc_raw = accuracy_score(all_true, all_raw) * 100.0
    acc_ref = accuracy_score(all_true, all_refined) * 100.0
    
    x = np.arange(n_cls)
    w = 0.38
    
    ax3.bar(x - w/2, f1_raw, width=w, label='Raw Model F1', color='#4a90e2', alpha=0.9, edgecolor='black', linewidth=0.6)
    ax3.bar(x + w/2, f1_ref, width=w, label='Refined HMM F1', color='#2ecc71', alpha=0.9, edgecolor='black', linewidth=0.6)
    
    for i in range(n_cls):
        ax3.text(x[i] - w/2, f1_raw[i] + 0.02, f"{f1_raw[i]:.2f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax3.text(x[i] + w/2, f1_ref[i] + 0.02, f"{f1_ref[i]:.2f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
        
    ax3.set_xticks(x)
    ax3.set_xticklabels(classes, rotation=20, ha='right', fontsize=9)
    ax3.set_ylabel('F1 Score', fontsize=10.5, fontweight='bold')
    ax3.set_ylim(0, 1.15)
    ax3.set_title('Per-Class F1 Score Comparison', fontsize=12, fontweight='bold', pad=10)
    ax3.legend(loc='upper right', framealpha=0.9)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    summary_text = (
        f"Macro F1 (Raw): {macro_raw_f1:.3f} | (Refined): {macro_ref_f1:.3f}\n"
        f"Frame Accuracy: {acc_raw:.1f}% (Raw) | {acc_ref:.1f}% (Refined)"
    )
    ax3.text(0.03, 0.95, summary_text, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#bdc3c7', alpha=0.9))
             
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if save_path.endswith('.png'):
            plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)

def plot_multiclass_roc_and_pr_curves(all_true, all_probs, save_path=None):
    """
    Plots multi-class One-vs-Rest ROC curves and Precision-Recall curves for all 5 mouse classes.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    classes = EVAL_CLASS_NAMES
    colors = EVAL_CLASS_COLORS
    n_cls = len(classes)
    
    # 1. ROC Curves
    for c in range(n_cls):
        binary_true = (all_true == c).astype(int)
        if np.sum(binary_true) == 0:
            continue
        fpr, tpr, _ = roc_curve(binary_true, all_probs[:, c])
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=colors[c], linewidth=2.2, label=f"{classes[c]} (AUC = {roc_auc:.3f})")
        
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.2, alpha=0.5, label='Random Chance (AUC = 0.500)')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
    ax1.set_title('One-vs-Rest ROC Curves', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='lower right', framealpha=0.9, fontsize=9)
    ax1.grid(True, alpha=0.25, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 2. Precision-Recall Curves
    for c in range(n_cls):
        binary_true = (all_true == c).astype(int)
        if np.sum(binary_true) == 0:
            continue
        precision, recall, _ = precision_recall_curve(binary_true, all_probs[:, c])
        ap = average_precision_score(binary_true, all_probs[:, c])
        prevalence = np.mean(binary_true)
        ax2.plot(recall, precision, color=colors[c], linewidth=2.2, label=f"{classes[c]} (AP = {ap:.3f})")
        ax2.axhline(prevalence, color=colors[c], linestyle=':', alpha=0.4, linewidth=1.0)
        
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax2.set_title('Precision-Recall Curves (Multi-class)', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax2.grid(True, alpha=0.25, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if save_path.endswith('.png'):
            plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)

def plot_hunt_detection_and_timing_errors(event_summary, matched_df, save_path=None, log_scale=False):
    """
    Plots hunt-event detection performance breakdown (TP, FP, FN)
    and transition timing error distributions (Enter, First Approach, Final Approach, Capture).
    If log_scale=True, plots absolute timing errors on a logarithmic y-axis.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw={'width_ratios': [1, 1.45]})
    
    # 1. Event Detection Breakdown Bar Chart
    categories = ['Detected Hunts\n(True Positives)', 'False Alarms\n(False Positives)', 'Missed Hunts\n(False Negatives)']
    counts = [event_summary['true_positives'], event_summary['false_positives'], event_summary['false_negatives']]
    bar_colors = ['#27ae60', '#e74c3c', '#e67e22']
    
    bars = ax1.bar(categories, counts, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.8, width=0.55)
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15, f"{count}",
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
                 
    ax1.set_ylabel('Number of Hunt Events', fontsize=11, fontweight='bold')
    title_suffix = " [Log Scale]" if log_scale else ""
    ax1.set_title(f'Hunt Event Detection Performance{title_suffix}', fontsize=12, fontweight='bold', pad=10)
    ax1.grid(axis='y', alpha=0.25, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    badge_text = (
        f"Sensitivity / Recall: {event_summary['recall']*100:.1f}%\n"
        f"Precision: {event_summary['precision']*100:.1f}%\n"
        f"Hunt Event F1: {event_summary['f1_score']*100:.1f}%\n"
        f"Ground Truth Hunts: {event_summary['total_true']}\n"
        f"Predicted Hunts: {event_summary['total_pred']}"
    )
    ax1.text(0.04, 0.95, badge_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.6', facecolor='#f4f6f7', edgecolor='#bdc3c7', alpha=0.9))
             
    # 2. Timing Error Distributions across 4 transitions
    if len(matched_df) > 0:
        timing_cols = ['enter_error_sec', 'first_approach_error_sec', 'final_approach_error_sec', 'capture_error_sec']
        labels = ['Hunt Enter', 'First Approach', 'Final Approach', 'Capture']
        data_to_plot = [matched_df[c].dropna().values for c in timing_cols]
        
        if log_scale:
            plot_data = [np.clip(np.abs(d), 0.01, None) for d in data_to_plot]
            ax2.boxplot(plot_data, patch_artist=True, tick_labels=labels,
                        boxprops=dict(facecolor='#d5dbdb', color='black', alpha=0.7),
                        medianprops=dict(color='#c0392b', linewidth=2.0),
                        whiskerprops=dict(color='black', linewidth=1.2),
                        capprops=dict(color='black', linewidth=1.2),
                        flierprops=dict(marker='o', markersize=4, alpha=0.3))
                        
            for idx, (raw_d, p_vals) in enumerate(zip(data_to_plot, plot_data)):
                jitter = np.random.normal(0, 0.05, size=len(p_vals))
                ax2.scatter(np.ones(len(p_vals))*(idx+1) + jitter, p_vals, color='#2980b9', alpha=0.5, s=25, zorder=3)
                
            ax2.set_yscale('log')
            all_p = [v for d in plot_data for v in d]
            if len(all_p) > 0:
                log_min = np.min(all_p)
                log_max = np.max(all_p)
                ax2.set_ylim(bottom=max(0.005, log_min * 0.4), top=log_max * 5.0)
                for idx, raw_d in enumerate(data_to_plot):
                    mae = np.mean(np.abs(raw_d)) if len(raw_d) > 0 else 0.0
                    med_abs = np.median(np.abs(raw_d)) if len(raw_d) > 0 else 0.0
                    badge_y = log_max * 1.5
                    ax2.text(idx + 1, badge_y,
                             f"MAE: {mae:.2f}s\nMed |Err|: {med_abs:.2f}s",
                             ha='center', va='bottom', fontsize=8, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, edgecolor='#7f8c8d'))
                             
            ax2.axhline(1.0, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)
            ax2.axhline(5.0, color='#e74c3c', linestyle='--', linewidth=1.2, alpha=0.7)
            ax2.set_ylabel('Absolute Timing Error in Seconds [Log Scale]', fontsize=11, fontweight='bold')
            ax2.set_title('Transition Timing Accuracy [Absolute Errors, Log Scale]', fontsize=12, fontweight='bold', pad=10)
            ax2.grid(axis='y', which='both', alpha=0.25, linestyle='--')
        else:
            ax2.boxplot(data_to_plot, patch_artist=True, tick_labels=labels,
                        boxprops=dict(facecolor='#d5dbdb', color='black', alpha=0.7),
                        medianprops=dict(color='#c0392b', linewidth=2.0),
                        whiskerprops=dict(color='black', linewidth=1.2),
                        capprops=dict(color='black', linewidth=1.2),
                        flierprops=dict(marker='o', markersize=4, alpha=0.3))
                        
            for idx, d in enumerate(data_to_plot):
                jitter = np.random.normal(0, 0.05, size=len(d))
                ax2.scatter(np.ones(len(d))*(idx+1) + jitter, d, color='#2980b9', alpha=0.5, s=25, zorder=3)
                
            all_vals = [v for d in data_to_plot for v in d]
            if len(all_vals) > 0:
                y_min = np.min(all_vals)
                y_max = np.max(all_vals)
                y_span = max(1.0, y_max - y_min)
                ax2.set_ylim(y_min - 0.08 * y_span, y_max + 0.22 * y_span)
                
                for idx, d in enumerate(data_to_plot):
                    mae = np.mean(np.abs(d)) if len(d) > 0 else 0.0
                    med = np.median(d) if len(d) > 0 else 0.0
                    badge_y = y_max + 0.05 * y_span
                    ax2.text(idx + 1, badge_y,
                             f"MAE: {mae:.2f}s\nMed: {med:+.2f}s",
                             ha='center', va='bottom', fontsize=8, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, edgecolor='#7f8c8d'))
                          
            ax2.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
            ax2.set_ylabel('Timing Error in Seconds (Predicted - Manual)', fontsize=11, fontweight='bold')
            ax2.set_title('Transition Timing Accuracy for Matched Hunts', fontsize=12, fontweight='bold', pad=10)
            ax2.grid(axis='y', alpha=0.25, linestyle='--')
            
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
    else:
        ax2.text(0.5, 0.5, "No matched hunts to evaluate timing errors.", ha='center', va='center')
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if save_path.endswith('.png'):
            plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Saved: {save_path}")
        
        if not log_scale and save_path.endswith('.png'):
            log_save = save_path.replace('.png', '_log.png')
            plot_hunt_detection_and_timing_errors(event_summary, matched_df, save_path=log_save, log_scale=True)
            
    plt.close(fig)

def plot_duration_correlations(matched_df, save_path=None):
    """
    Plots correlation scatter plots with identity line y=x and linear regression fit
    for Total Hunt Duration, Attack Duration, and First Approach Latency.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics_info = [
        ('true_duration_sec', 'pred_duration_sec', 'Total Hunt Duration (Enter -> Capture)', '#2980b9'),
        ('true_attack_sec', 'pred_attack_sec', 'Attack Duration (Final Approach -> Capture)', '#e74c3c'),
        ('true_latency_sec', 'pred_latency_sec', 'First Approach Latency (Enter -> First Approach)', '#f39c12'),
    ]
    
    for ax, (true_col, pred_col, title, color) in zip(axes, metrics_info):
        valid_mask = matched_df[true_col].notna() & matched_df[pred_col].notna() if (true_col in matched_df and pred_col in matched_df) else pd.Series([], dtype=bool)
        if int(valid_mask.sum()) > 1:
            x = matched_df.loc[valid_mask, true_col].values
            y = matched_df.loc[valid_mask, pred_col].values
            
            max_val = max(np.max(x), np.max(y)) * 1.1 if len(x) > 0 and len(y) > 0 else 10
            max_val = max(max_val, 2.0)
            ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, alpha=0.5, label='Identity (y = x)')
            
            ax.scatter(x, y, color=color, alpha=0.7, edgecolor='black', s=45, linewidth=0.6, label='Hunts')
            
            r, r2, p = calc_pearson_correlation(x, y)
            if len(x) >= 2 and np.std(x) > 0:
                slope, intercept = np.polyfit(x, y, 1)
                line_x = np.linspace(0, max_val, 100)
                ax.plot(line_x, slope * line_x + intercept, color='#2c3e50', linewidth=1.8,
                        label=f"Fit (y = {slope:.2f}x + {intercept:.2f})")
                        
            p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
            stats_text = f"R² = {r2:.3f}\nPearson r = {r:.3f}\n{p_str}\nN = {len(x)}"
            ax.text(0.05, 0.92, stats_text, transform=ax.transAxes, fontsize=9.5,
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='#fdfefe', edgecolor='#bdc3c7', alpha=0.9))
                    
            ax.set_xlim(0, max_val)
            ax.set_ylim(0, max_val)
            ax.set_xlabel('Ground Truth Manual (s)', fontsize=10.5, fontweight='bold')
            ax.set_ylabel('Model Predicted (s)', fontsize=10.5, fontweight='bold')
            ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.25, linestyle='--')
            ax.legend(loc='lower right', framealpha=0.9, fontsize=8.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        else:
            ax.text(0.5, 0.5, "Insufficient matched hunt data", ha='center', va='center')
            ax.set_title(title, fontsize=11, fontweight='bold')
            
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if save_path.endswith('.png'):
            plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)

def plot_per_video_performance(per_video_df, save_path=None):
    """
    Plots horizontal bar chart showing Macro F1 and Event Detection Rate per video.
    """
    if len(per_video_df) == 0:
        return
        
    fig, ax = plt.subplots(figsize=(13, max(4.5, len(per_video_df) * 0.45)))
    y_pos = np.arange(len(per_video_df))
    h = 0.38
    
    ax.barh(y_pos - h/2, per_video_df['refined_f1'], height=h, label='Frame Macro F1 (Refined)', color='#27ae60', alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.barh(y_pos + h/2, per_video_df['detection_recall'], height=h, label='Hunt Detection Rate (Sensitivity)', color='#2980b9', alpha=0.85, edgecolor='black', linewidth=0.5)
    
    mean_f1 = per_video_df['refined_f1'].mean()
    mean_det = per_video_df['detection_recall'].mean()
    ax.axvline(mean_f1, color='#27ae60', linestyle='--', linewidth=1.2, alpha=0.8, label=f"Avg F1: {mean_f1:.2f}")
    ax.axvline(mean_det, color='#2980b9', linestyle=':', linewidth=1.2, alpha=0.8, label=f"Avg Detection: {mean_det:.2f}")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(per_video_df['video_name'], fontsize=9.5)
    ax.set_xlabel('Performance Score (0.0 - 1.0)', fontsize=11, fontweight='bold')
    ax.set_xlim(0.0, 1.05)
    ax.set_title('Cross-Video Performance Breakdown', fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='lower right', framealpha=0.9, fontsize=9.5)
    ax.grid(axis='x', alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if save_path.endswith('.png'):
            plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)

def plot_master_evaluation_dashboard(all_true, all_raw, all_refined, all_probs, event_summary, matched_df, save_path=None, log_scale=False):
    """
    Creates a master 6-panel executive summary dashboard combining the core evaluation figures for mouse hunting.
    """
    fig = plt.figure(figsize=(20, 13))
    gs = GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.28)
    classes = EVAL_CLASS_NAMES
    colors = EVAL_CLASS_COLORS
    n_cls = len(classes)
    
    # --- PANEL A: Refined Confusion Matrix ---
    axA = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(all_true, all_refined, labels=list(range(n_cls)))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1e-10) * 100.0
    imA = axA.imshow(cm_norm, cmap='YlGn', vmin=0, vmax=100)
    plt.colorbar(imA, ax=axA, fraction=0.046, pad=0.04).set_label('Recall (%)', fontsize=9.5)
    axA.set_xticks(range(n_cls))
    axA.set_yticks(range(n_cls))
    axA.set_xticklabels(classes, rotation=20, ha='right', fontsize=9)
    axA.set_yticklabels(classes, fontsize=9)
    axA.set_xlabel('Predicted Class', fontsize=10.5, fontweight='bold')
    axA.set_ylabel('Ground Truth Class', fontsize=10.5, fontweight='bold')
    axA.set_title('A. Refined Predictions Confusion Matrix', fontsize=12, fontweight='bold', pad=8)
    for i in range(n_cls):
        for j in range(n_cls):
            val = cm_norm[i, j]
            axA.text(j, i, f"{val:.1f}%\n({cm[i, j]:,})", ha='center', va='center',
                     color="white" if val > 50 else "black", fontsize=8, fontweight='bold')
                     
    # --- PANEL B: Per-Class F1 Score Comparison ---
    axB = fig.add_subplot(gs[0, 1])
    _, _, f1_raw, _ = precision_recall_fscore_support(all_true, all_raw, labels=list(range(n_cls)), zero_division=0)
    _, _, f1_ref, _ = precision_recall_fscore_support(all_true, all_refined, labels=list(range(n_cls)), zero_division=0)
    x = np.arange(n_cls)
    w = 0.38
    axB.bar(x - w/2, f1_raw, width=w, label='Raw F1', color='#4a90e2', alpha=0.9, edgecolor='black', linewidth=0.5)
    axB.bar(x + w/2, f1_ref, width=w, label='Refined F1', color='#2ecc71', alpha=0.9, edgecolor='black', linewidth=0.5)
    for i in range(n_cls):
        axB.text(x[i] - w/2, f1_raw[i] + 0.02, f"{f1_raw[i]:.2f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
        axB.text(x[i] + w/2, f1_ref[i] + 0.02, f"{f1_ref[i]:.2f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
    axB.set_xticks(x)
    axB.set_xticklabels(classes, rotation=20, ha='right', fontsize=9)
    axB.set_ylabel('F1 Score', fontsize=10.5, fontweight='bold')
    axB.set_ylim(0, 1.15)
    axB.set_title('B. Per-Class F1 (Raw vs Refined)', fontsize=12, fontweight='bold', pad=8)
    axB.legend(loc='upper right', framealpha=0.9, fontsize=9)
    axB.grid(axis='y', alpha=0.25, linestyle='--')
    axB.spines['top'].set_visible(False)
    axB.spines['right'].set_visible(False)
    
    # --- PANEL C: Precision-Recall Curves ---
    axC = fig.add_subplot(gs[0, 2])
    for c in range(n_cls):
        bin_t = (all_true == c).astype(int)
        if np.sum(bin_t) > 0:
            p_curve, r_curve, _ = precision_recall_curve(bin_t, all_probs[:, c])
            ap = average_precision_score(bin_t, all_probs[:, c])
            axC.plot(r_curve, p_curve, color=colors[c], linewidth=2.0, label=f"{classes[c]} (AP={ap:.3f})")
    axC.set_xlim((0.0, 1.0))
    axC.set_ylim((0.0, 1.05))
    axC.set_xlabel('Recall', fontsize=10.5, fontweight='bold')
    axC.set_ylabel('Precision', fontsize=10.5, fontweight='bold')
    axC.set_title('C. Multiclass Precision-Recall Curves', fontsize=12, fontweight='bold', pad=8)
    axC.legend(loc='upper right', framealpha=0.9, fontsize=8.5)
    axC.grid(True, alpha=0.25, linestyle='--')
    axC.spines['top'].set_visible(False)
    axC.spines['right'].set_visible(False)
    
    # --- PANEL D: Hunt Event Detection Breakdown ---
    axD = fig.add_subplot(gs[1, 0])
    cats = ['Detected Hunts\n(TP)', 'False Alarms\n(FP)', 'Missed Hunts\n(FN)']
    cnts = [event_summary['true_positives'], event_summary['false_positives'], event_summary['false_negatives']]
    b_colors = ['#27ae60', '#e74c3c', '#e67e22']
    bars = axD.bar(cats, cnts, color=b_colors, alpha=0.85, edgecolor='black', linewidth=0.6, width=0.55)
    for b, ct in zip(bars, cnts):
        axD.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1, f"{ct}", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    axD.set_ylabel('Hunt Count', fontsize=10.5, fontweight='bold')
    axD.set_title('D. Hunt Event Detection Performance', fontsize=12, fontweight='bold', pad=8)
    axD.grid(axis='y', alpha=0.25, linestyle='--')
    axD.spines['top'].set_visible(False)
    axD.spines['right'].set_visible(False)
    badge = (f"Sensitivity / Recall: {event_summary['recall']*100:.1f}%\n"
             f"Precision: {event_summary['precision']*100:.1f}%\n"
             f"Hunt Event F1: {event_summary['f1_score']*100:.1f}%")
    axD.text(0.04, 0.95, badge, transform=axD.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f4f6f7', edgecolor='#bdc3c7', alpha=0.9))
             
    # --- PANEL E: Transition Timing Errors (seconds) across 4 transitions ---
    axE = fig.add_subplot(gs[1, 1])
    if len(matched_df) > 0:
        timing_cols = ['enter_error_sec', 'first_approach_error_sec', 'final_approach_error_sec', 'capture_error_sec']
        t_labels = ['Hunt Enter', 'First App', 'Final App', 'Capture']
        data_plot = [matched_df[col].dropna().values for col in timing_cols]
        
        if log_scale:
            plot_vals_e = [np.clip(np.abs(d), 0.01, None) for d in data_plot]
            axE.boxplot(plot_vals_e, patch_artist=True, tick_labels=t_labels,
                        boxprops=dict(facecolor='#d5dbdb', color='black', alpha=0.7),
                        medianprops=dict(color='#c0392b', linewidth=2.0))
            for idx, (d, pv) in enumerate(zip(data_plot, plot_vals_e)):
                jit = np.random.normal(0, 0.05, size=len(pv))
                axE.scatter(np.ones(len(pv))*(idx+1) + jit, pv, color='#2980b9', alpha=0.45, s=20, zorder=3)
            axE.set_yscale('log')
            all_p = [v for d in plot_vals_e for v in d]
            if len(all_p) > 0:
                log_max = np.max(all_p)
                log_min = np.min(all_p)
                axE.set_ylim(bottom=max(0.005, log_min * 0.4), top=log_max * 4.5)
                for idx, (d, pv) in enumerate(zip(data_plot, plot_vals_e)):
                    mae = np.mean(np.abs(d)) if len(d) > 0 else 0.0
                    axE.text(idx + 1, log_max * 1.5, f"MAE: {mae:.2f}s", ha='center', va='bottom', fontsize=8, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='#7f8c8d'))
            axE.set_ylabel('|Timing Error| (s) [Log Scale]', fontsize=10.5, fontweight='bold')
            axE.set_title('E. Timing Accuracy (|Err|, Log Scale)', fontsize=12, fontweight='bold', pad=8)
            axE.grid(axis='y', which='both', alpha=0.25, linestyle='--')
            axE.axhline(1.0, color='gray', linestyle=':', alpha=0.6)
            axE.axhline(5.0, color='#e74c3c', linestyle='--', alpha=0.7)
        else:
            axE.boxplot(data_plot, patch_artist=True, tick_labels=t_labels,
                        boxprops=dict(facecolor='#d5dbdb', color='black', alpha=0.7),
                        medianprops=dict(color='#c0392b', linewidth=2.0))
            for idx, d in enumerate(data_plot):
                jit = np.random.normal(0, 0.05, size=len(d))
                axE.scatter(np.ones(len(d))*(idx+1) + jit, d, color='#2980b9', alpha=0.45, s=20, zorder=3)
            all_vals = [v for d in data_plot for v in d]
            if len(all_vals) > 0:
                y_min = np.min(all_vals)
                y_max = np.max(all_vals)
                y_span = max(1.0, y_max - y_min)
                axE.set_ylim(y_min - 0.08 * y_span, y_max + 0.22 * y_span)
                for idx, d in enumerate(data_plot):
                    mae = np.mean(np.abs(d)) if len(d) > 0 else 0.0
                    badge_y = y_max + 0.05 * y_span
                    axE.text(idx + 1, badge_y,
                             f"MAE: {mae:.2f}s", ha='center', va='bottom', fontsize=8, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='#7f8c8d'))
            axE.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
            axE.set_ylabel('Timing Error (s) [Pred - True]', fontsize=10.5, fontweight='bold')
            axE.set_title('E. Transition Timing Accuracy', fontsize=12, fontweight='bold', pad=8)
            axE.grid(axis='y', alpha=0.25, linestyle='--')
            
        axE.spines['top'].set_visible(False)
        axE.spines['right'].set_visible(False)
        
    # --- PANEL F: Total Hunt Duration Scatter Correlation ---
    axF = fig.add_subplot(gs[1, 2])
    valid_mask = matched_df['true_duration_sec'].notna() & matched_df['pred_duration_sec'].notna() if ('true_duration_sec' in matched_df and 'pred_duration_sec' in matched_df) else pd.Series([], dtype=bool)
    if int(valid_mask.sum()) > 1:
        x = matched_df.loc[valid_mask, 'true_duration_sec'].values
        y = matched_df.loc[valid_mask, 'pred_duration_sec'].values
        max_v = max(np.max(x), np.max(y)) * 1.1 if len(x) > 0 and len(y) > 0 else 10
        axF.plot([0, max_v], [0, max_v], 'k--', linewidth=1.5, alpha=0.5, label='Identity (y=x)')
        axF.scatter(x, y, color='#2980b9', alpha=0.7, edgecolor='black', s=45, linewidth=0.6, label='Hunts')
        r, r2, p = calc_pearson_correlation(x, y)
        if len(x) >= 2 and np.std(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            line_x = np.linspace(0, max_v, 100)
            axF.plot(line_x, slope * line_x + intercept, color='#e74c3c', linewidth=1.8, label='Linear Fit')
        p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        axF.text(0.05, 0.92, f"R² = {r2:.3f} | r = {r:.3f}\n{p_str} | N = {len(x)}",
                 transform=axF.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#fdfefe', edgecolor='#bdc3c7', alpha=0.9))
        axF.set_xlim(0, max_v)
        axF.set_ylim(0, max_v)
        axF.set_xlabel('Ground Truth Manual Duration (s)', fontsize=10.5, fontweight='bold')
        axF.set_ylabel('Model Predicted Duration (s)', fontsize=10.5, fontweight='bold')
        axF.set_title('F. Total Hunt Duration Agreement', fontsize=12, fontweight='bold', pad=8)
        axF.legend(loc='lower right', framealpha=0.9, fontsize=8.5)
        axF.grid(True, alpha=0.25, linestyle='--')
        axF.spines['top'].set_visible(False)
        axF.spines['right'].set_visible(False)
        
    dashboard_title = "Mouse Hunt Model - Performance & Accuracy Evaluation Dashboard"
    if log_scale:
        dashboard_title += " [Log Scale Timing]"
    fig.suptitle(dashboard_title, fontsize=16, fontweight='bold', y=0.98)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if save_path.endswith('.png'):
            plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Master Dashboard saved: {save_path}")
        
        if not log_scale and save_path.endswith('.png'):
            log_save = save_path.replace('.png', '_log.png')
            plot_master_evaluation_dashboard(all_true, all_raw, all_refined, all_probs, event_summary, matched_df, save_path=log_save, log_scale=True)
            
    plt.close(fig)

def build_discrepancy_audit_table(all_matched, all_unmatched_true, all_unmatched_pred, timing_threshold_sec=5.0, duration_threshold_sec=3.0):
    """
    Builds a unified, actionable audit table listing all hunt discrepancies for mouse hunting:
    - Missed hunts (False Negatives)
    - Spurious detections (False Positives)
    - Matched hunts with large timing errors (>= timing_threshold_sec) or duration differences (>= duration_threshold_sec)
    """
    records = []
    
    # 1. Missed Hunts (False Negatives)
    for t in all_unmatched_true:
        fps = t.get('fps', 30.0)
        t_enter = _get_event_field(t, 'enter', 'released', 'Enter_Frame')
        t_first = _get_event_field(t, 'first_approach', 'First_Approach_Frame')
        t_final = _get_event_field(t, 'final_approach', 'approach_start', 'pursuit_start', 'Final_Approach_Frame')
        t_cap = _get_event_field(t, 'capture', 'Capture_Frame')
        t_dur = (t_cap - t_enter) / fps if (t_cap is not None and t_enter is not None) else np.nan
        records.append({
            'video': t.get('video', ''),
            'issue_type': 'Missed Hunt (FN)',
            'true_enter_frame': t_enter,
            'true_first_approach_frame': t_first,
            'true_final_approach_frame': t_final,
            'true_capture_frame': t_cap,
            'true_duration_sec': round(t_dur, 2) if pd.notna(t_dur) else np.nan,
            'pred_enter_frame': np.nan,
            'pred_first_approach_frame': np.nan,
            'pred_final_approach_frame': np.nan,
            'pred_capture_frame': np.nan,
            'pred_duration_sec': np.nan,
            'enter_error_sec': np.nan,
            'first_approach_error_sec': np.nan,
            'final_approach_error_sec': np.nan,
            'capture_error_sec': np.nan,
            'duration_diff_sec': np.nan,
            'notes': 'Manual hunt was not detected by model'
        })
        
    # 2. False Positives (Spurious Model Detections)
    for p in all_unmatched_pred:
        fps = p.get('fps', 30.0)
        p_enter = _get_event_field(p, 'enter', 'Enter_Frame')
        p_first = _get_event_field(p, 'first_approach', 'First_Approach_Frame')
        p_final = _get_event_field(p, 'final_approach', 'pursuit_start', 'Final_Approach_Frame')
        p_cap = _get_event_field(p, 'capture', 'Capture_Frame')
        p_dur = (p_cap - p_enter) / fps if (p_cap is not None and p_enter is not None) else np.nan
        records.append({
            'video': p.get('video', ''),
            'issue_type': 'False Positive (FP)',
            'true_enter_frame': np.nan,
            'true_first_approach_frame': np.nan,
            'true_final_approach_frame': np.nan,
            'true_capture_frame': np.nan,
            'true_duration_sec': np.nan,
            'pred_enter_frame': p_enter,
            'pred_first_approach_frame': p_first,
            'pred_final_approach_frame': p_final,
            'pred_capture_frame': p_cap,
            'pred_duration_sec': round(p_dur, 2) if pd.notna(p_dur) else np.nan,
            'enter_error_sec': np.nan,
            'first_approach_error_sec': np.nan,
            'final_approach_error_sec': np.nan,
            'capture_error_sec': np.nan,
            'duration_diff_sec': np.nan,
            'notes': 'Model detected hunt with no corresponding manual label'
        })
        
    # 3. Matched Hunts with Large Timing or Duration Discrepancies
    for m in all_matched:
        ent_err = m.get('enter_error_sec', np.nan)
        first_err = m.get('first_approach_error_sec', np.nan)
        final_err = m.get('final_approach_error_sec', np.nan)
        cap_err = m.get('capture_error_sec', np.nan)
        t_dur = m.get('true_duration_sec', np.nan)
        p_dur = m.get('pred_duration_sec', np.nan)
        dur_diff = abs(p_dur - t_dur) if (pd.notna(p_dur) and pd.notna(t_dur)) else np.nan
        
        has_timing_err = (
            (pd.notna(ent_err) and abs(ent_err) >= timing_threshold_sec) or
            (pd.notna(first_err) and abs(first_err) >= timing_threshold_sec) or
            (pd.notna(final_err) and abs(final_err) >= timing_threshold_sec) or
            (pd.notna(cap_err) and abs(cap_err) >= timing_threshold_sec)
        )
        has_dur_err = (pd.notna(dur_diff) and dur_diff >= duration_threshold_sec)
        
        if has_timing_err or has_dur_err:
            issues = []
            if pd.notna(ent_err) and abs(ent_err) >= timing_threshold_sec:
                issues.append(f"Enter err: {ent_err:+.1f}s")
            if pd.notna(first_err) and abs(first_err) >= timing_threshold_sec:
                issues.append(f"FirstApp err: {first_err:+.1f}s")
            if pd.notna(final_err) and abs(final_err) >= timing_threshold_sec:
                issues.append(f"FinalApp err: {final_err:+.1f}s")
            if pd.notna(cap_err) and abs(cap_err) >= timing_threshold_sec:
                issues.append(f"Capture err: {cap_err:+.1f}s")
            if has_dur_err:
                issues.append(f"Duration diff: {dur_diff:.1f}s")
                
            issue_label = "Timing & Duration Error" if (has_timing_err and has_dur_err) else ("Large Timing Error" if has_timing_err else "Large Duration Discrepancy")
            
            records.append({
                'video': m.get('video', ''),
                'issue_type': issue_label,
                'true_enter_frame': m.get('true_enter_frame'),
                'true_first_approach_frame': m.get('true_first_approach_frame'),
                'true_final_approach_frame': m.get('true_final_approach_frame'),
                'true_capture_frame': m.get('true_capture_frame'),
                'true_duration_sec': round(t_dur, 2) if pd.notna(t_dur) else np.nan,
                'pred_enter_frame': m.get('pred_enter_frame'),
                'pred_first_approach_frame': m.get('pred_first_approach_frame'),
                'pred_final_approach_frame': m.get('pred_final_approach_frame'),
                'pred_capture_frame': m.get('pred_capture_frame'),
                'pred_duration_sec': round(p_dur, 2) if pd.notna(p_dur) else np.nan,
                'enter_error_sec': round(ent_err, 2) if pd.notna(ent_err) else np.nan,
                'first_approach_error_sec': round(first_err, 2) if pd.notna(first_err) else np.nan,
                'final_approach_error_sec': round(final_err, 2) if pd.notna(final_err) else np.nan,
                'capture_error_sec': round(cap_err, 2) if pd.notna(cap_err) else np.nan,
                'duration_diff_sec': round(dur_diff, 2) if pd.notna(dur_diff) else np.nan,
                'notes': "; ".join(issues)
            })
            
    df = pd.DataFrame(records)
    if len(df) > 0:
        df['sort_frame'] = df['true_enter_frame'].fillna(df['pred_enter_frame'])
        df = df.sort_values(by=['video', 'sort_frame']).drop(columns=['sort_frame']).reset_index(drop=True)
    return df

def export_evaluation_metrics(excel_path, frame_metrics_df, event_summary, timing_stats_df, per_video_df, matched_df, discrepancies_df=None):
    """
    Exports all numerical evaluation metrics and the discrepancies audit table into an organized multi-sheet Excel workbook.
    """
    try:
        with pd.ExcelWriter(excel_path) as writer:
            summary_data = [
                {'Metric': 'Frame Classification Accuracy (Refined)', 'Value': f"{frame_metrics_df.loc['Refined_Overall', 'Accuracy']*100:.2f}%"},
                {'Metric': 'Frame Macro F1 Score (Refined)', 'Value': f"{frame_metrics_df.loc['Refined_Overall', 'F1_Score']:.4f}"},
                {'Metric': 'Frame Macro F1 Score (Raw)', 'Value': f"{frame_metrics_df.loc['Raw_Overall', 'F1_Score']:.4f}"},
                {'Metric': 'Hunt Event Detection Sensitivity / Recall', 'Value': f"{event_summary['recall']*100:.2f}%"},
                {'Metric': 'Hunt Event Detection Precision', 'Value': f"{event_summary['precision']*100:.2f}%"},
                {'Metric': 'Hunt Event Detection F1 Score', 'Value': f"{event_summary['f1_score']:.4f}"},
                {'Metric': 'Total Ground Truth Hunts', 'Value': event_summary['total_true']},
                {'Metric': 'Total Predicted Hunts', 'Value': event_summary['total_pred']},
                {'Metric': 'True Positives (Detected Hunts)', 'Value': event_summary['true_positives']},
                {'Metric': 'False Positives (False Alarms)', 'Value': event_summary['false_positives']},
                {'Metric': 'False Negatives (Missed Hunts)', 'Value': event_summary['false_negatives']},
                {'Metric': 'Discrepancies Flagged for Review', 'Value': len(discrepancies_df) if discrepancies_df is not None else 0},
            ]
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            if discrepancies_df is not None and len(discrepancies_df) > 0:
                discrepancies_df.to_excel(writer, sheet_name='Discrepancies_Audit', index=False)
                
            frame_metrics_df.to_excel(writer, sheet_name='Frame_Metrics')
            timing_stats_df.to_excel(writer, sheet_name='Timing_Accuracy', index=False)
            per_video_df.to_excel(writer, sheet_name='Per_Video_Breakdown', index=False)
            matched_df.to_excel(writer, sheet_name='Matched_Hunts_Detail', index=False)
            
        print(f"Metrics report exported: {excel_path}")
    except Exception as e:
        print(f"Error exporting metrics Excel: {e}")

def batch_process_dlc_folder(
    folder_path: Optional[str],
    model_path: str,
    dlc_files: Optional[Union[str, List[str]]] = None,
    device: Optional[torch.device] = None,
    label_folder: Optional[str] = None,
    label_format: Optional[str] = None,
    window_size: int = 6000,
    step_size: int = 1000,
    output_subdir: str = "excel_files",
    plot_subdir: str = "plots",
    prob_subdir: str = "probs",
    filter_poops: bool = True,
    poop_dist_thresh: float = 6.0,
    poop_thresh: float = 0.9,
    transition_weight: float = 5.0,
    temperature: float = 1.0,
    continuous_cricket_confidence: Optional[bool] = None
):
    """
    Executes model batch inference across all DLC tracking files in folder_path,
    or only the given dlc_files (folder_path then defaults to the first file's folder).
    Saves:
        - Class probabilities CSVs into prob_subdir
        - Structured analyzed event Excel sheets into output_subdir
        - Diagnostic prediction timeline plots into plot_subdir
    """
    if isinstance(dlc_files, str):
        dlc_files = [dlc_files]
    if not folder_path:
        if not dlc_files:
            raise ValueError("Provide folder_path and/or dlc_files.")
        folder_path = os.path.dirname(os.path.abspath(dlc_files[0]))

    pipeline = MouseInferencePipeline(
        model_weights_path=model_path,
        device=device,
        continuous_cricket_confidence=continuous_cricket_confidence
    )

    output_dir = os.path.join(folder_path, output_subdir) if output_subdir else folder_path
    plot_dir = os.path.join(folder_path, plot_subdir) if plot_subdir else folder_path
    prob_dir = os.path.join(folder_path, prob_subdir) if prob_subdir else folder_path
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(prob_dir, exist_ok=True)
    
    # Search for DLC files: prioritize .csv, falling back to .h5 only when no .csv exists
    all_candidates = glob.glob(os.path.join(folder_path, "*.csv")) + glob.glob(os.path.join(folder_path, "*.h5"))
    def _is_valid_dlc(f_path: str) -> bool:
        base_l = os.path.basename(f_path).lower()
        if 'dlc' not in base_l:
            return False
        if 'labeledinds' in base_l or 'labeledbouts' in base_l:
            return False
        if base_l.endswith('_probs.csv') or base_l.endswith('_analyzed.csv') or base_l.endswith('.xlsx'):
            return False
        return True

    if dlc_files:
        missing = [f for f in dlc_files if not os.path.exists(f)]
        if missing:
            raise FileNotFoundError(f"DLC file(s) not found: {missing}")
        all_candidates = list(dlc_files)
    valid_candidates = [f for f in all_candidates if _is_valid_dlc(f)]
    # Group by base stem so .csv is preferred over .h5
    seen_stems = {}
    for f in sorted(valid_candidates, key=lambda p: (0 if p.lower().endswith('.csv') else 1)):
        stem = re.sub(r'DLC.*(?:\.csv|\.h5)$', '', os.path.basename(f), flags=re.IGNORECASE).lower()
        if stem not in seen_stems:
            seen_stems[stem] = f

    dlc_files = sorted(list(seen_stems.values()))
    print(f"Found {len(dlc_files)} DLC tracking file(s) in {folder_path}")
    
    for dlc_file in dlc_files:
        filename = os.path.basename(dlc_file)
        clean_base = re.sub(r'DLC.*(?:\.csv|\.h5)$', '', filename, flags=re.IGNORECASE)
        if clean_base == filename:
            clean_base = os.path.splitext(filename)[0]
            
        print(f"\nProcessing {filename} (Base: {clean_base})...")
        
        # Determine label path
        label_path = None
        if label_folder:
            if os.path.isfile(label_folder):
                label_path = label_folder
            elif os.path.isdir(label_folder):
                cand_exact = os.path.join(label_folder, f"{clean_base}_analyzed.xlsx")
                if os.path.exists(cand_exact):
                    label_path = cand_exact
                else:
                    excel_candidates = glob.glob(os.path.join(label_folder, "*.xlsx")) + glob.glob(os.path.join(label_folder, "*.csv"))
                    diane_masters = [c for c in excel_candidates if any(k in os.path.basename(c).lower() for k in ('diane', 'master', 'cricket', 'exp'))]
                    if diane_masters:
                        label_path = diane_masters[0]
                    elif len(excel_candidates) == 1:
                        label_path = excel_candidates[0]
                    else:
                        for cand in excel_candidates:
                            if clean_base.lower() in os.path.basename(cand).lower():
                                label_path = cand
                                break
                                
        # Check for matching video file next to the DLC file
        dlc_dir = os.path.dirname(os.path.abspath(dlc_file))
        video_candidates = [
            os.path.join(dlc_dir, f"{clean_base}.avi"),
            os.path.join(dlc_dir, f"{clean_base}.mp4"),
            os.path.join(dlc_dir, f"{clean_base}_VIDEO.avi"),
            os.path.join(dlc_dir, f"{clean_base}_VIDEO.mp4"),
        ]
        video_file = next((v for v in video_candidates if os.path.exists(v)), None)
        if video_file is None:
            print(f"Warning: no video found for {clean_base}; FPS/width fall back to config defaults.")
        
        try:
            res = pipeline.run_on_session(
                dlc_path=dlc_file,
                video_path=video_file,
                label_path=label_path,
                refine_events=True,
                window_size=window_size,
                step_size=step_size,
                label_format=label_format,
                filter_poops=filter_poops,
                poop_dist_thresh=poop_dist_thresh,
                poop_thresh=poop_thresh,
                transition_weight=transition_weight,
                temperature=temperature,
                paradigm="mouse"
            )
            
            probs = res['probabilities']
            preds = res['raw_predictions']
            events = res['refined_events']
            refined_path = res.get('refined_path')
            fps = res.get('fps', 30.0)
            
            # 1. Save Probabilities CSV
            prob_path = os.path.join(prob_dir, f"{clean_base}_probs.csv")
            prob_df = pd.DataFrame(probs, columns=EVAL_CLASS_NAMES)
            prob_df.to_csv(prob_path, index=False)
            
            # 2. Save Analyzed Excel
            excel_path = os.path.join(output_dir, f"{clean_base}_analyzed.xlsx")
            generate_mouse_analyzed_excel(events, excel_path, fps)
            
            # 3. Save Diagnostic Inference Plot
            plot_path = os.path.join(plot_dir, f"{clean_base}_analyzed.png")
            plot_inference_results(
                probs=probs,
                preds=preds,
                refined_events=events,
                refined_path=refined_path,
                label_path=label_path,
                label_format=label_format,
                dlc_path=dlc_file,
                save_path=plot_path,
                title=clean_base,
                paradigm="mouse"
            )
            print(f"Completed inference for {clean_base} (Found {len(events)} hunt events).")
        except Exception as e:
            print(f"Error processing {clean_base}: {e}")
            import traceback
            traceback.print_exc()

def load_and_align_evaluation_data(
    folder_path: str,
    label_folder: str,
    label_format: Optional[str] = None,
    output_subdir: str = "excel_files",
    prob_subdir: str = "probs"
):
    """
    Scans folder_path, loads saved predictions (_probs.csv, _analyzed.xlsx),
    matches manual labels in label_folder, and returns aligned evaluation data per video.
    """
    output_dir = os.path.join(folder_path, output_subdir) if output_subdir else folder_path
    prob_dir = os.path.join(folder_path, prob_subdir) if prob_subdir else folder_path
    
    all_candidates = glob.glob(os.path.join(folder_path, "*.csv")) + glob.glob(os.path.join(folder_path, "*.h5"))
    def _is_valid_dlc_eval(f_path: str) -> bool:
        base_l = os.path.basename(f_path).lower()
        if 'dlc' not in base_l:
            return False
        if 'labeledinds' in base_l or 'labeledbouts' in base_l:
            return False
        if base_l.endswith('_probs.csv') or base_l.endswith('_analyzed.csv') or base_l.endswith('.xlsx'):
            return False
        return True

    valid_candidates = [f for f in all_candidates if _is_valid_dlc_eval(f)]
    seen_stems = {}
    for f in sorted(valid_candidates, key=lambda p: (0 if p.lower().endswith('.csv') else 1)):
        stem = re.sub(r'DLC.*(?:\.csv|\.h5)$', '', os.path.basename(f), flags=re.IGNORECASE).lower()
        if stem not in seen_stems:
            seen_stems[stem] = f

    dlc_files = sorted(list(seen_stems.values()))
    if len(dlc_files) == 0:
        raise FileNotFoundError(f"No DLC tracking files found in {folder_path}")
        
    aligned_data = []
    
    for dlc_file in dlc_files:
        filename = os.path.basename(dlc_file)
        clean_base = re.sub(r'DLC.*(?:\.csv|\.h5)$', '', filename, flags=re.IGNORECASE)
        if clean_base == filename:
            clean_base = os.path.splitext(filename)[0]
            
        # 1. Find probability file
        prob_candidates = [
            os.path.join(prob_dir, f"{clean_base}_probs.csv"),
            os.path.join(folder_path, f"{clean_base}_probs.csv"),
        ]
        prob_file = next((p for p in prob_candidates if os.path.exists(p)), None)
        if prob_file is None:
            print(f"Warning: No probability file found for {clean_base}. Skipping this video.")
            continue
            
        # 2. Find analyzed excel file
        excel_candidates = [
            os.path.join(output_dir, f"{clean_base}_analyzed.xlsx"),
            os.path.join(folder_path, f"{clean_base}_analyzed.xlsx")
        ]
        excel_file = next((p for p in excel_candidates if os.path.exists(p)), None)
        
        pred_events = []
        if excel_file and os.path.exists(excel_file):
            try:
                pred_df = pd.read_excel(excel_file)
                for _, row in pred_df.iterrows():
                    enter = int(row['Enter_Frame']) if pd.notna(row.get('Enter_Frame')) else 0
                    first_app = int(row['First_Approach_Frame']) if pd.notna(row.get('First_Approach_Frame')) else None
                    capture = int(row['Capture_Frame']) if pd.notna(row.get('Capture_Frame')) else None
                    cons_end = int(row['Consumption_End_Frame']) if pd.notna(row.get('Consumption_End_Frame')) else None
                    
                    final_app = None
                    for c_cand in ['Final_Approach_Frame', 'Pursuit_Frame', 'Pursuit_Start_Frame', 'Approach_Start_Frame', 'Approach Start']:
                        if c_cand in row and pd.notna(row[c_cand]):
                            try:
                                v = int(float(row[c_cand]))
                                if v > enter:
                                    final_app = v
                                    break
                            except (ValueError, TypeError): pass
                    if final_app is None:
                        for l_cand in ['Latency_to_Attack_Frames', 'Latency_to_Pursuit_Frames', 'Attack_Latency_Frames']:
                            if l_cand in row and pd.notna(row[l_cand]):
                                try:
                                    lat = int(float(row[l_cand]))
                                    if lat > 0:
                                        final_app = enter + lat
                                        break
                                except (ValueError, TypeError): pass

                    pred_events.append({
                        'enter': enter,
                        'first_approach': first_app,
                        'final_approach': final_app,
                        'pursuit_start': final_app,
                        'capture': capture,
                        'consumption_end': cons_end
                    })
            except Exception as e:
                print(f"Warning: Failed to read {excel_file}: {e}")
                
        # 3. Load probabilities
        prob_df = pd.read_csv(prob_file)
        if all(c in prob_df.columns for c in EVAL_CLASS_NAMES):
            probs_array = prob_df[EVAL_CLASS_NAMES].values
        else:
            probs_array = prob_df.values[:, :len(EVAL_CLASS_NAMES)]
        num_frames = len(probs_array)
        
        preds_raw = np.argmax(probs_array, axis=1)
        if len(pred_events) == 0:
            pred_events = refine_predictions(probs_array, preds_raw, paradigm="mouse")
        preds_refined = events_to_sequence(pred_events, num_frames, paradigm="mouse")
        
        # 4. Match Ground Truth Labels
        label_path = None
        if label_folder:
            if os.path.isfile(label_folder):
                label_path = label_folder
            elif os.path.isdir(label_folder):
                cand_exact = os.path.join(label_folder, f"{clean_base}_analyzed.xlsx")
                if os.path.exists(cand_exact):
                    label_path = cand_exact
                else:
                    excel_cands = glob.glob(os.path.join(label_folder, "*.xlsx")) + glob.glob(os.path.join(label_folder, "*.csv"))
                    diane_masters = [c for c in excel_cands if any(k in os.path.basename(c).lower() for k in ('diane', 'master', 'cricket', 'exp'))]
                    if diane_masters:
                        label_path = diane_masters[0]
                    elif len(excel_cands) == 1:
                        label_path = excel_cands[0]
                    else:
                        for cand in excel_cands:
                            if clean_base.lower() in os.path.basename(cand).lower():
                                label_path = cand
                                break
                                
        if label_path is None or not os.path.exists(label_path):
            print(f"Notice: No ground truth label found for {clean_base}. Excluding from evaluation results.")
            continue
            
        try:
            import dataset_utils
            _, _, video_fps = dataset_utils.get_video_params(dlc_file)
        except Exception:
            video_fps = 30.0
            
        y_true, true_hunts = load_ground_truth_sequence(
            label_path=label_path,
            dlc_path=dlc_file,
            fps=video_fps,
            num_frames=num_frames,
            label_format=label_format,
            paradigm="mouse"
        )
        
        if y_true is None or true_hunts is None or len(true_hunts) == 0:
            print(f"Notice: No corresponding log in master Excel for {clean_base}. Excluding from evaluation results.")
            continue
            
        aligned_data.append({
            'video_name': clean_base,
            'dlc_file': dlc_file,
            'fps': video_fps,
            'num_frames': num_frames,
            'probs': probs_array,
            'y_raw': preds_raw,
            'y_refined': preds_refined,
            'y_true': y_true,
            'true_hunts': true_hunts,
            'pred_events': pred_events
        })
        
    print(f"Successfully aligned {len(aligned_data)} video session(s) with manual labels.")
    return aligned_data

def batch_process_and_evaluate(
    folder_path: str,
    model_path: str,
    label_folder: str,
    device: Optional[torch.device] = None,
    label_format: Optional[str] = None,
    window_size: int = 6000,
    step_size: int = 1000,
    output_subdir: str = "excel_files",
    plot_subdir: str = "plots",
    prob_subdir: str = "probs",
    eval_subdir: str = "evaluation_results",
    rerun_inference: bool = True,
    filter_poops: bool = True,
    poop_dist_thresh: float = 6.0,
    poop_thresh: float = 0.9,
    transition_weight: float = 5.0,
    temperature: float = 1.0,
    timing_threshold_sec: float = 5.0,
    duration_threshold_sec: float = 3.0,
    continuous_cricket_confidence: Optional[bool] = None
):
    """
    Batch processes a folder of mouse DLC tracking files and comprehensively evaluates performance
    against ground truth manual labels (Diane format, Lucas, or Dan).
    
    Produces:
        - Frame-level Confusion Matrices & Per-Class F1 Comparisons
        - Multi-class One-vs-Rest ROC & Precision-Recall Curves
        - Hunt Event Detection Rates (Sensitivity, Precision, F1)
        - Transition Timing Accuracy (Hunt Enter, First App, Final App, Capture) [Linear & Log Scales]
        - Behavioral Metric Scatter Correlations (Total Duration, Attack Duration, First App Latency)
        - Per-Video Cross-Session Performance Breakdown
        - Master Executive Dashboard
        - Discrepancies Audit Log CSV
        - Multi-Sheet Evaluation Report Excel Workbook
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    eval_dir = os.path.join(folder_path, eval_subdir)
    os.makedirs(eval_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # Step 1: Execute Batch Inference (if requested)
    # -------------------------------------------------------------
    if rerun_inference:
        print("\n=======================================================")
        print("  STEP 1: Running Model Batch Inference on DLC Folder")
        print("=======================================================")
        batch_process_dlc_folder(
            folder_path=folder_path,
            model_path=model_path,
            device=device,
            label_folder=label_folder,
            label_format=label_format,
            window_size=window_size,
            step_size=step_size,
            output_subdir=output_subdir,
            plot_subdir=plot_subdir,
            prob_subdir=prob_subdir,
            filter_poops=filter_poops,
            poop_dist_thresh=poop_dist_thresh,
            poop_thresh=poop_thresh,
            transition_weight=transition_weight,
            temperature=temperature,
            continuous_cricket_confidence=continuous_cricket_confidence
        )
    else:
        print("\n[Info] Reusing existing inference outputs (rerun_inference=False).")
        
    # -------------------------------------------------------------
    # Step 2: Load and Align Predictions with Ground Truth
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("  STEP 2: Aligning Predictions with Ground Truth Labels")
    print("=======================================================")
    aligned_data = load_and_align_evaluation_data(
        folder_path=folder_path,
        label_folder=label_folder,
        label_format=label_format,
        output_subdir=output_subdir,
        prob_subdir=prob_subdir
    )
    
    if len(aligned_data) == 0:
        raise RuntimeError("No aligned data could be formed between predictions and manual labels.")
        
    # -------------------------------------------------------------
    # Step 3: Compute Comprehensive Performance Metrics
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("  STEP 3: Computing Frame & Event Accuracy Metrics")
    print("=======================================================")
    
    all_true = np.concatenate([d['y_true'] for d in aligned_data])
    all_raw = np.concatenate([d['y_raw'] for d in aligned_data])
    all_refined = np.concatenate([d['y_refined'] for d in aligned_data])
    all_probs = np.vstack([d['probs'] for d in aligned_data])
    
    all_matched = []
    all_unmatched_true = []
    all_unmatched_pred = []
    per_video_records = []
    
    for d in aligned_data:
        v_matched, v_unmatched_t, v_unmatched_p = match_hunt_events(
            d['true_hunts'], d['pred_events'], fps=d['fps'], video_name=d['video_name']
        )
        all_matched.extend(v_matched)
        all_unmatched_true.extend(v_unmatched_t)
        all_unmatched_pred.extend(v_unmatched_p)
        
        v_true = d['y_true']
        v_ref = d['y_refined']
        v_f1 = f1_score(v_true, v_ref, average='macro', zero_division=0)
        v_num_true = len(d['true_hunts'])
        v_num_pred = len(d['pred_events'])
        v_recall = len(v_matched) / v_num_true if v_num_true > 0 else (1.0 if v_num_pred == 0 else 0.0)
        
        per_video_records.append({
            'video_name': d['video_name'],
            'num_frames': d['num_frames'],
            'fps': d['fps'],
            'true_hunts': v_num_true,
            'pred_hunts': v_num_pred,
            'matched_hunts': len(v_matched),
            'detection_recall': v_recall,
            'refined_f1': v_f1
        })
        
    per_video_df = pd.DataFrame(per_video_records)
    matched_df = pd.DataFrame(all_matched)
    
    # Event Detection Summary
    total_true = len(all_matched) + len(all_unmatched_true)
    total_pred = len(all_matched) + len(all_unmatched_pred)
    tp = len(all_matched)
    fp = len(all_unmatched_pred)
    fn = len(all_unmatched_true)
    
    ev_recall = tp / total_true if total_true > 0 else 0.0
    ev_prec = tp / total_pred if total_pred > 0 else 0.0
    ev_f1 = (2 * ev_prec * ev_recall / (ev_prec + ev_recall)) if (ev_prec + ev_recall) > 0 else 0.0
    
    event_summary = {
        'total_true': total_true,
        'total_pred': total_pred,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'recall': ev_recall,
        'precision': ev_prec,
        'f1_score': ev_f1
    }
    
    # Frame Metrics DataFrame
    n_cls = len(EVAL_CLASS_NAMES)
    p_raw, r_raw, f1_raw, supp = precision_recall_fscore_support(all_true, all_raw, labels=list(range(n_cls)), zero_division=0)
    p_ref, r_ref, f1_ref, _ = precision_recall_fscore_support(all_true, all_refined, labels=list(range(n_cls)), zero_division=0)
    
    frame_metrics_data = {}
    for idx, cname in enumerate(EVAL_CLASS_NAMES):
        frame_metrics_data[f"Raw_{cname}"] = {'Precision': p_raw[idx], 'Recall': r_raw[idx], 'F1_Score': f1_raw[idx], 'Support_Frames': supp[idx], 'Accuracy': np.nan}
        frame_metrics_data[f"Refined_{cname}"] = {'Precision': p_ref[idx], 'Recall': r_ref[idx], 'F1_Score': f1_ref[idx], 'Support_Frames': supp[idx], 'Accuracy': np.nan}
        
    frame_metrics_data['Raw_Overall'] = {
        'Precision': np.mean(p_raw),
        'Recall': np.mean(r_raw),
        'F1_Score': f1_score(all_true, all_raw, average='macro', zero_division=0),
        'Support_Frames': len(all_true),
        'Accuracy': accuracy_score(all_true, all_raw)
    }
    frame_metrics_data['Refined_Overall'] = {
        'Precision': np.mean(p_ref),
        'Recall': np.mean(r_ref),
        'F1_Score': f1_score(all_true, all_refined, average='macro', zero_division=0),
        'Support_Frames': len(all_true),
        'Accuracy': accuracy_score(all_true, all_refined)
    }
    frame_metrics_df = pd.DataFrame.from_dict(frame_metrics_data, orient='index')
    
    # Timing Errors Summary across 4 transitions
    timing_rows = []
    if len(matched_df) > 0:
        transitions_to_eval = [
            ('enter_error_sec', 'Hunt Enter'),
            ('first_approach_error_sec', 'First Approach'),
            ('final_approach_error_sec', 'Final Approach'),
            ('capture_error_sec', 'Capture')
        ]
        for col, label in transitions_to_eval:
            if col in matched_df:
                vals = matched_df[col].dropna().values
                if len(vals) > 0:
                    timing_rows.append({
                        'Transition': label,
                        'MAE_Seconds': float(np.mean(np.abs(vals))),
                        'Median_Error_Seconds': float(np.median(vals)),
                        'Mean_Error_Seconds': float(np.mean(vals)),
                        'Std_Error_Seconds': float(np.std(vals)),
                        'IQR_Seconds': float(np.percentile(vals, 75) - np.percentile(vals, 25)),
                        'N_Hunts': len(vals)
                    })
    timing_stats_df = pd.DataFrame(timing_rows)
    
    # -------------------------------------------------------------
    # Step 4: Generate Professional Evaluation Plots
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("  STEP 4: Generating Publication-Grade Visualizations")
    print("=======================================================")
    
    fig_cm_path = os.path.join(eval_dir, "01_confusion_matrices_and_f1.png")
    plot_confusion_matrices_and_f1(all_true, all_raw, all_refined, save_path=fig_cm_path)
    
    fig_roc_path = os.path.join(eval_dir, "02_multiclass_roc_and_pr_curves.png")
    plot_multiclass_roc_and_pr_curves(all_true, all_probs, save_path=fig_roc_path)
    
    fig_det_path = os.path.join(eval_dir, "03_hunt_detection_and_timing_errors.png")
    plot_hunt_detection_and_timing_errors(event_summary, matched_df, save_path=fig_det_path)
    
    fig_corr_path = os.path.join(eval_dir, "04_behavioral_metric_correlations.png")
    plot_duration_correlations(matched_df, save_path=fig_corr_path)
    
    fig_vid_path = os.path.join(eval_dir, "05_per_video_performance.png")
    plot_per_video_performance(per_video_df, save_path=fig_vid_path)
    
    fig_dash_path = os.path.join(eval_dir, "00_model_evaluation_dashboard.png")
    plot_master_evaluation_dashboard(all_true, all_raw, all_refined, all_probs, event_summary, matched_df, save_path=fig_dash_path)
    
    # -------------------------------------------------------------
    # Step 5: Build Discrepancy Audit Table & Export Reports
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("  STEP 5: Generating Discrepancy Audit Log & Excel Reports")
    print("=======================================================")
    
    discrepancies_df = build_discrepancy_audit_table(
        all_matched=all_matched,
        all_unmatched_true=all_unmatched_true,
        all_unmatched_pred=all_unmatched_pred,
        timing_threshold_sec=timing_threshold_sec,
        duration_threshold_sec=duration_threshold_sec
    )
    
    audit_csv_path = os.path.join(eval_dir, "discrepancies_audit.csv")
    discrepancies_df.to_csv(audit_csv_path, index=False)
    print(f"Discrepancy audit log exported: {audit_csv_path}")
    
    excel_report_path = os.path.join(eval_dir, "evaluation_metrics.xlsx")
    export_evaluation_metrics(
        excel_path=excel_report_path,
        frame_metrics_df=frame_metrics_df,
        event_summary=event_summary,
        timing_stats_df=timing_stats_df,
        per_video_df=per_video_df,
        matched_df=matched_df,
        discrepancies_df=discrepancies_df
    )
    
    # -------------------------------------------------------------
    # Console Executive Summary
    # -------------------------------------------------------------
    print("\n" + "="*65)
    print("          MOUSE HUNT MODEL EVALUATION EXECUTIVE SUMMARY")
    print("="*65)
    print(f"Total Videos Evaluated       : {len(aligned_data)}")
    print(f"Total Frames Analyzed        : {len(all_true):,}")
    print(f"Refined Frame Accuracy       : {frame_metrics_df.loc['Refined_Overall', 'Accuracy']*100:.2f}%")
    print(f"Refined Frame Macro F1       : {frame_metrics_df.loc['Refined_Overall', 'F1_Score']:.4f} (Raw: {frame_metrics_df.loc['Raw_Overall', 'F1_Score']:.4f})")
    print(f"Hunt Event Detection Rate    : {event_summary['recall']*100:.1f}% ({tp}/{total_true} true hunts detected)")
    print(f"Hunt Detection Precision     : {event_summary['precision']*100:.1f}% ({fp} false alarms)")
    print(f"Hunt Event F1 Score          : {event_summary['f1_score']*100:.1f}%")
    print("="*65)
    
    return {
        'frame_metrics': frame_metrics_df,
        'event_summary': event_summary,
        'timing_stats': timing_stats_df,
        'per_video': per_video_df,
        'matched_hunts': matched_df,
        'discrepancies': discrepancies_df,
        'eval_dir': eval_dir,
        'master_dashboard': fig_dash_path,
        'metrics_excel': excel_report_path,
        'audit_csv': audit_csv_path
    }

if __name__ == "__main__":
    # Example usage analogous to RatHunt_Inference:
    model_weights = r"C:\Users\bcary\Documents\Work\CODE\CricketCode\LSTM_hunt_label_mouse\models\mouse_models\HuntLSTM_mouse_feat45_dim256_lay3_ep157_valf1=0.8410.pth"
    video_path = r"D:\Work\DATA\Diane\Exp30-32-DLCvids\20231025_DB_Exp30_day1-231025-091250_YFP-286-231025-091319_Cam1.avi"
    dlc_file = r"D:\Work\DATA\Diane\Exp30-32-DLCvids\20231025_DB_Exp30_day1-231025-091250_YFP-286-231025-091319_Cam1DLC_resnet50_CrickHunt_tailApr26shuffle1_1000000.csv"
    label_file = r"D:\Work\DATA\Diane\Exp30-32-DLCvids\MasterBehExcels\Cricket_hunting_adults_20240923-Exp30and41.xlsx"

    output_dir = r"C:\Users\bcary\Documents\Work\CODE\CricketCode\LSTM_hunt_label_mouse\vids"
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------
    # Example: Batch Process & Evaluate on a folder of mouse videos
    # -------------------------------------------------------------
    eval_results = batch_process_and_evaluate(
        folder_path=r"D:\Work\DATA\Diane\Exp30-32-DLCvids",
        model_path=model_weights,
        label_folder=r"D:\Work\DATA\Diane\Exp30-32-DLCvids\MasterBehExcels\Cricket_hunting_adults_20240923-Exp30and41.xlsx",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        label_format="diane_parts", # Auto-detects 'diane' or 'diane_parts'
        window_size=6000,
        step_size=1000,
        transition_weight=100.0,
        temperature=2,
        poop_thresh=0.92,
        poop_dist_thresh=6.0,
        filter_poops=True,
        output_subdir="excel_files",
        plot_subdir="plots",
        prob_subdir="probs",
        eval_subdir="evaluation_results",
        rerun_inference=True # Set to False to quickly re-plot without rerunning inference
    )

    # 1. Run inference on single session
    pipeline = MouseInferencePipeline(model_weights_path=model_weights)
    if os.path.exists(dlc_file):
        res = pipeline.run_on_session(
            dlc_path=dlc_file,
            video_path=video_path if os.path.exists(video_path) else None,
            label_path=label_file if os.path.exists(label_file) else None,
            refine_events=True,
            max_mem=0.8,
            transition_weight=100.0,
            temperature=2,
            poop_thresh=0.92,
            poop_dist_thresh=6.0,
            filter_poops=True,
        )
        probs = res['probabilities']
        preds = res['raw_predictions']
        refined_events = res['refined_events']

        print(f"\nFound {len(refined_events)} refined hunt events via HMM Viterbi decoding:")
        for h_i, ev in enumerate(refined_events):
            print(f"  Hunt {h_i+1}: Enter={ev['enter']}, FirstApp={ev['first_approach']}, Strike={ev['pursuit_start']}, Capture={ev['capture']}, EatEnd={ev['consumption_end']}")

        # 2. Plot inference results (Probabilities + Raw Predictions vs HMM Viterbi Refined + Manual Labels)
        plot_inference_results(
            probs, preds,
            refined_events=refined_events,
            refined_path=res.get('refined_path'),
            label_path=label_file if os.path.exists(label_file) else None,
            dlc_path=dlc_file
        )
        plt.show()

        # 3. Render hunting video clip (optional)
        start_frame = 99000
        end_frame = 101500
        clip_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_{start_frame}_{end_frame}_analyzed.mp4"
        output_path = os.path.join(output_dir, clip_name)

        if os.path.exists(video_path):
            create_hunting_video_clip(
                video_path=video_path,
                probs=probs,
                preds=preds,
                start_frame=start_frame,
                end_frame=end_frame,
                output_path=output_path,
                trailing_frames=300,
                dlc_path=dlc_file,
                circle_radius=4,
                include_blurry_cricket=True,
                poop_dist_thresh=6,
                filter_poops=False,
                plot_poops=True
            )
    else:
        print(f"File not found: {dlc_file}")



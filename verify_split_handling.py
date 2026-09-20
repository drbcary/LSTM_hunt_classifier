"""
verify_split_handling.py
Verification tool for multi-part split video handling in mouse hunting sessions.
Plots whole-day hunting timelines with manual 5-class behavioral step labels
overlayed on animal speed and/or tail angle for split sessions.

Supports both:
  1. 'diane_parts' format: Split videos have frame numbers resetting to 0 for each part.
                           Plots stacked subplots for each part on its own timeline.
  2. 'diane' format (original): Split videos stitched into a single continuous timeline.
                                Plots the full continuous day with stitch boundaries marked.
"""

import os
import glob
import argparse
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dataset_utils
from dataset_utils import BASE_SERVER_PATH
from dataset.matcher import DatasetMatcher
from labels.diane_parser import DianeExcelLabelParser
from pose.dlc_adapter import extract_filename_timestamp

# 5-Class Mouse Behavior metadata
CLASS_NAMES = ['Background', 'Cricket Present', 'First Approach', 'Attack', 'Eating']
CLASS_COLORS = {
    0: ('gray', 0.0),
    1: ('gold', 0.18),
    2: ('darkorange', 0.25),
    3: ('crimson', 0.35),
    4: ('forestgreen', 0.22)
}


def build_5class_targets(hunts: List[Dict[str, Any]], total_frames: int, video_fps: float) -> np.ndarray:
    """Builds a frame-by-frame 1D array of 5-class behavioral state ground truth."""
    targets = np.zeros(total_frames, dtype=int)
    for h in hunts:
        enter = int(h['enter'])
        capture = int(h['capture'])
        first_app = int(h.get('first_approach') or enter)
        pursuit = int(h.get('pursuit_start') or h.get('final_approach') or first_app)
        cons_end = int(h.get('consumption_end') or (capture + int(10 * video_fps)))

        local_enter = max(0, min(total_frames, enter))
        local_first = max(0, min(total_frames, first_app))
        local_pursuit = max(0, min(total_frames, pursuit))
        local_capture = max(0, min(total_frames, capture))
        local_cons = max(0, min(total_frames, cons_end))

        if local_first > local_enter:
            targets[local_enter:local_first] = 1
        if local_pursuit > local_first:
            targets[local_first:local_pursuit] = 2
        if local_capture > local_pursuit:
            targets[local_pursuit:local_capture] = 3
        if local_cons > local_capture:
            targets[local_capture:local_cons] = 4

    return targets


def plot_day_split_verification(
    animal: str,
    day: int,
    session_parts_data: List[Dict[str, Any]],
    save_dir: str = "figs/split_verification",
    features_to_plot: Tuple[str, ...] = ("speed", "tail_angle"),
    show_plot: bool = False,
    verbose: bool = True
) -> List[str]:
    """
    Plots a multi-panel figure for a single day's multi-part session in 'diane_parts' format.
    Each video part is plotted on its own subplot timeline starting from frame 0,
    showing the animal feature trace (speed or tail angle) on the left axis
    and the 5-class manual behavioral step line overlayed on the right axis.
    """
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []
    num_parts = len(session_parts_data)

    feature_specs = []
    if "speed" in features_to_plot or "both" in features_to_plot:
        feature_specs.append({
            'name': 'speed',
            'title': 'Mouse Speed',
            'candidates': ['predator_speed'],
            'ylabel': 'Normalized Speed',
            'color': 'royalblue',
            'suffix': 'speed'
        })
    if "tail_angle" in features_to_plot or "tail" in features_to_plot or "both" in features_to_plot:
        feature_specs.append({
            'name': 'tail_angle',
            'title': 'Distal Tail Angle',
            'candidates': ['distal_3pt_tail_angle', 'distal_3pt_angle', 'distal_tail_angle'],
            'ylabel': 'Angle (deg / 180)',
            'color': 'darkviolet',
            'suffix': 'tail_angle'
        })

    for spec in feature_specs:
        fig, axes = plt.subplots(num_parts, 1, figsize=(16, max(4.0, 3.5 * num_parts)), squeeze=False)
        fig.suptitle(
            f"Multi-Part Split Verification (diane_parts): {spec['title']} vs 5-Class Manual Labels\n"
            f"Animal: {animal} | Day: {day} (Total Parts: {num_parts})",
            fontsize=13, fontweight='bold', y=0.99
        )

        for p_idx, part_info in enumerate(session_parts_data):
            ax = axes[p_idx, 0]
            dlc_path = part_info['dlc_path']
            video_path = part_info.get('video_path')
            hunts = part_info['hunts']
            part_num = part_info.get('part', p_idx + 1)

            # Load DLC and extract features
            width, height, fps = dataset_utils.get_video_params(dlc_path, video_path=video_path, verbose=False)
            pose = dataset_utils.load_mouse_dlc(dlc_path, fps)
            if pose is None:
                ax.text(0.5, 0.5, f"Could not load DLC for Part {part_num}", ha='center', va='center')
                continue

            feat_mat, manifest = dataset_utils.generate_mouse_features(pose, width, fps)
            total_frames = len(feat_mat)
            time_sec = np.arange(total_frames) / fps

            # Find matching feature column
            feat_idx = None
            for cand in spec['candidates']:
                if cand in manifest.feature_names:
                    feat_idx = manifest.feature_names.index(cand)
                    break

            if feat_idx is not None:
                feat_trace = feat_mat[:, feat_idx]
                ax.plot(time_sec, feat_trace, color=spec['color'], linewidth=1.2, label=spec['title'])
            else:
                ax.text(0.5, 0.5, f"Feature {spec['title']} not found", ha='center', va='center')

            ax.set_ylabel(spec['ylabel'], color=spec['color'], fontsize=10, fontweight='bold')
            ax.tick_params(axis='y', labelcolor=spec['color'])
            ax.set_xlabel("Time from video start (seconds)", fontsize=9)
            ax.set_xlim(0, time_sec[-1] if len(time_sec) > 0 else 1)
            ax.grid(True, alpha=0.25)

            # Build and plot ground truth behavioral labels
            targets = build_5class_targets(hunts, total_frames, fps)
            ax2 = ax.twinx()
            ax2.step(time_sec, targets, where='post', color='black', linewidth=1.8, label='Manual 5-Class Label')
            ax2.set_yticks([0, 1, 2, 3, 4])
            ax2.set_yticklabels(CLASS_NAMES, fontsize=8)
            ax2.set_ylabel("Behavior Class", color='black', fontsize=10)
            ax2.set_ylim(-0.5, 4.5)

            # Shading by behavior epoch
            for c_idx, (c_col, c_alpha) in CLASS_COLORS.items():
                if c_alpha > 0:
                    mask = (targets == c_idx)
                    if np.any(mask):
                        ax2.fill_between(time_sec, -0.5, 4.5, where=mask, color=c_col, alpha=c_alpha)

            # Annotate individual cricket releases
            for h in hunts:
                c_num = h.get('metadata', {}).get('cricket') or h.get('cricket', '')
                c_lbl = f"C{c_num}" if c_num else "Hunt"
                h_enter_sec = h['enter'] / fps
                h_cap_sec = h['capture'] / fps
                ax2.axvline(h_enter_sec, color='goldenrod', linestyle='--', alpha=0.6, linewidth=1.0)
                ax2.axvline(h_cap_sec, color='crimson', linestyle=':', alpha=0.7, linewidth=1.2)
                ax.text(
                    h_enter_sec, ax.get_ylim()[1] * 0.92, f" {c_lbl}",
                    color='black', fontsize=8, fontweight='bold', rotation=0,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.4, edgecolor='none')
                )

            file_base = os.path.basename(dlc_path)
            ax.set_title(
                f"Part {part_num} ({len(hunts)} hunts, {total_frames} frames / {total_frames/fps:.1f}s) - {file_base}",
                fontsize=10, loc='left', fontweight='bold'
            )

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        out_path = os.path.join(save_dir, f"split_verification_{animal}_Day{day}_parts_{spec['suffix']}.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        saved_paths.append(out_path)
        if verbose:
            print(f"  [SAVED DIANE_PARTS] {out_path}")
        if show_plot:
            plt.show()
        plt.close(fig)

    return saved_paths


def plot_day_stitched_verification(
    animal: str,
    day: int,
    session_item: Dict[str, Any],
    save_dir: str = "figs/split_verification",
    features_to_plot: Tuple[str, ...] = ("speed", "tail_angle"),
    show_plot: bool = False,
    verbose: bool = True
) -> List[str]:
    """
    Plots the whole day's hunt for an original stitched Diane session.
    Shows the continuous timeline with stitch boundaries marked where video splits were concatenated.
    """
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []

    sorted_files = session_item.get('dlc_paths') or [session_item['dlc_path']]
    if isinstance(sorted_files, str):
        sorted_files = [sorted_files]

    num_parts = len(sorted_files)
    hunts = session_item.get('hunts', [])

    # Calculate stitch boundary frame indices by reading individual file lengths
    part_lengths = []
    for f in sorted_files:
        try:
            if f.endswith('.h5'):
                part_df = pd.read_hdf(f)
            else:
                part_df = pd.read_csv(f, header=[0, 1, 2], index_col=0)
            part_lengths.append(len(part_df))
        except Exception:
            part_lengths.append(0)

    stitch_boundaries = []
    running = 0
    for pl in part_lengths[:-1]:
        running += pl
        stitch_boundaries.append(running)

    width, height, fps = dataset_utils.get_video_params(sorted_files[0], video_path=session_item.get('video_path'), verbose=False)
    pose = dataset_utils.load_mouse_dlc(sorted_files, fps)
    if pose is None:
        if verbose:
            print(f"  [ERROR] Could not load stitched DLC for {animal} Day {day}")
        return []

    feat_mat, manifest = dataset_utils.generate_mouse_features(pose, width, fps)
    total_frames = len(feat_mat)
    time_sec = np.arange(total_frames) / fps

    feature_specs = []
    if "speed" in features_to_plot or "both" in features_to_plot:
        feature_specs.append({
            'name': 'speed',
            'title': 'Mouse Speed',
            'candidates': ['predator_speed'],
            'ylabel': 'Normalized Speed',
            'color': 'royalblue',
            'suffix': 'speed'
        })
    if "tail_angle" in features_to_plot or "tail" in features_to_plot or "both" in features_to_plot:
        feature_specs.append({
            'name': 'tail_angle',
            'title': 'Distal Tail Angle',
            'candidates': ['distal_3pt_tail_angle', 'distal_3pt_angle', 'distal_tail_angle'],
            'ylabel': 'Angle (deg / 180)',
            'color': 'darkviolet',
            'suffix': 'tail_angle'
        })

    for spec in feature_specs:
        fig, ax = plt.subplots(figsize=(18, 5.5))
        fig.suptitle(
            f"Original Stitched Diane Verification: {spec['title']} vs 5-Class Manual Labels\n"
            f"Animal: {animal} | Day: {day} (Stitched across {num_parts} video parts | Total Frames: {total_frames})",
            fontsize=13, fontweight='bold', y=0.99
        )

        feat_idx = None
        for cand in spec['candidates']:
            if cand in manifest.feature_names:
                feat_idx = manifest.feature_names.index(cand)
                break

        if feat_idx is not None:
            feat_trace = feat_mat[:, feat_idx]
            ax.plot(time_sec, feat_trace, color=spec['color'], linewidth=1.2, label=spec['title'])
        else:
            ax.text(0.5, 0.5, f"Feature {spec['title']} not found", ha='center', va='center')

        ax.set_ylabel(spec['ylabel'], color=spec['color'], fontsize=10, fontweight='bold')
        ax.tick_params(axis='y', labelcolor=spec['color'])
        ax.set_xlabel("Time from session start (seconds)", fontsize=10)
        ax.set_xlim(0, time_sec[-1] if len(time_sec) > 0 else 1)
        ax.grid(True, alpha=0.25)

        targets = build_5class_targets(hunts, total_frames, fps)
        ax2 = ax.twinx()
        ax2.step(time_sec, targets, where='post', color='black', linewidth=1.8, label='Manual 5-Class Label')
        ax2.set_yticks([0, 1, 2, 3, 4])
        ax2.set_yticklabels(CLASS_NAMES, fontsize=8)
        ax2.set_ylabel("Behavior Class", color='black', fontsize=10)
        ax2.set_ylim(-0.5, 4.5)

        # Shading
        for c_idx, (c_col, c_alpha) in CLASS_COLORS.items():
            if c_alpha > 0:
                mask = (targets == c_idx)
                if np.any(mask):
                    ax2.fill_between(time_sec, -0.5, 4.5, where=mask, color=c_col, alpha=c_alpha)

        # Annotate cricket releases
        for h in hunts:
            c_num = h.get('metadata', {}).get('cricket') or h.get('cricket', '')
            c_lbl = f"C{c_num}" if c_num else "Hunt"
            h_enter_sec = h['enter'] / fps
            h_cap_sec = h['capture'] / fps
            ax2.axvline(h_enter_sec, color='goldenrod', linestyle='--', alpha=0.6, linewidth=1.0)
            ax2.axvline(h_cap_sec, color='crimson', linestyle=':', alpha=0.7, linewidth=1.2)
            ax.text(
                h_enter_sec, ax.get_ylim()[1] * 0.90, f" {c_lbl}",
                color='black', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.4, edgecolor='none')
            )

        # Mark stitch boundaries with vertical lines and annotations
        for s_i, sb_frame in enumerate(stitch_boundaries):
            sb_sec = sb_frame / fps
            ax.axvline(sb_sec, color='darkmagenta', linestyle='-.', linewidth=2.0, alpha=0.85)
            ax.text(
                sb_sec, ax.get_ylim()[1] * 0.96, f" Stitch: Part {s_i+1} \u2192 {s_i+2} (frame {sb_frame})",
                color='darkmagenta', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='square,pad=0.2', facecolor='lavender', alpha=0.85, edgecolor='darkmagenta')
            )

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        out_path = os.path.join(save_dir, f"split_verification_{animal}_Day{day}_stitched_{spec['suffix']}.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        saved_paths.append(out_path)
        if verbose:
            print(f"  [SAVED STITCHED] {out_path}")
        if show_plot:
            plt.show()
        plt.close(fig)

    return saved_paths


def verify_split_handling(
    dlc_path: str,
    excel_path: str,
    save_dir: str = "figs/split_verification",
    format_type: str = "diane_parts",
    num_examples: int = 2,
    feature: str = "both",
    show_plot: bool = False,
    verbose: bool = True
) -> List[str]:
    """
    Finds sessions with video splitting in the dataset, verifies that split
    handling aligns labels with features correctly, and produces verification plots.

    Parameters:
      dlc_path: Path to DLC file or directory containing DLC files.
      excel_path: Path to Diane master Excel file.
      save_dir: Output directory for plots (default: 'figs/split_verification').
      format_type: 'diane_parts' (frame resets, unstitched), 'diane' (original stitched), or 'both'.
      num_examples: Number of split session examples to find and plot.
      feature: 'speed', 'tail_angle', or 'both'.
      show_plot: Show interactive matplotlib window.
      verbose: Print progress diagnostics.
    """
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Master Excel file not found: {excel_path}")

    # Determine dlc directory and optional single-session filter
    target_animal = None
    target_day = None

    if os.path.isfile(dlc_path):
        dlc_dir = os.path.dirname(dlc_path)
        matcher = DatasetMatcher()
        info = matcher.parse_filename(dlc_path)
        if info:
            target_animal = str(info['animal_id']).upper()
            target_day = int(info['day'])
            if verbose:
                print(f"Targeting specific session from file: Animal {target_animal}, Day {target_day}")
    else:
        dlc_dir = dlc_path

    if not os.path.exists(dlc_dir):
        raise FileNotFoundError(f"DLC directory not found: {dlc_dir}")

    all_saved = []
    features_tuple = ("speed", "tail_angle") if feature == "both" else (feature,)

    formats_to_run = ["diane_parts", "diane"] if format_type == "both" else [format_type]

    for fmt in formats_to_run:
        if verbose:
            print(f"\n=======================================================")
            print(f"Running Split Verification for format: '{fmt}'")
            print(f"  Excel: {excel_path}")
            print(f"  DLC Dir: {dlc_dir}")
            print(f"=======================================================")

        if fmt == "diane_parts":
            paired = dataset_utils.pair_diane_parts_dataset(
                master_excel_path=excel_path,
                dlc_dir=dlc_dir,
                verbose=verbose
            )
            if not paired:
                if verbose:
                    print("No paired sessions found for diane_parts.")
                continue

            sessions_by_key: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
            for p in paired:
                key = (p['animal'], p['day'])
                if key not in sessions_by_key:
                    sessions_by_key[key] = []
                sessions_by_key[key].append(p)

            for key in sessions_by_key:
                sessions_by_key[key].sort(key=lambda x: x.get('part', 1))

            split_keys = [k for k, parts in sessions_by_key.items() if len(parts) > 1]
            if verbose:
                print(f"Found {len(split_keys)} multi-part split session(s) in 'diane_parts': {split_keys}")

            if target_animal and target_day:
                clean_target = target_animal.replace("YFP-", "").replace("YFP_", "").replace("YFP", "").replace("YB-", "").replace("YB_", "").replace("YB", "").strip()
                matched_keys = [
                    k for k in sessions_by_key.keys()
                    if str(k[0]).replace("YFP-", "").replace("YFP_", "").replace("YFP", "").replace("YB-", "").replace("YB_", "").replace("YB", "").strip() == clean_target
                    and k[1] == target_day
                ]
                candidate_keys = matched_keys if matched_keys else split_keys
            else:
                candidate_keys = split_keys

            if not candidate_keys:
                print("No multi-part split sessions found in diane_parts. Selecting fallback sessions...")
                candidate_keys = list(sessions_by_key.keys())[:num_examples]

            selected_keys = candidate_keys[:num_examples]
            for animal, day in selected_keys:
                parts_data = sessions_by_key[(animal, day)]
                saved = plot_day_split_verification(
                    animal=animal,
                    day=day,
                    session_parts_data=parts_data,
                    save_dir=save_dir,
                    features_to_plot=features_tuple,
                    show_plot=show_plot,
                    verbose=verbose
                )
                all_saved.extend(saved)

        elif fmt == "diane":
            paired = dataset_utils.pair_diane_dataset(
                master_excel_path=excel_path,
                dlc_dir=dlc_dir,
                split_parts=False,
                verbose=verbose
            )
            if not paired:
                if verbose:
                    print("No paired sessions found for original diane.")
                continue

            # Find stitched sessions (is_stitched=True or len(dlc_paths) > 1)
            stitched_sessions = [p for p in paired if p.get('is_stitched') or len(p.get('dlc_paths', [])) > 1]
            if verbose:
                print(f"Found {len(stitched_sessions)} stitched multi-part session(s) in original 'diane'.")

            if target_animal and target_day:
                clean_target = target_animal.replace("YFP-", "").replace("YFP_", "").replace("YFP", "").replace("YB-", "").replace("YB_", "").replace("YB", "").strip()
                matched_sessions = [
                    p for p in stitched_sessions
                    if str(p['animal']).replace("YFP-", "").replace("YFP_", "").replace("YFP", "").replace("YB-", "").replace("YB_", "").replace("YB", "").strip() == clean_target
                    and p['day'] == target_day
                ]
                candidate_sessions = matched_sessions if matched_sessions else stitched_sessions
            else:
                candidate_sessions = stitched_sessions

            if not candidate_sessions:
                print("No stitched sessions found in original diane. Selecting fallback sessions...")
                candidate_sessions = paired[:num_examples]

            selected_sessions = candidate_sessions[:num_examples]
            for sess in selected_sessions:
                animal = sess['animal']
                day = sess['day']
                saved = plot_day_stitched_verification(
                    animal=animal,
                    day=day,
                    session_item=sess,
                    save_dir=save_dir,
                    features_to_plot=features_tuple,
                    show_plot=show_plot,
                    verbose=verbose
                )
                all_saved.extend(saved)

    if verbose:
        print(f"\n[DONE] Verification complete. {len(all_saved)} plot(s) saved to '{save_dir}'.")

    return all_saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate verification test plots for multi-part video split handling.")
    parser.add_argument(
        "--format", type=str, default="diane_parts", choices=["diane_parts", "diane", "both"],
        help="Format to verify: 'diane_parts' (resetting frames), 'diane' (stitched), or 'both' (default: diane_parts)"
    )
    parser.add_argument(
        "--dlc_path", type=str,
        default=os.path.join(BASE_SERVER_PATH, r"DianeBissen\Cricket hunting videos\Craniotomy hunting videos_DLC"),
        help="Path to DLC tracking CSV file or directory of DLC files"
    )
    parser.add_argument(
        "--excel_path", type=str,
        default=os.path.join(BASE_SERVER_PATH, r"DianeBissen\Cricket hunting videos\Cricket_hunting_all_hunting_videos_labeling_Diane_BCFixed_MostUpdated.xlsx"),
        help="Path to Diane master Excel file"
    )
    parser.add_argument(
        "--save_dir", type=str,
        default="figs/split_verification",
        help="Output directory to save verification figures (default: figs/split_verification)"
    )
    parser.add_argument(
        "--num_examples", type=int, default=2,
        help="Number of split examples to find and plot (default: 2)"
    )
    parser.add_argument(
        "--feature", type=str, default="both", choices=["speed", "tail_angle", "tail", "both"],
        help="Feature to overlay: 'speed', 'tail_angle', or 'both' (default: both)"
    )
    parser.add_argument(
        "--show", action="store_true", default=False,
        help="Show interactive plot window in addition to saving"
    )

    args = parser.parse_args()

    verify_split_handling(
        dlc_path=args.dlc_path,
        excel_path=args.excel_path,
        save_dir=args.save_dir,
        format_type=args.format,
        num_examples=args.num_examples,
        feature=args.feature,
        show_plot=args.show,
        verbose=True
    )

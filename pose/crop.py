"""
Undoes DeepLabCut analysis cropping so pose coordinates line up with the full video frame.

When DLC analyzes a video with cropping enabled, the output x/y are relative to the crop box.
The crop is recovered (in order of preference) from:
  1. DLC's `<video><scorer>_meta.pickle` written next to every analysis output
  2. An optional `'crop_offset'` entry in the matching dataset dict in dataset_utils,
     either (x1, y1) or DLC's full (x1, x2, y1, y2) (the latter also gives the crop size)
If neither is found, no offset is applied.

Inference uses the crop size (when known) as the effective video width for feature
normalization, since the crop box approximates the arena rather than the full camera frame.
"""
import os
import glob
import pickle
from typing import Optional, Tuple, NamedTuple

import pandas as pd


class CropOffset(NamedTuple):
    dx: float
    dy: float
    source: str  # 'meta', 'dataset', or 'none'
    width: Optional[float] = None   # Crop box size in pixels, if the analysis was cropped and it is known
    height: Optional[float] = None

    @property
    def is_applied(self) -> bool:
        return self.dx != 0 or self.dy != 0


NO_CROP = CropOffset(0.0, 0.0, 'none')


def find_dlc_meta_pickle(dlc_path: str) -> Optional[str]:
    """Locates the `_meta.pickle` DLC writes alongside an analysis .csv/.h5 (also for `_filtered` outputs)."""
    stem = os.path.splitext(dlc_path)[0]
    if stem.endswith('_filtered'):
        stem = stem[:-len('_filtered')]
    exact = stem + '_meta.pickle'
    if os.path.exists(exact):
        return exact

    # Fall back to a unique meta pickle for the same video in the folder
    base = os.path.basename(stem)
    if 'DLC' not in base:
        return None
    video_stem = base.split('DLC')[0]
    matches = glob.glob(os.path.join(glob.escape(os.path.dirname(dlc_path)), glob.escape(video_stem) + 'DLC*_meta.pickle'))
    return matches[0] if len(matches) == 1 else None


class _MetaUnpickler(pickle.Unpickler):
    """
    DLC meta pickles embed ruamel.yaml config types (e.g. ScalarFloat, CommentedMap).
    Substitute plain builtin subclasses so they load without ruamel installed.
    """
    _STAND_INS = (('Float', float), ('Int', int), ('Bool', int), ('Seq', list), ('Map', dict), ('Str', str))

    def find_class(self, module, name):
        if module.startswith('ruamel'):
            base = next((b for key, b in self._STAND_INS if key in name), object)
            return type(name, (base,), {'__setstate__': lambda self, state: None})
        return super().find_class(module, name)


def _crop_from_params(params, source: str) -> CropOffset:
    """Builds a CropOffset from (x1, y1) or DLC-style (x1, x2, y1, y2)."""
    if len(params) == 4:
        x1, x2, y1, y2 = (float(v) for v in params)
        return CropOffset(x1, y1, source, x2 - x1, y2 - y1)
    return CropOffset(float(params[0]), float(params[1]), source)


def read_meta_crop(meta_path: str) -> CropOffset:
    """Returns the crop recorded in a DLC meta pickle (zero offset if the analysis was not cropped)."""
    with open(meta_path, 'rb') as f:
        meta = _MetaUnpickler(f).load()
    data = meta.get('data', meta)
    if not data.get('cropping', False):
        return CropOffset(0.0, 0.0, 'meta')
    return _crop_from_params(data['cropping_parameters'], 'meta')


def resolve_crop_offset(dlc_path: str) -> CropOffset:
    meta_path = find_dlc_meta_pickle(dlc_path)
    if meta_path:
        try:
            return read_meta_crop(meta_path)
        except Exception as e:
            print(f"Warning: could not read DLC crop from {meta_path}: {e}")

    try:
        from dataset_utils import get_dataset_crop_offset
    except ImportError:
        return NO_CROP
    ds_offset = get_dataset_crop_offset(dlc_path)
    if ds_offset is not None:
        return _crop_from_params(ds_offset, 'dataset')
    return NO_CROP


def apply_crop_offset(df: pd.DataFrame, offset: CropOffset) -> pd.DataFrame:
    """Shifts every bodypart's x/y column in a DLC DataFrame by the crop offset (in place)."""
    if not offset.is_applied:
        return df
    x_cols = [c for c in df.columns if c[-1] == 'x']
    y_cols = [c for c in df.columns if c[-1] == 'y']
    df.loc[:, x_cols] = df.loc[:, x_cols] + offset.dx
    df.loc[:, y_cols] = df.loc[:, y_cols] + offset.dy
    return df


def read_dlc_table(dlc_path: str) -> Tuple[pd.DataFrame, CropOffset]:
    """Reads a DLC .csv/.h5 and returns it in full-frame coordinates along with the offset applied."""
    if dlc_path.endswith('.h5'):
        df = pd.read_hdf(dlc_path)
    else:
        df = pd.read_csv(dlc_path, header=[0, 1, 2], index_col=0)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    offset = resolve_crop_offset(dlc_path)
    if offset.is_applied:
        print(f"Applied DLC crop offset (+{offset.dx:g}, +{offset.dy:g}) from {offset.source} to {os.path.basename(dlc_path)}")
    return apply_crop_offset(df, offset), offset


def effective_video_width(dlc_path: str, video_width: float) -> float:
    """Width used to normalize pose features: the DLC crop width if the analysis was cropped, else video_width."""
    crop = resolve_crop_offset(dlc_path)
    if crop.width:
        print(f"Using DLC crop width {crop.width:g}px (from {crop.source}) instead of video width {video_width:g}px for feature scaling")
        return crop.width
    return video_width

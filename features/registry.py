import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Any

@dataclass
class FeatureItem:
    name: str
    group: str
    description: str = ""

@dataclass
class FeatureManifest:
    items: List[FeatureItem] = field(default_factory=list)

    @property
    def feature_names(self) -> List[str]:
        return [item.name for item in self.items]

    @property
    def num_features(self) -> int:
        return len(self.items)

    def get_group(self, group_name: str) -> List[str]:
        return [item.name for item in self.items if item.group == group_name]

class FeatureRegistry:
    """
    Central registry for composable feature calculation blocks.
    Ensures training and inference always share identical feature manifests and indices.
    """
    def __init__(self):
        self._extractors: Dict[str, Callable] = {}

    def register(self, group_name: str):
        def decorator(fn: Callable):
            self._extractors[group_name] = fn
            return fn
        return decorator

    def compute(
        self,
        pose_data: Any,
        video_width: float,
        video_fps: float,
        enabled_blocks: List[str],
        config: Any = None
    ) -> Tuple[np.ndarray, FeatureManifest]:
        """
        Executes all enabled feature extractors, concatenates the resulting columns,
        and constructs the corresponding FeatureManifest.
        """
        feature_arrays = []
        manifest_items = []

        for block in enabled_blocks:
            if block not in self._extractors:
                continue
            extractor_fn = self._extractors[block]
            block_data, block_items = extractor_fn(
                pose_data=pose_data,
                video_width=video_width,
                video_fps=video_fps,
                config=config
            )
            if block_data.ndim == 1:
                block_data = block_data[:, np.newaxis]
            feature_arrays.append(block_data)
            manifest_items.extend(block_items)

        if not feature_arrays:
            raise ValueError(f"No features generated for enabled blocks: {enabled_blocks}")

        full_matrix = np.column_stack(feature_arrays)
        full_matrix = np.nan_to_num(full_matrix, nan=0.0)
        return full_matrix, FeatureManifest(items=manifest_items)

feature_registry = FeatureRegistry()

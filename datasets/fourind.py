"""Generic 4ind high-resolution anomaly dataset loader.

The private dataset layout is not encoded into the model.  A CSV manifest is
preferred and may contain: category, split, label, image, mask.  MVTec-style
folders are also discovered when no manifest is supplied.
"""

from __future__ import annotations

import os

from .highres_base import (
    DatasetSplit,
    HighResolutionAnomalyDataset,
    discover_mvtec_style,
    read_manifest,
)


class FourIndDataset(HighResolutionAnomalyDataset):
    def __init__(
        self,
        source,
        classname="default",
        split=DatasetSplit.TRAIN,
        manifest_path=None,
        **kwargs,
    ):
        split_value = split.value if isinstance(split, DatasetSplit) else str(split)
        manifest_candidates = [
            manifest_path,
            os.path.join(source, "metadata.csv"),
            os.path.join(source, "split.csv"),
            os.path.join(source, "manifest.csv"),
        ]
        manifest = next(
            (path for path in manifest_candidates if path and os.path.isfile(path)),
            None,
        )
        if manifest:
            records = read_manifest(manifest, source, classname, split_value)
        else:
            records = discover_mvtec_style(source, classname, split_value)
        super().__init__(
            records=records,
            name=f"fourind_{classname}",
            split=DatasetSplit(split_value),
            **kwargs,
        )


__all__ = ["DatasetSplit", "FourIndDataset"]

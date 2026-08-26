"""VisA loader for the global/local SimpleNet pipeline."""

from __future__ import annotations

import os
from pathlib import Path

from .highres_base import (
    DatasetSplit,
    HighResolutionAnomalyDataset,
    discover_mvtec_style,
    read_manifest,
)


_CLASSNAMES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]


class VisADataset(HighResolutionAnomalyDataset):
    """Read raw official VisA data or prepared MVTec-style data.

    Raw data requires the official ``split_csv/1cls.csv``.  Prepared data can
    instead use ``<root>/<category>/{train,test,ground_truth}``.
    """

    def __init__(
        self,
        source,
        classname,
        split=DatasetSplit.TRAIN,
        manifest_path=None,
        **kwargs,
    ):
        if classname not in _CLASSNAMES:
            raise ValueError(f"Unknown VisA category {classname!r}.")
        split_value = split.value if isinstance(split, DatasetSplit) else str(split)
        manifest_candidates = [
            manifest_path,
            os.path.join(source, "split_csv", "1cls.csv"),
            os.path.join(source, "1cls.csv"),
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
            name=f"visa_{classname}",
            split=DatasetSplit(split_value),
            **kwargs,
        )


__all__ = ["DatasetSplit", "VisADataset"]

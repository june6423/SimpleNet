"""Whole-image VisA adapter for the unchanged SimpleNet baselines.

The original MVTec loader uses Resize(short side) followed by a square center
crop.  That discards a material part of a VisA image.  This adapter instead
letterboxes the complete image into a square model input and keeps the mask at
the original resolution for a common high-resolution evaluator.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import PIL.Image
import PIL.ImageEnhance
import torch

from .highres_base import (
    DatasetSplit,
    IMAGENET_MEAN,
    IMAGENET_STD,
    image_to_normalized_tensor,
    letterbox,
    read_manifest,
)
from .visa import _CLASSNAMES


class VisALegacyDataset(torch.utils.data.Dataset):
    """Official one-class VisA split with a whole-image square input."""

    def __init__(
        self,
        source,
        classname,
        split=DatasetSplit.TRAIN,
        imagesize=320,
        manifest_path=None,
        brightness_factor=0.0,
        contrast_factor=0.0,
        saturation_factor=0.0,
        seed=0,
        **_kwargs,
    ):
        super().__init__()
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
        if manifest is None:
            raise FileNotFoundError("VisA split_csv/1cls.csv was not found.")

        self.records = read_manifest(
            manifest, source, classname, split_value
        )
        if not self.records:
            raise ValueError(f"No VisA records for {classname} split={split_value}.")
        if split_value == DatasetSplit.TRAIN.value and any(
            record.is_anomaly for record in self.records
        ):
            raise ValueError("VisA one-class training split contains anomalies.")

        self.source = str(source)
        self.classname = classname
        self.split = DatasetSplit(split_value)
        self.input_size = int(imagesize)
        self.imagesize = (3, self.input_size, self.input_size)
        self.name = f"visa_{classname}"
        self.seed = int(seed)
        self.brightness_factor = max(0.0, float(brightness_factor))
        self.contrast_factor = max(0.0, float(contrast_factor))
        self.saturation_factor = max(0.0, float(saturation_factor))
        self.transform_mean = IMAGENET_MEAN
        self.transform_std = IMAGENET_STD
        self.data_to_iterate = [
            (
                record.category,
                "anomaly" if record.is_anomaly else "good",
                record.image_path,
                record.mask_path,
            )
            for record in self.records
        ]

    def __len__(self):
        return len(self.records)

    def _augment(self, image: PIL.Image.Image) -> PIL.Image.Image:
        if self.split != DatasetSplit.TRAIN:
            return image
        if self.brightness_factor:
            image = PIL.ImageEnhance.Brightness(image).enhance(
                random.uniform(
                    1 - self.brightness_factor, 1 + self.brightness_factor
                )
            )
        if self.contrast_factor:
            image = PIL.ImageEnhance.Contrast(image).enhance(
                random.uniform(1 - self.contrast_factor, 1 + self.contrast_factor)
            )
        if self.saturation_factor:
            image = PIL.ImageEnhance.Color(image).enhance(
                random.uniform(
                    1 - self.saturation_factor, 1 + self.saturation_factor
                )
            )
        return image

    def _content_box(self, width: int, height: int) -> torch.Tensor:
        scale = min(self.input_size / width, self.input_size / height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        offset_x = (self.input_size - resized_width) // 2
        offset_y = (self.input_size - resized_height) // 2
        return torch.tensor(
            [
                offset_x,
                offset_y,
                offset_x + resized_width,
                offset_y + resized_height,
            ],
            dtype=torch.int64,
        )

    @staticmethod
    def _mask(record, height: int, width: int) -> torch.Tensor:
        if record.mask_path:
            mask = PIL.Image.open(record.mask_path).convert("L")
            if mask.size != (width, height):
                mask = mask.resize(
                    (width, height), PIL.Image.Resampling.NEAREST
                )
            array = (np.asarray(mask) > 0).astype(np.uint8)
            return torch.from_numpy(array).unsqueeze(0)
        return torch.zeros((1, height, width), dtype=torch.uint8)

    def __getitem__(self, index):
        record = self.records[index]
        image = PIL.Image.open(record.image_path).convert("RGB")
        width, height = image.size
        model_image = letterbox(image, self.input_size, self.input_size)
        model_image = self._augment(model_image)

        result = {
            "image": image_to_normalized_tensor(model_image),
            "classname": record.category,
            "anomaly": "anomaly" if record.is_anomaly else "good",
            "is_anomaly": int(record.is_anomaly),
            "image_name": Path(record.image_path).name,
            "image_path": record.image_path,
            "original_size": torch.tensor([height, width], dtype=torch.int64),
            "content_box": self._content_box(width, height),
        }
        if self.split == DatasetSplit.TEST:
            result["mask"] = self._mask(record, height, width)
        return result


__all__ = ["DatasetSplit", "VisALegacyDataset"]

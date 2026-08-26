"""Whole-image letterbox adapter for unchanged SimpleNet 4ind baselines."""

from __future__ import annotations

import os
import random
from pathlib import Path

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


class FourIndLegacyDataset(torch.utils.data.Dataset):
    """Read the portable 4ind manifest and keep the full strip in one view."""

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
        split_value = split.value if isinstance(split, DatasetSplit) else str(split)
        if not manifest_path or not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"4ind manifest not found: {manifest_path}")
        self.records = read_manifest(manifest_path, source, classname, split_value)
        if not self.records:
            raise ValueError(f"No 4ind records for {classname} split={split_value}.")
        if split_value == DatasetSplit.TRAIN.value and any(
            record.is_anomaly for record in self.records
        ):
            raise ValueError("4ind one-class training split contains anomalies.")

        self.source = str(source)
        self.classname = classname
        self.split = DatasetSplit(split_value)
        self.input_size = int(imagesize)
        self.imagesize = (3, self.input_size, self.input_size)
        self.name = f"fourind_{classname}"
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
                random.uniform(1 - self.brightness_factor, 1 + self.brightness_factor)
            )
        if self.contrast_factor:
            image = PIL.ImageEnhance.Contrast(image).enhance(
                random.uniform(1 - self.contrast_factor, 1 + self.contrast_factor)
            )
        if self.saturation_factor:
            image = PIL.ImageEnhance.Color(image).enhance(
                random.uniform(1 - self.saturation_factor, 1 + self.saturation_factor)
            )
        return image

    def __getitem__(self, index):
        record = self.records[index]
        image = PIL.Image.open(record.image_path).convert("RGB")
        model_image = letterbox(image, self.input_size, self.input_size)
        model_image = self._augment(model_image)
        return {
            "image": image_to_normalized_tensor(model_image),
            "classname": record.category,
            "anomaly": record.label,
            "is_anomaly": int(record.is_anomaly),
            "image_name": Path(record.image_path).name,
            "image_path": record.image_path,
        }


__all__ = ["DatasetSplit", "FourIndLegacyDataset"]

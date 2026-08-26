"""Shared high-resolution dataset utilities for global/local SimpleNet."""

from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import PIL.Image
import PIL.ImageEnhance
import torch
from torchvision.transforms import functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LETTERBOX_COLOR = tuple(round(channel * 255) for channel in IMAGENET_MEAN)


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass(frozen=True)
class ImageRecord:
    category: str
    split: str
    label: str
    image_path: str
    mask_path: Optional[str] = None

    @property
    def is_anomaly(self) -> bool:
        return self.label.lower() not in {"good", "normal", "ok", "0"}


def _tile_starts(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def make_tile_boxes(
    width: int,
    height: int,
    tile_width: int,
    tile_height: int,
    stride_x: int,
    stride_y: int,
) -> List[Tuple[int, int, int, int]]:
    xs = _tile_starts(width, tile_width, stride_x)
    ys = _tile_starts(height, tile_height, stride_y)
    boxes = []
    for y in ys:
        for x in xs:
            boxes.append(
                (x, y, min(x + tile_width, width), min(y + tile_height, height))
            )
    return boxes


def letterbox(image: PIL.Image.Image, target_height: int, target_width: int) -> PIL.Image.Image:
    source_width, source_height = image.size
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = image.resize((resized_width, resized_height), PIL.Image.Resampling.BILINEAR)
    canvas = PIL.Image.new("RGB", (target_width, target_height), LETTERBOX_COLOR)
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def image_to_normalized_tensor(image: PIL.Image.Image) -> torch.Tensor:
    tensor = TF.to_tensor(image)
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


def image_to_uint8_tensor(image: PIL.Image.Image) -> torch.Tensor:
    """Keep test tiles compact until a tile mini-batch is scored."""
    return TF.pil_to_tensor(image)


class HighResolutionAnomalyDataset(torch.utils.data.Dataset):
    """Base dataset returning one global view and local crops.

    Training returns one sampled local crop per logical item.  Test returns all
    tiles and their source coordinates; test loaders therefore use batch size 1.
    """

    def __init__(
        self,
        records: Sequence[ImageRecord],
        name: str,
        split: DatasetSplit,
        imagesize: int = 320,
        global_height: int = 512,
        global_width: int = 768,
        tile_height: int = 512,
        tile_width: int = 512,
        tile_stride_y: int = 384,
        tile_stride_x: int = 384,
        train_tiles_per_image: int = 1,
        brightness_factor: float = 0.0,
        contrast_factor: float = 0.0,
        saturation_factor: float = 0.0,
        seed: int = 0,
        **_kwargs,
    ):
        super().__init__()
        self.records = list(records)
        self.name = name
        self.split = split
        self.local_input_size = int(imagesize)
        self.imagesize = (3, self.local_input_size, self.local_input_size)
        self.global_height = int(global_height)
        self.global_width = int(global_width)
        self.tile_height = int(tile_height)
        self.tile_width = int(tile_width)
        self.tile_stride_y = int(tile_stride_y)
        self.tile_stride_x = int(tile_stride_x)
        self.train_tiles_per_image = max(1, int(train_tiles_per_image))
        self.brightness_factor = max(0.0, float(brightness_factor))
        self.contrast_factor = max(0.0, float(contrast_factor))
        self.saturation_factor = max(0.0, float(saturation_factor))
        self.seed = int(seed)
        self.transform_std = IMAGENET_STD
        self.transform_mean = IMAGENET_MEAN
        self.has_pixel_masks = any(record.mask_path for record in self.records)

        if not self.records:
            raise ValueError(f"No images found for {name} split={split.value}.")
        if split == DatasetSplit.TRAIN and any(r.is_anomaly for r in self.records):
            raise ValueError("High-resolution one-class training records must be normal.")
        for value, field in (
            (self.global_height, "global_height"),
            (self.global_width, "global_width"),
            (self.tile_height, "tile_height"),
            (self.tile_width, "tile_width"),
            (self.tile_stride_y, "tile_stride_y"),
            (self.tile_stride_x, "tile_stride_x"),
        ):
            if value <= 0:
                raise ValueError(f"{field} must be positive.")

    def __len__(self) -> int:
        multiplier = self.train_tiles_per_image if self.split == DatasetSplit.TRAIN else 1
        return len(self.records) * multiplier

    def _photometric_augment(self, image: PIL.Image.Image) -> PIL.Image.Image:
        if self.split != DatasetSplit.TRAIN:
            return image
        if self.brightness_factor:
            factor = random.uniform(1 - self.brightness_factor, 1 + self.brightness_factor)
            image = PIL.ImageEnhance.Brightness(image).enhance(factor)
        if self.contrast_factor:
            factor = random.uniform(1 - self.contrast_factor, 1 + self.contrast_factor)
            image = PIL.ImageEnhance.Contrast(image).enhance(factor)
        if self.saturation_factor:
            factor = random.uniform(1 - self.saturation_factor, 1 + self.saturation_factor)
            image = PIL.ImageEnhance.Color(image).enhance(factor)
        return image

    def _global_tensor(self, image: PIL.Image.Image) -> torch.Tensor:
        view = letterbox(image, self.global_height, self.global_width)
        view = self._photometric_augment(view)
        return image_to_normalized_tensor(view)

    def _local_tensor(self, image: PIL.Image.Image, box: Tuple[int, int, int, int]) -> torch.Tensor:
        tile = image.crop(box)
        tile = letterbox(tile, self.local_input_size, self.local_input_size)
        tile = self._photometric_augment(tile)
        return image_to_normalized_tensor(tile)

    def _local_uint8_tensor(
        self, image: PIL.Image.Image, box: Tuple[int, int, int, int]
    ) -> torch.Tensor:
        tile = image.crop(box)
        tile = letterbox(tile, self.local_input_size, self.local_input_size)
        return image_to_uint8_tensor(tile)

    def _mask_tensor(self, record: ImageRecord, height: int, width: int) -> torch.Tensor:
        if record.mask_path:
            mask = PIL.Image.open(record.mask_path).convert("L")
            mask_array = (np.asarray(mask) > 0).astype(np.uint8)
            if mask_array.shape != (height, width):
                mask = mask.resize((width, height), PIL.Image.Resampling.NEAREST)
                mask_array = (np.asarray(mask) > 0).astype(np.uint8)
            return torch.from_numpy(mask_array).unsqueeze(0)
        return torch.zeros((1, height, width), dtype=torch.uint8)

    def __getitem__(self, index: int) -> Dict[str, object]:
        if self.split == DatasetSplit.TRAIN:
            record_index = index // self.train_tiles_per_image
        else:
            record_index = index
        record = self.records[record_index]
        image = PIL.Image.open(record.image_path).convert("RGB")
        width, height = image.size
        boxes = make_tile_boxes(
            width,
            height,
            self.tile_width,
            self.tile_height,
            self.tile_stride_x,
            self.tile_stride_y,
        )
        global_image = self._global_tensor(image)
        common = {
            "global_image": global_image,
            "original_size": (height, width),
            "classname": record.category,
            "anomaly": record.label,
            "is_anomaly": int(record.is_anomaly),
            "image_name": os.path.basename(record.image_path),
            "image_path": record.image_path,
            "mask_valid": self.has_pixel_masks,
        }

        if self.split == DatasetSplit.TRAIN:
            # Worker-specific random seeds are managed by PyTorch's DataLoader.
            box = boxes[random.randrange(len(boxes))]
            common["local_image"] = self._local_tensor(image, box)
            common["tile_box"] = torch.tensor(box, dtype=torch.int64)
            return common

        common["local_tiles"] = torch.stack(
            [self._local_uint8_tensor(image, box) for box in boxes]
        )
        common["tile_boxes"] = torch.tensor(boxes, dtype=torch.int64)
        common["mask"] = self._mask_tensor(record, height, width)
        return common


def read_manifest(
    manifest_path: str,
    root: str,
    category: Optional[str],
    split: str,
) -> List[ImageRecord]:
    records = []
    with open(manifest_path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"split", "label"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Manifest {manifest_path} must contain split,label and image/path columns."
            )
        for row in reader:
            row_category = row.get("object") or row.get("category") or "default"
            if category and row_category != category:
                continue
            if row["split"].lower() != split.lower():
                continue
            image_rel = row.get("image") or row.get("path") or row.get("image_path")
            if not image_rel:
                raise ValueError(f"Manifest row has no image path: {row}")
            mask_rel = row.get("mask") or row.get("mask_path") or None
            image_path = image_rel if os.path.isabs(image_rel) else os.path.join(root, image_rel)
            mask_path = None
            if mask_rel:
                mask_path = mask_rel if os.path.isabs(mask_rel) else os.path.join(root, mask_rel)
            records.append(
                ImageRecord(
                    category=row_category,
                    split=row["split"],
                    label=row["label"],
                    image_path=image_path,
                    mask_path=mask_path,
                )
            )
    return records


def discover_mvtec_style(
    root: str, category: str, split: str
) -> List[ImageRecord]:
    category_root = Path(root) / category
    split_root = category_root / split
    if not split_root.exists():
        split_root = Path(root) / split
    if not split_root.exists():
        return []

    records = []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for label_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        for image_path in sorted(label_dir.rglob("*")):
            if image_path.suffix.lower() not in image_extensions:
                continue
            mask_path = None
            if label_dir.name.lower() not in {"good", "normal", "ok"}:
                ground_truth = category_root / "ground_truth" / label_dir.name
                candidates = [
                    ground_truth / f"{image_path.stem}_mask.png",
                    ground_truth / f"{image_path.stem}.png",
                ]
                mask_path = next((str(p) for p in candidates if p.exists()), None)
            records.append(
                ImageRecord(
                    category=category,
                    split=split,
                    label=label_dir.name,
                    image_path=str(image_path),
                    mask_path=mask_path,
                )
            )
    return records

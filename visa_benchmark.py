"""Per-category VisA benchmark runner for three SimpleNet variants.

The baseline implementation modules are imported unchanged.  This runner owns
the fixed-epoch training loop so the official test split is evaluated once,
instead of being used for checkpoint selection at every meta epoch.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics as sklearn_metrics

import backbones
import simplenet
import simplenet_gl
import simplenet_plus
import utils
from datasets.highres_base import DatasetSplit
from datasets.visa import VisADataset
from datasets.visa_legacy import VisALegacyDataset


LOGGER = logging.getLogger("visa_benchmark")
METHODS = ("simplenet", "simplenet_plus", "global_local")
CATEGORIES = (
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
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--category", choices=CATEGORIES, required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--imagesize", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--meta-epochs", type=int, default=40)
    parser.add_argument("--gan-epochs", type=int, default=4)
    parser.add_argument("--pixel-bins", type=int, default=65536)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def build_loader(dataset, batch_size: int, workers: int, shuffle: bool):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
    )
    if workers > 0:
        kwargs.update(prefetch_factor=2, persistent_workers=True)
    return torch.utils.data.DataLoader(**kwargs)


def select_smoke_records(dataset, train: bool):
    if train:
        dataset.records = dataset.records[: min(16, len(dataset.records))]
    else:
        normal = [record for record in dataset.records if not record.is_anomaly][:4]
        anomaly = [record for record in dataset.records if record.is_anomaly][:4]
        dataset.records = normal + anomaly
    if hasattr(dataset, "data_to_iterate"):
        dataset.data_to_iterate = [
            (
                record.category,
                "anomaly" if record.is_anomaly else "good",
                record.image_path,
                record.mask_path,
            )
            for record in dataset.records
        ]


def build_datasets(args):
    augmentation = 0.0 if args.method == "simplenet" else 0.1
    if args.method == "global_local":
        common = dict(
            source=args.data_path,
            classname=args.category,
            imagesize=args.imagesize,
            global_height=512,
            global_width=768,
            tile_height=512,
            tile_width=512,
            tile_stride_y=384,
            tile_stride_x=384,
            train_tiles_per_image=1,
            seed=args.seed,
        )
        train_dataset = VisADataset(
            split=DatasetSplit.TRAIN,
            brightness_factor=augmentation,
            contrast_factor=augmentation,
            saturation_factor=augmentation,
            **common,
        )
        test_dataset = VisADataset(split=DatasetSplit.TEST, **common)
    else:
        common = dict(
            source=args.data_path,
            classname=args.category,
            imagesize=args.imagesize,
            seed=args.seed,
        )
        train_dataset = VisALegacyDataset(
            split=DatasetSplit.TRAIN,
            brightness_factor=augmentation,
            contrast_factor=augmentation,
            saturation_factor=augmentation,
            **common,
        )
        test_dataset = VisALegacyDataset(split=DatasetSplit.TEST, **common)
    if args.smoke:
        select_smoke_records(train_dataset, train=True)
        select_smoke_records(test_dataset, train=False)
    return train_dataset, test_dataset, augmentation


def build_baseline_model(args, input_shape, device):
    module = simplenet if args.method == "simplenet" else simplenet_plus
    backbone = backbones.load("wideresnet50")
    backbone.name = "wideresnet50"
    backbone.seed = args.seed
    model = module.SimpleNet(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=device,
        input_shape=input_shape,
        pretrain_embed_dimension=1536,
        target_embed_dimension=1536,
        patchsize=3,
        embedding_size=256,
        meta_epochs=args.meta_epochs,
        aed_meta_epochs=1 if args.meta_epochs > 1 else 0,
        gan_epochs=args.gan_epochs,
        noise_std=0.015,
        mix_noise=1,
        dsc_layers=2,
        dsc_hidden=1024,
        dsc_margin=0.5,
        dsc_lr=2e-4,
        train_backbone=False,
        cos_lr=False,
        pre_proj=1,
        proj_layer_type=0,
    )
    return model


def build_global_local_model(args, input_shape, device):
    backbone = backbones.load("wideresnet50")
    backbone.name = "wideresnet50"
    backbone.seed = args.seed
    model = simplenet_gl.GlobalLocalSimpleNet(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=device,
        input_shape=input_shape,
        pretrain_embed_dimension=1536,
        target_embed_dimension=1536,
        patchsize=3,
        embedding_size=256,
        meta_epochs=args.meta_epochs,
        aed_meta_epochs=0,
        gan_epochs=args.gan_epochs,
        noise_std=0.015,
        mix_noise=1,
        dsc_layers=2,
        dsc_hidden=1024,
        dsc_margin=0.5,
        dsc_lr=2e-4,
        train_backbone=False,
        cos_lr=False,
        pre_proj=1,
        proj_layer_type=0,
        tile_batch_size=32,
        exact_pixel_auroc=False,
        legacy_gaussian_sigma=4.0,
    )
    return model


def baseline_checkpoint_state(model, method: str, next_meta_epoch: int):
    state = {
        "method": method,
        "next_meta_epoch": next_meta_epoch,
        "discriminator": model.discriminator.state_dict(),
        "dsc_opt": model.dsc_opt.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    if model.pre_proj > 0:
        state["pre_projection"] = model.pre_projection.state_dict()
        state["proj_opt"] = model.proj_opt.state_dict()
    if method == "simplenet_plus":
        state.update(
            teacher=model.teacher.state_dict(),
            noise_generator=model.noise_generator.state_dict(),
            g_opt=model.g_opt.state_dict(),
        )
    return state


def atomic_torch_save(state, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_baseline_checkpoint(model, path: Path, method: str, load_optimizer: bool):
    # This is a trusted, locally generated progress file and includes NumPy RNG
    # state, which is intentionally outside PyTorch's weights-only allowlist.
    state = torch.load(path, map_location=model.device, weights_only=False)
    model.discriminator.load_state_dict(state["discriminator"])
    if model.pre_proj > 0:
        model.pre_projection.load_state_dict(state["pre_projection"])
    if method == "simplenet_plus" and "teacher" in state:
        model.teacher.load_state_dict(state["teacher"])
        model.noise_generator.load_state_dict(state["noise_generator"])
    if load_optimizer:
        model.dsc_opt.load_state_dict(state["dsc_opt"])
        if model.pre_proj > 0:
            model.proj_opt.load_state_dict(state["proj_opt"])
        if method == "simplenet_plus" and "g_opt" in state:
            model.g_opt.load_state_dict(state["g_opt"])
        if "torch_rng" in state:
            torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and "cuda_rng" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if "numpy_rng" in state:
            np.random.set_state(state["numpy_rng"])
    return int(state.get("next_meta_epoch", 0))


def train_baseline(model, train_loader, job_dir: Path, args):
    final_path = job_dir / "checkpoint_final.pth"
    progress_path = job_dir / "checkpoint_progress.pth"
    if final_path.exists():
        load_baseline_checkpoint(
            model, final_path, args.method, load_optimizer=False
        )
        LOGGER.info("Loaded final checkpoint %s", final_path)
        return 0.0, True

    start_meta_epoch = 0
    if progress_path.exists():
        start_meta_epoch = load_baseline_checkpoint(
            model, progress_path, args.method, load_optimizer=True
        )
        LOGGER.info("Resuming at meta epoch %d", start_meta_epoch)

    started = time.perf_counter()
    for meta_epoch in range(start_meta_epoch, args.meta_epochs):
        LOGGER.info(
            "Fixed-epoch training %s/%s meta=%d/%d",
            args.method,
            args.category,
            meta_epoch + 1,
            args.meta_epochs,
        )
        if args.method == "simplenet_plus":
            model._train_discriminator(
                train_loader, meta_epoch, teacher=(meta_epoch > 0)
            )
            if meta_epoch == 0:
                model.teacher.load_state_dict(model.discriminator.state_dict())
        else:
            model._train_discriminator(train_loader)
        atomic_torch_save(
            baseline_checkpoint_state(model, args.method, meta_epoch + 1),
            progress_path,
        )

    atomic_torch_save(
        baseline_checkpoint_state(model, args.method, args.meta_epochs),
        final_path,
    )
    return time.perf_counter() - started, start_meta_epoch > 0


def restore_letterbox_map(
    anomaly_map: np.ndarray,
    content_box: torch.Tensor,
    original_size: torch.Tensor,
) -> np.ndarray:
    x1, y1, x2, y2 = [int(value) for value in content_box.flatten().tolist()]
    height, width = [int(value) for value in original_size.flatten().tolist()]
    cropped = np.asarray(anomaly_map, dtype=np.float32)[y1:y2, x1:x2]
    if cropped.size == 0:
        raise ValueError(f"Empty letterbox content crop: {(x1, y1, x2, y2)}")
    restored = F.interpolate(
        torch.from_numpy(cropped)[None, None],
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return restored.numpy()


def histogram_auroc(
    positive_hist: np.ndarray, negative_hist: np.ndarray
) -> float:
    positives = int(positive_hist.sum())
    negatives = int(negative_hist.sum())
    if positives == 0 or negatives == 0:
        return math.nan
    tpr = np.cumsum(positive_hist[::-1], dtype=np.float64) / positives
    fpr = np.cumsum(negative_hist[::-1], dtype=np.float64) / negatives
    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))
    return float(np.trapz(tpr, fpr))


def update_histograms(
    anomaly_map: np.ndarray,
    mask: np.ndarray,
    score_min: float,
    score_max: float,
    positive_hist: np.ndarray,
    negative_hist: np.ndarray,
):
    if anomaly_map.shape != mask.shape:
        raise ValueError(f"Map/mask mismatch: {anomaly_map.shape} vs {mask.shape}")
    scale = (len(positive_hist) - 1) / max(score_max - score_min, 1e-12)
    bins = np.clip(
        ((anomaly_map - score_min) * scale).astype(np.int64),
        0,
        len(positive_hist) - 1,
    )
    positive_hist += np.bincount(
        bins[mask > 0].ravel(), minlength=len(positive_hist)
    )
    negative_hist += np.bincount(
        bins[mask <= 0].ravel(), minlength=len(negative_hist)
    )


def evaluate_baseline(model, test_loader, device, pixel_bins: int):
    labels = []
    scores = []
    score_min = math.inf
    score_max = -math.inf
    model_seconds = 0.0
    image_count = 0
    end_to_end_start = time.perf_counter()

    for data in test_loader:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_start = time.perf_counter()
        batch_scores, batch_maps, _ = model._predict(data["image"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        anomaly_map = restore_letterbox_map(
            batch_maps[0], data["content_box"], data["original_size"]
        )
        model_seconds += time.perf_counter() - inference_start
        if not np.isfinite(anomaly_map).all():
            raise FloatingPointError("Non-finite baseline anomaly map.")
        score_min = min(score_min, float(anomaly_map.min()))
        score_max = max(score_max, float(anomaly_map.max()))
        scores.append(float(np.asarray(batch_scores[0]).reshape(-1)[0]))
        labels.append(int(data["is_anomaly"].flatten()[0]))
        image_count += 1

    end_to_end_seconds = time.perf_counter() - end_to_end_start
    positive_hist = np.zeros(pixel_bins, dtype=np.int64)
    negative_hist = np.zeros(pixel_bins, dtype=np.int64)

    for data in test_loader:
        _, batch_maps, _ = model._predict(data["image"])
        anomaly_map = restore_letterbox_map(
            batch_maps[0], data["content_box"], data["original_size"]
        )
        mask = data["mask"][0, 0].numpy()
        update_histograms(
            anomaly_map,
            mask,
            score_min,
            score_max,
            positive_hist,
            negative_hist,
        )

    return {
        "image_auroc": float(sklearn_metrics.roc_auc_score(labels, scores)),
        "pixel_auroc": histogram_auroc(positive_hist, negative_hist),
        "score_min": score_min,
        "score_max": score_max,
        "seconds_per_image": model_seconds / max(1, image_count),
        "images_per_second": image_count / max(model_seconds, 1e-12),
        "end_to_end_seconds_per_image": end_to_end_seconds / max(1, image_count),
    }


def evaluate_global_local(model, test_loader, pixel_bins: int):
    labels = []
    scores = []
    score_min = math.inf
    score_max = -math.inf
    image_count = 0
    model._model_inference_seconds = 0.0
    model._model_inference_images = 0
    end_to_end_start = time.perf_counter()

    for prediction in model._iter_predictions(test_loader):
        anomaly_map = prediction["map"]
        if not np.isfinite(anomaly_map).all():
            raise FloatingPointError("Non-finite global/local anomaly map.")
        score_min = min(score_min, float(anomaly_map.min()))
        score_max = max(score_max, float(anomaly_map.max()))
        scores.append(float(prediction["score"]))
        labels.append(int(prediction["label"]))
        image_count += 1

    end_to_end_seconds = time.perf_counter() - end_to_end_start
    model_seconds = model._model_inference_seconds
    positive_hist = np.zeros(pixel_bins, dtype=np.int64)
    negative_hist = np.zeros(pixel_bins, dtype=np.int64)

    for prediction in model._iter_predictions(test_loader):
        mask = prediction["mask"]
        if mask is None:
            continue
        update_histograms(
            prediction["map"],
            mask,
            score_min,
            score_max,
            positive_hist,
            negative_hist,
        )

    return {
        "image_auroc": float(sklearn_metrics.roc_auc_score(labels, scores)),
        "pixel_auroc": histogram_auroc(positive_hist, negative_hist),
        "score_min": score_min,
        "score_max": score_max,
        "seconds_per_image": model_seconds / max(1, image_count),
        "images_per_second": image_count / max(model_seconds, 1e-12),
        "end_to_end_seconds_per_image": end_to_end_seconds / max(1, image_count),
    }


def atomic_json_dump(data: Dict[str, object], path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary, path)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(args.data_path)
    device = torch.device(f"cuda:{args.gpu}")
    utils.fix_seeds(args.seed, with_torch=True, with_cuda=True)

    job_dir = Path(args.results_root) / args.method / args.category
    job_dir.mkdir(parents=True, exist_ok=True)
    result_path = job_dir / "job_result.json"
    if result_path.exists():
        LOGGER.info("Completed result already exists: %s", result_path)
        return

    train_dataset, test_dataset, augmentation = build_datasets(args)
    train_loader = build_loader(
        train_dataset, args.batch_size, args.num_workers, shuffle=True
    )
    test_loader = build_loader(
        test_dataset, 1, args.num_workers, shuffle=False
    )
    model_dir = job_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    wall_start = time.perf_counter()
    if args.method == "global_local":
        model = build_global_local_model(args, train_dataset.imagesize, device)
        model.set_model_dir(str(model_dir), f"visa_{args.category}")
        train_start = time.perf_counter()
        model.train(train_loader, None)
        train_seconds = time.perf_counter() - train_start
        resumed = False
        evaluation = evaluate_global_local(model, test_loader, args.pixel_bins)
        input_policy = "global_512x768_local_512_stride384_to_320"
    else:
        model = build_baseline_model(args, train_dataset.imagesize, device)
        model.set_model_dir(str(model_dir), f"visa_{args.category}")
        train_seconds, resumed = train_baseline(
            model, train_loader, job_dir, args
        )
        evaluation = evaluate_baseline(
            model, test_loader, device, args.pixel_bins
        )
        input_policy = "whole_image_letterbox_320"

    result = {
        "method": args.method,
        "category": args.category,
        "gpu": args.gpu,
        "seed": args.seed,
        "train_images": len(train_dataset),
        "test_images": len(test_dataset),
        "imagesize": args.imagesize,
        "meta_epochs": args.meta_epochs,
        "gan_epochs": args.gan_epochs,
        "augmentation": augmentation,
        "input_policy": input_policy,
        "selection_policy": "fixed_final_epoch_test_once",
        "pixel_auroc_mode": f"hist{args.pixel_bins}_dynamic_range",
        "train_seconds_this_process": train_seconds,
        "resumed": resumed,
        "wall_seconds_this_process": time.perf_counter() - wall_start,
        **evaluation,
    }
    for key in (
        "image_auroc",
        "pixel_auroc",
        "seconds_per_image",
        "end_to_end_seconds_per_image",
    ):
        if not math.isfinite(float(result[key])):
            raise FloatingPointError(f"Non-finite result {key}={result[key]}")
    atomic_json_dump(result, result_path)
    LOGGER.info("RESULT %s", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

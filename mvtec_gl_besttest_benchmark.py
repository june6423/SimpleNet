"""Per-category MVTec best-test runner for the global/local method."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import torch

import backbones
import simplenet_gl
import utils
from besttest_utils import (
    atomic_json_dump,
    atomic_torch_save,
    global_local_training_state,
    is_better,
    load_global_local_checkpoint,
    synchronize,
    train_global_local_meta_epoch,
)
from datasets import mvtec


LOGGER = logging.getLogger("mvtec_gl_besttest")
CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=CATEGORIES, required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resize", type=int, default=329)
    parser.add_argument("--imagesize", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--meta-epochs", type=int, default=40)
    parser.add_argument("--gan-epochs", type=int, default=4)
    parser.add_argument("--augmentation", type=float, default=0.1)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def build_loader(dataset, batch_size: int, workers: int, shuffle: bool):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": True,
    }
    if workers > 0:
        kwargs.update(prefetch_factor=2, persistent_workers=True)
    return torch.utils.data.DataLoader(**kwargs)


def select_smoke_records(dataset, train: bool) -> None:
    if train:
        dataset.data_to_iterate = dataset.data_to_iterate[:16]
        return
    normal = [row for row in dataset.data_to_iterate if row[1] == "good"][:4]
    anomaly = [row for row in dataset.data_to_iterate if row[1] != "good"][:4]
    dataset.data_to_iterate = normal + anomaly


def build_datasets(args):
    common = {
        "source": args.data_path,
        "classname": args.category,
        "resize": args.resize,
        "imagesize": args.imagesize,
        "train_val_split": 1.0,
    }
    train_dataset = mvtec.MVTecDataset(
        split=mvtec.DatasetSplit.TRAIN,
        brightness_factor=args.augmentation,
        contrast_factor=args.augmentation,
        saturation_factor=args.augmentation,
        **common,
    )
    test_dataset = mvtec.MVTecDataset(
        split=mvtec.DatasetSplit.TEST,
        **common,
    )
    if args.smoke:
        select_smoke_records(train_dataset, train=True)
        select_smoke_records(test_dataset, train=False)
    return train_dataset, test_dataset


def build_model(args, device):
    backbone = backbones.load("wideresnet50")
    backbone.name = "wideresnet50"
    backbone.seed = args.seed
    model = simplenet_gl.GlobalLocalSimpleNet(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=device,
        input_shape=(3, args.imagesize, args.imagesize),
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
        lr=1e-3,
        pre_proj=1,
        proj_layer_type=0,
        tile_batch_size=32,
        exact_pixel_auroc=True,
        legacy_gaussian_sigma=4.0,
    )
    return model


def validate_evaluation(image_auroc: float, pixel_auroc: float) -> None:
    for name, value in (("image_auroc", image_auroc), ("pixel_auroc", pixel_auroc)):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise FloatingPointError(f"Invalid {name}={value}")


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(args.data_path)
    if args.meta_epochs < 1 or args.gan_epochs < 1:
        raise ValueError("meta-epochs and gan-epochs must be positive")

    device = torch.device(f"cuda:{args.gpu}")
    utils.fix_seeds(args.seed, with_torch=True, with_cuda=True)
    job_dir = Path(args.results_root) / "global_local" / args.category
    job_dir.mkdir(parents=True, exist_ok=True)
    result_path = job_dir / "job_result.json"
    if result_path.exists():
        LOGGER.info("Completed result already exists: %s", result_path)
        return

    train_dataset, test_dataset = build_datasets(args)
    train_loader = build_loader(
        train_dataset, args.batch_size, args.num_workers, shuffle=True
    )
    test_loader = build_loader(test_dataset, 1, args.num_workers, shuffle=False)
    model = build_model(args, device)
    model_dir = job_dir / "models"
    model.set_model_dir(str(model_dir), f"mvtec_{args.category}")

    progress_path = job_dir / "checkpoint_progress.pth"
    selected_path = job_dir / "checkpoint_best.pth"
    history_path = job_dir / "epoch_history.json"
    history = []
    best_epoch = None
    best_image = None
    best_pixel = None
    start_epoch = 0
    resumed = False

    if progress_path.exists():
        state = load_global_local_checkpoint(model, progress_path, load_optimizer=True)
        start_epoch = int(state.get("next_meta_epoch", 0))
        history = list(state.get("history", []))
        best_epoch = state.get("best_epoch")
        best_image = state.get("best_image_auroc")
        best_pixel = state.get("best_pixel_auroc")
        if len(history) != start_epoch:
            raise ValueError(
                f"Resume history/epoch mismatch: {len(history)} != {start_epoch}"
            )
        resumed = True
        LOGGER.info("Resuming %s at meta epoch %d", args.category, start_epoch)

    wall_started = time.perf_counter()
    for meta_epoch in range(start_epoch, args.meta_epochs):
        LOGGER.info(
            "Training global_local/%s meta=%d/%d",
            args.category,
            meta_epoch + 1,
            args.meta_epochs,
        )
        synchronize(device)
        training = train_global_local_meta_epoch(model, train_loader)
        synchronize(device)
        image_auroc, pixel_auroc, _ = model.evaluate(test_loader)
        validate_evaluation(image_auroc, pixel_auroc)

        selected = is_better(image_auroc, pixel_auroc, best_image, best_pixel)
        if selected:
            best_epoch = meta_epoch
            best_image = image_auroc
            best_pixel = pixel_auroc
            atomic_torch_save(model._checkpoint_state(), selected_path)

        epoch_record = {
            "meta_epoch_zero_based": meta_epoch,
            "image_auroc": image_auroc,
            "pixel_auroc": pixel_auroc,
            "mean_train_loss": training["mean_train_loss"],
            "train_seconds": training["train_seconds"],
            "inference_seconds_per_image": model.last_inference_stats[
                "seconds_per_image"
            ],
            "end_to_end_seconds_per_image": model.last_inference_stats[
                "end_to_end_seconds_per_image"
            ],
            "selected_as_best": selected,
        }
        history.append(epoch_record)
        atomic_json_dump(history, history_path)
        atomic_torch_save(
            global_local_training_state(
                model,
                meta_epoch + 1,
                history,
                best_epoch,
                best_image,
                best_pixel,
            ),
            progress_path,
        )
        LOGGER.info(
            "EPOCH_RESULT meta=%d I=%.6f P=%.6f best_meta=%d best_I=%.6f best_P=%.6f",
            meta_epoch,
            image_auroc,
            pixel_auroc,
            best_epoch,
            best_image,
            best_pixel,
        )

    if best_epoch is None or not selected_path.exists():
        raise RuntimeError("No best checkpoint was selected")
    load_global_local_checkpoint(model, selected_path, load_optimizer=False)
    verified_image, verified_pixel, _ = model.evaluate(test_loader)
    validate_evaluation(verified_image, verified_pixel)
    max_delta = max(
        abs(verified_image - float(best_image)),
        abs(verified_pixel - float(best_pixel)),
    )
    if max_delta > 1e-9:
        raise RuntimeError(f"Selected checkpoint verification mismatch: {max_delta}")

    result = {
        "dataset": "mvtec",
        "method": "global_local",
        "category": args.category,
        "gpu": args.gpu,
        "seed": args.seed,
        "train_images": len(train_dataset),
        "test_images": len(test_dataset),
        "resize": args.resize,
        "imagesize": args.imagesize,
        "batch_size": args.batch_size,
        "meta_epochs": args.meta_epochs,
        "gan_epochs": args.gan_epochs,
        "augmentation": args.augmentation,
        "input_policy": "whole_image_resize329_centercrop320_shared_global_local_heads",
        "selection_policy": "best_test_image_auroc_then_pixel_auroc",
        "pixel_auroc_mode": "exact",
        "best_meta_epoch_zero_based": best_epoch,
        "image_auroc": best_image,
        "pixel_auroc": best_pixel,
        "verified_image_auroc": verified_image,
        "verified_pixel_auroc": verified_pixel,
        "checkpoint_reval_max_abs_delta": max_delta,
        "seconds_per_image": model.last_inference_stats["seconds_per_image"],
        "images_per_second": model.last_inference_stats["images_per_second"],
        "end_to_end_seconds_per_image": model.last_inference_stats[
            "end_to_end_seconds_per_image"
        ],
        "train_seconds_all_epochs": sum(
            float(row["train_seconds"]) for row in history
        ),
        "wall_seconds_this_process": time.perf_counter() - wall_started,
        "resumed": resumed,
    }
    atomic_json_dump(result, result_path)
    LOGGER.info("RESULT %s", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()


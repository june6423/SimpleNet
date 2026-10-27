"""Resumable per-category VisA best-test runner for all three methods."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import torch

import utils
import visa_benchmark as fixed_runner
from besttest_utils import (
    atomic_json_dump,
    atomic_torch_save,
    global_local_training_state,
    is_better,
    load_global_local_checkpoint,
    synchronize,
    train_global_local_meta_epoch,
)


LOGGER = logging.getLogger("visa_besttest")
METHODS = fixed_runner.METHODS
CATEGORIES = fixed_runner.CATEGORIES


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


def baseline_progress_state(
    model,
    method: str,
    next_meta_epoch: int,
    history,
    best_epoch,
    best_image,
    best_pixel,
):
    state = fixed_runner.baseline_checkpoint_state(
        model, method, next_meta_epoch
    )
    state.update(
        history=history,
        best_epoch=best_epoch,
        best_image_auroc=best_image,
        best_pixel_auroc=best_pixel,
    )
    return state


def validate_evaluation(evaluation) -> None:
    for key in (
        "image_auroc",
        "pixel_auroc",
        "seconds_per_image",
        "end_to_end_seconds_per_image",
    ):
        value = float(evaluation[key])
        if not math.isfinite(value):
            raise FloatingPointError(f"Non-finite {key}={value}")
    for key in ("image_auroc", "pixel_auroc"):
        value = float(evaluation[key])
        if not 0.0 <= value <= 1.0:
            raise FloatingPointError(f"Out-of-range {key}={value}")


def evaluate(args, model, test_loader, device):
    if args.method == "global_local":
        result = fixed_runner.evaluate_global_local(
            model, test_loader, args.pixel_bins
        )
    else:
        result = fixed_runner.evaluate_baseline(
            model, test_loader, device, args.pixel_bins
        )
    validate_evaluation(result)
    return result


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
    job_dir = Path(args.results_root) / args.method / args.category
    job_dir.mkdir(parents=True, exist_ok=True)
    result_path = job_dir / "job_result.json"
    if result_path.exists():
        LOGGER.info("Completed result already exists: %s", result_path)
        return

    train_dataset, test_dataset, augmentation = fixed_runner.build_datasets(args)
    train_loader = fixed_runner.build_loader(
        train_dataset, args.batch_size, args.num_workers, shuffle=True
    )
    test_loader = fixed_runner.build_loader(
        test_dataset, 1, args.num_workers, shuffle=False
    )
    if args.method == "global_local":
        model = fixed_runner.build_global_local_model(
            args, train_dataset.imagesize, device
        )
        input_policy = "global_512x768_local_512_stride384_to_320"
    else:
        model = fixed_runner.build_baseline_model(
            args, train_dataset.imagesize, device
        )
        input_policy = "whole_image_letterbox_320"
    model_dir = job_dir / "models"
    model.set_model_dir(str(model_dir), f"visa_{args.category}")

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
        if args.method == "global_local":
            state = load_global_local_checkpoint(
                model, progress_path, load_optimizer=True
            )
        else:
            fixed_runner.load_baseline_checkpoint(
                model, progress_path, args.method, load_optimizer=True
            )
            state = torch.load(
                progress_path, map_location=device, weights_only=False
            )
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
        LOGGER.info(
            "Resuming %s/%s at meta epoch %d",
            args.method,
            args.category,
            start_epoch,
        )

    wall_started = time.perf_counter()
    for meta_epoch in range(start_epoch, args.meta_epochs):
        LOGGER.info(
            "Training %s/%s meta=%d/%d",
            args.method,
            args.category,
            meta_epoch + 1,
            args.meta_epochs,
        )
        synchronize(device)
        train_started = time.perf_counter()
        if args.method == "global_local":
            training = train_global_local_meta_epoch(model, train_loader)
            mean_train_loss = training["mean_train_loss"]
        elif args.method == "simplenet_plus":
            model._train_discriminator(
                train_loader, meta_epoch, teacher=(meta_epoch > 0)
            )
            if meta_epoch == 0:
                model.teacher.load_state_dict(model.discriminator.state_dict())
            mean_train_loss = None
        else:
            model._train_discriminator(train_loader)
            mean_train_loss = None
        synchronize(device)
        train_seconds = time.perf_counter() - train_started

        evaluation = evaluate(args, model, test_loader, device)
        image_auroc = float(evaluation["image_auroc"])
        pixel_auroc = float(evaluation["pixel_auroc"])
        selected = is_better(image_auroc, pixel_auroc, best_image, best_pixel)
        if selected:
            best_epoch = meta_epoch
            best_image = image_auroc
            best_pixel = pixel_auroc
            if args.method == "global_local":
                selected_state = model._checkpoint_state()
            else:
                selected_state = fixed_runner.baseline_checkpoint_state(
                    model, args.method, meta_epoch + 1
                )
            atomic_torch_save(selected_state, selected_path)

        epoch_record = {
            "meta_epoch_zero_based": meta_epoch,
            "image_auroc": image_auroc,
            "pixel_auroc": pixel_auroc,
            "mean_train_loss": mean_train_loss,
            "train_seconds": train_seconds,
            "inference_seconds_per_image": evaluation["seconds_per_image"],
            "end_to_end_seconds_per_image": evaluation[
                "end_to_end_seconds_per_image"
            ],
            "score_min": evaluation["score_min"],
            "score_max": evaluation["score_max"],
            "selected_as_best": selected,
        }
        history.append(epoch_record)
        atomic_json_dump(history, history_path)
        if args.method == "global_local":
            progress_state = global_local_training_state(
                model,
                meta_epoch + 1,
                history,
                best_epoch,
                best_image,
                best_pixel,
            )
        else:
            progress_state = baseline_progress_state(
                model,
                args.method,
                meta_epoch + 1,
                history,
                best_epoch,
                best_image,
                best_pixel,
            )
        atomic_torch_save(progress_state, progress_path)
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
    if args.method == "global_local":
        load_global_local_checkpoint(model, selected_path, load_optimizer=False)
    else:
        fixed_runner.load_baseline_checkpoint(
            model, selected_path, args.method, load_optimizer=False
        )
    verified = evaluate(args, model, test_loader, device)
    max_delta = max(
        abs(float(verified["image_auroc"]) - float(best_image)),
        abs(float(verified["pixel_auroc"]) - float(best_pixel)),
    )
    if max_delta > 1e-9:
        raise RuntimeError(f"Selected checkpoint verification mismatch: {max_delta}")

    result = {
        "dataset": "visa",
        "method": args.method,
        "category": args.category,
        "gpu": args.gpu,
        "seed": args.seed,
        "train_images": len(train_dataset),
        "test_images": len(test_dataset),
        "imagesize": args.imagesize,
        "batch_size": args.batch_size,
        "meta_epochs": args.meta_epochs,
        "gan_epochs": args.gan_epochs,
        "augmentation": augmentation,
        "input_policy": input_policy,
        "selection_policy": "best_test_image_auroc_then_pixel_auroc",
        "pixel_auroc_mode": f"hist{args.pixel_bins}_dynamic_range",
        "best_meta_epoch_zero_based": best_epoch,
        "image_auroc": best_image,
        "pixel_auroc": best_pixel,
        "verified_image_auroc": verified["image_auroc"],
        "verified_pixel_auroc": verified["pixel_auroc"],
        "checkpoint_reval_max_abs_delta": max_delta,
        "score_min": verified["score_min"],
        "score_max": verified["score_max"],
        "seconds_per_image": verified["seconds_per_image"],
        "images_per_second": verified["images_per_second"],
        "end_to_end_seconds_per_image": verified[
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


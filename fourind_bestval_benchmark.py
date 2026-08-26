"""Resumable, per-product 4ind best-validation benchmark for three methods.

This runner intentionally does not modify the SimpleNet model files.  It uses
the existing model implementations, selects a checkpoint only from validation
I-AUROC, and evaluates the test split once after selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from sklearn import metrics as sklearn_metrics

import backbones
import simplenet_gl
import utils
import visa_benchmark as baseline_utils
from besttest_utils import (
    atomic_json_dump,
    atomic_torch_save,
    global_local_training_state,
    synchronize,
    train_global_local_meta_epoch,
)
from datasets.fourind import DatasetSplit, FourIndDataset
from datasets.fourind_legacy import FourIndLegacyDataset


LOGGER = logging.getLogger("fourind_bestval")
METHODS = ("simplenet", "simplenet_plus", "global_local")
CATEGORIES = ("KQG27542", "KQG27824")
SELECTION_POLICY = "best_val_image_auroc_keep_earliest_on_tie"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--category", choices=CATEGORIES, required=True)
    parser.add_argument(
        "--data-root",
        default=os.environ.get("FOURIND_DATA_ROOT"),
        help="Extracted dataset root; may also be set by FOURIND_DATA_ROOT.",
    )
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--meta-epochs", type=int, default=40)
    parser.add_argument("--gan-epochs", type=int, default=4)
    parser.add_argument("--imagesize", type=int, default=320)
    parser.add_argument("--global-height", type=int, default=128)
    parser.add_argument("--global-width", type=int, default=2048)
    parser.add_argument("--tile-height", type=int, default=512)
    parser.add_argument("--tile-width", type=int, default=512)
    parser.add_argument("--tile-stride-y", type=int, default=384)
    parser.add_argument("--tile-stride-x", type=int, default=384)
    parser.add_argument("--tile-batch-size", type=int, default=32)
    parser.add_argument("--train-tiles-per-image", type=int, default=1)
    parser.add_argument("--augmentation", type=float, default=0.1)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configuration_sha256(args: argparse.Namespace, manifest_sha256: str) -> str:
    fields = {
        "method": args.method,
        "category": args.category,
        "manifest_sha256": manifest_sha256,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "gan_epochs": args.gan_epochs,
        "imagesize": args.imagesize,
        "global_height": args.global_height,
        "global_width": args.global_width,
        "tile_height": args.tile_height,
        "tile_width": args.tile_width,
        "tile_stride_y": args.tile_stride_y,
        "tile_stride_x": args.tile_stride_x,
        "tile_batch_size": args.tile_batch_size,
        "train_tiles_per_image": args.train_tiles_per_image,
        "augmentation": 0.0 if args.method == "simplenet" else args.augmentation,
        "smoke": args.smoke,
    }
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_loader(dataset, batch_size: int, workers: int, shuffle: bool):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if workers > 0:
        kwargs.update(prefetch_factor=2, persistent_workers=True)
    return torch.utils.data.DataLoader(**kwargs)


def select_smoke_records(dataset, train: bool) -> None:
    if train:
        dataset.records = dataset.records[: min(4, len(dataset.records))]
        return
    normal = [record for record in dataset.records if not record.is_anomaly][:2]
    anomaly = [record for record in dataset.records if record.is_anomaly][:2]
    dataset.records = normal + anomaly
    if len(normal) < 1 or len(anomaly) < 1:
        raise ValueError(f"Smoke split {dataset.split.value} needs both classes.")


def build_datasets(args: argparse.Namespace):
    augmentation = 0.0 if args.method == "simplenet" else args.augmentation
    common = {
        "source": args.data_root,
        "classname": args.category,
        "manifest_path": args.manifest_path,
        "imagesize": args.imagesize,
        "seed": args.seed,
    }
    dataset_class = (
        FourIndDataset if args.method == "global_local" else FourIndLegacyDataset
    )
    if args.method == "global_local":
        common.update(
            global_height=args.global_height,
            global_width=args.global_width,
            tile_height=args.tile_height,
            tile_width=args.tile_width,
            tile_stride_y=args.tile_stride_y,
            tile_stride_x=args.tile_stride_x,
            train_tiles_per_image=args.train_tiles_per_image,
        )
    train_dataset = dataset_class(
        split=DatasetSplit.TRAIN,
        brightness_factor=augmentation,
        contrast_factor=augmentation,
        saturation_factor=augmentation,
        **common,
    )
    val_dataset = dataset_class(split=DatasetSplit.VAL, **common)
    test_dataset = dataset_class(split=DatasetSplit.TEST, **common)
    if args.smoke:
        select_smoke_records(train_dataset, train=True)
        select_smoke_records(val_dataset, train=False)
        select_smoke_records(test_dataset, train=False)
    return train_dataset, val_dataset, test_dataset, augmentation


def build_model(args: argparse.Namespace, device: torch.device):
    if args.method != "global_local":
        return baseline_utils.build_baseline_model(
            args, (3, args.imagesize, args.imagesize), device
        )
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
        tile_batch_size=args.tile_batch_size,
        exact_pixel_auroc=False,
        legacy_gaussian_sigma=0.0,
    )
    return model


def _validate_auc(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise FloatingPointError(f"Invalid {name}={value}")


def summarize_image_scores(
    labels: List[int],
    scores: List[float],
    anomaly_types: List[str],
    model_seconds: float,
    end_to_end_seconds: float,
) -> Dict[str, object]:
    if not labels or not (
        len(labels) == len(scores) == len(anomaly_types)
    ):
        raise ValueError(
            "Prediction lengths differ: "
            f"labels={len(labels)} scores={len(scores)} "
            f"anomalies={len(anomaly_types)}"
        )
    if len(set(labels)) != 2:
        raise ValueError("Image-level AUROC requires both normal and anomaly samples.")
    image_auroc = float(sklearn_metrics.roc_auc_score(labels, scores))
    _validate_auc("image_auroc", image_auroc)

    normal_indices = [index for index, label in enumerate(labels) if label == 0]
    defects: Dict[str, Dict[str, object]] = {}
    for defect in sorted(
        {name for name, label in zip(anomaly_types, labels) if label == 1}
    ):
        defect_indices = [
            index
            for index, (name, label) in enumerate(zip(anomaly_types, labels))
            if label == 1 and name == defect
        ]
        selected = normal_indices + defect_indices
        selected_labels = [labels[index] for index in selected]
        selected_scores = [scores[index] for index in selected]
        auc = float(sklearn_metrics.roc_auc_score(selected_labels, selected_scores))
        _validate_auc(f"{defect}_image_auroc", auc)
        defects[defect] = {
            "image_auroc": auc,
            "normal_images": len(normal_indices),
            "anomaly_images": len(defect_indices),
        }

    counts = Counter("normal" if label == 0 else "anomaly" for label in labels)
    image_count = len(labels)
    stats = {
        "images": image_count,
        "model_seconds": model_seconds,
        "seconds_per_image": model_seconds / max(1, image_count),
        "images_per_second": image_count / max(model_seconds, 1e-12),
        "end_to_end_seconds": end_to_end_seconds,
        "end_to_end_seconds_per_image": end_to_end_seconds / max(1, image_count),
    }
    return {
        "image_auroc": image_auroc,
        "image_count": image_count,
        "normal_images": counts["normal"],
        "anomaly_images": counts["anomaly"],
        "score_min": min(scores),
        "score_max": max(scores),
        "per_defect": defects,
        **stats,
    }


def evaluate_global_local_image_level(model, dataloader: Iterable) -> Dict[str, object]:
    labels: List[int] = []
    scores: List[float] = []
    anomaly_types: List[str] = []
    model._model_inference_seconds = 0.0
    model._model_inference_images = 0
    end_to_end_start = time.perf_counter()

    for prediction in model._iter_predictions(dataloader):
        score = float(prediction["score"])
        if not math.isfinite(score):
            raise FloatingPointError(f"Non-finite image score: {score}")
        labels.append(int(prediction["label"]))
        scores.append(score)
        anomaly_types.append(str(prediction["anomaly"]))

    result = summarize_image_scores(
        labels=labels,
        scores=scores,
        anomaly_types=anomaly_types,
        model_seconds=float(model._model_inference_seconds),
        end_to_end_seconds=time.perf_counter() - end_to_end_start,
    )
    model.last_inference_stats = {
        key: result[key]
        for key in (
            "images",
            "model_seconds",
            "seconds_per_image",
            "images_per_second",
            "end_to_end_seconds",
            "end_to_end_seconds_per_image",
        )
    }
    return result


@torch.inference_mode()
def evaluate_baseline_image_level(
    model, dataloader: Iterable, device: torch.device
) -> Dict[str, object]:
    labels: List[int] = []
    scores: List[float] = []
    anomaly_types: List[str] = []
    model_seconds = 0.0
    end_to_end_start = time.perf_counter()

    for data in dataloader:
        synchronize(device)
        inference_start = time.perf_counter()
        batch_scores, _, _ = model._predict(data["image"])
        synchronize(device)
        model_seconds += time.perf_counter() - inference_start
        for score in batch_scores:
            value = float(np.asarray(score).reshape(-1)[0])
            if not math.isfinite(value):
                raise FloatingPointError(f"Non-finite image score: {value}")
            scores.append(value)
        labels.extend(int(value) for value in data["is_anomaly"].cpu().tolist())
        anomalies = data["anomaly"]
        anomaly_types.extend(
            str(value) for value in (anomalies if isinstance(anomalies, list) else [anomalies])
        )

    return summarize_image_scores(
        labels=labels,
        scores=scores,
        anomaly_types=anomaly_types,
        model_seconds=model_seconds,
        end_to_end_seconds=time.perf_counter() - end_to_end_start,
    )


def evaluate_image_level(
    args: argparse.Namespace, model, dataloader: Iterable, device: torch.device
) -> Dict[str, object]:
    if args.method == "global_local":
        return evaluate_global_local_image_level(model, dataloader)
    return evaluate_baseline_image_level(model, dataloader, device)


def is_better_val(candidate: float, best: Optional[float]) -> bool:
    """Use validation I-AUROC only; exact ties keep the earlier epoch."""
    return best is None or candidate > best + 1e-12


def progress_state(
    method: str,
    model,
    next_meta_epoch: int,
    history: List[Dict[str, object]],
    best_epoch: Optional[int],
    best_val_image_auroc: Optional[float],
    manifest_sha256: str,
    configuration_sha256_value: str,
    category: str,
) -> Dict[str, object]:
    if method == "global_local":
        state = global_local_training_state(
            model=model,
            next_meta_epoch=next_meta_epoch,
            history=history,
            best_epoch=best_epoch,
            best_image_auroc=best_val_image_auroc,
            best_pixel_auroc=0.0 if best_val_image_auroc is not None else None,
        )
    else:
        state = baseline_utils.baseline_checkpoint_state(
            model, method, next_meta_epoch
        )
    state.update(
        {
            "dataset": "4ind",
            "method": method,
            "category": category,
            "selection_policy": SELECTION_POLICY,
            "best_val_image_auroc": best_val_image_auroc,
            "best_epoch": best_epoch,
            "history": history,
            "manifest_sha256": manifest_sha256,
            "configuration_sha256": configuration_sha256_value,
        }
    )
    return state


def load_checkpoint(model, path: Path, method: str, load_optimizer: bool):
    # Loading the full progress state directly onto CUDA also moves the saved
    # CPU RNG byte tensor, which makes torch.set_rng_state fail.  Load on CPU;
    # module/optimizer load_state_dict moves parameter state as required.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if method == "global_local":
        model.discriminator.load_state_dict(state["local_discriminator"])
        model.global_discriminator.load_state_dict(state["global_discriminator"])
        if model.pre_proj > 0:
            model.pre_projection.load_state_dict(state["local_projection"])
            model.global_projection.load_state_dict(state["global_projection"])
        if model.train_backbone and "backbone" in state:
            model.backbone.load_state_dict(state["backbone"])
        calibration = state.get("calibration", {})
        for name, buffer_name in (
            ("local_median", "local_score_median"),
            ("local_scale", "local_score_scale"),
            ("global_median", "global_score_median"),
            ("global_scale", "global_score_scale"),
        ):
            if name in calibration:
                getattr(model, buffer_name).fill_(float(calibration[name]))
        if load_optimizer:
            optimizers = state["optimizers"]
            model.dsc_opt.load_state_dict(optimizers["local_discriminator"])
            model.global_dsc_opt.load_state_dict(optimizers["global_discriminator"])
            if model.pre_proj > 0:
                model.proj_opt.load_state_dict(optimizers["local_projection"])
                model.global_proj_opt.load_state_dict(optimizers["global_projection"])
            if model.train_backbone:
                model.backbone_opt.load_state_dict(optimizers["backbone"])
    else:
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

    if load_optimizer:
        if "torch_rng" in state:
            torch.set_rng_state(state["torch_rng"].cpu())
        if torch.cuda.is_available() and "cuda_rng" in state:
            torch.cuda.set_rng_state_all(
                [rng.cpu() for rng in state["cuda_rng"]]
            )
        if "numpy_rng" in state:
            np.random.set_state(state["numpy_rng"])
    return state


def selected_checkpoint_state(model, method: str, next_meta_epoch: int):
    if method == "global_local":
        return model._checkpoint_state()
    return baseline_utils.baseline_checkpoint_state(
        model, method, next_meta_epoch
    )


def train_one_meta_epoch(
    args: argparse.Namespace, model, train_loader, meta_epoch: int
):
    if args.method == "global_local":
        return train_global_local_meta_epoch(model, train_loader)

    started = time.perf_counter()
    if args.method == "simplenet_plus":
        model._train_discriminator(
            train_loader, meta_epoch, teacher=(meta_epoch > 0)
        )
        if meta_epoch == 0:
            model.teacher.load_state_dict(model.discriminator.state_dict())
    else:
        model._train_discriminator(train_loader)
    return {
        "mean_train_loss": None,
        "train_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.data_root:
        raise ValueError("Set --data-root or FOURIND_DATA_ROOT.")
    data_root = Path(args.data_root).expanduser().resolve()
    manifest_path = Path(args.manifest_path).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if args.meta_epochs < 1 or args.gan_epochs < 1:
        raise ValueError("meta-epochs and gan-epochs must be positive.")
    if args.gpu < 0 or not torch.cuda.is_available():
        raise RuntimeError("4ind training requires a CUDA GPU.")
    if args.gpu >= torch.cuda.device_count():
        raise ValueError(
            f"GPU {args.gpu} is unavailable; count={torch.cuda.device_count()}"
        )

    args.data_root = str(data_root)
    args.manifest_path = str(manifest_path)
    manifest_digest = file_sha256(manifest_path)
    configuration_digest = configuration_sha256(args, manifest_digest)
    device = torch.device(f"cuda:{args.gpu}")
    utils.fix_seeds(args.seed, with_torch=True, with_cuda=True)

    job_dir = (
        Path(args.results_root).expanduser().resolve() / args.method / args.category
    )
    job_dir.mkdir(parents=True, exist_ok=True)
    result_path = job_dir / "job_result.json"
    if result_path.exists():
        LOGGER.info("Completed result already exists: %s", result_path)
        return
    atomic_json_dump(vars(args), job_dir / "args.json")

    train_dataset, val_dataset, test_dataset, augmentation = build_datasets(args)
    LOGGER.info(
        "DATA method=%s category=%s train=%d val=%d test=%d manifest=%s",
        args.method,
        args.category,
        len(train_dataset.records),
        len(val_dataset.records),
        len(test_dataset.records),
        manifest_digest,
    )
    train_loader = build_loader(
        train_dataset, args.batch_size, args.num_workers, shuffle=True
    )
    val_loader = build_loader(val_dataset, 1, args.num_workers, shuffle=False)
    test_loader = build_loader(test_dataset, 1, args.num_workers, shuffle=False)

    model = build_model(args, device)
    model_dir = job_dir / "models"
    model.set_model_dir(str(model_dir), f"fourind_{args.method}_{args.category}")

    progress_path = job_dir / "checkpoint_progress.pth"
    selected_path = job_dir / "checkpoint_best_val.pth"
    history_path = job_dir / "epoch_history.json"
    history: List[Dict[str, object]] = []
    best_epoch: Optional[int] = None
    best_val: Optional[float] = None
    start_epoch = 0
    resumed = False

    if progress_path.exists():
        state = load_checkpoint(
            model, progress_path, args.method, load_optimizer=True
        )
        if state.get("manifest_sha256") != manifest_digest:
            raise ValueError("Manifest changed since the progress checkpoint was saved.")
        saved_configuration = state.get("configuration_sha256")
        if saved_configuration and saved_configuration != configuration_digest:
            raise ValueError("Training configuration changed since checkpoint save.")
        if state.get("method") != args.method:
            raise ValueError("Progress checkpoint method mismatch.")
        if state.get("category") != args.category:
            raise ValueError("Progress checkpoint category mismatch.")
        start_epoch = int(state.get("next_meta_epoch", 0))
        history = list(state.get("history", []))
        best_epoch = state.get("best_epoch")
        best_val = state.get("best_val_image_auroc")
        if len(history) != start_epoch:
            raise ValueError(
                f"Resume history/epoch mismatch: {len(history)} != {start_epoch}"
            )
        if start_epoch > args.meta_epochs:
            raise ValueError(
                f"Progress epoch {start_epoch} exceeds requested {args.meta_epochs}."
            )
        resumed = True
        LOGGER.info("Resuming %s at meta epoch %d", args.category, start_epoch)

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
        training = train_one_meta_epoch(args, model, train_loader, meta_epoch)
        synchronize(device)
        validation = evaluate_image_level(args, model, val_loader, device)
        val_image_auroc = float(validation["image_auroc"])
        selected = is_better_val(val_image_auroc, best_val)
        if selected:
            best_epoch = meta_epoch
            best_val = val_image_auroc
            atomic_torch_save(
                selected_checkpoint_state(model, args.method, meta_epoch + 1),
                selected_path,
            )

        epoch_record = {
            "meta_epoch_zero_based": meta_epoch,
            "val_image_auroc": val_image_auroc,
            "mean_train_loss": training["mean_train_loss"],
            "train_seconds": training["train_seconds"],
            "val_seconds_per_image": validation["seconds_per_image"],
            "val_end_to_end_seconds_per_image": validation[
                "end_to_end_seconds_per_image"
            ],
            "val_score_min": validation["score_min"],
            "val_score_max": validation["score_max"],
            "selected_as_best": selected,
        }
        history.append(epoch_record)
        atomic_json_dump(history, history_path)
        atomic_torch_save(
            progress_state(
                method=args.method,
                model=model,
                next_meta_epoch=meta_epoch + 1,
                history=history,
                best_epoch=best_epoch,
                best_val_image_auroc=best_val,
                manifest_sha256=manifest_digest,
                configuration_sha256_value=configuration_digest,
                category=args.category,
            ),
            progress_path,
        )
        LOGGER.info(
            "EPOCH_RESULT meta=%d val_I=%.6f best_meta=%d best_val_I=%.6f",
            meta_epoch,
            val_image_auroc,
            best_epoch,
            best_val,
        )

    if best_epoch is None or best_val is None or not selected_path.exists():
        raise RuntimeError("No best-validation checkpoint was selected.")
    load_checkpoint(model, selected_path, args.method, load_optimizer=False)
    verified_val = evaluate_image_level(args, model, val_loader, device)
    val_reval_delta = abs(float(verified_val["image_auroc"]) - best_val)
    if val_reval_delta > 1e-9:
        raise RuntimeError(
            f"Best-validation checkpoint verification mismatch: {val_reval_delta}"
        )

    LOGGER.info("Evaluating held-out test split once with best-validation checkpoint.")
    test_result = evaluate_image_level(args, model, test_loader, device)
    result = {
        "dataset": "4ind",
        "method": args.method,
        "category": args.category,
        "gpu": args.gpu,
        "seed": args.seed,
        "smoke": args.smoke,
        "manifest_sha256": manifest_digest,
        "configuration_sha256": configuration_digest,
        "train_images": len(train_dataset.records),
        "val_images": len(val_dataset.records),
        "test_images": len(test_dataset.records),
        "meta_epochs": args.meta_epochs,
        "gan_epochs": args.gan_epochs,
        "batch_size": args.batch_size,
        "augmentation": augmentation,
        "input_policy": (
            (
                f"global_{args.global_height}x{args.global_width}_local_"
                f"{args.tile_height}x{args.tile_width}_stride_"
                f"{args.tile_stride_y}x{args.tile_stride_x}_to_{args.imagesize}"
            )
            if args.method == "global_local"
            else f"whole_image_letterbox_{args.imagesize}"
        ),
        "selection_policy": SELECTION_POLICY,
        "test_evaluations_during_training": 0,
        "best_meta_epoch_zero_based": best_epoch,
        "best_val_image_auroc": best_val,
        "verified_val_image_auroc": verified_val["image_auroc"],
        "checkpoint_val_reval_abs_delta": val_reval_delta,
        "val_per_defect": verified_val["per_defect"],
        "test_image_auroc": test_result["image_auroc"],
        "test_normal_images": test_result["normal_images"],
        "test_anomaly_images": test_result["anomaly_images"],
        "test_per_defect": test_result["per_defect"],
        "test_score_min": test_result["score_min"],
        "test_score_max": test_result["score_max"],
        "test_seconds_per_image": test_result["seconds_per_image"],
        "test_images_per_second": test_result["images_per_second"],
        "test_end_to_end_seconds_per_image": test_result[
            "end_to_end_seconds_per_image"
        ],
        "train_seconds_all_epochs": sum(
            float(row["train_seconds"]) for row in history
        ),
        "validation_seconds_all_epochs": sum(
            float(row["val_end_to_end_seconds_per_image"]) * len(val_dataset.records)
            for row in history
        ),
        "wall_seconds_this_process": time.perf_counter() - wall_started,
        "resumed": resumed,
        "pixel_metrics_available": False,
    }
    atomic_json_dump(result, result_path)
    LOGGER.info("RESULT %s", json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

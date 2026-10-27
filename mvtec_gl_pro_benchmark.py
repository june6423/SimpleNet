"""Evaluate PRO-AUROC for a saved MVTec global/local best checkpoint.

This is an evaluation-only runner.  It intentionally does not train or select
a checkpoint, and it writes to a result tree separate from the best-test run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics as sklearn_metrics

import metrics
import utils
from besttest_utils import load_global_local_checkpoint
from datasets import mvtec
from mvtec_gl_besttest_benchmark import CATEGORIES, build_loader, build_model


# Keep compatibility local instead of modifying the baseline metrics module.
if "bool" not in np.__dict__:
    np.bool = np.bool_


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=CATEGORIES, required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resize", type=int, default=329)
    parser.add_argument("--imagesize", type=int, default=320)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--meta-epochs", type=int, default=40)
    parser.add_argument("--gan-epochs", type=int, default=4)
    parser.add_argument("--augmentation", type=float, default=0.1)
    parser.add_argument("--num-thresholds", type=int, default=200)
    return parser.parse_args()


def atomic_json_dump(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_test_loader(args):
    dataset = mvtec.MVTecDataset(
        source=args.data_path,
        classname=args.category,
        resize=args.resize,
        imagesize=args.imagesize,
        split=mvtec.DatasetSplit.TEST,
        train_val_split=1.0,
    )
    return dataset, build_loader(dataset, batch_size=1, workers=args.num_workers, shuffle=False)


def collect_predictions(model, dataloader):
    scores, labels, anomaly_maps, masks = [], [], [], []
    model._model_inference_seconds = 0.0
    model._model_inference_images = 0
    for prediction in model._iter_predictions(dataloader):
        mask = prediction["mask"]
        if mask is None:
            raise RuntimeError("MVTec PRO evaluation requires a mask for every test image")
        anomaly_map = np.asarray(prediction["map"], dtype=np.float32)
        mask = np.asarray(mask)
        if anomaly_map.shape != mask.shape:
            mask = F.interpolate(
                torch.from_numpy(mask)[None, None].float(),
                size=anomaly_map.shape,
                mode="nearest",
            )[0, 0].numpy()
        scores.append(float(prediction["score"]))
        labels.append(int(prediction["label"]))
        anomaly_maps.append(anomaly_map)
        masks.append((mask > 0).astype(np.uint8))
    return (
        np.asarray(scores, dtype=np.float64),
        np.asarray(labels, dtype=np.uint8),
        np.stack(anomaly_maps),
        np.stack(masks),
    )


def main():
    args = parse_args()
    if args.num_thresholds != 200:
        raise ValueError("The declared comparison protocol requires 200 thresholds")
    source_dir = Path(args.source_root) / "global_local" / args.category
    checkpoint_path = source_dir / "checkpoint_best.pth"
    source_result_path = source_dir / "job_result.json"
    if not checkpoint_path.is_file() or not source_result_path.is_file():
        raise FileNotFoundError(f"Missing source checkpoint/result in {source_dir}")

    output_dir = Path(args.results_root) / args.category
    output_path = output_dir / "job_result.json"
    if output_path.exists():
        print(f"Completed PRO result already exists: {output_path}")
        return

    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    device = torch.device(f"cuda:{args.gpu}")
    utils.fix_seeds(args.seed, with_torch=True, with_cuda=True)
    dataset, test_loader = build_test_loader(args)
    model = build_model(args, device)
    load_global_local_checkpoint(model, checkpoint_path, load_optimizer=False)

    started = time.perf_counter()
    scores, labels, anomaly_maps, masks = collect_predictions(model, test_loader)
    image_auroc = float(sklearn_metrics.roc_auc_score(labels, scores))
    pixel_auroc = float(
        sklearn_metrics.roc_auc_score(masks.ravel(), anomaly_maps.ravel())
    )
    pro_auroc = float(metrics.compute_pro(masks, anomaly_maps, num_th=200))
    wall_seconds = time.perf_counter() - started

    for name, value in (
        ("image_auroc", image_auroc),
        ("pixel_auroc", pixel_auroc),
        ("pro_auroc", pro_auroc),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise FloatingPointError(f"Invalid {name}={value}")
    image_delta = abs(image_auroc - float(source_result["image_auroc"]))
    pixel_delta = abs(pixel_auroc - float(source_result["pixel_auroc"]))
    if max(image_delta, pixel_delta) > 1e-9:
        raise RuntimeError(
            "Checkpoint correspondence failed: "
            f"image_delta={image_delta}, pixel_delta={pixel_delta}"
        )

    result = {
        "dataset": "mvtec",
        "method": "global_local",
        "category": args.category,
        "gpu": args.gpu,
        "seed": args.seed,
        "resize": args.resize,
        "imagesize": args.imagesize,
        "test_images": len(dataset),
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "source_best_meta_epoch_zero_based": source_result[
            "best_meta_epoch_zero_based"
        ],
        "selection_policy": source_result["selection_policy"],
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "pro_auroc": pro_auroc,
        "checkpoint_image_auroc_delta": image_delta,
        "checkpoint_pixel_auroc_delta": pixel_delta,
        "pro_protocol": {
            "implementation": "metrics.compute_pro",
            "num_thresholds": 200,
            "dilation_kernel": "5x5 rectangle",
            "fpr_limit": 0.3,
            "map_input": "selected-checkpoint fused anomaly map",
        },
        "wall_seconds": wall_seconds,
    }
    atomic_json_dump(result, output_path)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

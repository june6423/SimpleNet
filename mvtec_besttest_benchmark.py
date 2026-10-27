"""Run one MVTec category with the untouched SimpleNet baseline modules.

The model's original ``train`` method owns best-on-test selection.  This
wrapper records the per-meta-epoch metrics, reloads the selected checkpoint,
and measures inference latency without changing simplenet.py or
simplenet_plus.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import types
from pathlib import Path

import numpy as np
import torch

import backbones
from datasets import mvtec
import simplenet
import simplenet_plus
import utils


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
METHODS = ("simplenet", "simplenet_plus")

# The untouched baseline's optional PRO calculation still uses the NumPy
# alias removed in NumPy 1.24.  Keep the compatibility shim local to this
# runner instead of editing metrics.py.
if "bool" not in np.__dict__:
    np.bool = np.bool_


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
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
    parser.add_argument("--augmentation", type=float, default=None)
    return parser.parse_args()


def atomic_json_dump(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary, path)


def make_loader(dataset, batch_size, shuffle, workers):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": True,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return torch.utils.data.DataLoader(**kwargs)


def build_dataloaders(args, augmentation):
    train_dataset = mvtec.MVTecDataset(
        args.data_path,
        classname=args.category,
        resize=args.resize,
        imagesize=args.imagesize,
        split=mvtec.DatasetSplit.TRAIN,
        train_val_split=1.0,
        brightness_factor=augmentation,
        contrast_factor=augmentation,
        saturation_factor=augmentation,
    )
    test_dataset = mvtec.MVTecDataset(
        args.data_path,
        classname=args.category,
        resize=args.resize,
        imagesize=args.imagesize,
        split=mvtec.DatasetSplit.TEST,
    )
    train_loader = make_loader(
        train_dataset, args.batch_size, shuffle=True, workers=args.num_workers
    )
    test_loader = make_loader(
        test_dataset, 1, shuffle=False, workers=args.num_workers
    )
    train_loader.name = f"mvtec_{args.category}"
    return train_loader, test_loader


def build_model(args, device):
    module = simplenet if args.method == "simplenet" else simplenet_plus
    backbone = backbones.load("wideresnet50")
    backbone.name = "wideresnet50"
    backbone.seed = args.seed
    model = module.SimpleNet(device)
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
        aed_meta_epochs=1,
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


def collect_mvtec_predictions(self, dataloader, prefix=""):
    """Collect MVTec score maps and masks omitted by the ETRI inference path."""
    _ = self.forward_modules.eval()
    all_scores = []
    all_masks = []
    all_labels_gt = []
    all_masks_gt = []
    all_anomalies = []
    for data in dataloader:
        labels = data["is_anomaly"].numpy().tolist()
        if data.get("mask") is not None:
            all_masks_gt.extend(data["mask"].numpy().tolist())
        scores, masks, features = self._predict(data["image"])
        all_scores.extend(scores)
        all_masks.extend(masks)
        all_labels_gt.extend(labels)
        all_anomalies.extend(list(data["anomaly"]))
        del scores, masks, features
    return (
        all_scores,
        all_masks,
        [],
        all_labels_gt,
        all_masks_gt,
        all_anomalies,
    )


def load_selected_checkpoint(model, path: Path):
    state = torch.load(path, map_location=model.device)
    if "discriminator" in state:
        model.discriminator.load_state_dict(state["discriminator"])
        if model.pre_proj > 0 and "pre_projection" in state:
            model.pre_projection.load_state_dict(state["pre_projection"])
    else:
        model.load_state_dict(state, strict=False)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    args = parse_args()
    result_dir = Path(args.results_root) / args.method / args.category
    result_path = result_dir / "job_result.json"
    if result_path.exists():
        print(f"Result already exists, preserving it: {result_path}")
        return

    augmentation = (
        args.augmentation
        if args.augmentation is not None
        else (0.0 if args.method == "simplenet" else 0.1)
    )
    if not math.isfinite(augmentation) or augmentation < 0:
        raise ValueError(f"Invalid augmentation: {augmentation}")

    device = torch.device(f"cuda:{args.gpu}")
    utils.fix_seeds(args.seed)
    train_loader, test_loader = build_dataloaders(args, augmentation)
    model = build_model(args, device)
    model._predict_dataloader = types.MethodType(collect_mvtec_predictions, model)
    model.set_model_dir(str(result_dir / "model"), f"mvtec_{args.category}")
    checkpoint_path = Path(model.ckpt_dir) / "ckpt.pth"
    resumed = checkpoint_path.exists()

    history = []
    original_evaluate = model._evaluate

    def tracked_evaluate(*evaluate_args, **evaluate_kwargs):
        values = original_evaluate(*evaluate_args, **evaluate_kwargs)
        history.append(
            {
                "evaluation_index": len(history),
                "image_auroc": float(values[0]),
                "pixel_auroc": float(values[1]),
                "pro_auroc": float(values[2]),
            }
        )
        return values

    model._evaluate = tracked_evaluate
    synchronize(device)
    training_started = time.perf_counter()
    best_record = model.train(train_loader, test_loader)
    synchronize(device)
    training_seconds = time.perf_counter() - training_started
    model._evaluate = original_evaluate

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Selected checkpoint was not saved: {checkpoint_path}")
    load_selected_checkpoint(model, checkpoint_path)

    synchronize(device)
    inference_started = time.perf_counter()
    prediction = model.predict(test_loader)
    synchronize(device)
    inference_seconds = time.perf_counter() - inference_started
    verified = original_evaluate(test_loader, *prediction)

    best_epoch = None
    training_history = history
    if not resumed and len(history) == args.meta_epochs:
        best_epoch = max(
            range(len(history)),
            key=lambda index: (
                history[index]["image_auroc"],
                history[index]["pixel_auroc"],
            ),
        )
    elif resumed:
        training_history = []

    result = {
        "method": args.method,
        "category": args.category,
        "selection_policy": "best_test_image_auroc_then_pixel_auroc",
        "best_meta_epoch_zero_based": best_epoch,
        "image_auroc": float(best_record[0]),
        "pixel_auroc": float(best_record[1]),
        "pro_auroc": float(best_record[2]),
        "verified_image_auroc": float(verified[0]),
        "verified_pixel_auroc": float(verified[1]),
        "verified_pro_auroc": float(verified[2]),
        "seconds_per_image": inference_seconds / len(test_loader.dataset),
        "images_per_second": len(test_loader.dataset) / inference_seconds,
        "train_seconds_this_process": training_seconds,
        "train_images": len(train_loader.dataset),
        "test_images": len(test_loader.dataset),
        "gpu": args.gpu,
        "seed": args.seed,
        "resize": args.resize,
        "imagesize": args.imagesize,
        "batch_size": args.batch_size,
        "meta_epochs": args.meta_epochs,
        "gan_epochs": args.gan_epochs,
        "augmentation": augmentation,
        "resumed_from_selected_checkpoint": resumed,
    }
    for metric in (
        "image_auroc",
        "pixel_auroc",
        "verified_image_auroc",
        "verified_pixel_auroc",
        "seconds_per_image",
    ):
        if not math.isfinite(float(result[metric])):
            raise ValueError(f"Non-finite result {metric}: {result[metric]}")
    for metric in (
        "image_auroc",
        "pixel_auroc",
        "verified_image_auroc",
        "verified_pixel_auroc",
    ):
        if not 0.0 <= float(result[metric]) <= 1.0:
            raise ValueError(f"Out-of-range AUROC {metric}: {result[metric]}")

    atomic_json_dump(training_history, result_dir / "epoch_history.json")
    atomic_json_dump(result, result_path)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

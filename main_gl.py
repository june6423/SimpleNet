"""Standalone launcher for the global/local Gaussian SimpleNet variant."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import os
from pathlib import Path

import torch

import backbones
import simplenet_gl
import utils


LOGGER = logging.getLogger(__name__)

DATASETS = {
    "mvtec": ("datasets.mvtec", "MVTecDataset"),
    "visa": ("datasets.visa", "VisADataset"),
    "fourind": ("datasets.fourind", "FourIndDataset"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/evaluate overlap-tiled global/local SimpleNet."
    )
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--manifest-path")
    parser.add_argument("--results-path", default="results_global_local")
    parser.add_argument("--run-name", default="run")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--backbone", default="wideresnet50")
    parser.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    parser.add_argument("--pretrain-embed-dimension", type=int, default=1536)
    parser.add_argument("--target-embed-dimension", type=int, default=1536)
    parser.add_argument("--patchsize", type=int, default=3)
    parser.add_argument("--meta-epochs", type=int, default=40)
    parser.add_argument("--gan-epochs", type=int, default=4)
    parser.add_argument("--noise-std", type=float, default=0.015)
    parser.add_argument("--global-noise-std", type=float)
    parser.add_argument("--local-noise-std", type=float)
    parser.add_argument("--dsc-hidden", type=int, default=1024)
    parser.add_argument("--dsc-layers", type=int, default=2)
    parser.add_argument("--dsc-margin", type=float, default=0.5)
    parser.add_argument("--dsc-lr", type=float, default=2e-4)
    parser.add_argument("--adapter-layers", type=int, default=1)
    parser.add_argument("--mix-noise", type=int, default=1)

    parser.add_argument("--resize", type=int, default=329)
    parser.add_argument("--imagesize", type=int, default=320)
    parser.add_argument("--global-height", type=int)
    parser.add_argument("--global-width", type=int)
    parser.add_argument("--tile-height", type=int, default=512)
    parser.add_argument("--tile-width", type=int, default=512)
    parser.add_argument("--tile-stride-y", type=int, default=384)
    parser.add_argument("--tile-stride-x", type=int, default=384)
    parser.add_argument("--tile-batch-size", type=int, default=32)
    parser.add_argument("--train-tiles-per-image", type=int, default=1)
    parser.add_argument("--brightness", type=float, default=0.1)
    parser.add_argument("--contrast", type=float, default=0.1)
    parser.add_argument("--saturation", type=float, default=0.1)
    return parser.parse_args()


def build_dataset(args, category, split):
    module_name, class_name = DATASETS[args.dataset]
    module = importlib.import_module(module_name)
    dataset_class = getattr(module, class_name)
    split_enum = module.DatasetSplit
    common = dict(
        source=args.data_path,
        classname=category,
        split=split_enum(split),
        imagesize=args.imagesize,
        resize=args.resize,
        seed=args.seed,
        brightness_factor=args.brightness if split == "train" else 0.0,
        contrast_factor=args.contrast if split == "train" else 0.0,
        saturation_factor=args.saturation if split == "train" else 0.0,
    )
    if args.dataset != "mvtec":
        default_global_size = (128, 2048) if args.dataset == "fourind" else (512, 768)
        common.update(
            manifest_path=args.manifest_path,
            global_height=args.global_height or default_global_size[0],
            global_width=args.global_width or default_global_size[1],
            tile_height=args.tile_height,
            tile_width=args.tile_width,
            tile_stride_y=args.tile_stride_y,
            tile_stride_x=args.tile_stride_x,
            train_tiles_per_image=args.train_tiles_per_image,
        )
    return dataset_class(**common)


def build_loader(dataset, batch_size, workers, shuffle):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    if workers > 0:
        kwargs.update(prefetch_factor=2, persistent_workers=True)
    return torch.utils.data.DataLoader(**kwargs)


def build_model(args, input_shape, device):
    backbone = backbones.load(args.backbone)
    backbone.name = args.backbone
    backbone.seed = args.seed
    model = simplenet_gl.GlobalLocalSimpleNet(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=args.layers,
        device=device,
        input_shape=input_shape,
        pretrain_embed_dimension=args.pretrain_embed_dimension,
        target_embed_dimension=args.target_embed_dimension,
        patchsize=args.patchsize,
        embedding_size=256,
        meta_epochs=args.meta_epochs,
        aed_meta_epochs=0,
        gan_epochs=args.gan_epochs,
        noise_std=args.noise_std,
        global_noise_std=args.global_noise_std,
        local_noise_std=args.local_noise_std,
        mix_noise=args.mix_noise,
        dsc_layers=args.dsc_layers,
        dsc_hidden=None if args.dsc_layers == 1 else args.dsc_hidden,
        dsc_margin=args.dsc_margin,
        dsc_lr=args.dsc_lr,
        train_backbone=False,
        cos_lr=False,
        lr=1e-3,
        pre_proj=args.adapter_layers,
        proj_layer_type=0,
        tile_batch_size=args.tile_batch_size,
        exact_pixel_auroc=(args.dataset == "mvtec"),
        legacy_gaussian_sigma=4.0,
    )
    return model


def main():
    args = parse_args()
    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(args.data_path)
    device = (
        torch.device(f"cuda:{args.gpu}")
        if args.gpu >= 0 and torch.cuda.is_available()
        else torch.device("cpu")
    )
    utils.fix_seeds(
        args.seed, with_torch=True, with_cuda=(device.type == "cuda")
    )
    result_root = Path(args.results_path) / args.run_name
    result_root.mkdir(parents=True, exist_ok=True)
    with open(result_root / "args.json", "w", encoding="utf-8") as stream:
        json.dump(vars(args), stream, ensure_ascii=False, indent=2, sort_keys=True)
    rows = []

    for category in args.categories:
        LOGGER.info("Preparing %s/%s", args.dataset, category)
        train_dataset = build_dataset(args, category, "train")
        test_dataset = build_dataset(args, category, "test")
        train_loader = build_loader(
            train_dataset, args.batch_size, args.num_workers, shuffle=True
        )
        test_loader = build_loader(
            test_dataset, 1, args.num_workers, shuffle=False
        )
        model = build_model(args, train_dataset.imagesize, device)
        model_dir = result_root / category / "models"
        model.set_model_dir(str(model_dir), f"{args.dataset}_{category}")
        image_auroc, pixel_auroc, _ = model.train(train_loader, test_loader)
        rows.append(
            {
                "dataset": args.dataset,
                "category": category,
                "train_images": len(train_dataset),
                "test_images": len(test_dataset),
                "image_auroc": image_auroc,
                "pixel_auroc": pixel_auroc,
                "pixel_auroc_mode": (
                    "exact" if args.dataset == "mvtec" else "hist4096"
                ),
                "seconds_per_image": model.last_inference_stats.get(
                    "seconds_per_image", float("nan")
                ),
                "images_per_second": model.last_inference_stats.get(
                    "images_per_second", float("nan")
                ),
                "end_to_end_seconds_per_image": model.last_inference_stats.get(
                    "end_to_end_seconds_per_image", float("nan")
                ),
            }
        )

    result_file = result_root / "results.csv"
    with open(result_file, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Stored results in %s", result_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

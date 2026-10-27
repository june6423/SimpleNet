"""Post-hoc F1 evaluation for SimpleNet / SimpleNet++ VisA runs.

A copy of ``mvtec_f1_eval.py`` for VisA with the MVTec input geometry
(``datasets/visa_mvtecstyle.py``).  The F1 computation and the output keys are
unchanged.  It rebuilds the model exactly as ``main.py`` / ``main_ours.py`` do,
restores the best-I-AUROC checkpoint saved by ``SimpleNet.train``, re-runs
inference on the VisA test split, and reports image-level and pixel-level
F1-max next to AUROC.

Usage (inside the docker image, cwd = repository root):

  python3 visa_f1_eval.py eval --method simplenet --category candle \
      --results-root /results --data-root /data/visa --gpu 0

  python3 visa_f1_eval.py merge --results-root /results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics as skm

import backbones
import simplenet
import simplenet_plus
from datasets.visa_mvtecstyle import VisAMVTecStyleDataset, DatasetSplit

METHODS = ("simplenet", "simplenet_plus")
LOG_PROJECT = "visa"


def run_dir(results_root: str, method: str, category: str) -> Path:
    return Path(results_root) / LOG_PROJECT / method / category


def f1_max(labels: np.ndarray, scores: np.ndarray):
    precision, recall, thresholds = skm.precision_recall_curve(labels, scores)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    # precision_recall_curve appends a final (p=1, r=0) point with no threshold.
    best = int(np.argmax(f1[:-1]))
    return {
        "f1_max": float(f1[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "threshold": float(thresholds[best]),
    }


def build_model(method: str, device: torch.device, input_shape, meta_epochs: int, gan_epochs: int):
    module = simplenet if method == "simplenet" else simplenet_plus
    backbone = backbones.load("wideresnet50")
    backbone.name, backbone.seed = "wideresnet50", None
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
        meta_epochs=meta_epochs,
        aed_meta_epochs=1,
        gan_epochs=gan_epochs,
        noise_std=0.015,
        dsc_layers=2,
        dsc_hidden=1024,
        dsc_margin=0.5,
        dsc_lr=0.0002,
        auto_noise=0,
        train_backbone=False,
        cos_lr=False,
        pre_proj=1,
        proj_layer_type=0,
        mix_noise=1,
    )
    return model


def training_csv_auroc(path: Path):
    """Return the instance AUROC that main.py wrote for this run, if present."""
    if not path.is_file():
        return None
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    header = rows[0]
    col = header.index("instance_auroc")
    for row in rows[1:]:
        if row and row[0] != "Mean":
            return float(row[col])
    return None


def evaluate(args: argparse.Namespace) -> None:
    out_dir = run_dir(args.results_root, args.method, args.category)
    ckpt = out_dir / "models" / "0" / f"{LOG_PROJECT}_{args.category}" / "ckpt.pth"
    if not ckpt.is_file():
        sys.exit(f"Checkpoint not found: {ckpt}")

    device = torch.device(f"cuda:{args.gpu}")
    test_dataset = VisAMVTecStyleDataset(
        args.data_root,
        classname=args.category,
        resize=args.resize,
        imagesize=args.imagesize,
        split=DatasetSplit.TEST,
    )
    loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    model = build_model(args.method, device, test_dataset.imagesize, args.meta_epochs, args.gan_epochs)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.discriminator.load_state_dict(state["discriminator"])
    model.pre_projection.load_state_dict(state["pre_projection"])

    scores, labels, seg_maps, gt_masks, defect_types = [], [], [], [], []
    for data in loader:
        batch_scores, batch_masks, _ = model._predict(data["image"])
        scores.extend(float(np.asarray(s).reshape(-1)[0]) for s in batch_scores)
        seg_maps.extend(np.asarray(m, dtype=np.float32) for m in batch_masks)
        labels.extend(int(v) for v in data["is_anomaly"].tolist())
        gt_masks.extend((data["mask"][:, 0].numpy() > 0.5).astype(np.uint8))
        defect_types.extend(data["anomaly"])

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if not np.all(np.isfinite(scores)):
        sys.exit("Non-finite image scores.")

    image_auroc = float(skm.roc_auc_score(labels, scores))
    image_f1 = f1_max(labels, scores)

    flat_seg = np.stack(seg_maps).ravel()
    flat_gt = np.stack(gt_masks).ravel()
    pixel_auroc = float(skm.roc_auc_score(flat_gt, flat_seg))
    pixel_f1 = f1_max(flat_gt, flat_seg)

    per_defect = {}
    normal = labels == 0
    for defect in sorted(set(defect_types) - {"good"}):
        sel = normal | (np.asarray(defect_types) == defect)
        per_defect[defect] = {
            "n": int((np.asarray(defect_types) == defect).sum()),
            "image_auroc": float(skm.roc_auc_score(labels[sel], scores[sel])),
            "image_f1_max": f1_max(labels[sel], scores[sel])["f1_max"],
        }

    csv_auroc = training_csv_auroc(out_dir / "results.csv")
    result = {
        "method": args.method,
        "category": args.category,
        "checkpoint": str(ckpt),
        "n_test": int(len(labels)),
        "n_test_anomaly": int(labels.sum()),
        "image_auroc": image_auroc,
        "image_f1_max": image_f1["f1_max"],
        "image_f1_precision": image_f1["precision"],
        "image_f1_recall": image_f1["recall"],
        "image_f1_threshold": image_f1["threshold"],
        "pixel_auroc": pixel_auroc,
        "pixel_f1_max": pixel_f1["f1_max"],
        "training_csv_instance_auroc": csv_auroc,
        "auroc_matches_training_csv": (
            None if csv_auroc is None else bool(abs(csv_auroc - image_auroc) < 1e-3)
        ),
        "per_defect": per_defect,
        "resize": args.resize,
        "imagesize": args.imagesize,
        "meta_epochs": args.meta_epochs,
        "gan_epochs": args.gan_epochs,
        "seed": args.seed,
    }
    with (out_dir / "f1_eval.json").open("w") as stream:
        json.dump(result, stream, indent=2)
    print(
        f"{args.method:<15}{args.category:<12}I-AUROC={image_auroc:.4f} "
        f"I-F1max={image_f1['f1_max']:.4f} P-AUROC={pixel_auroc:.4f} "
        f"P-F1max={pixel_f1['f1_max']:.4f} csv_match={result['auroc_matches_training_csv']}"
    )


def merge(args: argparse.Namespace) -> None:
    root = Path(args.results_root)
    rows = []
    for method in METHODS:
        for path in sorted((root / LOG_PROJECT / method).glob("*/f1_eval.json")):
            with path.open() as stream:
                rows.append(json.load(stream))
    if not rows:
        sys.exit("No f1_eval.json found.")

    columns = ["method", "category", "n_test", "image_auroc", "image_f1_max",
               "pixel_auroc", "pixel_f1_max", "auroc_matches_training_csv"]
    out_csv = root / f"{LOG_PROJECT}_f1_summary.csv"
    with out_csv.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[c] for c in columns])
        for method in METHODS:
            sub = [r for r in rows if r["method"] == method]
            if sub:
                writer.writerow([method, f"MEAN({len(sub)})", sum(r["n_test"] for r in sub)]
                                + [np.mean([r[c] for r in sub]) for c in columns[3:7]]
                                + [all(r["auroc_matches_training_csv"] for r in sub)])

    categories = sorted({r["category"] for r in rows})
    by = {(r["method"], r["category"]): r for r in rows}
    print(f"{'category':<12}" + "".join(f"{m+'/I-F1':>18}{m+'/P-F1':>18}" for m in METHODS))
    for cat in categories:
        line = f"{cat:<12}"
        for m in METHODS:
            r = by.get((m, cat))
            line += (f"{r['image_f1_max']:>18.4f}{r['pixel_f1_max']:>18.4f}" if r else f"{'-':>18}{'-':>18}")
        print(line)
    for m in METHODS:
        sub = [r for r in rows if r["method"] == m]
        if sub:
            print(f"{m}: n={len(sub)} mean I-AUROC={np.mean([r['image_auroc'] for r in sub]):.4f} "
                  f"I-F1max={np.mean([r['image_f1_max'] for r in sub]):.4f} "
                  f"P-AUROC={np.mean([r['pixel_auroc'] for r in sub]):.4f} "
                  f"P-F1max={np.mean([r['pixel_f1_max'] for r in sub]):.4f}")
    print(f"Wrote {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("eval")
    ev.add_argument("--method", choices=METHODS, required=True)
    ev.add_argument("--category", required=True)
    ev.add_argument("--results-root", required=True)
    ev.add_argument("--data-root", required=True)
    ev.add_argument("--gpu", type=int, default=0)
    ev.add_argument("--resize", type=int, default=256)
    ev.add_argument("--imagesize", type=int, default=224)
    ev.add_argument("--meta-epochs", type=int, default=40)
    ev.add_argument("--gan-epochs", type=int, default=4)
    ev.add_argument("--batch-size", type=int, default=8)
    ev.add_argument("--seed", type=int, default=0)
    mg = sub.add_parser("merge")
    mg.add_argument("--results-root", required=True)
    args = parser.parse_args()
    if args.command == "eval":
        evaluate(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()

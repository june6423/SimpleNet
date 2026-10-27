"""F1-max evaluation of the 320 best-test checkpoints of the three methods.

The model and the test loader come from the builders of ``visa_benchmark.py``,
which ``visa_besttest_benchmark.py`` uses: the whole-image letterbox loader
(``VisALegacyDataset``) for SimpleNet and SimpleNet++, and the high-resolution
global/local loader (``VisADataset``) for "Ours".  The script loads
``checkpoint_best.pth`` of each job and does not train.

Image level: AUROC and F1-max (``f1_max`` of ``mvtec_f1_eval.py``) from the
image scores.  Pixel level: the score maps at the original mask resolution are
counted in two PIXEL_BINS-bin histograms over the common score range of the
category, with the rule of ``visa_benchmark.py`` (a first pass finds the
range, a second pass fills the histograms).  Pixel AUROC comes from
``histogram_auroc``, and pixel F1-max from the F1 at every bin edge.

``--dataset mvtec`` is reserved for the MVTec revision; it is not implemented.

Usage (inside the docker image, cwd = repository root):

  python3 visa_f1_eval_320.py eval --method simplenet --category candle \
      --checkpoint-root /workspace/visa_320_checkpoints \
      --data-root /workspace/visa-anomaly --results-root /results --gpu 0

  python3 visa_f1_eval_320.py merge --results-root /results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics as skm

import utils
import visa_benchmark as fixed_runner
from besttest_utils import load_global_local_checkpoint
from mvtec_f1_eval import f1_max

DATASETS = ("visa", "mvtec")
METHODS = fixed_runner.METHODS
INPUT_POLICY = {
    "simplenet": "whole_image_letterbox_320",
    "simplenet_plus": "whole_image_letterbox_320",
    "global_local": "global_512x768_local_512_stride384_to_320",
}


def histogram_f1_max(positive_hist, negative_hist, score_min, score_max):
    """F1-max over the bin edges of two pixel histograms (threshold: bin >= k)."""
    tp = np.cumsum(positive_hist[::-1], dtype=np.float64)
    fp = np.cumsum(negative_hist[::-1], dtype=np.float64)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = tp / max(float(positive_hist.sum()), 1.0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    best = int(np.argmax(f1))
    bin_index = len(positive_hist) - 1 - best
    scale = (len(positive_hist) - 1) / max(score_max - score_min, 1e-12)
    return {
        "f1_max": float(f1[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "threshold": float(score_min + bin_index / scale),
    }


class TileCounter:
    """Pass the loader items through and record the tile count and global view size."""

    def __init__(self, loader):
        self.loader = loader
        self.tiles = []
        self.global_sizes = set()

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for item in self.loader:
            self.tiles.append(int(item["local_tiles"].shape[1]))
            self.global_sizes.add(tuple(int(v) for v in item["global_image"].shape[-2:]))
            yield item


def predictions(method, model, loader, device):
    """Yield (image score, label, anomaly map at the original size, mask) per test image."""
    if method == "global_local":
        for prediction in model._iter_predictions(loader):
            yield (
                float(prediction["score"]),
                int(prediction["label"]),
                prediction["map"],
                prediction["mask"],
            )
        return
    with torch.no_grad():
        for data in loader:
            batch_scores, batch_maps, _ = model._predict(data["image"])
            anomaly_map = fixed_runner.restore_letterbox_map(
                batch_maps[0], data["content_box"], data["original_size"]
            )
            yield (
                float(np.asarray(batch_scores[0]).reshape(-1)[0]),
                int(data["is_anomaly"].flatten()[0]),
                anomaly_map,
                data["mask"][0, 0].numpy(),
            )


def reference_image_auroc(path: Path, method: str, category: str):
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["method"] == method and row["category"] == category:
                return float(row["image_auroc"])
    return None


def evaluate(args: argparse.Namespace) -> None:
    if args.dataset != "visa":
        sys.exit(f"--dataset {args.dataset} is not implemented yet.")
    checkpoint = Path(args.checkpoint_root) / args.method / args.category / "checkpoint_best.pth"
    if not checkpoint.is_file():
        sys.exit(f"Checkpoint not found: {checkpoint}")
    reference_csv = Path(args.reference_csv or Path(args.checkpoint_root) / "results.csv")
    reference = reference_image_auroc(reference_csv, args.method, args.category)
    if reference is None:
        sys.exit(f"No {args.method}/{args.category} row in {reference_csv}")
    out_dir = Path(args.results_root) / args.dataset / args.method / args.category
    out_dir.mkdir(parents=True, exist_ok=True)

    # The runner's argument names, so that its builders can be reused.
    runner_args = argparse.Namespace(
        method=args.method,
        category=args.category,
        data_path=args.data_root,
        seed=args.seed,
        imagesize=args.imagesize,
        meta_epochs=args.meta_epochs,
        gan_epochs=args.gan_epochs,
        smoke=False,
    )
    device = torch.device(f"cuda:{args.gpu}")
    utils.fix_seeds(args.seed, with_torch=True, with_cuda=True)
    train_dataset, test_dataset, _augmentation = fixed_runner.build_datasets(runner_args)
    loader = fixed_runner.build_loader(test_dataset, 1, args.num_workers, shuffle=False)
    if args.method == "global_local":
        model = fixed_runner.build_global_local_model(runner_args, train_dataset.imagesize, device)
        load_global_local_checkpoint(model, checkpoint, load_optimizer=False)
        loader = TileCounter(loader)
    else:
        model = fixed_runner.build_baseline_model(runner_args, train_dataset.imagesize, device)
        fixed_runner.load_baseline_checkpoint(model, checkpoint, args.method, load_optimizer=False)

    # Pass 1: image scores and the score range of the maps.
    scores, labels = [], []
    score_min, score_max = math.inf, -math.inf
    for score, label, anomaly_map, _mask in predictions(args.method, model, loader, device):
        if not np.isfinite(anomaly_map).all():
            sys.exit("Non-finite anomaly map.")
        scores.append(score)
        labels.append(label)
        score_min = min(score_min, float(anomaly_map.min()))
        score_max = max(score_max, float(anomaly_map.max()))
    tile_counts = list(loader.tiles) if args.method == "global_local" else None

    # Pass 2: pixel histograms over the common range.
    positive_hist = np.zeros(args.pixel_bins, dtype=np.int64)
    negative_hist = np.zeros(args.pixel_bins, dtype=np.int64)
    pass2_scores = []
    for score, _label, anomaly_map, mask in predictions(args.method, model, loader, device):
        pass2_scores.append(score)
        fixed_runner.update_histograms(
            anomaly_map, mask, score_min, score_max, positive_hist, negative_hist
        )

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if not np.all(np.isfinite(scores)):
        sys.exit("Non-finite image scores.")
    image_auroc = float(skm.roc_auc_score(labels, scores))
    image_f1 = f1_max(labels, scores)
    pixel_f1 = histogram_f1_max(positive_hist, negative_hist, score_min, score_max)

    result = {
        "dataset": args.dataset,
        "method": args.method,
        "category": args.category,
        "checkpoint": str(checkpoint),
        "loader": "VisADataset" if args.method == "global_local" else "VisALegacyDataset",
        "input_policy": INPUT_POLICY[args.method],
        "imagesize": args.imagesize,
        "n_test": int(len(labels)),
        "n_test_anomaly": int(labels.sum()),
        "image_auroc": image_auroc,
        "image_f1_max": image_f1["f1_max"],
        "image_f1_threshold": image_f1["threshold"],
        "image_f1_precision": image_f1["precision"],
        "image_f1_recall": image_f1["recall"],
        "pixel_auroc": fixed_runner.histogram_auroc(positive_hist, negative_hist),
        "pixel_f1_max": pixel_f1["f1_max"],
        "pixel_f1_threshold": pixel_f1["threshold"],
        "pixel_f1_precision": pixel_f1["precision"],
        "pixel_f1_recall": pixel_f1["recall"],
        "pixel_f1_method": f"histogram_{args.pixel_bins}_original_resolution",
        "pixel_bins": args.pixel_bins,
        "pixel_score_min": score_min,
        "pixel_score_max": score_max,
        "pixel_positive_count": int(positive_hist.sum()),
        "pixel_negative_count": int(negative_hist.sum()),
        "pass2_max_abs_image_score_diff": float(
            np.max(np.abs(np.asarray(pass2_scores) - scores))
        ),
        "reference_csv": str(reference_csv),
        "reference_image_auroc": reference,
        "auroc_matches_reference": bool(abs(reference - image_auroc) <= 1e-3),
        "seed": args.seed,
    }
    if args.method == "global_local":
        result.update(
            {
                "global_size": [list(size) for size in sorted(loader.global_sizes)],
                "tile_size": [test_dataset.tile_height, test_dataset.tile_width],
                "tile_stride": [test_dataset.tile_stride_y, test_dataset.tile_stride_x],
                "tiles_per_image": {"min": min(tile_counts), "max": max(tile_counts)},
            }
        )
    with (out_dir / "f1_eval.json").open("w") as stream:
        json.dump(result, stream, indent=2)
    print(
        f"{args.method:<15}{args.category:<12}I-AUROC={image_auroc:.4f} "
        f"I-F1max={image_f1['f1_max']:.4f} P-AUROC={result['pixel_auroc']:.4f} "
        f"P-F1max={pixel_f1['f1_max']:.4f} ref_match={result['auroc_matches_reference']}"
    )


def merge(args: argparse.Namespace) -> None:
    root = Path(args.results_root)
    rows = []
    for method in METHODS:
        for path in sorted((root / args.dataset / method).glob("*/f1_eval.json")):
            with path.open() as stream:
                rows.append(json.load(stream))
    if not rows:
        sys.exit("No f1_eval.json found.")

    columns = ["method", "category", "n_test", "image_auroc", "image_f1_max",
               "pixel_auroc", "pixel_f1_max", "reference_image_auroc", "auroc_matches_reference"]
    out_csv = root / f"{args.dataset}_f1_summary.csv"
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
                                + ["", all(r["auroc_matches_reference"] for r in sub)])
    for method in METHODS:
        sub = [r for r in rows if r["method"] == method]
        if sub:
            print(f"{method}: n={len(sub)} mean I-AUROC={np.mean([r['image_auroc'] for r in sub]):.4f} "
                  f"I-F1max={np.mean([r['image_f1_max'] for r in sub]):.4f} "
                  f"P-AUROC={np.mean([r['pixel_auroc'] for r in sub]):.4f} "
                  f"P-F1max={np.mean([r['pixel_f1_max'] for r in sub]):.4f}")
    print(f"Wrote {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("eval")
    ev.add_argument("--dataset", choices=DATASETS, default="visa")
    ev.add_argument("--method", choices=METHODS, required=True)
    ev.add_argument("--category", required=True)
    ev.add_argument("--checkpoint-root", required=True)
    ev.add_argument("--reference-csv", help="Default: <checkpoint-root>/results.csv.")
    ev.add_argument("--data-root", required=True)
    ev.add_argument("--results-root", required=True)
    ev.add_argument("--gpu", type=int, default=0)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--imagesize", type=int, default=320)
    ev.add_argument("--meta-epochs", type=int, default=40)
    ev.add_argument("--gan-epochs", type=int, default=4)
    ev.add_argument("--pixel-bins", type=int, default=65536)
    ev.add_argument("--num-workers", type=int, default=4)
    mg = sub.add_parser("merge")
    mg.add_argument("--dataset", choices=DATASETS, default="visa")
    mg.add_argument("--results-root", required=True)
    args = parser.parse_args()
    if args.command == "eval":
        evaluate(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()

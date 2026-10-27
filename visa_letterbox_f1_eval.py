"""Post-hoc F1 evaluation for SimpleNet / SimpleNet++ VisA runs with the
whole-image letterbox input (``datasets/visa_legacy.py``).

The image-level part is the same as in ``visa_f1_eval.py``.  The masks keep
the original image resolution, and the images have different sizes, so the
pixel-level part has two modes:

  original-hist    Restore each anomaly map from the letterbox content box to
                   the original image size (as ``visa_benchmark.py`` does) and
                   accumulate PIXEL_BINS-bin histograms against the original
                   mask.  Pixel AUROC and F1-max come from the histograms.
  letterbox-exact  Resize each original mask (nearest) to the letterbox content
                   box, crop the map to the same box, and compute exact pixel
                   AUROC and F1-max over all content pixels.

``visa_f1_eval.py merge`` writes ``visa_f1_summary.csv`` for these runs too.

Usage (inside the docker image, cwd = repository root):

  python3 visa_letterbox_f1_eval.py --method simplenet --category candle \
      --results-root /results --data-root /data/visa --gpu 0 \
      --imagesize 320 --pixel-mode original-hist
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics as skm

from datasets.visa_legacy import DatasetSplit, VisALegacyDataset
from visa_benchmark import histogram_auroc, update_histograms
from visa_f1_eval import METHODS, build_model, f1_max, run_dir, training_csv_auroc

# The dataset name under which main.py / main_ours.py register the loader.
DATASET_NAME = "visa_letterbox"
PIXEL_MODES = ("original-hist", "letterbox-exact")


def histogram_f1_max(positive_hist, negative_hist, score_min, score_max):
    """F1-max over the bin thresholds of two pixel histograms."""
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


def pixel_original_hist(dataset, crops, sizes, bins):
    score_min = min(float(crop.min()) for crop in crops)
    score_max = max(float(crop.max()) for crop in crops)
    positive_hist = np.zeros(bins, dtype=np.int64)
    negative_hist = np.zeros(bins, dtype=np.int64)
    for record, crop, (height, width) in zip(dataset.records, crops, sizes):
        restored = F.interpolate(
            torch.from_numpy(crop)[None, None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        mask = dataset._mask(record, height, width)[0].numpy()
        update_histograms(restored, mask, score_min, score_max, positive_hist, negative_hist)
    pixel_f1 = histogram_f1_max(positive_hist, negative_hist, score_min, score_max)
    return histogram_auroc(positive_hist, negative_hist), pixel_f1


def pixel_letterbox_exact(dataset, crops):
    flat_maps, flat_masks = [], []
    for record, crop in zip(dataset.records, crops):
        height, width = crop.shape
        flat_maps.append(crop.ravel())
        flat_masks.append(dataset._mask(record, height, width)[0].numpy().ravel())
    flat_seg = np.concatenate(flat_maps)
    flat_gt = np.concatenate(flat_masks)
    return float(skm.roc_auc_score(flat_gt, flat_seg)), f1_max(flat_gt, flat_seg)


def evaluate(args: argparse.Namespace) -> None:
    out_dir = run_dir(args.results_root, args.method, args.category)
    ckpt = out_dir / "models" / "0" / f"{DATASET_NAME}_{args.category}" / "ckpt.pth"
    if not ckpt.is_file():
        sys.exit(f"Checkpoint not found: {ckpt}")

    device = torch.device(f"cuda:{args.gpu}")
    test_dataset = VisALegacyDataset(
        args.data_root,
        classname=args.category,
        imagesize=args.imagesize,
        split=DatasetSplit.TEST,
    )
    loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=2
    )

    model = build_model(args.method, device, test_dataset.imagesize, args.meta_epochs, args.gan_epochs)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.discriminator.load_state_dict(state["discriminator"])
    model.pre_projection.load_state_dict(state["pre_projection"])

    # Keep only the content box of each map at the model resolution.
    scores, labels, defect_types, crops, sizes = [], [], [], [], []
    for data in loader:
        batch_scores, batch_masks, _ = model._predict(data["image"])
        x1, y1, x2, y2 = [int(v) for v in data["content_box"].flatten().tolist()]
        height, width = [int(v) for v in data["original_size"].flatten().tolist()]
        scores.append(float(np.asarray(batch_scores[0]).reshape(-1)[0]))
        crops.append(np.asarray(batch_masks[0], dtype=np.float32)[y1:y2, x1:x2].copy())
        sizes.append((height, width))
        labels.append(int(data["is_anomaly"].flatten()[0]))
        defect_types.append(data["anomaly"][0])

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if not np.all(np.isfinite(scores)):
        sys.exit("Non-finite image scores.")
    image_auroc = float(skm.roc_auc_score(labels, scores))
    image_f1 = f1_max(labels, scores)

    if args.pixel_mode == "original-hist":
        pixel_auroc, pixel_f1 = pixel_original_hist(test_dataset, crops, sizes, args.pixel_bins)
    else:
        pixel_auroc, pixel_f1 = pixel_letterbox_exact(test_dataset, crops)

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
        "pixel_f1_threshold": pixel_f1["threshold"],
        "pixel_mode": args.pixel_mode,
        "pixel_bins": args.pixel_bins if args.pixel_mode == "original-hist" else None,
        "training_csv_instance_auroc": csv_auroc,
        "auroc_matches_training_csv": (
            None if csv_auroc is None else bool(abs(csv_auroc - image_auroc) < 1e-3)
        ),
        "per_defect": per_defect,
        "input_policy": f"whole_image_letterbox_{args.imagesize}",
        "resize": None,
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
        f"P-F1max={pixel_f1['f1_max']:.4f} ({args.pixel_mode}) "
        f"csv_match={result['auroc_matches_training_csv']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--imagesize", type=int, required=True)
    parser.add_argument("--pixel-mode", choices=PIXEL_MODES, required=True)
    parser.add_argument("--pixel-bins", type=int, default=4096)
    parser.add_argument("--meta-epochs", type=int, default=40)
    parser.add_argument("--gan-epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()

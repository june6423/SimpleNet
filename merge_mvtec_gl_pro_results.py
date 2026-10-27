"""Merge per-category MVTec global/local PRO-AUROC results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

from mvtec_gl_besttest_benchmark import CATEGORIES


TEXTURES = ("carpet", "grid", "leather", "tile", "wood")
OBJECTS = tuple(category for category in CATEGORIES if category not in TEXTURES)


def atomic_json_dump(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    root = Path(args.results_root)
    rows, missing = [], []
    for category in CATEGORIES:
        path = root / category / "job_result.json"
        if not path.exists():
            missing.append(category)
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("category") != category:
            raise ValueError(f"Category mismatch in {path}")
        for key in ("image_auroc", "pixel_auroc", "pro_auroc"):
            value = float(row[key])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Invalid {key}={value} in {path}")
        if max(
            float(row["checkpoint_image_auroc_delta"]),
            float(row["checkpoint_pixel_auroc_delta"]),
        ) > 1e-9:
            raise ValueError(f"Checkpoint re-evaluation mismatch in {path}")
        rows.append(row)
    if missing and not args.allow_partial:
        raise RuntimeError(f"Missing categories: {missing}")

    fields = [
        "category",
        "image_auroc",
        "pixel_auroc",
        "pro_auroc",
        "source_best_meta_epoch_zero_based",
        "checkpoint_image_auroc_delta",
        "checkpoint_pixel_auroc_delta",
        "test_images",
        "wall_seconds",
        "source_checkpoint",
        "source_checkpoint_sha256",
        "selection_policy",
    ]
    csv_tmp = root / "results.csv.tmp"
    with csv_tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(csv_tmp, root / "results.csv")

    by_category = {row["category"]: row for row in rows}
    mean = lambda categories, key: sum(
        float(by_category[c][key]) for c in categories if c in by_category
    ) / sum(c in by_category for c in categories)
    summary = {
        "dataset": "mvtec",
        "method": "global_local",
        "complete": not missing and len(rows) == len(CATEGORIES),
        "completed_categories": len(rows),
        "expected_categories": len(CATEGORIES),
        "missing": missing,
        "macro_image_auroc": mean(CATEGORIES, "image_auroc") if rows else None,
        "macro_pixel_auroc": mean(CATEGORIES, "pixel_auroc") if rows else None,
        "macro_pro_auroc": mean(CATEGORIES, "pro_auroc") if rows else None,
        "texture_pro_auroc": mean(TEXTURES, "pro_auroc") if any(c in by_category for c in TEXTURES) else None,
        "object_pro_auroc": mean(OBJECTS, "pro_auroc") if any(c in by_category for c in OBJECTS) else None,
        "checkpoint_reval_max_abs_delta": max(
            [
                max(
                    float(row["checkpoint_image_auroc_delta"]),
                    float(row["checkpoint_pixel_auroc_delta"]),
                )
                for row in rows
            ],
            default=None,
        ),
        "protocol": {
            "implementation": "metrics.compute_pro",
            "num_thresholds": 200,
            "dilation_kernel": "5x5 rectangle",
            "fpr_limit": 0.3,
            "checkpoint_policy": "reuse I-AUROC then P-AUROC selected checkpoint",
        },
    }
    atomic_json_dump(summary, root / "summary.json")
    print(f"Merged {len(rows)}/{len(CATEGORIES)} PRO results into {root / 'results.csv'}")


if __name__ == "__main__":
    main()

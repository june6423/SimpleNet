"""Validate and merge the new MVTec/VisA best-test experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from mvtec_gl_besttest_benchmark import CATEGORIES as MVTEC_CATEGORIES
from visa_besttest_benchmark import CATEGORIES as VISA_CATEGORIES
from visa_besttest_benchmark import METHODS as VISA_METHODS


CONFIG = {
    "mvtec": (("global_local",), MVTEC_CATEGORIES),
    "visa": (VISA_METHODS, VISA_CATEGORIES),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=CONFIG)
    parser.add_argument("results_root")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def validate_history(root: Path, row) -> None:
    history_path = root / row["method"] / row["category"] / "epoch_history.json"
    with history_path.open(encoding="utf-8") as stream:
        history = json.load(stream)
    if len(history) != int(row["meta_epochs"]):
        raise ValueError(
            f"Incomplete history {row['method']}/{row['category']}: "
            f"{len(history)} != {row['meta_epochs']}"
        )
    expected_index = max(
        range(len(history)),
        key=lambda index: (
            float(history[index]["image_auroc"]),
            float(history[index]["pixel_auroc"]),
        ),
    )
    if expected_index != int(row["best_meta_epoch_zero_based"]):
        raise ValueError(
            f"Best epoch mismatch {row['method']}/{row['category']}: "
            f"{expected_index} != {row['best_meta_epoch_zero_based']}"
        )
    selected = history[expected_index]
    delta = max(
        abs(float(selected["image_auroc"]) - float(row["image_auroc"])),
        abs(float(selected["pixel_auroc"]) - float(row["pixel_auroc"])),
        abs(float(row["verified_image_auroc"]) - float(row["image_auroc"])),
        abs(float(row["verified_pixel_auroc"]) - float(row["pixel_auroc"])),
    )
    if delta > 1e-9:
        raise ValueError(
            f"Metric/checkpoint mismatch {row['method']}/{row['category']}: {delta}"
        )


def main():
    args = parse_args()
    methods, categories = CONFIG[args.dataset]
    root = Path(args.results_root)
    rows = []
    for path in sorted(root.glob("*/*/job_result.json")):
        with path.open(encoding="utf-8") as stream:
            row = json.load(stream)
        if row.get("dataset") != args.dataset:
            raise ValueError(f"Unexpected dataset in {path}: {row.get('dataset')}")
        rows.append(row)
    if not rows:
        raise ValueError("No job_result.json files found")

    keys = [(row["method"], row["category"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate method/category results detected")
    expected = {(method, category) for method in methods for category in categories}
    missing = sorted(expected - set(keys))
    extra = sorted(set(keys) - expected)
    if extra:
        raise ValueError(f"Unexpected result keys: {extra}")
    if missing and not args.allow_partial:
        raise ValueError(f"Missing {len(missing)} results: {missing}")

    for row in rows:
        for field in (
            "image_auroc",
            "pixel_auroc",
            "verified_image_auroc",
            "verified_pixel_auroc",
            "seconds_per_image",
            "end_to_end_seconds_per_image",
        ):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"Non-finite {field} in {row}")
        validate_history(root, row)

    rows.sort(key=lambda row: (methods.index(row["method"]), row["category"]))
    fieldnames = list(rows[0])
    for row in rows:
        if set(row) != set(fieldnames):
            raise ValueError(f"Schema mismatch for {row['method']}/{row['category']}")
    with (root / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    summary = {
        "complete": not missing,
        "completed_jobs": len(rows),
        "expected_jobs": len(expected),
        "missing": missing,
        "dataset": args.dataset,
        "selection_policy": "best_test_image_auroc_then_pixel_auroc",
        "methods": {},
    }
    for method in methods:
        method_rows = grouped.get(method, [])
        if not method_rows:
            continue
        count = len(method_rows)
        test_images = sum(int(row["test_images"]) for row in method_rows)
        best_epochs = [int(row["best_meta_epoch_zero_based"]) for row in method_rows]
        summary["methods"][method] = {
            "categories": count,
            "macro_image_auroc": sum(float(row["image_auroc"]) for row in method_rows)
            / count,
            "macro_pixel_auroc": sum(float(row["pixel_auroc"]) for row in method_rows)
            / count,
            "weighted_seconds_per_image": sum(
                int(row["test_images"]) * float(row["seconds_per_image"])
                for row in method_rows
            )
            / test_images,
            "weighted_end_to_end_seconds_per_image": sum(
                int(row["test_images"])
                * float(row["end_to_end_seconds_per_image"])
                for row in method_rows
            )
            / test_images,
            "test_images": test_images,
            "best_epoch_zero_based_min": min(best_epochs),
            "best_epoch_zero_based_max": max(best_epochs),
            "best_epoch_zero_based_mean": sum(best_epochs) / len(best_epochs),
            "checkpoint_reval_max_abs_delta": max(
                float(row["checkpoint_reval_max_abs_delta"])
                for row in method_rows
            ),
        }
    with (root / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"Merged {len(rows)}/{len(expected)} results into {root / 'results.csv'}")


if __name__ == "__main__":
    main()


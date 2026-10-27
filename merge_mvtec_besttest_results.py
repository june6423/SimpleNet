"""Merge per-category MVTec best-test results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from mvtec_besttest_benchmark import CATEGORIES, METHODS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.results_root)
    rows = []
    for path in sorted(root.glob("*/*/job_result.json")):
        with path.open(encoding="utf-8") as stream:
            rows.append(json.load(stream))
    if not rows:
        raise ValueError("No job_result.json files found.")

    keys = [(row["method"], row["category"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate method/category results detected.")
    expected = {(method, category) for method in METHODS for category in CATEGORIES}
    missing = sorted(expected - set(keys))
    extra = sorted(set(keys) - expected)
    if extra:
        raise ValueError(f"Unexpected result keys: {extra}")
    if missing and not args.allow_partial:
        raise ValueError(f"Missing {len(missing)} results: {missing}")

    rows.sort(key=lambda row: (METHODS.index(row["method"]), row["category"]))
    fieldnames = list(rows[0])
    for row in rows:
        if set(row) != set(fieldnames):
            raise ValueError(f"Schema mismatch for {row['method']}/{row['category']}")
        for field in (
            "image_auroc",
            "pixel_auroc",
            "verified_image_auroc",
            "verified_pixel_auroc",
            "seconds_per_image",
        ):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"Non-finite {field} in {row}")

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
        "selection_policy": "best_test_image_auroc_then_pixel_auroc",
        "methods": {},
    }
    for method in METHODS:
        method_rows = grouped.get(method, [])
        if not method_rows:
            continue
        total_test_images = sum(int(row["test_images"]) for row in method_rows)
        summary["methods"][method] = {
            "categories": len(method_rows),
            "macro_image_auroc": sum(float(row["image_auroc"]) for row in method_rows)
            / len(method_rows),
            "macro_pixel_auroc": sum(float(row["pixel_auroc"]) for row in method_rows)
            / len(method_rows),
            "weighted_seconds_per_image": sum(
                int(row["test_images"]) * float(row["seconds_per_image"])
                for row in method_rows
            )
            / total_test_images,
            "test_images": total_test_images,
        }
    with (root / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"Merged {len(rows)}/{len(expected)} results into {root / 'results.csv'}")


if __name__ == "__main__":
    main()

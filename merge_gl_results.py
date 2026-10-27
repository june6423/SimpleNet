"""Merge independent global-local benchmark result files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    return parser.parse_args()


def finite_mean(rows, key):
    values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
    return sum(values) / len(values) if values else math.nan


def weighted_mean(rows, key, weight_key):
    pairs = [
        (float(row[key]), int(row[weight_key]))
        for row in rows
        if math.isfinite(float(row[key])) and int(row[weight_key]) > 0
    ]
    total_weight = sum(weight for _, weight in pairs)
    if total_weight == 0:
        return math.nan
    return sum(value * weight for value, weight in pairs) / total_weight


def main():
    args = parse_args()
    rows = []
    fieldnames = None
    for input_path in args.input:
        with open(input_path, newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise ValueError(f"CSV schema mismatch: {input_path}")
            rows.extend(reader)

    rows.sort(key=lambda row: row["category"])
    if len({row["category"] for row in rows}) != len(rows):
        raise ValueError("Duplicate categories found while merging results.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "categories": len(rows),
        "train_images": sum(int(row["train_images"]) for row in rows),
        "test_images": sum(int(row["test_images"]) for row in rows),
        "macro_image_auroc": finite_mean(rows, "image_auroc"),
        "macro_pixel_auroc": finite_mean(rows, "pixel_auroc"),
        "weighted_seconds_per_image": weighted_mean(
            rows, "seconds_per_image", "test_images"
        ),
        "weighted_end_to_end_seconds_per_image": weighted_mean(
            rows, "end_to_end_seconds_per_image", "test_images"
        ),
    }
    with open(args.summary, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()

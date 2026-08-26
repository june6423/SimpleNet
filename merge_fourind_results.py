"""Validate and merge all 4ind method/product best-validation results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

from fourind_manifest import EXPECTED_FIXED_SPLITS, EXPECTED_TRAIN_POOLS


METHODS = ("simplenet", "simplenet_plus", "global_local")
CATEGORIES = ("KQG27542", "KQG27824")
SELECTION_POLICY = "best_val_image_auroc_keep_earliest_on_tie"


def atomic_json_dump(data: Dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary, path)


def atomic_csv_dump(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def validate_auc(name: str, value: object) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"Invalid {name}={value}")
    return number


def expected_best_epoch(history: List[Dict[str, object]]) -> Tuple[int, float]:
    if not history:
        raise ValueError("Empty epoch history.")
    best_epoch = 0
    best_value = validate_auc("history[0].val_image_auroc", history[0]["val_image_auroc"])
    for index, row in enumerate(history[1:], start=1):
        value = validate_auc(f"history[{index}].val_image_auroc", row["val_image_auroc"])
        if value > best_value + 1e-12:
            best_epoch = index
            best_value = value
    return best_epoch, best_value


def validate_job(
    method: str, category: str, result: Dict[str, object], history
) -> None:
    if result.get("dataset") != "4ind" or result.get("method") != method:
        raise ValueError(f"{method}/{category}: dataset/method mismatch.")
    if result.get("category") != category:
        raise ValueError(f"{method}/{category}: category mismatch.")
    if result.get("selection_policy") != SELECTION_POLICY:
        raise ValueError(f"{method}/{category}: selection policy mismatch.")
    if int(result.get("test_evaluations_during_training", -1)) != 0:
        raise ValueError(f"{method}/{category}: test was evaluated during training.")
    if bool(result.get("pixel_metrics_available", True)):
        raise ValueError(f"{method}/{category}: pixel metrics must be marked unavailable.")
    is_smoke = bool(result.get("smoke", int(result.get("train_images", 0)) < 100))
    if not is_smoke:
        expected_counts = {
            "train_images": int(math.floor(EXPECTED_TRAIN_POOLS[category] * 0.1)),
            "val_images": EXPECTED_FIXED_SPLITS[(category, "val")],
            "test_images": EXPECTED_FIXED_SPLITS[(category, "test")],
        }
        for key, expected in expected_counts.items():
            if int(result[key]) != expected:
                raise ValueError(
                    f"{method}/{category}: {key} expected {expected}, "
                    f"found {result[key]}."
                )

    meta_epochs = int(result["meta_epochs"])
    if len(history) != meta_epochs:
        raise ValueError(f"{category}: history {len(history)} != {meta_epochs} epochs.")
    for index, row in enumerate(history):
        if int(row["meta_epoch_zero_based"]) != index:
            raise ValueError(f"{category}: non-contiguous epoch history at {index}.")

    expected_epoch, expected_auc = expected_best_epoch(history)
    if int(result["best_meta_epoch_zero_based"]) != expected_epoch:
        raise ValueError(f"{category}: best epoch does not follow best-val policy.")
    best_val = validate_auc(f"{category}.best_val", result["best_val_image_auroc"])
    if abs(best_val - expected_auc) > 1e-12:
        raise ValueError(f"{category}: stored best validation AUROC mismatch.")
    validate_auc(f"{category}.verified_val", result["verified_val_image_auroc"])
    validate_auc(f"{category}.test", result["test_image_auroc"])
    if float(result["checkpoint_val_reval_abs_delta"]) > 1e-9:
        raise ValueError(f"{category}: checkpoint validation re-evaluation mismatch.")

    for split_key in ("val_per_defect", "test_per_defect"):
        for defect, metrics in dict(result.get(split_key, {})).items():
            validate_auc(f"{category}.{split_key}.{defect}", metrics["image_auroc"])
            if int(metrics["anomaly_images"]) < 1:
                raise ValueError(f"{category}.{split_key}.{defect}: empty defect class.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.results_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    missing: List[str] = []
    manifest_hashes = set()

    for method in METHODS:
        for category in CATEGORIES:
            job_dir = root / method / category
            result_path = job_dir / "job_result.json"
            history_path = job_dir / "epoch_history.json"
            if not result_path.is_file() or not history_path.is_file():
                missing.append(f"{method}/{category}")
                continue
            with result_path.open(encoding="utf-8") as stream:
                result = json.load(stream)
            with history_path.open(encoding="utf-8") as stream:
                history = json.load(stream)
            validate_job(method, category, result, history)
            manifest_hashes.add(result["manifest_sha256"])
            rows.append(
                {
                    "dataset": "4ind",
                    "method": method,
                    "category": category,
                    "smoke": bool(
                        result.get("smoke", int(result.get("train_images", 0)) < 100)
                    ),
                    "train_images": result["train_images"],
                    "val_images": result["val_images"],
                    "test_images": result["test_images"],
                    "best_meta_epoch_zero_based": result[
                        "best_meta_epoch_zero_based"
                    ],
                    "best_val_image_auroc": result["best_val_image_auroc"],
                    "test_image_auroc": result["test_image_auroc"],
                    "test_seconds_per_image": result["test_seconds_per_image"],
                    "test_end_to_end_seconds_per_image": result[
                        "test_end_to_end_seconds_per_image"
                    ],
                    "test_normal_images": result["test_normal_images"],
                    "test_anomaly_images": result["test_anomaly_images"],
                    "test_per_defect_json": json.dumps(
                        result["test_per_defect"],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "selection_policy": result["selection_policy"],
                    "manifest_sha256": result["manifest_sha256"],
                }
            )

    if missing and not args.allow_partial:
        raise FileNotFoundError(f"Missing completed jobs: {', '.join(missing)}")
    if len(manifest_hashes) > 1:
        raise ValueError("Per-product jobs used different manifests.")

    rows.sort(key=lambda row: (str(row["method"]), str(row["category"])))
    atomic_csv_dump(rows, root / "results.csv")
    summary: Dict[str, object] = {
        "dataset": "4ind",
        "methods": list(METHODS),
        "expected_jobs": len(METHODS) * len(CATEGORIES),
        "completed_jobs": len(rows),
        "missing_jobs": missing,
        "complete": not missing,
        "selection_policy": SELECTION_POLICY,
        "test_evaluations_during_training": 0,
        "pixel_metrics_available": False,
        "manifest_sha256": next(iter(manifest_hashes), None),
        "jobs": rows,
    }
    if rows:
        method_summaries = {}
        for method in METHODS:
            method_rows = [row for row in rows if row["method"] == method]
            if not method_rows:
                continue
            method_summaries[method] = {
                "completed_products": len(method_rows),
                "macro_best_val_image_auroc": mean(
                    float(row["best_val_image_auroc"]) for row in method_rows
                ),
                "macro_test_image_auroc": mean(
                    float(row["test_image_auroc"]) for row in method_rows
                ),
                "macro_test_seconds_per_image": mean(
                    float(row["test_seconds_per_image"]) for row in method_rows
                ),
                "macro_test_end_to_end_seconds_per_image": mean(
                    float(row["test_end_to_end_seconds_per_image"])
                    for row in method_rows
                ),
            }
        summary["method_summaries"] = method_summaries
    atomic_json_dump(summary, root / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

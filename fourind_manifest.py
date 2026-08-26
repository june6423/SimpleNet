"""Build a portable, leakage-aware manifest for the 4ind dataset.

The generated CSV stores image paths relative to ``--data-root``.  The same
manifest can therefore be copied to a server and paired with a different
absolute dataset root at runtime.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PRODUCT_PATTERN = re.compile(r"(KQG\d+)", re.IGNORECASE)
EXPECTED_PRODUCTS = ("KQG27542", "KQG27824")
DEFAULT_EXCLUDED_BLOCK = "20241222_20_결함_2"
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}
EXPECTED_SCANNED_AFTER_DEFAULT_EXCLUSION = 146_840
EXPECTED_TRAIN_POOLS = {"KQG27542": 19_302, "KQG27824": 24_005}
EXPECTED_FIXED_SPLITS = {
    ("KQG27542", "val"): 1_384,
    ("KQG27542", "test"): 6_463,
    ("KQG27824", "val"): 529,
    ("KQG27824", "test"): 3_358,
}


@dataclass(frozen=True)
class Candidate:
    category: str
    split: str
    label: str
    relative_path: str
    source_block: str
    capture_date: str


def _compact_name(value: str) -> str:
    return re.sub(r"[\s._-]+", "", value).lower()


def canonical_label(raw_label: str) -> Optional[str]:
    """Map the observed Korean folder spellings to stable manifest labels."""
    compact = _compact_name(raw_label)
    if compact in {"정상", "good", "normal", "ok", "0"}:
        return "good"
    if compact in {"라벨링불가", "판단불가", "멈춤"}:
        return None

    aliases = {
        "1흑점": "black_spot",
        "흑점": "black_spot",
        "2덴트": "dent",
        "덴트": "dent",
        "3스크레치": "scratch",
        "3스크래치": "scratch",
        "스크레치": "scratch",
        "스크래치": "scratch",
        "4얼룩": "stain",
        "얼룩": "stain",
        "5펑쳐": "puncture",
        "펑쳐": "puncture",
        "7라인": "line",
        "라인": "line",
    }
    return aliases.get(compact, "__unknown__")


def canonical_product(path_parts: Sequence[str]) -> Optional[str]:
    for part in path_parts:
        match = PRODUCT_PATTERN.search(part)
        if match:
            return match.group(1).upper()
    return None


def split_for_date(capture_date: str) -> Optional[str]:
    if "20241221" <= capture_date <= "20241223":
        return "train"
    if "20241224" <= capture_date <= "20241225":
        return "val"
    if "20241226" <= capture_date <= "20241227":
        return "test"
    return None


def stable_rank(relative_path: str, seed: int) -> bytes:
    payload = f"{seed}\0{relative_path}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def select_across_time(
    rows: Sequence[Candidate], quota: int, seed: int
) -> List[Candidate]:
    """Choose one deterministic item from each contiguous temporal bin."""
    if quota <= 0:
        return []
    ordered = sorted(rows, key=lambda row: (row.capture_date, row.relative_path))
    if quota >= len(ordered):
        return ordered
    selected = []
    for index in range(quota):
        start = int(math.floor(index * len(ordered) / quota))
        end = int(math.floor((index + 1) * len(ordered) / quota))
        temporal_bin = ordered[start:end]
        selected.append(
            min(
                temporal_bin,
                key=lambda row: stable_rank(row.relative_path, seed),
            )
        )
    return selected


def _allocate_stratified_quotas(
    group_sizes: Dict[str, int], fraction: float
) -> Dict[str, int]:
    """Allocate an exact product-level fraction using largest remainders."""
    target = int(math.floor(sum(group_sizes.values()) * fraction))
    quotas = {
        group: int(math.floor(size * fraction))
        for group, size in group_sizes.items()
    }
    remaining = target - sum(quotas.values())
    remainders = sorted(
        group_sizes,
        key=lambda group: (
            -(group_sizes[group] * fraction - quotas[group]),
            group,
        ),
    )
    for group in remainders[:remaining]:
        quotas[group] += 1
    return quotas


def select_train_fraction(
    candidates: Iterable[Candidate], fraction: float, seed: int
) -> Tuple[List[Candidate], Dict[str, Dict[str, int]]]:
    by_product: DefaultDict[str, DefaultDict[str, List[Candidate]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for candidate in candidates:
        if candidate.split != "train" or candidate.label != "good":
            raise ValueError("Train sampling received a non-normal train candidate.")
        by_product[candidate.category][candidate.source_block].append(candidate)

    selected: List[Candidate] = []
    sampling_summary: Dict[str, Dict[str, int]] = {}
    for product, groups in sorted(by_product.items()):
        group_sizes = {group: len(rows) for group, rows in groups.items()}
        quotas = _allocate_stratified_quotas(group_sizes, fraction)
        product_selected: List[Candidate] = []
        for group, rows in sorted(groups.items()):
            product_selected.extend(select_across_time(rows, quotas[group], seed))
        selected.extend(product_selected)
        sampling_summary[product] = {
            "pool": sum(group_sizes.values()),
            "selected": len(product_selected),
            "source_blocks": len(groups),
        }
    return selected, sampling_summary


def scan_dataset(
    data_root: Path,
    excluded_blocks: Sequence[str],
) -> Tuple[List[Candidate], Counter, Counter]:
    excluded = set(excluded_blocks)
    candidates: List[Candidate] = []
    scanned = Counter()
    skipped = Counter()

    for directory, dirnames, filenames in os.walk(data_root):
        dirnames[:] = sorted(name for name in dirnames if name not in excluded)
        directory_path = Path(directory)
        relative_directory = directory_path.relative_to(data_root)
        if any(part in excluded for part in relative_directory.parts):
            continue

        for filename in sorted(filenames):
            image_path = directory_path / filename
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            scanned["images"] += 1
            relative_path = image_path.relative_to(data_root)
            if not relative_path.parts:
                skipped["invalid_relative_path"] += 1
                continue
            source_block = relative_path.parts[0]
            if source_block in excluded:
                skipped["excluded_block"] += 1
                continue

            label = canonical_label(image_path.parent.name)
            if label is None:
                skipped["ambiguous_label"] += 1
                continue
            if label == "__unknown__":
                skipped[f"unknown_label:{image_path.parent.name}"] += 1
                continue

            product = canonical_product(relative_path.parts)
            if product is None:
                skipped["missing_product"] += 1
                continue

            capture_date = image_path.stem[:8]
            if len(capture_date) != 8 or not capture_date.isdigit():
                skipped["invalid_timestamp"] += 1
                continue
            split = split_for_date(capture_date)
            if split is None:
                skipped["outside_split_dates"] += 1
                continue
            if split == "train" and label != "good":
                skipped["train_period_anomaly"] += 1
                continue

            candidates.append(
                Candidate(
                    category=product,
                    split=split,
                    label=label,
                    relative_path=relative_path.as_posix(),
                    source_block=source_block,
                    capture_date=capture_date,
                )
            )
    return candidates, scanned, skipped


def validate_records(
    records: Sequence[Candidate], expected_products: Sequence[str]
) -> None:
    expected = set(expected_products)
    found = {record.category for record in records}
    if found != expected:
        raise ValueError(f"Product mismatch: expected={sorted(expected)} found={sorted(found)}")

    for product in sorted(expected):
        product_records = [row for row in records if row.category == product]
        train = [row for row in product_records if row.split == "train"]
        if not train or any(row.label != "good" for row in train):
            raise ValueError(f"{product}: train must contain normal images only.")
        for split in ("val", "test"):
            split_records = [row for row in product_records if row.split == split]
            labels = {row.label for row in split_records}
            if "good" not in labels or len(labels) < 2:
                raise ValueError(
                    f"{product}/{split}: both normal and anomaly images are required."
                )


def validate_known_inventory(
    records: Sequence[Candidate], scanned: Counter, train_fraction: float
) -> None:
    """Fail early when the supposedly complete server extraction is partial."""
    scanned_images = int(scanned["images"])
    if scanned_images != EXPECTED_SCANNED_AFTER_DEFAULT_EXCLUSION:
        raise ValueError(
            "Extracted inventory mismatch after excluding "
            f"{DEFAULT_EXCLUDED_BLOCK}: expected "
            f"{EXPECTED_SCANNED_AFTER_DEFAULT_EXCLUSION}, found {scanned_images}. "
            "Use --allow-incomplete only for a local PoC fixture."
        )

    counts = Counter((row.category, row.split) for row in records)
    for product, pool_size in EXPECTED_TRAIN_POOLS.items():
        expected = int(math.floor(pool_size * train_fraction))
        actual = counts[(product, "train")]
        if actual != expected:
            raise ValueError(
                f"{product}/train mismatch: expected {expected}, found {actual}."
            )
    for key, expected in EXPECTED_FIXED_SPLITS.items():
        actual = counts[key]
        if actual != expected:
            raise ValueError(
                f"{key[0]}/{key[1]} mismatch: expected {expected}, found {actual}."
            )


def write_manifest(records: Sequence[Candidate], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    fields = ["category", "split", "label", "image", "source_block", "capture_date"]
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in sorted(
            records,
            key=lambda row: (
                row.category,
                SPLIT_ORDER[row.split],
                row.capture_date,
                row.relative_path,
            ),
        ):
            writer.writerow(
                {
                    "category": record.category,
                    "split": record.split,
                    "label": record.label,
                    "image": record.relative_path,
                    "source_block": record.source_block,
                    "capture_date": record.capture_date,
                }
            )
    os.replace(temporary, output_path)


def build_summary(
    records: Sequence[Candidate],
    scanned: Counter,
    skipped: Counter,
    sampling: Dict[str, Dict[str, int]],
    data_root: Path,
    manifest_path: Path,
    train_fraction: float,
    seed: int,
    excluded_blocks: Sequence[str],
) -> Dict[str, object]:
    counts: DefaultDict[str, Counter] = defaultdict(Counter)
    defect_counts: DefaultDict[str, Counter] = defaultdict(Counter)
    for row in records:
        counts[row.category][row.split] += 1
        counts[row.category][f"{row.split}_{'normal' if row.label == 'good' else 'anomaly'}"] += 1
        if row.label != "good":
            defect_counts[row.category][f"{row.split}:{row.label}"] += 1
    return {
        "data_root_at_generation": str(data_root.resolve()),
        "manifest": str(manifest_path.resolve()),
        "paths_are_relative_to_data_root": True,
        "seed": seed,
        "train_fraction": train_fraction,
        "excluded_blocks": list(excluded_blocks),
        "scanned": dict(scanned),
        "skipped": dict(sorted(skipped.items())),
        "sampling": sampling,
        "counts": {key: dict(value) for key, value in sorted(counts.items())},
        "defect_counts": {
            key: dict(value) for key, value in sorted(defect_counts.items())
        },
        "manifest_rows": len(records),
        "expected_complete_inventory": {
            "images_after_exclusion": EXPECTED_SCANNED_AFTER_DEFAULT_EXCLUSION,
            "train_pools_before_sampling": EXPECTED_TRAIN_POOLS,
            "fixed_val_test_counts": {
                f"{product}/{split}": count
                for (product, split), count in EXPECTED_FIXED_SPLITS.items()
            },
        },
        "selection": (
            "normal-only train from 2024-12-21..23, deterministic temporal-bin "
            "sampling stratified by source block; full usable val 2024-12-24..25; "
            "full usable test 2024-12-26..27"
        ),
    }


def atomic_json_dump(data: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True, help="Output CSV manifest path.")
    parser.add_argument("--summary-path")
    parser.add_argument("--train-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--exclude-block",
        action="append",
        default=[DEFAULT_EXCLUDED_BLOCK],
        help="Exact directory name to exclude; may be supplied more than once.",
    )
    parser.add_argument(
        "--expected-products", nargs="+", default=list(EXPECTED_PRODUCTS)
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Skip product/split completeness validation (PoC fixtures only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    if not 0.0 < args.train_fraction <= 1.0:
        raise ValueError("--train-fraction must be in (0, 1].")

    candidates, scanned, skipped = scan_dataset(data_root, args.exclude_block)
    train_candidates = [row for row in candidates if row.split == "train"]
    selected_train, sampling = select_train_fraction(
        train_candidates, args.train_fraction, args.seed
    )
    records = selected_train + [row for row in candidates if row.split != "train"]
    if not args.allow_incomplete:
        validate_records(records, args.expected_products)
        validate_known_inventory(records, scanned, args.train_fraction)

    output_path = Path(args.output).expanduser().resolve()
    summary_path = (
        Path(args.summary_path).expanduser().resolve()
        if args.summary_path
        else output_path.with_suffix(".summary.json")
    )
    write_manifest(records, output_path)
    summary = build_summary(
        records=records,
        scanned=scanned,
        skipped=skipped,
        sampling=sampling,
        data_root=data_root,
        manifest_path=output_path,
        train_fraction=args.train_fraction,
        seed=args.seed,
        excluded_blocks=args.exclude_block,
    )
    atomic_json_dump(summary, summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Per-stage inference profile of the 4ind global-local model and baselines.

This script does not modify the model files.  It rebuilds each model from the
``args.json`` of its 4ind job, loads ``checkpoint_best_val.pth``, and times
inference on the first test images of the job manifest.  The global-local
stages are timed by an instance-level wrapper of ``_predict_highres_item``;
the unchanged ``_iter_predictions`` loop still records ``model_total``.

With ``--variant NAME`` the script profiles one global-local inference variant
(fp16 autocast, channels_last, tile batch size, tile stride) and writes
``variants/NAME.json``.  ``--merge-variants`` compares the variant scores with
the first variant and writes ``variants_<category>.json``.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

import utils
from fourind_bestval_benchmark import (
    build_datasets,
    build_loader,
    build_model,
    file_sha256,
    load_checkpoint,
)


STAGES = (
    "loader",
    "global_branch",
    "local_backbone_heads",
    "stitching",
    "fusion_and_transfer",
    "model_total",
    "end_to_end",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--category", default="KQG27824")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-count", type=int, default=100)
    parser.add_argument("--warmup-images", type=int, default=5)
    parser.add_argument("--baseline-batch-sizes", default="1,8")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--variant", help="Profile one global-local variant under this name.")
    parser.add_argument("--autocast", action="store_true", help="fp16 autocast for backbone and heads.")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--tile-batch-size", type=int, help="Default: the value in args.json.")
    parser.add_argument("--tile-stride", type=int, help="Default: the values in args.json.")
    parser.add_argument(
        "--nvidia-smi-start", action="store_true",
        help="With --variant, write nvidia-smi-start.txt after the model is on the GPU.",
    )
    parser.add_argument(
        "--baselines-only", action="store_true",
        help="Profile only the baselines on the images of images_<category>.txt.",
    )
    parser.add_argument(
        "--merge-variants",
        help="Comma-separated variant names.  The first one is the reference.",
    )
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_nvidia_smi(path: Path) -> None:
    done = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    path.write_text(done.stdout + done.stderr, encoding="utf-8")


def host_load() -> Dict[str, object]:
    """Record the CPU load, the thread settings, and the use of the visible GPU."""
    gpu_index = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            gpu_index,
            "--query-gpu=index,utilization.gpu,memory.used",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    # Plain nproc returns OMP_NUM_THREADS when it is set; --all gives the core count.
    nproc = subprocess.run(["nproc", "--all"], capture_output=True, text=True)
    return {
        "unix_time": time.time(),
        "loadavg": Path("/proc/loadavg").read_text(encoding="utf-8").strip(),
        "nproc": nproc.stdout.strip(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "torch_num_threads": torch.get_num_threads(),
        "gpu": (gpu.stdout + gpu.stderr).strip(),
    }


def summary_ms(values: List[float]) -> Dict[str, float]:
    return {
        "mean": 1000.0 * statistics.fmean(values),
        "median": 1000.0 * statistics.median(values),
    }


class TimedLoader:
    """Record how long each ``next()`` on the data loader waits."""

    def __init__(self, loader):
        self.loader = loader
        self.waits: List[float] = []

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        iterator = iter(self.loader)
        while True:
            start = time.perf_counter()
            try:
                item = next(iterator)
            except StopIteration:
                return
            self.waits.append(time.perf_counter() - start)
            yield item


def branch_score_maps(model, images: torch.Tensor, branch: str, autocast: bool) -> torch.Tensor:
    """``_score_maps(images, branch, normalize=True)``, optionally under fp16 autocast.

    With autocast, the backbone and the head run in fp16.  The score grid is
    cast to float32 before the upsample and the median/MAD normalization.
    """
    if not autocast:
        return model._score_maps(images, branch, normalize=True)
    with torch.autocast("cuda", dtype=torch.float16):
        score_grid = model._raw_score_grid(images, branch)
    return model._score_maps_from_grid(
        score_grid.float(), images.shape[-2:], branch, normalize=True
    )


@torch.inference_mode()
def timed_predict_highres_item(
    model,
    data_item,
    timings: List[Dict[str, float]],
    autocast: bool = False,
    channels_last: bool = False,
):
    """``GlobalLocalSimpleNet._predict_highres_item`` with a timer per stage.

    The arithmetic is the same as ``_predict_highres_item`` and
    ``_stitch_local_maps``.  The global upsample moves into the fusion stage;
    it does not depend on the local map.  ``autocast`` and ``channels_last``
    change only the backbone and head passes.
    """
    device = model.device
    model._set_inference_mode()
    original_size = model._original_size_from_batch(data_item)
    global_image = data_item["global_image"]
    local_tiles = data_item["local_tiles"]
    tile_boxes = data_item["tile_boxes"]
    if channels_last:
        global_image = global_image.contiguous(memory_format=torch.channels_last)

    synchronize(device)
    start = time.perf_counter()
    global_map = branch_score_maps(model, global_image, "global", autocast)
    synchronize(device)
    global_branch = time.perf_counter() - start

    backbone_heads = 0.0
    stitching = 0.0
    start = time.perf_counter()
    original_h, original_w = original_size
    score_sum = torch.zeros((original_h, original_w), device=device)
    weight_sum = torch.zeros_like(score_sum)
    local_tiles = local_tiles.squeeze(0) if local_tiles.ndim == 5 else local_tiles
    tile_boxes = tile_boxes.squeeze(0) if tile_boxes.ndim == 3 else tile_boxes
    synchronize(device)
    stitching += time.perf_counter() - start

    for batch_start in range(0, len(local_tiles), model.tile_batch_size):
        start = time.perf_counter()
        end = min(batch_start + model.tile_batch_size, len(local_tiles))
        tile_batch = local_tiles[batch_start:end]
        if tile_batch.dtype == torch.uint8:
            tile_batch = tile_batch.to(
                device, dtype=torch.float32, non_blocking=True
            ).div_(255.0)
            mean = tile_batch.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
            std = tile_batch.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
            tile_batch = (tile_batch - mean) / std
        if channels_last:
            tile_batch = tile_batch.contiguous(memory_format=torch.channels_last)
        tile_maps = branch_score_maps(model, tile_batch, "local", autocast)
        synchronize(device)
        backbone_heads += time.perf_counter() - start

        start = time.perf_counter()
        for tile_map, box in zip(tile_maps, tile_boxes[batch_start:end]):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            box_h, box_w = y2 - y1, x2 - x1
            resized = F.interpolate(
                tile_map[None, None],
                size=(box_h, box_w),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            weight = model._tile_weight(box_h, box_w, resized.device)
            score_sum[y1:y2, x1:x2] += resized * weight
            weight_sum[y1:y2, x1:x2] += weight
        synchronize(device)
        stitching += time.perf_counter() - start

    start = time.perf_counter()
    local_map = score_sum / weight_sum.clamp_min(1e-6)
    synchronize(device)
    stitching += time.perf_counter() - start

    start = time.perf_counter()
    global_map = F.interpolate(
        global_map.unsqueeze(1),
        size=original_size,
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    fused_map = torch.maximum(global_map, local_map)
    image_score = float(torch.max(fused_map).cpu())
    anomaly_map = fused_map.float().cpu().numpy()
    synchronize(device)
    fusion = time.perf_counter() - start

    timings.append(
        {
            "global_branch": global_branch,
            "local_backbone_heads": backbone_heads,
            "stitching": stitching,
            "fusion_and_transfer": fusion,
            "tiles": int(len(local_tiles)),
        }
    )
    return image_score, anomaly_map


def run_global_local(
    model, loader, instrumented: bool, autocast: bool = False, channels_last: bool = False
) -> List[Dict[str, float]]:
    """Run the unchanged ``_iter_predictions`` loop once over ``loader``."""
    timed_loader = TimedLoader(loader)
    timings: List[Dict[str, float]] = []
    if instrumented:
        model._predict_highres_item = functools.partial(
            timed_predict_highres_item,
            model,
            timings=timings,
            autocast=autocast,
            channels_last=channels_last,
        )
    else:
        model.__dict__.pop("_predict_highres_item", None)
    model._model_inference_seconds = 0.0
    model._model_inference_images = 0

    rows: List[Dict[str, float]] = []
    previous_model_seconds = 0.0
    last = time.perf_counter()
    for prediction in model._iter_predictions(timed_loader):
        now = time.perf_counter()
        rows.append(
            {
                "score": float(prediction["score"]),
                "model_total": model._model_inference_seconds - previous_model_seconds,
                "end_to_end": now - last,
            }
        )
        previous_model_seconds = model._model_inference_seconds
        last = now
    model.__dict__.pop("_predict_highres_item", None)

    for index, row in enumerate(rows):
        row["loader"] = timed_loader.waits[index]
        if instrumented:
            row.update(timings[index])
    return rows


@torch.inference_mode()
def run_baseline(model, loader, device: torch.device) -> List[Dict[str, float]]:
    """Time ``_predict`` per batch as ``evaluate_baseline_image_level`` does."""
    timed_loader = TimedLoader(loader)
    rows: List[Dict[str, float]] = []
    first_index = 0
    last = time.perf_counter()
    for data in timed_loader:
        synchronize(device)
        start = time.perf_counter()
        scores, _, _ = model._predict(data["image"])
        synchronize(device)
        model_seconds = time.perf_counter() - start
        now = time.perf_counter()
        rows.append(
            {
                "first_index": first_index,
                "images": len(scores),
                "loader": timed_loader.waits[-1],
                "model_total": model_seconds,
                "end_to_end": now - last,
            }
        )
        first_index += len(scores)
        last = now
    return rows


def job_args(results_root: Path, method: str, category: str, data_root: str, manifest_path: str):
    job_dir = results_root / method / category
    args = argparse.Namespace(**json.loads((job_dir / "args.json").read_text(encoding="utf-8")))
    args.data_root = data_root
    args.manifest_path = manifest_path
    job_result = json.loads((job_dir / "job_result.json").read_text(encoding="utf-8"))
    return args, job_dir, job_result


def load_job(args, job_dir: Path, image_count: int, device: torch.device):
    utils.fix_seeds(args.seed, with_torch=True, with_cuda=True)
    model = build_model(args, device)
    checkpoint = job_dir / "checkpoint_best_val.pth"
    load_checkpoint(model, checkpoint, args.method, load_optimizer=False)
    _train, _val, test_dataset, _augmentation = build_datasets(args)
    test_dataset.records = test_dataset.records[:image_count]
    if len(test_dataset.records) != image_count:
        raise ValueError(f"Only {len(test_dataset.records)} test images for {args.category}.")
    return model, test_dataset, checkpoint


def environment(device: torch.device) -> Dict[str, object]:
    return {
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def stage_summary(rows: List[Dict[str, float]], warmup: int) -> Dict[str, object]:
    measured = rows[warmup:]
    stages = {stage: summary_ms([row[stage] for row in measured]) for stage in STAGES}
    sum_1_to_5 = sum(stages[stage]["mean"] for stage in STAGES[:5])
    sum_2_to_5 = sum(stages[stage]["mean"] for stage in STAGES[1:5])
    model_total = stages["model_total"]["mean"]
    return {
        "measured_images": len(measured),
        "stages_ms": stages,
        "sum_mean_stages_1_to_5_ms": sum_1_to_5,
        "sum_mean_stages_2_to_5_ms": sum_2_to_5,
        "sum_1_to_5_over_model_total": sum_1_to_5 / model_total,
        "sum_2_to_5_over_model_total": sum_2_to_5 / model_total,
    }


def write_or_check_images(path: Path, image_paths: List[str]) -> None:
    """Write the image list, or check it against the list that is already there."""
    if path.exists():
        if path.read_text(encoding="utf-8").splitlines() != image_paths:
            raise ValueError(f"The images differ from {path}.")
        return
    path.write_text("\n".join(image_paths) + "\n", encoding="utf-8")


def profile_variant(cli, device: torch.device, output_dir: Path, results_root: Path,
                    manifest_sha256: str) -> None:
    """Profile one global-local variant with the instrumented path."""
    args, job_dir, job_result = job_args(
        results_root, "global_local", cli.category, cli.data_root, cli.manifest_path
    )
    if job_result["manifest_sha256"] != manifest_sha256:
        raise ValueError("The manifest differs from the one in job_result.json.")
    if cli.tile_batch_size:
        args.tile_batch_size = cli.tile_batch_size
    if cli.tile_stride:
        args.tile_stride_x = args.tile_stride_y = cli.tile_stride
    model, dataset, checkpoint = load_job(args, job_dir, cli.image_count, device)
    write_or_check_images(
        output_dir / f"images_{cli.category}.txt",
        [record.image_path for record in dataset.records],
    )
    if cli.channels_last:
        model.backbone.to(memory_format=torch.channels_last)
    loader = build_loader(dataset, 1, cli.num_workers, shuffle=False)
    if cli.nvidia_smi_start:
        write_nvidia_smi(output_dir / "nvidia-smi-start.txt")

    torch.cuda.reset_peak_memory_stats(device)
    load_start = host_load()
    rows = run_global_local(
        model, loader, instrumented=True,
        autocast=cli.autocast, channels_last=cli.channels_last,
    )
    load_end = host_load()
    variants_dir = output_dir / "variants"
    variants_dir.mkdir(exist_ok=True)
    write_json(
        variants_dir / f"{cli.variant}.json",
        {
            "variant": cli.variant,
            "autocast_fp16": cli.autocast,
            "channels_last": cli.channels_last,
            "tile_batch_size": model.tile_batch_size,
            "tile_stride": [args.tile_stride_y, args.tile_stride_x],
            "tile_counts": sorted({row["tiles"] for row in rows}),
            "checkpoint": str(checkpoint),
            "manifest_sha256": manifest_sha256,
            **environment(device),
            "image_count": len(rows),
            "warmup_images": cli.warmup_images,
            **stage_summary(rows, cli.warmup_images),
            "max_memory_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "max_memory_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            "scores": [row["score"] for row in rows],
            "host_load_start": load_start,
            "host_load_end": load_end,
            "per_image_seconds": rows,
        },
    )


def merge_variants(output_dir: Path, names: List[str], category: str) -> None:
    """Compare each variant with the first one and write variants_<category>.json."""
    variants = {
        name: json.loads((output_dir / "variants" / f"{name}.json").read_text(encoding="utf-8"))
        for name in names
    }
    reference = variants[names[0]]
    reference_scores = np.asarray(reference["scores"])
    reference_ranks = stats.rankdata(reference_scores, method="ordinal")
    for data in variants.values():
        scores = np.asarray(data["scores"])
        ranks = stats.rankdata(scores, method="ordinal")
        data["against_reference"] = {
            "reference": names[0],
            "max_abs_score_diff": float(np.max(np.abs(scores - reference_scores))),
            "pearson": float(stats.pearsonr(reference_scores, scores)[0]),
            "spearman": float(stats.spearmanr(reference_scores, scores)[0]),
            "images_with_rank_change_over_5": int(np.sum(np.abs(ranks - reference_ranks) > 5)),
            "speedup_model_total": (
                reference["stages_ms"]["model_total"]["mean"]
                / data["stages_ms"]["model_total"]["mean"]
            ),
            "speedup_local_backbone_heads": (
                reference["stages_ms"]["local_backbone_heads"]["mean"]
                / data["stages_ms"]["local_backbone_heads"]["mean"]
            ),
        }
    write_json(
        output_dir / f"variants_{category}.json",
        {"category": category, "reference": names[0], "variants": variants},
    )


def profile_ours(cli, device: torch.device, output_dir: Path, results_root: Path,
                 manifest_sha256: str) -> List[str]:
    """Profile "Ours": one unchanged pass, one instrumented pass, and their consistency."""
    warmup = cli.warmup_images
    args, job_dir, job_result = job_args(
        results_root, "global_local", cli.category, cli.data_root, cli.manifest_path
    )
    if job_result["manifest_sha256"] != manifest_sha256:
        raise ValueError("The manifest differs from the one in job_result.json.")
    model, dataset, checkpoint = load_job(args, job_dir, cli.image_count, device)
    image_paths = [record.image_path for record in dataset.records]
    (output_dir / f"images_{cli.category}.txt").write_text(
        "\n".join(image_paths) + "\n", encoding="utf-8"
    )
    loader = build_loader(dataset, 1, cli.num_workers, shuffle=False)
    write_nvidia_smi(output_dir / "nvidia-smi-start.txt")

    load_start = host_load()
    unchanged = run_global_local(model, loader, instrumented=False)
    instrumented = run_global_local(model, loader, instrumented=True)
    load_end = host_load()
    unchanged_scores = [row["score"] for row in unchanged]
    instrumented_scores = [row["score"] for row in instrumented]
    write_json(
        output_dir / f"consistency_{cli.category}.json",
        {
            "image_count": len(unchanged_scores),
            "unchanged_scores": unchanged_scores,
            "instrumented_scores": instrumented_scores,
            "max_abs_diff": max(
                abs(a - b) for a, b in zip(unchanged_scores, instrumented_scores)
            ),
        },
    )

    write_json(
        output_dir / f"ours_{cli.category}.json",
        {
            "method": "global_local",
            "category": cli.category,
            "checkpoint": str(checkpoint),
            "manifest_path": cli.manifest_path,
            "manifest_sha256": manifest_sha256,
            **environment(device),
            "image_count": len(instrumented),
            "warmup_images": warmup,
            "tile_batch_size": model.tile_batch_size,
            "tile_counts": sorted({row["tiles"] for row in instrumented}),
            "num_workers": cli.num_workers,
            **stage_summary(instrumented, warmup),
            "unchanged_model_total_ms": summary_ms(
                [row["model_total"] for row in unchanged[warmup:]]
            ),
            "recorded_test_seconds_per_image": job_result["test_seconds_per_image"],
            "host_load_start": load_start,
            "host_load_end": load_end,
            "per_image_seconds": instrumented,
        },
    )
    del model, loader
    torch.cuda.empty_cache()
    return image_paths


def main() -> None:
    cli = parse_args()
    output_dir = Path(cli.output_dir)
    if cli.merge_variants:
        merge_variants(output_dir, cli.merge_variants.split(","), cli.category)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Profiling requires a CUDA GPU.")
    torch.set_num_threads(cli.num_threads)
    device = torch.device("cuda:0")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_root = Path(cli.results_root)
    manifest_sha256 = file_sha256(Path(cli.manifest_path))
    if cli.variant:
        profile_variant(cli, device, output_dir, results_root, manifest_sha256)
        return
    batch_sizes = [int(value) for value in cli.baseline_batch_sizes.split(",")]
    warmup = cli.warmup_images

    if cli.baselines_only:
        image_paths = (output_dir / f"images_{cli.category}.txt").read_text(
            encoding="utf-8"
        ).splitlines()
    else:
        image_paths = profile_ours(cli, device, output_dir, results_root, manifest_sha256)

    # SimpleNet and SimpleNet++ baselines on the same images.
    for method, output_name in (("simplenet", "simplenet"), ("simplenet_plus", "simplenet_plus")):
        args, job_dir, job_result = job_args(
            results_root, method, cli.category, cli.data_root, cli.manifest_path
        )
        if job_result["manifest_sha256"] != manifest_sha256:
            raise ValueError(f"The manifest of {method} differs from job_result.json.")
        model, dataset, checkpoint = load_job(args, job_dir, cli.image_count, device)
        if [record.image_path for record in dataset.records] != image_paths:
            raise ValueError(f"The {method} images differ from the global-local images.")
        by_batch_size = {}
        load_start = host_load()
        for batch_size in batch_sizes:
            loader = build_loader(dataset, batch_size, cli.num_workers, shuffle=False)
            rows = run_baseline(model, loader, device)
            # Leave out each batch that holds a warm-up image.
            kept = [row for row in rows if row["first_index"] >= warmup]
            images = sum(row["images"] for row in kept)
            by_batch_size[f"batch_size_{batch_size}"] = {
                "batch_size": batch_size,
                "measured_images": images,
                "model_total_ms_per_image": {
                    "mean": 1000.0 * sum(row["model_total"] for row in kept) / images,
                    "median": 1000.0 * statistics.median(
                        row["model_total"] / row["images"] for row in kept
                    ),
                },
                "end_to_end_ms_per_image": {
                    "mean": 1000.0 * sum(row["end_to_end"] for row in kept) / images,
                    "median": 1000.0 * statistics.median(
                        row["end_to_end"] / row["images"] for row in kept
                    ),
                },
                "per_batch_seconds": rows,
            }
            del loader
        load_end = host_load()
        write_json(
            output_dir / f"{output_name}_{cli.category}.json",
            {
                "method": method,
                "category": cli.category,
                "checkpoint": str(checkpoint),
                "manifest_sha256": manifest_sha256,
                **environment(device),
                "image_count": len(dataset.records),
                "warmup_images": warmup,
                "num_workers": cli.num_workers,
                "recorded_test_seconds_per_image": job_result["test_seconds_per_image"],
                "host_load_start": load_start,
                "host_load_end": load_end,
                **by_batch_size,
            },
        )
        del model
        torch.cuda.empty_cache()

    write_nvidia_smi(output_dir / "nvidia-smi-end.txt")


if __name__ == "__main__":
    main()

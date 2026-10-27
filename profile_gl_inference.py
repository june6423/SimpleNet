"""Per-stage inference profile of the 4ind global-local model and baselines.

This script does not modify the model files.  It rebuilds each model from the
``args.json`` of its 4ind job, loads ``checkpoint_best_val.pth``, and times
inference on the first test images of the job manifest.  The global-local
stages are timed by an instance-level wrapper of ``_predict_highres_item``;
the unchanged ``_iter_predictions`` loop still records ``model_total``.
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

import torch
import torch.nn.functional as F

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
    nproc = subprocess.run(["nproc"], capture_output=True, text=True)
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


@torch.inference_mode()
def timed_predict_highres_item(model, data_item, timings: List[Dict[str, float]]):
    """``GlobalLocalSimpleNet._predict_highres_item`` with a timer per stage.

    The arithmetic is the same as ``_predict_highres_item`` and
    ``_stitch_local_maps``.  The global upsample moves into the fusion stage;
    it does not depend on the local map.
    """
    device = model.device
    model._set_inference_mode()
    original_size = model._original_size_from_batch(data_item)
    local_tiles = data_item["local_tiles"]
    tile_boxes = data_item["tile_boxes"]

    synchronize(device)
    start = time.perf_counter()
    global_map = model._score_maps(data_item["global_image"], "global", normalize=True)
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
        tile_maps = model._score_maps(tile_batch, "local", normalize=True)
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


def run_global_local(model, loader, instrumented: bool) -> List[Dict[str, float]]:
    """Run the unchanged ``_iter_predictions`` loop once over ``loader``."""
    timed_loader = TimedLoader(loader)
    timings: List[Dict[str, float]] = []
    if instrumented:
        model._predict_highres_item = functools.partial(
            timed_predict_highres_item, model, timings=timings
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


def main() -> None:
    cli = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Profiling requires a CUDA GPU.")
    torch.set_num_threads(cli.num_threads)
    device = torch.device("cuda:0")
    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_root = Path(cli.results_root)
    manifest_sha256 = file_sha256(Path(cli.manifest_path))
    batch_sizes = [int(value) for value in cli.baseline_batch_sizes.split(",")]
    warmup = cli.warmup_images

    # Global-local ("Ours").
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

    measured = instrumented[warmup:]
    stages = {stage: summary_ms([row[stage] for row in measured]) for stage in STAGES}
    sum_1_to_5 = sum(stages[stage]["mean"] for stage in STAGES[:5])
    sum_2_to_5 = sum(stages[stage]["mean"] for stage in STAGES[1:5])
    model_total = stages["model_total"]["mean"]
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
            "measured_images": len(measured),
            "tile_batch_size": model.tile_batch_size,
            "tile_counts": sorted({row["tiles"] for row in instrumented}),
            "num_workers": cli.num_workers,
            "stages_ms": stages,
            "sum_mean_stages_1_to_5_ms": sum_1_to_5,
            "sum_mean_stages_2_to_5_ms": sum_2_to_5,
            "sum_1_to_5_over_model_total": sum_1_to_5 / model_total,
            "sum_2_to_5_over_model_total": sum_2_to_5 / model_total,
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

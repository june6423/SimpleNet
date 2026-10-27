"""Inference time of SimpleNet, SimpleNet++, DeGLAD, and DeGLAD-S on one GPU.

``run`` times one (dataset, category) pair at batch size 1 over the first
``--image-count`` test images, in the order SN_A, SN_B, SNP_A, SNP_B, GL, GLS:

* boundary A: the unchanged baseline ``_predict``;
* boundary B: ``predict_boundary_b``, the same calls up to the patch scores and
  then only the score-map part of ``RescaleSegmentor.convert_to_segmentation``
  (no feature upsample and no feature host copy);
* GL: the unchanged DeGLAD ``_iter_predictions`` (mode ``old``, fp32);
* GLS: DeGLAD-S, ``use_deglad_s`` of ``visa_f1_eval_320.py``.

Each method uses the loader, input size, and checkpoint of the earlier runs:
4IND as ``fourind_bestval_benchmark.py`` (the first test images of the job
manifest), MVTec and ViSA as ``visa_f1_eval_320.py``.  ``run`` writes
``raw/<dataset>_<category>.json`` and, for each baseline, the image scores of
both boundaries to ``scores/<dataset>_<category>_<method>.json``.  ``merge``
writes ``speed_compare.csv`` and ``speed_compare_summary.csv``.
"""

from __future__ import annotations

import argparse
import csv
import functools
import itertools
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

import fourind_bestval_benchmark as fourind_runner
import mvtec_besttest_benchmark as mvtec_baseline_runner
import utils
import visa_benchmark as visa_runner
import visa_f1_eval_320 as f1_eval
from profile_gl_inference import job_args, load_job

CONFIGS = (
    ("SN_A", "simplenet", "A"),
    ("SN_B", "simplenet", "B"),
    ("SNP_A", "simplenet_plus", "A"),
    ("SNP_B", "simplenet_plus", "B"),
    ("GL", "global_local", "DeGLAD"),
    ("GLS", "global_local", "DeGLAD-S"),
)
BASELINES = ("simplenet", "simplenet_plus")
WARMUP_IMAGES = 5


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def host_load() -> dict:
    """CPU load, thread settings, and the use of the visible GPU."""
    gpu = subprocess.run(
        ["nvidia-smi", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
         "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
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


def predict_boundary_b(model, images):
    """Baseline ``_predict`` up to the patch scores, then the score map only.

    The lines follow ``SimpleNet._predict`` (``simplenet.py``) and
    ``simplenet_plus.SimpleNet._predict``, which are the same.  The score-map
    part of ``convert_to_segmentation`` is ``common.py`` lines 84 to 93 and
    115 to 117; lines 95 to 113 (feature upsample and host copy) are left out.
    """
    images = images.to(torch.float).to(model.device)
    _ = model.forward_modules.eval()

    batchsize = images.shape[0]
    if model.pre_proj > 0:
        model.pre_projection.eval()
    model.discriminator.eval()
    with torch.no_grad():
        features, patch_shapes = model._embed(images, provide_patch_shapes=True, evaluation=True)
        if model.pre_proj > 0:
            features = model.pre_projection(features)

        patch_scores = image_scores = -model.discriminator(features)
        patch_scores = patch_scores.cpu().numpy()
        image_scores = image_scores.cpu().numpy()

        image_scores = model.patch_maker.unpatch_scores(image_scores, batchsize=batchsize)
        image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
        image_scores = model.patch_maker.score(image_scores)

        patch_scores = model.patch_maker.unpatch_scores(patch_scores, batchsize=batchsize)
        scales = patch_shapes[0]
        patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

        segmentor = model.anomaly_segmentor
        _scores = torch.from_numpy(patch_scores).to(segmentor.device).unsqueeze(1)
        _scores = F.interpolate(
            _scores, size=segmentor.target_size, mode="bilinear", align_corners=False
        ).squeeze(1)
        patch_scores = _scores.cpu().numpy()
        masks = [
            ndimage.gaussian_filter(patch_score, sigma=segmentor.smoothing)
            for patch_score in patch_scores
        ]
    return list(image_scores), masks


class FirstItem:
    """Pass the loader items through; record the input size of the first item."""

    def __init__(self, loader):
        self.loader = loader
        self.input_size = None

    def __iter__(self):
        for item in self.loader:
            if self.input_size is None:
                if "global_image" in item:
                    tiles = item["local_tiles"]
                    self.input_size = (
                        f"global {tuple(item['global_image'].shape[-2:])}, "
                        f"{tiles.shape[1]} tiles {tuple(tiles.shape[-2:])}"
                    )
                else:
                    self.input_size = str(tuple(item["image"].shape[-2:]))
            yield item


def time_baseline(model, loader, device, boundary):
    """Per-image model seconds and image scores at boundary A or B (batch size 1)."""
    predict = model._predict if boundary == "A" else functools.partial(predict_boundary_b, model)
    times, scores = [], []
    for data in loader:
        synchronize(device)
        start = time.perf_counter()
        batch_scores = predict(data["image"])[0]
        synchronize(device)
        times.append(time.perf_counter() - start)
        scores.append(float(np.asarray(batch_scores[0]).reshape(-1)[0]))
    return times, scores


def time_global_local(model, loader):
    """Per-image synchronized model seconds of ``_iter_predictions`` and image scores."""
    times, scores = [], []
    for prediction in f1_eval.timed_iter_predictions(model, loader, times):
        scores.append(float(prediction["score"]))
    return times, scores


def stats_ms(times):
    measured = [1000.0 * t for t in times[WARMUP_IMAGES:]]
    return {
        "mean_ms": statistics.fmean(measured),
        "median_ms": statistics.median(measured),
        "min_ms": min(measured),
        "max_ms": max(measured),
    }


def runner_args(cli, method):
    """The argument names of the MVTec and ViSA runners, as visa_f1_eval_320.py sets them."""
    args = argparse.Namespace(
        method=method, category=cli.category, data_path=cli.data_root, seed=cli.seed,
        imagesize=cli.imagesize, meta_epochs=cli.meta_epochs, gan_epochs=cli.gan_epochs,
        smoke=False,
    )
    if cli.dataset == "mvtec":
        args.resize = cli.resize
        args.batch_size = 8
        args.num_workers = cli.num_workers
        args.augmentation = 0.0
    return args


def load_method(cli, method, device):
    """Model, test DataLoader (batch size 1), loader name, and checkpoint of one method."""
    if cli.dataset == "4ind":
        args, job_dir, _job_result = job_args(
            Path(cli.checkpoint_root), method, cli.category, cli.data_root, cli.manifest_path
        )
        model, dataset, checkpoint = load_job(args, job_dir, cli.image_count, device)
        loader = fourind_runner.build_loader(dataset, 1, cli.num_workers, shuffle=False)
        return model, loader, type(dataset).__name__, checkpoint

    utils.fix_seeds(cli.seed, with_torch=True, with_cuda=True)
    f1_args = argparse.Namespace(
        dataset=cli.dataset, method=method, category=cli.category,
        checkpoint_root=cli.checkpoint_root, reference_csv=None,
    )
    checkpoint, _reference_csv = f1_eval.checkpoint_paths(f1_args)
    if method == "global_local":
        cli.method = method
        model, loader = f1_eval.build_global_local(cli, checkpoint, device)
        name = type(loader.dataset).__name__
    elif cli.dataset == "mvtec":
        args = runner_args(cli, method)
        _train_loader, loader = mvtec_baseline_runner.build_dataloaders(args, 0.0)
        model = mvtec_baseline_runner.build_model(args, device)
        mvtec_baseline_runner.load_selected_checkpoint(model, checkpoint)
        name = type(loader.dataset).__name__
    else:
        args = runner_args(cli, method)
        train_dataset, test_dataset, _augmentation = visa_runner.build_datasets(args)
        loader = visa_runner.build_loader(test_dataset, 1, cli.num_workers, shuffle=False)
        model = visa_runner.build_baseline_model(args, train_dataset.imagesize, device)
        visa_runner.load_baseline_checkpoint(model, checkpoint, method, load_optimizer=False)
        name = type(test_dataset).__name__
    return model, loader, name, checkpoint


def run(cli) -> None:
    torch.set_num_threads(cli.num_threads)
    device = torch.device("cuda:0")
    out = Path(cli.output_dir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    rows = []
    scores = {}
    for method in (*BASELINES, "global_local"):
        model, loader, loader_name, checkpoint = load_method(cli, method, device)
        with torch.inference_mode():
            for config, config_method, boundary in CONFIGS:
                if config_method != method:
                    continue
                if config == "GLS":
                    f1_eval.use_deglad_s(model)
                torch.cuda.reset_peak_memory_stats(device)
                # The same first images of the shuffle-free loader for every configuration.
                first = FirstItem(itertools.islice(loader, cli.image_count))
                load_start = host_load()
                if method == "global_local":
                    times, config_scores = time_global_local(model, first)
                else:
                    times, config_scores = time_baseline(model, first, device, boundary)
                load_end = host_load()
                scores[config] = config_scores
                rows.append(
                    {
                        "dataset": cli.dataset,
                        "category": cli.category,
                        "config": config,
                        "method": method,
                        "boundary": boundary,
                        "image_count": len(times),
                        "measured_images": len(times) - WARMUP_IMAGES,
                        "batch_size": 1,
                        **stats_ms(times),
                        "checkpoint": str(checkpoint),
                        "loader": loader_name,
                        "input_size": first.input_size,
                        "max_memory_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
                        "max_memory_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                        "torch_version": torch.__version__,
                        "host_load_start": load_start,
                        "host_load_end": load_end,
                        "per_image_seconds": times,
                    }
                )
        if method in BASELINES:
            prefix = "SN" if method == "simplenet" else "SNP"
            a, b = np.asarray(scores[f"{prefix}_A"]), np.asarray(scores[f"{prefix}_B"])
            with (out / "scores" / f"{cli.dataset}_{cli.category}_{method}.json").open("w") as stream:
                json.dump(
                    {
                        "dataset": cli.dataset,
                        "category": cli.category,
                        "method": method,
                        "image_count": len(a),
                        "max_abs_diff": float(np.max(np.abs(a - b))),
                        "scores_a": a.tolist(),
                        "scores_b": b.tolist(),
                    },
                    stream,
                    indent=2,
                )
        del model, loader
        torch.cuda.empty_cache()
    with (out / "raw" / f"{cli.dataset}_{cli.category}.json").open("w") as stream:
        json.dump({"rows": rows, "gl_scores": scores["GL"], "gls_scores": scores["GLS"]}, stream, indent=2)
    for row in rows:
        print(f"{row['dataset']:<6}{row['category']:<12}{row['config']:<6}images={row['image_count']} "
              f"mean={row['mean_ms']:.2f}ms median={row['median_ms']:.2f}ms")


ROW_COLUMNS = (
    "dataset", "category", "config", "method", "boundary", "image_count", "measured_images",
    "batch_size", "mean_ms", "median_ms", "min_ms", "max_ms", "checkpoint", "loader",
    "input_size", "max_memory_reserved_mib", "max_memory_allocated_mib", "torch_version",
)


def merge(cli) -> None:
    out = Path(cli.output_dir)
    rows = []
    for path in sorted((out / "raw").glob("*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["rows"])
    with (out / "speed_compare.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(ROW_COLUMNS)
        for row in rows:
            writer.writerow([row[c] for c in ROW_COLUMNS])

    with (out / "speed_compare_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", "config", "categories", "mean_ms", "median_ms", "gls_speedup"])
        for dataset in dict.fromkeys(row["dataset"] for row in rows):
            sub = [row for row in rows if row["dataset"] == dataset]
            means = {
                config: (
                    statistics.fmean(r["mean_ms"] for r in sub if r["config"] == config),
                    statistics.fmean(r["median_ms"] for r in sub if r["config"] == config),
                    sum(r["config"] == config for r in sub),
                )
                for config, _method, _boundary in CONFIGS
            }
            for config, (mean_ms, median_ms, count) in means.items():
                writer.writerow([dataset, config, count, mean_ms, median_ms, mean_ms / means["GLS"][0]])
    print(f"Wrote {out / 'speed_compare.csv'} and {out / 'speed_compare_summary.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rn = sub.add_parser("run")
    rn.add_argument("--dataset", choices=("4ind", "mvtec", "visa"), required=True)
    rn.add_argument("--category", required=True)
    rn.add_argument("--checkpoint-root", required=True,
                    help="4ind: the results root of the job directories.")
    rn.add_argument("--data-root", required=True)
    rn.add_argument("--manifest-path", help="4ind only.")
    rn.add_argument("--output-dir", required=True)
    rn.add_argument("--image-count", type=int, default=100)
    rn.add_argument("--num-workers", type=int, default=4)
    rn.add_argument("--num-threads", type=int, default=4)
    rn.add_argument("--seed", type=int, default=0)
    rn.add_argument("--resize", type=int, default=329, help="MVTec only.")
    rn.add_argument("--imagesize", type=int, default=320)
    rn.add_argument("--meta-epochs", type=int, default=40)
    rn.add_argument("--gan-epochs", type=int, default=4)
    mg = sub.add_parser("merge")
    mg.add_argument("--output-dir", required=True)
    cli = parser.parse_args()
    if cli.command == "run":
        run(cli)
    else:
        merge(cli)


if __name__ == "__main__":
    main()

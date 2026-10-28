"""DeGLAD-S on the whole 4ind test split of one category.

The script loads ``checkpoint_best_val.pth`` of the global_local job, builds the
TEST split with ``fourind_bestval_benchmark.py`` (batch size 1, no shuffle),
switches the model to DeGLAD-S with ``use_deglad_s`` of ``visa_f1_eval_320.py``,
and runs the unchanged ``_iter_predictions`` loop over every test image.  The
metrics come from ``summarize_image_scores`` of ``fourind_bestval_benchmark.py``
plus the image F1-max of ``mvtec_f1_eval.f1_max``.  The reference block copies
the recorded test result of ``job_result.json``.  Outputs:
``deglad_s_4ind_<category>.json`` and ``scores_<category>.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

import fourind_bestval_benchmark as fourind_runner
import utils
import visa_f1_eval_320 as f1_eval
from mvtec_f1_eval import f1_max
from profile_gl_inference import job_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--category", default="KQG27824")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    torch.set_num_threads(cli.num_threads)
    device = torch.device("cuda:0")
    out = Path(cli.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results_root = Path(cli.results_root)

    args, job_dir, job_result = job_args(
        results_root, "global_local", cli.category, cli.data_root, cli.manifest_path
    )
    manifest_sha256 = fourind_runner.file_sha256(Path(cli.manifest_path))
    job_result_path = job_dir / "job_result.json"
    reference = {
        "path": str(job_result_path),
        "sha256": fourind_runner.file_sha256(job_result_path),
        **{
            key: job_result[key]
            for key in (
                "test_image_auroc", "test_images", "test_per_defect",
                "test_seconds_per_image", "manifest_sha256",
            )
        },
    }

    utils.fix_seeds(args.seed, with_torch=True, with_cuda=True)
    model = fourind_runner.build_model(args, device)
    checkpoint = job_dir / "checkpoint_best_val.pth"
    fourind_runner.load_checkpoint(model, checkpoint, "global_local", load_optimizer=False)
    _train, _val, test_dataset, _augmentation = fourind_runner.build_datasets(args)
    loader = fourind_runner.build_loader(test_dataset, 1, cli.num_workers, shuffle=False)
    f1_eval.use_deglad_s(model)

    labels, scores, anomaly_types, times = [], [], [], []
    model._model_inference_seconds = 0.0
    model._model_inference_images = 0
    start = time.perf_counter()
    with torch.inference_mode():
        for prediction in f1_eval.timed_iter_predictions(model, loader, times):
            labels.append(int(prediction["label"]))
            scores.append(float(prediction["score"]))
            anomaly_types.append(str(prediction["anomaly"]))
    end_to_end = time.perf_counter() - start
    if len(scores) != len(test_dataset.records):
        raise ValueError(f"{len(scores)} predictions for {len(test_dataset.records)} test records.")

    result = fourind_runner.summarize_image_scores(
        labels=labels,
        scores=scores,
        anomaly_types=anomaly_types,
        model_seconds=float(model._model_inference_seconds),
        end_to_end_seconds=end_to_end,
    )
    image_f1 = f1_max(np.asarray(labels), np.asarray(scores))
    payload = {
        "dataset": "4ind",
        "method": "deglad_s",
        "category": cli.category,
        "checkpoint": str(checkpoint),
        "manifest_path": cli.manifest_path,
        "manifest_sha256": manifest_sha256,
        "split": "test",
        "batch_size": 1,
        "feature_interpolation_mode": model.feature_interpolation_mode,
        "feature_interpolation_fp16": bool(model.feature_interpolation_fp16),
        "autocast_fp16": True,
        "channels_last": True,
        "tile_stride": [args.tile_stride_y, args.tile_stride_x],
        "tile_batch_size": model.tile_batch_size,
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        **result,
        "image_f1_max": image_f1["f1_max"],
        "image_f1_threshold": image_f1["threshold"],
        "image_f1_precision": image_f1["precision"],
        "image_f1_recall": image_f1["recall"],
        "reference": reference,
        "image_auroc_diff": result["image_auroc"] - reference["test_image_auroc"],
        "per_defect_image_auroc_diff": {
            defect: values["image_auroc"] - reference["test_per_defect"][defect]["image_auroc"]
            for defect, values in result["per_defect"].items()
            if defect in reference["test_per_defect"]
        },
        "manifest_matches_reference": manifest_sha256 == reference["manifest_sha256"],
    }
    with (out / f"deglad_s_4ind_{cli.category}.json").open("w") as stream:
        json.dump(payload, stream, indent=2)

    # The loader keeps the record order (batch size 1, no shuffle).
    with (out / f"scores_{cli.category}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_path", "label", "anomaly_type", "score", "model_seconds"])
        for record, label, anomaly, score, seconds in zip(
            test_dataset.records, labels, anomaly_types, scores, times
        ):
            writer.writerow([record.image_path, label, anomaly, score, seconds])
    print(
        f"{cli.category}: images={result['image_count']} image_auroc={result['image_auroc']:.4f} "
        f"(reference {reference['test_image_auroc']:.4f}) image_f1_max={image_f1['f1_max']:.4f} "
        f"seconds_per_image={result['seconds_per_image']:.4f}"
    )


if __name__ == "__main__":
    main()

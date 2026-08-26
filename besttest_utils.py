"""Shared helpers for resumable epoch-wise best-test benchmarks.

This module is intentionally separate from the three model implementations.
It provides the experiment-only training loop required to inspect learning
dynamics without changing ``simplenet.py``, ``simplenet_plus.py``, or
``simplenet_gl.py``.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


def atomic_torch_save(state: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def atomic_json_dump(data: Dict[str, object], path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary, path)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def is_better(
    candidate_image: float,
    candidate_pixel: float,
    best_image: Optional[float],
    best_pixel: Optional[float],
) -> bool:
    """Apply the declared test I-AUROC then P-AUROC selection policy."""
    if best_image is None or best_pixel is None:
        return True
    return (candidate_image, candidate_pixel) > (best_image, best_pixel)


def _global_local_optimizer_state(model) -> Dict[str, object]:
    state = {
        "local_discriminator": model.dsc_opt.state_dict(),
        "global_discriminator": model.global_dsc_opt.state_dict(),
    }
    if model.pre_proj > 0:
        state["local_projection"] = model.proj_opt.state_dict()
        state["global_projection"] = model.global_proj_opt.state_dict()
    if model.train_backbone:
        state["backbone"] = model.backbone_opt.state_dict()
    return state


def global_local_training_state(
    model,
    next_meta_epoch: int,
    history: List[Dict[str, object]],
    best_epoch: Optional[int],
    best_image_auroc: Optional[float],
    best_pixel_auroc: Optional[float],
) -> Dict[str, object]:
    state = model._checkpoint_state()
    state.update(
        {
            "next_meta_epoch": int(next_meta_epoch),
            "history": history,
            "best_epoch": best_epoch,
            "best_image_auroc": best_image_auroc,
            "best_pixel_auroc": best_pixel_auroc,
            "optimizers": _global_local_optimizer_state(model),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
        }
    )
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    return state


def _load_global_local_weights(model, state: Dict[str, object]) -> None:
    model.discriminator.load_state_dict(state["local_discriminator"])
    model.global_discriminator.load_state_dict(state["global_discriminator"])
    if model.pre_proj > 0:
        model.pre_projection.load_state_dict(state["local_projection"])
        model.global_projection.load_state_dict(state["global_projection"])
    if model.train_backbone and "backbone" in state:
        model.backbone.load_state_dict(state["backbone"])
    calibration = state.get("calibration", {})
    for name, buffer_name in (
        ("local_median", "local_score_median"),
        ("local_scale", "local_score_scale"),
        ("global_median", "global_score_median"),
        ("global_scale", "global_score_scale"),
    ):
        if name in calibration:
            getattr(model, buffer_name).fill_(float(calibration[name]))


def load_global_local_checkpoint(
    model, path: Path, load_optimizer: bool
) -> Dict[str, object]:
    state = torch.load(path, map_location=model.device, weights_only=False)
    _load_global_local_weights(model, state)
    if load_optimizer:
        optimizers = state["optimizers"]
        model.dsc_opt.load_state_dict(optimizers["local_discriminator"])
        model.global_dsc_opt.load_state_dict(optimizers["global_discriminator"])
        if model.pre_proj > 0:
            model.proj_opt.load_state_dict(optimizers["local_projection"])
            model.global_proj_opt.load_state_dict(optimizers["global_projection"])
        if model.train_backbone:
            model.backbone_opt.load_state_dict(optimizers["backbone"])
        if "torch_rng" in state:
            torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and "cuda_rng" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if "numpy_rng" in state:
            np.random.set_state(state["numpy_rng"])
    return state


def train_global_local_meta_epoch(
    model, training_data: Iterable
) -> Dict[str, float]:
    """Run one meta epoch while preserving the original branch update recipe."""
    model.forward_modules.eval()
    model.discriminator.train()
    model.global_discriminator.train()
    if model.pre_proj > 0:
        model.pre_projection.train()
        model.global_projection.train()

    calibration = {"local": [], "global": []}
    all_losses: List[float] = []
    started = time.perf_counter()

    for gan_epoch in range(model.gan_epochs):
        for data_item in training_data:
            global_images, local_images, reuse_features = model._get_training_views(
                data_item
            )
            global_images = global_images.to(
                model.device, dtype=torch.float32, non_blocking=True
            )
            local_images = local_images.to(
                model.device, dtype=torch.float32, non_blocking=True
            )
            model._zero_branch_optimizers()

            if reuse_features:
                raw_features = model._embed(local_images, evaluation=False)[0]
                local_loss, local_scores = model._branch_loss_from_features(
                    raw_features, "local"
                )
                global_loss, global_scores = model._branch_loss_from_features(
                    raw_features, "global"
                )
            else:
                local_raw = model._embed(local_images, evaluation=False)[0]
                global_raw = model._embed(global_images, evaluation=False)[0]
                local_loss, local_scores = model._branch_loss_from_features(
                    local_raw, "local"
                )
                global_loss, global_scores = model._branch_loss_from_features(
                    global_raw, "global"
                )

            loss = (
                model.local_loss_weight * local_loss
                + model.global_loss_weight * global_loss
            )
            loss.backward()
            model._step_branch_optimizers()
            all_losses.append(float(loss.detach().cpu()))

            if gan_epoch == model.gan_epochs - 1:
                model._append_calibration_samples(calibration["local"], local_scores)
                model._append_calibration_samples(calibration["global"], global_scores)

    model._set_score_calibration(calibration)
    return {
        "mean_train_loss": float(np.mean(all_losses)) if all_losses else math.nan,
        "train_seconds": time.perf_counter() - started,
    }


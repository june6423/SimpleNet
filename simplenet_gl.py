"""Global-local SimpleNet with overlap tiling and Gaussian feature noise.

This module intentionally lives next to, rather than replacing, ``simplenet.py``
and ``simplenet_plus.py``.  It reuses their feature extraction primitives while
keeping an independent training, checkpointing, and high-resolution inference
path.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections import OrderedDict
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from scipy import ndimage
from sklearn import metrics as sklearn_metrics

import common
import simplenet


LOGGER = logging.getLogger(__name__)


class GlobalLocalSimpleNet(simplenet.SimpleNet):
    """A low-cost global/local extension of SimpleNet.

    The backbone and multi-level feature aggregation are shared.  Each branch
    owns only a small linear adapter and discriminator.  Gaussian feature
    negatives are used during training and discarded at inference.
    """

    def load(
        self,
        *args,
        global_noise_std: Optional[float] = None,
        local_noise_std: Optional[float] = None,
        global_loss_weight: float = 0.5,
        local_loss_weight: float = 0.5,
        tile_batch_size: int = 32,
        score_smoothing: int = 0,
        calibration_sample_limit: int = 200_000,
        pixel_histogram_bins: int = 4096,
        early_exit_backbone: bool = True,
        exact_pixel_auroc: bool = False,
        legacy_gaussian_sigma: float = 4.0,
        **kwargs,
    ):
        super().load(*args, **kwargs)

        # Restore the intended stop-at-last-feature behavior for this variant
        # without changing the shared baseline feature aggregator.
        if early_exit_backbone and not self.train_backbone:
            self._install_early_exit_hook()

        if global_loss_weight < 0 or local_loss_weight < 0:
            raise ValueError("Branch loss weights must be non-negative.")
        weight_sum = global_loss_weight + local_loss_weight
        if weight_sum <= 0:
            raise ValueError("At least one branch loss weight must be positive.")

        self.global_loss_weight = global_loss_weight / weight_sum
        self.local_loss_weight = local_loss_weight / weight_sum
        self.global_noise_std = (
            self.noise_std if global_noise_std is None else global_noise_std
        )
        self.local_noise_std = (
            self.noise_std if local_noise_std is None else local_noise_std
        )
        self.tile_batch_size = max(1, int(tile_batch_size))
        self.score_smoothing = max(0, int(score_smoothing))
        self.calibration_sample_limit = max(1024, int(calibration_sample_limit))
        self.pixel_histogram_bins = max(128, int(pixel_histogram_bins))
        self.exact_pixel_auroc = bool(exact_pixel_auroc)
        self.legacy_gaussian_sigma = max(0.0, float(legacy_gaussian_sigma))

        # The inherited projection/discriminator form the local branch.  A
        # second lightweight head handles the different global feature scale.
        self.global_projection = None
        if self.pre_proj > 0:
            self.global_projection = simplenet.Projection(
                self.target_embed_dimension,
                self.target_embed_dimension,
                self.pre_proj,
                kwargs.get("proj_layer_type", 0),
            ).to(self.device)
            self.global_proj_opt = torch.optim.AdamW(
                self.global_projection.parameters(), lr=self.lr * 0.1
            )

        dsc_layers = kwargs.get("dsc_layers", 2)
        dsc_hidden = kwargs.get("dsc_hidden")
        self.global_discriminator = simplenet.Discriminator(
            self.target_embed_dimension,
            n_layers=dsc_layers,
            hidden=dsc_hidden,
        ).to(self.device)
        self.global_dsc_opt = torch.optim.Adam(
            self.global_discriminator.parameters(),
            lr=self.dsc_lr,
            weight_decay=1e-5,
        )

        # Robust train-normal calibration.  MAD is stored already multiplied
        # by 1.4826, making it comparable to a standard deviation for a normal
        # score distribution.
        self.register_buffer("local_score_median", torch.tensor(0.0))
        self.register_buffer("local_score_scale", torch.tensor(1.0))
        self.register_buffer("global_score_median", torch.tensor(0.0))
        self.register_buffer("global_score_scale", torch.tensor(1.0))
        self.last_inference_stats: Dict[str, float] = {}
        self._model_inference_seconds = 0.0
        self._model_inference_images = 0
        self._weight_cache: Dict[Tuple[int, int, str], torch.Tensor] = {}

    def _install_early_exit_hook(self):
        extract_layer = self.layers_to_extract_from[-1]
        if "." in extract_layer:
            extract_block, extract_idx = extract_layer.split(".", maxsplit=1)
            network_layer = self.backbone.__dict__["_modules"][extract_block]
            if extract_idx.isnumeric():
                network_layer = network_layer[int(extract_idx)]
            else:
                network_layer = network_layer.__dict__["_modules"][extract_idx]
        else:
            network_layer = self.backbone.__dict__["_modules"][extract_layer]
        if isinstance(network_layer, torch.nn.Sequential):
            network_layer = network_layer[-1]

        def stop_after_last_requested_layer(_module, _inputs, _output):
            raise common.LastLayerToExtractReachedException()

        self.backbone.hook_handles.append(
            network_layer.register_forward_hook(stop_after_last_requested_layer)
        )

    # ------------------------------------------------------------------
    # Branch feature and score helpers
    # ------------------------------------------------------------------
    def _adapt(self, features: torch.Tensor, branch: str) -> torch.Tensor:
        if self.pre_proj <= 0:
            return features
        if branch == "local":
            return self.pre_projection(features)
        if branch == "global":
            return self.global_projection(features)
        raise ValueError(f"Unknown branch: {branch}")

    def _branch_discriminator(self, branch: str) -> torch.nn.Module:
        if branch == "local":
            return self.discriminator
        if branch == "global":
            return self.global_discriminator
        raise ValueError(f"Unknown branch: {branch}")

    def _set_inference_mode(self):
        self.forward_modules.eval()
        self.discriminator.eval()
        self.global_discriminator.eval()
        if self.pre_proj > 0:
            self.pre_projection.eval()
            self.global_projection.eval()

    def _embed_and_adapt(
        self, images: torch.Tensor, branch: str, evaluation: bool
    ) -> Tuple[torch.Tensor, List[List[int]]]:
        features, patch_shapes = self._embed(
            images,
            provide_patch_shapes=True,
            evaluation=evaluation,
        )
        return self._adapt(features, branch), patch_shapes

    def _sample_gaussian_noise(
        self, features: torch.Tensor, branch: str
    ) -> torch.Tensor:
        base_std = (
            self.local_noise_std if branch == "local" else self.global_noise_std
        )
        if self.mix_noise <= 1:
            return torch.randn_like(features) * base_std

        noise_level = torch.randint(
            0, self.mix_noise, (features.shape[0],), device=features.device
        )
        scales = base_std * torch.pow(
            torch.tensor(1.1, device=features.device), noise_level
        )
        return torch.randn_like(features) * scales.unsqueeze(1)

    def _branch_loss_from_features(
        self, raw_features: torch.Tensor, branch: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        true_features = self._adapt(raw_features, branch)
        fake_features = true_features + self._sample_gaussian_noise(
            true_features, branch
        )
        discriminator = self._branch_discriminator(branch)
        scores = discriminator(torch.cat([true_features, fake_features], dim=0))
        true_scores = scores[: len(true_features)]
        fake_scores = scores[len(true_features) :]

        margin = self.dsc_margin
        true_loss = torch.clamp(-true_scores + margin, min=0).mean()
        fake_loss = torch.clamp(fake_scores + margin, min=0).mean()
        return true_loss + fake_loss, -true_scores.detach().flatten()

    def _raw_score_grid_from_features(
        self,
        raw_features: torch.Tensor,
        patch_shapes: List[List[int]],
        batch_size: int,
        branch: str,
    ) -> torch.Tensor:
        features = self._adapt(raw_features, branch)
        scores = -self._branch_discriminator(branch)(features)
        patch_h, patch_w = patch_shapes[0]
        return scores.reshape(batch_size, patch_h, patch_w)

    def _raw_score_grid(
        self, images: torch.Tensor, branch: str
    ) -> torch.Tensor:
        images = images.to(self.device, dtype=torch.float32, non_blocking=True)
        raw_features, patch_shapes = self._embed(
            images, provide_patch_shapes=True, evaluation=True
        )
        return self._raw_score_grid_from_features(
            raw_features, patch_shapes, images.shape[0], branch
        )

    def _score_maps_from_grid(
        self,
        score_grid: torch.Tensor,
        input_size: Tuple[int, int],
        branch: str,
        normalize: bool,
    ) -> torch.Tensor:
        maps = F.interpolate(
            score_grid.unsqueeze(1),
            size=input_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        if self.score_smoothing > 1:
            kernel = self.score_smoothing
            if kernel % 2 == 0:
                kernel += 1
            maps = F.avg_pool2d(
                maps.unsqueeze(1), kernel, stride=1, padding=kernel // 2
            ).squeeze(1)

        if normalize:
            median = getattr(self, f"{branch}_score_median")
            scale = getattr(self, f"{branch}_score_scale")
            maps = (maps - median) / scale.clamp_min(1e-6)
        return maps

    def _score_maps(
        self,
        images: torch.Tensor,
        branch: str,
        normalize: bool = True,
    ) -> torch.Tensor:
        input_size = images.shape[-2:]
        score_grid = self._raw_score_grid(images, branch)
        return self._score_maps_from_grid(
            score_grid, input_size, branch, normalize
        )

    # ------------------------------------------------------------------
    # Training and checkpointing
    # ------------------------------------------------------------------
    @staticmethod
    def _get_training_views(
        data_item: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        if "global_image" in data_item and "local_image" in data_item:
            return data_item["global_image"], data_item["local_image"], False
        image = data_item["image"]
        return image, image, True

    def _zero_branch_optimizers(self):
        self.dsc_opt.zero_grad()
        self.global_dsc_opt.zero_grad()
        if self.pre_proj > 0:
            self.proj_opt.zero_grad()
            self.global_proj_opt.zero_grad()
        if self.train_backbone:
            self.backbone_opt.zero_grad()

    def _step_branch_optimizers(self):
        if self.pre_proj > 0:
            self.proj_opt.step()
            self.global_proj_opt.step()
        self.dsc_opt.step()
        self.global_dsc_opt.step()
        if self.train_backbone:
            self.backbone_opt.step()

    def _append_calibration_samples(
        self, target: List[torch.Tensor], samples: torch.Tensor
    ):
        current = sum(x.numel() for x in target)
        remaining = self.calibration_sample_limit - current
        if remaining <= 0:
            return
        if samples.numel() > remaining:
            indices = torch.randperm(samples.numel(), device=samples.device)[:remaining]
            samples = samples[indices]
        target.append(samples.float().cpu())

    def _train_global_local(self, training_data: Iterable) -> Dict[str, List[torch.Tensor]]:
        self.forward_modules.eval()
        self.discriminator.train()
        self.global_discriminator.train()
        if self.pre_proj > 0:
            self.pre_projection.train()
            self.global_projection.train()

        calibration = {"local": [], "global": []}
        total_steps = self.meta_epochs * self.gan_epochs
        progress = tqdm.tqdm(total=total_steps, desc="Training global/local heads")

        for _meta_epoch in range(self.meta_epochs):
            for _gan_epoch in range(self.gan_epochs):
                epoch_losses = []
                for data_item in training_data:
                    global_images, local_images, reuse_features = self._get_training_views(
                        data_item
                    )
                    global_images = global_images.to(
                        self.device, dtype=torch.float32, non_blocking=True
                    )
                    local_images = local_images.to(
                        self.device, dtype=torch.float32, non_blocking=True
                    )
                    self._zero_branch_optimizers()

                    if reuse_features:
                        raw_features = self._embed(
                            local_images, evaluation=False
                        )[0]
                        local_loss, local_scores = self._branch_loss_from_features(
                            raw_features, "local"
                        )
                        global_loss, global_scores = self._branch_loss_from_features(
                            raw_features, "global"
                        )
                    else:
                        local_raw = self._embed(
                            local_images, evaluation=False
                        )[0]
                        global_raw = self._embed(
                            global_images, evaluation=False
                        )[0]
                        local_loss, local_scores = self._branch_loss_from_features(
                            local_raw, "local"
                        )
                        global_loss, global_scores = self._branch_loss_from_features(
                            global_raw, "global"
                        )

                    loss = (
                        self.local_loss_weight * local_loss
                        + self.global_loss_weight * global_loss
                    )
                    loss.backward()
                    self._step_branch_optimizers()
                    epoch_losses.append(float(loss.detach().cpu()))

                    if (
                        _meta_epoch == self.meta_epochs - 1
                        and _gan_epoch == self.gan_epochs - 1
                    ):
                        self._append_calibration_samples(
                            calibration["local"], local_scores
                        )
                        self._append_calibration_samples(
                            calibration["global"], global_scores
                        )

                mean_loss = float(np.mean(epoch_losses)) if epoch_losses else math.nan
                progress.set_postfix(loss=f"{mean_loss:.5f}")
                progress.update(1)
        progress.close()
        return calibration

    def _set_score_calibration(self, calibration: Dict[str, List[torch.Tensor]]):
        for branch in ("local", "global"):
            if not calibration[branch]:
                continue
            values = torch.cat(calibration[branch])
            median = torch.median(values)
            mad = torch.median(torch.abs(values - median)) * 1.4826
            if not torch.isfinite(mad) or mad < 1e-6:
                mad = torch.std(values).clamp_min(1e-6)
            getattr(self, f"{branch}_score_median").copy_(median.to(self.device))
            getattr(self, f"{branch}_score_scale").copy_(mad.to(self.device))

    def _checkpoint_state(self) -> Dict[str, object]:
        state = {
            "local_discriminator": OrderedDict(
                (k, v.detach().cpu()) for k, v in self.discriminator.state_dict().items()
            ),
            "global_discriminator": OrderedDict(
                (k, v.detach().cpu())
                for k, v in self.global_discriminator.state_dict().items()
            ),
            "calibration": {
                "local_median": float(self.local_score_median.cpu()),
                "local_scale": float(self.local_score_scale.cpu()),
                "global_median": float(self.global_score_median.cpu()),
                "global_scale": float(self.global_score_scale.cpu()),
            },
            "config": {
                "tile_batch_size": self.tile_batch_size,
                "global_noise_std": self.global_noise_std,
                "local_noise_std": self.local_noise_std,
            },
        }
        if self.pre_proj > 0:
            state["local_projection"] = OrderedDict(
                (k, v.detach().cpu()) for k, v in self.pre_projection.state_dict().items()
            )
            state["global_projection"] = OrderedDict(
                (k, v.detach().cpu())
                for k, v in self.global_projection.state_dict().items()
            )
        if self.train_backbone:
            state["backbone"] = OrderedDict(
                (k, v.detach().cpu()) for k, v in self.backbone.state_dict().items()
            )
        return state

    def _load_checkpoint(self, checkpoint_path: str):
        state = torch.load(checkpoint_path, map_location=self.device)
        self.discriminator.load_state_dict(state["local_discriminator"])
        self.global_discriminator.load_state_dict(state["global_discriminator"])
        if self.pre_proj > 0:
            self.pre_projection.load_state_dict(state["local_projection"])
            self.global_projection.load_state_dict(state["global_projection"])
        if self.train_backbone and "backbone" in state:
            self.backbone.load_state_dict(state["backbone"])
        calibration = state.get("calibration", {})
        for name, buffer_name in (
            ("local_median", "local_score_median"),
            ("local_scale", "local_score_scale"),
            ("global_median", "global_score_median"),
            ("global_scale", "global_score_scale"),
        ):
            if name in calibration:
                getattr(self, buffer_name).fill_(float(calibration[name]))

    def train(self, training_data, test_data=None):
        """Train fixed epochs and evaluate the test set once at the end."""
        checkpoint_path = os.path.join(self.ckpt_dir, "global_local_ckpt.pth")
        if os.path.exists(checkpoint_path):
            LOGGER.info("Loading global/local checkpoint: %s", checkpoint_path)
            self._load_checkpoint(checkpoint_path)
        else:
            calibration = self._train_global_local(training_data)
            self._set_score_calibration(calibration)
            torch.save(self._checkpoint_state(), checkpoint_path)

        if test_data is None:
            return math.nan, math.nan, math.nan
        return self.evaluate(test_data)

    # ------------------------------------------------------------------
    # High-resolution inference
    # ------------------------------------------------------------------
    def _tile_weight(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        key = (height, width, str(device))
        if key not in self._weight_cache:
            wy = torch.hann_window(height, periodic=False, device=device).clamp_min(0.05)
            wx = torch.hann_window(width, periodic=False, device=device).clamp_min(0.05)
            self._weight_cache[key] = torch.outer(wy, wx)
        return self._weight_cache[key]

    def _stitch_local_maps(
        self,
        local_tiles: torch.Tensor,
        tile_boxes: torch.Tensor,
        original_size: Tuple[int, int],
    ) -> torch.Tensor:
        original_h, original_w = original_size
        score_sum = torch.zeros((original_h, original_w), device=self.device)
        weight_sum = torch.zeros_like(score_sum)
        local_tiles = local_tiles.squeeze(0) if local_tiles.ndim == 5 else local_tiles
        tile_boxes = tile_boxes.squeeze(0) if tile_boxes.ndim == 3 else tile_boxes

        for start in range(0, len(local_tiles), self.tile_batch_size):
            end = min(start + self.tile_batch_size, len(local_tiles))
            tile_batch = local_tiles[start:end]
            if tile_batch.dtype == torch.uint8:
                tile_batch = tile_batch.to(
                    self.device, dtype=torch.float32, non_blocking=True
                ).div_(255.0)
                mean = tile_batch.new_tensor(
                    (0.485, 0.456, 0.406)
                )[None, :, None, None]
                std = tile_batch.new_tensor(
                    (0.229, 0.224, 0.225)
                )[None, :, None, None]
                tile_batch = (tile_batch - mean) / std
            tile_maps = self._score_maps(tile_batch, "local", normalize=True)
            for tile_map, box in zip(tile_maps, tile_boxes[start:end]):
                x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                box_h, box_w = y2 - y1, x2 - x1
                resized = F.interpolate(
                    tile_map[None, None],
                    size=(box_h, box_w),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                weight = self._tile_weight(box_h, box_w, resized.device)
                score_sum[y1:y2, x1:x2] += resized * weight
                weight_sum[y1:y2, x1:x2] += weight
        return score_sum / weight_sum.clamp_min(1e-6)

    @staticmethod
    def _original_size_from_batch(data_item: Dict[str, object]) -> Tuple[int, int]:
        size = data_item["original_size"]
        if isinstance(size, torch.Tensor):
            size = size.squeeze(0).tolist()
        elif isinstance(size, (list, tuple)) and len(size) == 2:
            size = [v.item() if isinstance(v, torch.Tensor) else v for v in size]
        return int(size[0]), int(size[1])

    @torch.inference_mode()
    def _predict_highres_item(self, data_item: Dict[str, object]) -> Tuple[float, np.ndarray]:
        self._set_inference_mode()
        original_size = self._original_size_from_batch(data_item)
        global_image = data_item["global_image"]
        local_tiles = data_item["local_tiles"]
        tile_boxes = data_item["tile_boxes"]

        global_map = self._score_maps(global_image, "global", normalize=True)
        global_map = F.interpolate(
            global_map.unsqueeze(1),
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        local_map = self._stitch_local_maps(
            local_tiles, tile_boxes, original_size
        )
        fused_map = torch.maximum(global_map, local_map)
        image_score = float(torch.max(fused_map).cpu())
        return image_score, fused_map.float().cpu().numpy()

    @torch.inference_mode()
    def _predict_legacy_batch(
        self, images: torch.Tensor
    ) -> Tuple[List[float], List[np.ndarray]]:
        self._set_inference_mode()
        images = images.to(self.device, dtype=torch.float32, non_blocking=True)
        raw_features, patch_shapes = self._embed(
            images, provide_patch_shapes=True, evaluation=True
        )
        local_grid = self._raw_score_grid_from_features(
            raw_features, patch_shapes, images.shape[0], "local"
        )
        global_grid = self._raw_score_grid_from_features(
            raw_features, patch_shapes, images.shape[0], "global"
        )
        local_maps = self._score_maps_from_grid(
            local_grid, images.shape[-2:], "local", normalize=True
        )
        global_maps = self._score_maps_from_grid(
            global_grid, images.shape[-2:], "global", normalize=True
        )
        fused = torch.maximum(local_maps, global_maps)
        scores = fused.flatten(1).max(1).values.cpu().tolist()
        maps = [m.float().cpu().numpy() for m in fused]
        if self.legacy_gaussian_sigma > 0:
            maps = [
                ndimage.gaussian_filter(m, sigma=self.legacy_gaussian_sigma)
                for m in maps
            ]
        return scores, maps

    def _iter_predictions(self, dataloader: Iterable) -> Iterator[Dict[str, object]]:
        self._set_inference_mode()

        for data_item in tqdm.tqdm(dataloader, desc="Global/local inference", leave=False):
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            inference_start = time.perf_counter()
            highres = "global_image" in data_item and "local_tiles" in data_item
            if highres:
                if data_item["global_image"].shape[0] != 1:
                    raise ValueError("High-resolution inference requires test batch_size=1.")
                score, anomaly_map = self._predict_highres_item(data_item)
                scores, maps = [score], [anomaly_map]
            else:
                scores, maps = self._predict_legacy_batch(data_item["image"])
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self._model_inference_seconds += time.perf_counter() - inference_start
            self._model_inference_images += len(scores)

            labels = data_item.get("is_anomaly")
            labels = labels.cpu().tolist() if isinstance(labels, torch.Tensor) else labels
            labels = labels if isinstance(labels, list) else [labels]
            anomalies = data_item.get("anomaly", [""] * len(scores))
            anomalies = anomalies if isinstance(anomalies, list) else [anomalies]

            masks = data_item.get("mask")
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
                if masks.ndim == 4:
                    masks = masks[:, 0]
            mask_valid = data_item.get("mask_valid", True)
            if isinstance(mask_valid, torch.Tensor):
                mask_valid = bool(mask_valid.flatten()[0].item())

            for index, (score, anomaly_map) in enumerate(zip(scores, maps)):
                mask = None
                if mask_valid and masks is not None:
                    mask = masks[index]
                yield {
                    "score": score,
                    "map": anomaly_map,
                    "label": int(labels[index]),
                    "anomaly": anomalies[index],
                    "mask": mask,
                }

    def predict(self, data, prefix=""):
        if not isinstance(data, torch.utils.data.DataLoader):
            scores, maps = self._predict_legacy_batch(data)
            return scores, maps, []

        scores, maps, labels, masks, anomalies = [], [], [], [], []
        for prediction in self._iter_predictions(data):
            scores.append(prediction["score"])
            maps.append(prediction["map"])
            labels.append(prediction["label"])
            anomalies.append(prediction["anomaly"])
            if prediction["mask"] is not None:
                masks.append(prediction["mask"])
        return scores, maps, [], labels, masks, anomalies

    @staticmethod
    def _histogram_pixel_auroc(
        positive_hist: np.ndarray, negative_hist: np.ndarray
    ) -> float:
        positives = positive_hist.sum()
        negatives = negative_hist.sum()
        if positives == 0 or negatives == 0:
            return math.nan
        # Threshold moves from high anomaly score to low anomaly score.
        tpr = np.cumsum(positive_hist[::-1]) / positives
        fpr = np.cumsum(negative_hist[::-1]) / negatives
        tpr = np.concatenate([[0.0], tpr])
        fpr = np.concatenate([[0.0], fpr])
        return float(np.trapz(tpr, fpr))

    def evaluate(self, dataloader) -> Tuple[float, float, float]:
        start_time = time.perf_counter()
        self._model_inference_seconds = 0.0
        self._model_inference_images = 0
        scores: List[float] = []
        labels: List[int] = []
        positive_hist = np.zeros(self.pixel_histogram_bins, dtype=np.int64)
        negative_hist = np.zeros_like(positive_hist)
        exact_pixel_scores: List[np.ndarray] = []
        exact_pixel_labels: List[np.ndarray] = []
        pixel_min, pixel_max = -12.0, 12.0
        image_count = 0

        for prediction in self._iter_predictions(dataloader):
            image_count += 1
            scores.append(prediction["score"])
            labels.append(prediction["label"])
            mask = prediction["mask"]
            if mask is None:
                continue
            anomaly_map = prediction["map"]
            if anomaly_map.shape != mask.shape:
                resized_mask = F.interpolate(
                    torch.from_numpy(mask)[None, None].float(),
                    size=anomaly_map.shape,
                    mode="nearest",
                )[0, 0].numpy()
            else:
                resized_mask = mask
            clipped = np.clip(anomaly_map, pixel_min, pixel_max)
            bin_ids = ((clipped - pixel_min) / (pixel_max - pixel_min)
                       * (self.pixel_histogram_bins - 1)).astype(np.int64)
            positive_hist += np.bincount(
                bin_ids[resized_mask > 0].ravel(), minlength=self.pixel_histogram_bins
            )
            negative_hist += np.bincount(
                bin_ids[resized_mask <= 0].ravel(), minlength=self.pixel_histogram_bins
            )
            if self.exact_pixel_auroc:
                exact_pixel_scores.append(anomaly_map.astype(np.float32).ravel())
                exact_pixel_labels.append((resized_mask > 0).astype(np.uint8).ravel())

        elapsed = time.perf_counter() - start_time
        self.last_inference_stats = {
            "images": image_count,
            "model_seconds": self._model_inference_seconds,
            "seconds_per_image": self._model_inference_seconds
            / max(1, self._model_inference_images),
            "images_per_second": self._model_inference_images
            / max(self._model_inference_seconds, 1e-12),
            "end_to_end_seconds": elapsed,
            "end_to_end_seconds_per_image": elapsed / max(1, image_count),
        }

        if len(set(labels)) < 2:
            image_auroc = math.nan
        else:
            image_auroc = float(sklearn_metrics.roc_auc_score(labels, scores))
        if self.exact_pixel_auroc and exact_pixel_scores:
            flat_labels = np.concatenate(exact_pixel_labels)
            flat_scores = np.concatenate(exact_pixel_scores)
            pixel_auroc = (
                float(sklearn_metrics.roc_auc_score(flat_labels, flat_scores))
                if np.unique(flat_labels).size == 2
                else math.nan
            )
            pixel_mode = "exact"
        else:
            pixel_auroc = self._histogram_pixel_auroc(
                positive_hist, negative_hist
            )
            pixel_mode = f"hist{self.pixel_histogram_bins}"
        LOGGER.info(
            "I-AUROC=%.5f P-AUROC(%s)=%.5f latency=%.4fs/image",
            image_auroc,
            pixel_mode,
            pixel_auroc,
            self.last_inference_stats["seconds_per_image"],
        )
        return image_auroc, pixel_auroc, -1.0


# Backwards-friendly alias used by the existing launcher naming convention.
SimpleNet = GlobalLocalSimpleNet

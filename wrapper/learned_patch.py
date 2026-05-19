from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
    from ultralytics import YOLO
except ImportError:
    torch = None
    F = None
    YOLO = None


@dataclass
class PatchTrainingConfig:
    model_path: str
    patch_size: int = 64
    image_size: int = 640
    phone_class_id: int = 67
    learning_rate: float = 0.05
    steps: int = 300
    batch_size: int = 4
    min_scale: float = 0.6
    max_scale: float = 1.4
    placement: str = "random"
    patch_shape: str = "circle"
    seed: int = 42
    topk_anchors: int = 50
    device: str | None = None


class LearnedPatchTrainer:
    """
    Train a learnable adversarial patch directly against a local YOLO model.

    The current objective is a practical first surrogate for your project:
    minimize the model's confidence for the `cell phone` class on patched images.
    """

    def __init__(self, config: PatchTrainingConfig):
        if torch is None or YOLO is None:
            raise ImportError(
                "learned_patch.py requires torch and ultralytics installed in the current Python environment."
            )

        self.config = config
        self.device = self._resolve_device(config.device)
        self.rng = random.Random(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self.yolo = YOLO(config.model_path)
        self.model = self.yolo.model
        self.model.to(self.device)

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.num_classes = len(getattr(self.yolo, "names", {})) or 80
        self.patch = torch.nn.Parameter(
            torch.rand(1, 3, config.patch_size, config.patch_size, device=self.device)
        )
        self.optimizer = torch.optim.Adam([self.patch], lr=config.learning_rate)
        self.history: list[dict[str, float]] = []

    def _resolve_device(self, requested: str | None) -> str:
        if requested:
            return requested
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_image_tensor(self, image_path: str | Path) -> "torch.Tensor":
        image = Image.open(image_path).convert("RGB")
        image = image.resize(
            (self.config.image_size, self.config.image_size),
            Image.Resampling.BILINEAR,
        )
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _build_base_mask(self) -> "torch.Tensor":
        size = self.config.patch_size
        if self.config.patch_shape == "square":
            return torch.ones((1, 1, size, size), device=self.device)

        yy, xx = torch.meshgrid(
            torch.arange(size, device=self.device),
            torch.arange(size, device=self.device),
            indexing="ij",
        )
        center = (size - 1) / 2.0
        radius = size / 2.0
        mask = (((xx - center) ** 2 + (yy - center) ** 2) <= radius**2).float()
        return mask.unsqueeze(0).unsqueeze(0)

    def _sample_patch_geometry(self, image_h: int, image_w: int) -> tuple[int, int, int]:
        scale = self.rng.uniform(self.config.min_scale, self.config.max_scale)
        patch_size = max(8, min(int(round(self.config.patch_size * scale)), min(image_h, image_w)))

        if self.config.placement == "center":
            x = max(0, (image_w - patch_size) // 2)
            y = max(0, (image_h - patch_size) // 2)
            return x, y, patch_size

        x = self.rng.randint(0, max(0, image_w - patch_size))
        y = self.rng.randint(0, max(0, image_h - patch_size))
        return x, y, patch_size

    def _apply_patch_batch(self, images: "torch.Tensor") -> "torch.Tensor":
        patched = images.clone()
        _, _, image_h, image_w = patched.shape

        base_mask = self._build_base_mask()

        for batch_idx in range(patched.shape[0]):
            x, y, patch_size = self._sample_patch_geometry(image_h, image_w)

            resized_patch = F.interpolate(
                self.patch,
                size=(patch_size, patch_size),
                mode="bilinear",
                align_corners=False,
            )
            resized_mask = F.interpolate(
                base_mask,
                size=(patch_size, patch_size),
                mode="bilinear",
                align_corners=False,
            )

            roi = patched[batch_idx : batch_idx + 1, :, y : y + patch_size, x : x + patch_size]
            patched_roi = (1.0 - resized_mask) * roi + resized_mask * resized_patch
            patched[batch_idx : batch_idx + 1, :, y : y + patch_size, x : x + patch_size] = patched_roi

        return patched.clamp(0.0, 1.0)

    def _extract_phone_scores(self, raw_output) -> "torch.Tensor":
        """
        Extract phone-class scores from the raw Ultralytics output in a way
        that preserves gradient flow to the patch.
        """
        if isinstance(raw_output, dict):
            one2many = raw_output.get("one2many")
            if isinstance(one2many, dict):
                scores = one2many.get("scores")
                if isinstance(scores, torch.Tensor) and scores.ndim == 3:
                    # Expected layout here is [B, num_classes, num_anchors]
                    return scores[:, self.config.phone_class_id, :]

        raise ValueError(
            "Could not extract gradient-preserving phone scores from YOLO output. "
            "Expected raw_output['one2many']['scores'] to be a tensor."
        )

    def _phone_score_loss(self, raw_output) -> "torch.Tensor":
        phone_scores = self._extract_phone_scores(raw_output)

        # Scores in the raw head are logits, so we map them to probabilities.
        phone_scores = torch.sigmoid(phone_scores)
        topk = min(self.config.topk_anchors, phone_scores.shape[-1])
        strongest_scores = torch.topk(phone_scores, k=topk, dim=-1).values
        return strongest_scores.mean()

    def _sample_batch(self, image_paths: list[str | Path]) -> "torch.Tensor":
        batch_paths = [self.rng.choice(image_paths) for _ in range(self.config.batch_size)]
        tensors = [self._load_image_tensor(path) for path in batch_paths]
        return torch.cat(tensors, dim=0)

    def train(self, image_paths: Iterable[str | Path]) -> list[dict[str, float]]:
        image_paths = [Path(path) for path in image_paths]
        if not image_paths:
            raise ValueError("No training images were provided for patch optimization.")

        # We need training-mode raw outputs from the detection head, but we still
        # keep model weights frozen so only the patch is updated.
        self.model.train()

        for step in range(1, self.config.steps + 1):
            batch = self._sample_batch(image_paths)
            patched_batch = self._apply_patch_batch(batch)

            self.optimizer.zero_grad()
            raw_output = self.model(patched_batch)
            loss = self._phone_score_loss(raw_output)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                self.patch.data.clamp_(0.0, 1.0)

            history_item = {
                "step": float(step),
                "loss": float(loss.detach().cpu().item()),
            }
            self.history.append(history_item)

            if step % 25 == 0 or step == 1 or step == self.config.steps:
                print(f"[patch-train] step {step:04d}/{self.config.steps} loss={history_item['loss']:.6f}")

        return self.history

    def save_patch(self, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        patch_image = (
            self.patch.detach()
            .cpu()
            .squeeze(0)
            .permute(1, 2, 0)
            .numpy()
            * 255.0
        ).clip(0, 255).astype(np.uint8)

        Image.fromarray(patch_image).save(output_path)
        return output_path

    def save_metadata(self, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "config": self.config.__dict__,
            "history": self.history,
        }

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)

        return output_path

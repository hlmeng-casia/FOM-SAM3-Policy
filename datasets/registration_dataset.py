"""Registration image contract for FO Memory learning.

Each JSONL row is either a positive image with a binary mask and xyxy box, or
an absent/hard-negative image supervised only by SAM3's native presence head.
No private dataset-specific object list or augmentation pipeline is assumed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class RegistrationSample:
    image_path: Path
    presence: bool
    mask_path: Path | None = None
    box_xyxy: tuple[float, float, float, float] | None = None

    def load_image(self) -> Image.Image:
        with Image.open(self.image_path) as image:
            return image.convert("RGB")

    def load_mask(self) -> np.ndarray | None:
        if self.mask_path is None:
            return None
        with Image.open(self.mask_path) as image:
            mask = np.asarray(image.convert("L")) > 0
        if not bool(mask.any()):
            raise ValueError(f"positive mask is empty: {self.mask_path}")
        return mask


class RegistrationDataset(Dataset[RegistrationSample]):
    """Strict JSONL loader with paths relative to the configured data root."""

    def __init__(self, root: str | Path, annotations: str | Path) -> None:
        self.root = Path(root).resolve()
        annotation_path = Path(annotations)
        if not annotation_path.is_absolute():
            annotation_path = self.root / annotation_path
        rows = [
            json.loads(line)
            for line in annotation_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError(f"registration annotations are empty: {annotation_path}")
        samples: list[RegistrationSample] = []
        for index, row in enumerate(rows):
            presence = bool(row.get("presence", True))
            image_value = row.get("image")
            if not image_value:
                raise ValueError(f"row {index} has no image")
            image_path = (self.root / str(image_value)).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            mask_path = None
            box = None
            if presence:
                if not row.get("mask") or not isinstance(row.get("box_xyxy"), list):
                    raise ValueError(
                        f"positive row {index} requires mask and box_xyxy"
                    )
                if len(row["box_xyxy"]) != 4:
                    raise ValueError(f"row {index} box_xyxy must have four values")
                mask_path = (self.root / str(row["mask"])).resolve()
                if not mask_path.is_file():
                    raise FileNotFoundError(mask_path)
                box = tuple(map(float, row["box_xyxy"]))
            elif row.get("mask") or row.get("box_xyxy"):
                raise ValueError(
                    f"absent row {index} must not declare mask or box_xyxy"
                )
            samples.append(RegistrationSample(image_path, presence, mask_path, box))
        if not any(sample.presence for sample in samples):
            raise ValueError("registration set requires at least one positive image")
        self.samples = tuple(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> RegistrationSample:
        return self.samples[index]


def normalized_box_from_mask(mask: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a non-empty HW mask to normalized cxcywh."""
    selector = torch.from_numpy(np.asarray(mask, dtype=bool))
    ys, xs = torch.nonzero(selector, as_tuple=True)
    if xs.numel() == 0:
        raise ValueError("cannot create a box from an empty mask")
    height, width = selector.shape
    x1, x2 = xs.min().float(), xs.max().float() + 1
    y1, y2 = ys.min().float(), ys.max().float() + 1
    return torch.stack(
        [
            (x1 + x2) / (2 * width),
            (y1 + y2) / (2 * height),
            (x2 - x1) / width,
            (y2 - y1) / height,
        ]
    ).to(device=device)

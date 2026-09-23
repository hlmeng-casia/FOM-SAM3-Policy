"""Minimal cached SAM3 feature/mask/box dataset for policy training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class PolicyCacheDataset(Dataset[dict[str, torch.Tensor]]):
    """Load one `.pt` sample per manifest row and enforce the public shapes."""

    def __init__(
        self,
        root: str | Path,
        manifest: str | Path,
        *,
        num_cameras: int = 2,
        objects_per_camera: int = 2,
        action_steps: int,
        feature_channels: int = 256,
        feature_hw: tuple[int, int] = (24, 24),
        state_history: int = 2,
        state_dim: int = 10,
        action_dim: int = 10,
    ) -> None:
        self.root = Path(root).resolve()
        manifest_path = Path(manifest)
        if not manifest_path.is_absolute():
            manifest_path = self.root / manifest_path
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = value.get("samples") if isinstance(value, dict) else value
        if not isinstance(rows, list) or not rows:
            raise ValueError("policy cache manifest requires a non-empty samples list")
        self.paths = tuple((self.root / str(row)).resolve() for row in rows)
        missing = [path for path in self.paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(missing[0])
        height, width = map(int, feature_hw)
        self.expected = {
            "features": (num_cameras, feature_channels, height, width),
            "masks": (num_cameras, objects_per_camera, height, width),
            "boxes": (num_cameras, objects_per_camera, 4),
            "state": (state_history, state_dim),
            "actions": (action_steps, action_dim),
        }

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        payload: Any = torch.load(self.paths[index], map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError(f"cache sample must be a dict: {self.paths[index]}")
        output: dict[str, torch.Tensor] = {}
        for key, expected_shape in self.expected.items():
            if key not in payload:
                raise KeyError(f"{self.paths[index]} is missing {key}")
            value = torch.as_tensor(payload[key], dtype=torch.float32)
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"{self.paths[index]} {key} shape {tuple(value.shape)} != {expected_shape}"
                )
            output[key] = value
        if not bool(torch.isfinite(output["features"]).all()):
            raise ValueError(f"non-finite SAM3 feature: {self.paths[index]}")
        output["masks"] = output["masks"].clamp(0, 1)
        output["boxes"] = output["boxes"].clamp(0, 1)
        return output

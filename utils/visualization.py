"""Minimal localization visualization used by the release demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def save_overlay(
    image: Image.Image,
    mask: np.ndarray,
    box_cxcywh: np.ndarray,
    score: float,
    output: str | Path,
) -> None:
    """Save a red mask overlay and normalized box without GUI dependencies."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    selector = np.asarray(mask, dtype=bool)
    overlay = rgb.copy()
    overlay[selector] = (0.45 * overlay[selector] + 0.55 * np.array([255, 40, 40])).astype(
        np.uint8
    )
    rendered = Image.fromarray(overlay)
    width, height = rendered.size
    cx, cy, box_w, box_h = map(float, box_cxcywh)
    xyxy = (
        (cx - box_w / 2) * width,
        (cy - box_h / 2) * height,
        (cx + box_w / 2) * width,
        (cy + box_h / 2) * height,
    )
    draw = ImageDraw.Draw(rendered)
    draw.rectangle(xyxy, outline=(50, 255, 80), width=3)
    draw.text((xyxy[0], max(0, xyxy[1] - 14)), f"{score:.3f}", fill=(50, 255, 80))
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(destination)

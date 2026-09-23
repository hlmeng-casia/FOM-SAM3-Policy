"""Configuration, hashing, and reproducibility helpers."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


RELEASE_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load one YAML config and return it with its absolute source path."""
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"config must contain a mapping: {source}")
    return value, source


def resolve_path(value: str | Path, config_path: str | Path) -> Path:
    """Resolve a config path relative to the release root, never the shell cwd."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (RELEASE_ROOT / path).resolve()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

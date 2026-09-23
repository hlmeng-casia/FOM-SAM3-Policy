#!/usr/bin/env python3
"""Fold a train-time Private Linear checkpoint into a reusable FO Memory."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if "clean_release" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "clean_release", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot initialize the release package")
    package = importlib.util.module_from_spec(spec)
    sys.modules["clean_release"] = package
    spec.loader.exec_module(package)

from clean_release.models.fo_memory import (  # noqa: E402
    PrivateLinearFOMemory,
    build_memory_index,
    save_memory_artifact,
)
from clean_release.utils.misc import load_config, resolve_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to memory.yaml")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    run_dir = resolve_path(config["output"]["run_dir"], config_path)
    checkpoint_path = run_dir / str(config["output"]["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("method") != PrivateLinearFOMemory.method_name:
        raise ValueError("checkpoint is not a Private Linear FO Memory")
    state = checkpoint["state_dict"]
    model = PrivateLinearFOMemory(
        state["base_text"],
        state["text_mask"],
        state["geometry_tokens"],
        state["geometry_mask"],
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    prompt = str(checkpoint["prompt"])
    artifact_dir = run_dir / str(config["output"]["artifact_dir"])
    sam_config = config["sam3"]
    record = save_memory_artifact(
        artifact_dir,
        model=model,
        prompt=prompt,
        sam_checkpoint=resolve_path(sam_config["checkpoint"], config_path),
        model_resolution=int(sam_config["model_resolution"]),
    )
    configured_records = config["output"].get("artifact_records", [])
    records = [record, *[resolve_path(value, config_path) for value in configured_records]]
    bank_dir = resolve_path(config["output"]["bank_dir"], config_path)
    index = build_memory_index(records, bank_dir / "index.json")
    print(f"Exported static FO Memory: {record}")
    print(f"Updated memory index: {index}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run frozen SAM3 localization from one reusable FO Memory."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from PIL import Image

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

from clean_release.models.fo_memory import load_memory_artifact, lookup_memory  # noqa: E402
from clean_release.sam3.interface import FrozenSAM3  # noqa: E402
from clean_release.utils.misc import atomic_json, load_config, resolve_path  # noqa: E402
from clean_release.utils.visualization import save_overlay  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to sam3.yaml")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    sam = config["sam3"]
    runner = FrozenSAM3(
        source=resolve_path(sam["source"], config_path),
        checkpoint=resolve_path(sam["checkpoint"], config_path),
        device=str(config["device"]),
        model_resolution=int(sam["model_resolution"]),
        use_fa3=bool(sam["use_fa3"]),
        compile_model=bool(sam["compile"]),
    )
    record = lookup_memory(
        resolve_path(config["memory"]["bank_index"], config_path),
        str(config["memory"]["target_prompt"]),
    )
    artifact = load_memory_artifact(
        record,
        checkpoint_sha256=runner.checkpoint_sha256,
        model_resolution=runner.model_resolution,
    )
    image_path = resolve_path(config["inference"]["images"][0], config_path)
    with Image.open(image_path) as value:
        image = value.convert("RGB")
    inference = config["inference"]
    result = runner.localize(
        image,
        artifact,
        score_threshold=float(inference["score_threshold"]),
        mask_threshold=float(inference["mask_threshold"]),
        mask_nms_iou=float(inference["mask_nms_iou"]),
    )
    output_dir = resolve_path(inference["output_dir"], config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "prompt": artifact.prompt,
        "memory_id": artifact.memory_id,
        "present": result.present,
        "score": result.score,
        "box_cxcywh": None if result.box_cxcywh is None else result.box_cxcywh.tolist(),
        "query_index": result.query_index,
        "rejection_reason": result.rejection_reason,
    }
    atomic_json(output_dir / "localization.json", summary)
    if result.present:
        assert result.mask is not None and result.box_cxcywh is not None
        save_overlay(
            image,
            result.mask,
            result.box_cxcywh,
            result.score,
            output_dir / "localization.png",
        )
    runner.close()
    print(summary)


if __name__ == "__main__":
    main()

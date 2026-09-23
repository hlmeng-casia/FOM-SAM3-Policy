#!/usr/bin/env python3
"""End-to-end FO Memory -> SAM3 localization -> FSAE policy condition demo."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
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
from clean_release.models.policy_adapter import (  # noqa: E402
    ACTPolicyAdapter,
    DiffusionPolicyAdapter,
    build_policy,
)
from clean_release.sam3.interface import FrozenSAM3  # noqa: E402
from clean_release.utils.misc import atomic_json, load_config, resolve_path  # noqa: E402


def _runner(config: dict, config_path: Path, resolution: int) -> FrozenSAM3:
    sam = config["sam3"]
    return FrozenSAM3(
        source=resolve_path(sam["source"], config_path),
        checkpoint=resolve_path(sam["checkpoint"], config_path),
        device=str(config["device"]),
        model_resolution=int(resolution),
        use_fa3=bool(sam["use_fa3"]),
        compile_model=bool(sam["compile"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to sam3.yaml")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    image_paths = [resolve_path(value, config_path) for value in config["inference"]["images"]]
    images = []
    for path in image_paths:
        with Image.open(path) as value:
            images.append(value.convert("RGB"))
    prompts = [str(config["memory"]["target_prompt"]), str(config["memory"]["context_prompt"])]
    index_path = resolve_path(config["memory"]["bank_index"], config_path)
    record_paths = [lookup_memory(index_path, prompt) for prompt in prompts]

    detector = _runner(config, config_path, int(config["sam3"]["model_resolution"]))
    artifacts = [
        load_memory_artifact(
            path,
            checkpoint_sha256=detector.checkpoint_sha256,
            model_resolution=detector.model_resolution,
        )
        for path in record_paths
    ]
    inference = config["inference"]
    detections = []
    rejected = []
    for camera, image in enumerate(images):
        rows = []
        for prompt, artifact in zip(prompts, artifacts):
            result = detector.localize(
                image,
                artifact,
                score_threshold=float(inference["score_threshold"]),
                mask_threshold=float(inference["mask_threshold"]),
                mask_nms_iou=float(inference["mask_nms_iou"]),
            )
            rows.append(result)
            if not result.present:
                rejected.append(
                    {"camera": camera, "prompt": prompt, "reason": result.rejection_reason, "score": result.score}
                )
        detections.append(rows)
    output_dir = resolve_path(inference["output_dir"], config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    if rejected:
        atomic_json(output_dir / "demo_summary.json", {"status": "rejected", "objects": rejected})
        detector.close()
        print(f"Localization rejected {len(rejected)} required object(s); no policy condition generated.")
        return
    detector.close()

    feature_runner = _runner(config, config_path, int(config["sam3"]["policy_resolution"]))
    features = []
    masks = []
    boxes = []
    level = int(config["sam3"]["fpn_level"])
    for image, rows in zip(images, detections):
        feature = feature_runner.extract_fpn(image, level=level).float()
        features.append(feature)
        camera_masks = []
        camera_boxes = []
        for row in rows:
            assert row.mask is not None and row.box_cxcywh is not None
            mask = torch.from_numpy(row.mask).float()[None, None]
            mask = F.interpolate(mask, size=feature.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            camera_masks.append(mask)
            camera_boxes.append(torch.from_numpy(row.box_cxcywh).float())
        masks.append(torch.stack(camera_masks))
        boxes.append(torch.stack(camera_boxes))
    feature_runner.close()

    policy_config_path = resolve_path(config["policy"]["config"], config_path)
    policy_config, _ = load_config(policy_config_path)
    expected_cameras = int(policy_config["fsae"]["num_cameras"])
    if len(images) != expected_cameras:
        raise ValueError(f"demo has {len(images)} images but policy expects {expected_cameras} cameras")
    policy = build_policy(policy_config)
    checkpoint_path = resolve_path(config["policy"]["checkpoint"], config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("backend") != policy_config["active_backend"]:
        raise ValueError("policy checkpoint backend differs from policy config")
    policy.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(str(config["device"]))
    policy = policy.to(device).eval()
    feature_batch = torch.stack(features)[None].to(device)
    mask_batch = torch.stack(masks)[None].to(device)
    box_batch = torch.stack(boxes)[None].to(device)
    state = torch.zeros(
        (1, int(policy_config["state_history"]), int(policy_config["state_dim"])),
        device=device,
    )
    with torch.no_grad():
        if isinstance(policy, DiffusionPolicyAdapter):
            condition = policy.condition(feature_batch, mask_batch, box_batch, state)
        elif isinstance(policy, ACTPolicyAdapter):
            condition = policy.condition_tokens(feature_batch, mask_batch, box_batch, state)
        else:
            raise TypeError(type(policy).__name__)
    torch.save({"policy_condition": condition.cpu()}, output_dir / "policy_condition.pt")
    summary = {
        "status": "ok",
        "backend": str(policy_config["active_backend"]),
        "prompts": prompts,
        "condition_shape": list(condition.shape),
        "condition_file": "policy_condition.pt",
    }
    atomic_json(output_dir / "demo_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()

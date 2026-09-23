#!/usr/bin/env python3
"""CPU-only shape, gradient, and compact-memory smoke checks."""

from __future__ import annotations

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

from clean_release.models.fo_memory import PrivateLinearFOMemory, reconstruct_prompt  # noqa: E402
from clean_release.models.fsae import FSAE, ObjectROIAlign, TargetContextRelationGate  # noqa: E402
from clean_release.models.policy_adapter import (  # noqa: E402
    ACTPolicyAdapter,
    DiffusionPolicyAdapter,
    act_loss,
)


def inputs(batch: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.randn(batch, 2, 256, 24, 24)
    masks = torch.zeros(batch, 2, 2, 24, 24)
    masks[:, :, 0, 3:12, 4:13] = 1
    masks[:, :, 1, 10:22, 8:21] = 1
    boxes = torch.tensor([0.35, 0.3, 0.4, 0.4, 0.6, 0.65, 0.55, 0.5])
    boxes = boxes.reshape(1, 1, 2, 4).repeat(batch, 2, 1, 1)
    return features, masks, boxes


def main() -> None:
    base = torch.randn(32, 256)
    mask = torch.zeros(32, dtype=torch.bool)
    mask[7:] = True
    memory = PrivateLinearFOMemory(base, mask, torch.randn(1, 256), torch.zeros(1, dtype=torch.bool))
    dense, prompt_mask = reconstruct_prompt(memory.folded_tensors())
    prompt, expected_mask = memory()
    assert torch.allclose(dense, prompt[:, 0].detach())
    assert torch.equal(prompt_mask, expected_mask)

    features, masks, boxes = inputs()
    aligned = ObjectROIAlign()(features[:, 0], masks[:, 0], boxes[:, 0])
    gate = TargetContextRelationGate()
    enhanced = gate(aligned)
    assert torch.allclose(enhanced[:, 0], aligned.features[:, 0])
    assert torch.allclose(enhanced[:, 1], aligned.features[:, 1])

    dp_fsae = FSAE(backend="dp")
    dp_visual = dp_fsae(features, masks, boxes)
    assert dp_visual.shape == (1, 144)
    zero_visual = dp_fsae(features, torch.zeros_like(masks), torch.zeros_like(boxes))
    assert torch.isfinite(zero_visual).all()

    act_fsae = FSAE(backend="act")
    act_tokens = act_fsae(features, masks, boxes)
    assert act_tokens.shape == (1, 256, 512)

    small_dp_fsae = FSAE(backend="dp")
    dp = DiffusionPolicyAdapter(
        small_dp_fsae, down_dims=(32, 64, 128), groups=8, kernel=3, time_dim=32
    )
    actions = torch.randn(1, 24, 10)
    state = torch.randn(1, 2, 10)
    assert dp.condition(features, masks, boxes, state).shape == (1, 164)
    output = dp(actions, torch.tensor([2]), features, masks, boxes, state)
    assert output.shape == actions.shape
    output.square().mean().backward()

    small_act_fsae = FSAE(backend="act", act_token_grid_size=2, act_dim=64)
    act = ACTPolicyAdapter(
        small_act_fsae,
        chunk_size=4,
        dim_model=64,
        heads=8,
        encoder_layers=1,
        decoder_layers=1,
        feedforward_dim=128,
        latent_dim=8,
    ).train()
    assert act.condition_tokens(features, masks, boxes, torch.randn(1, 2, 10)).shape == (
        1,
        18,
        64,
    )
    target = torch.randn(1, 4, 10)
    prediction, mean, log_variance = act(
        features, masks, boxes, torch.randn(1, 2, 10), target
    )
    loss, _ = act_loss(prediction, target, mean, log_variance, 1.0)
    loss.backward()
    assert prediction.shape == target.shape
    print("clean_release smoke checks passed")


if __name__ == "__main__":
    main()

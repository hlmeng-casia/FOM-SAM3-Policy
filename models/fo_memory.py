"""Private Linear FO Memory.

Paper equation (valid SAM3 text-token slots only):

    T_v^FO = W_i (C_v + Delta_v) + b_i,

where the native SAM3 token C_v and every SAM3 parameter are frozen, Delta is
zero initialized, and W/b start as identity/zero. Export folds the train-time
parameters into reusable static prompt tokens; inference never runs W or Delta.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..utils.misc import atomic_json, file_sha256


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "fo_memory"


class PrivateLinearFOMemory(nn.Module):
    """Optimize one object's native valid prompt tokens and a private affine map."""

    method_name = "private_linear"

    def __init__(
        self,
        base_text: Tensor,
        text_mask: Tensor,
        geometry_tokens: Tensor,
        geometry_mask: Tensor,
    ) -> None:
        super().__init__()
        if base_text.shape != (32, 256):
            raise ValueError(f"base_text must be [32,256], got {tuple(base_text.shape)}")
        if text_mask.shape != (32,):
            raise ValueError(f"text_mask must be [32], got {tuple(text_mask.shape)}")
        if geometry_tokens.ndim != 2 or geometry_tokens.shape[-1] != 256:
            raise ValueError("geometry_tokens must be [G,256]")
        if geometry_mask.shape != geometry_tokens.shape[:1]:
            raise ValueError("geometry_mask must be [G]")
        valid_positions = torch.nonzero(~text_mask.bool(), as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            raise ValueError("native SAM3 prompt contains no valid text token")
        self.register_buffer("base_text", base_text.detach().float())
        self.register_buffer("text_mask", text_mask.detach().bool())
        self.register_buffer("geometry_tokens", geometry_tokens.detach().float())
        self.register_buffer("geometry_mask", geometry_mask.detach().bool())
        self.register_buffer("valid_positions", valid_positions.long())
        self.valid_delta = nn.Parameter(torch.zeros(valid_positions.numel(), 256))
        self.private_linear = nn.Linear(256, 256, bias=True)
        with torch.no_grad():
            nn.init.eye_(self.private_linear.weight)
            nn.init.zeros_(self.private_linear.bias)

    def adapted_valid_tokens(self) -> Tensor:
        base = self.base_text.index_select(0, self.valid_positions)
        return self.private_linear(base + self.valid_delta).to(base.dtype)

    def dense_text(self) -> Tensor:
        # Padding slots are masked by SAM3. Keeping them zero makes the folded
        # artifact byte-for-byte equivalent to the train-time prompt.
        return self.base_text.new_zeros(self.base_text.shape).index_copy(
            0, self.valid_positions, self.adapted_valid_tokens()
        )

    def forward(self) -> tuple[Tensor, Tensor]:
        """Return prompt `[P,1,256]` and bool padding mask `[1,P]`."""
        tokens = torch.cat([self.dense_text(), self.geometry_tokens], dim=0)
        mask = torch.cat([self.text_mask, self.geometry_mask], dim=0)
        return tokens[:, None, :], mask[None]

    def optimizer_groups(
        self, token_lr: float, affine_lr: float, weight_decay: float
    ) -> list[dict[str, Any]]:
        return [
            {
                "params": [self.valid_delta],
                "lr": float(token_lr),
                "weight_decay": float(weight_decay),
            },
            {
                "params": [self.private_linear.weight],
                "lr": float(affine_lr),
                "weight_decay": float(weight_decay),
            },
            {
                "params": [self.private_linear.bias],
                "lr": float(affine_lr),
                "weight_decay": 0.0,
            },
        ]

    def folded_tensors(self) -> dict[str, Tensor]:
        """Return the static, losslessly folded inference payload."""
        return {
            "valid_text_tokens": self.adapted_valid_tokens().detach().float().cpu(),
            "valid_text_positions": self.valid_positions.detach().cpu(),
            "text_mask": self.text_mask[None].detach().cpu(),
            "geometry_tokens": self.geometry_tokens[None].detach().float().cpu(),
            "geometry_mask": self.geometry_mask[None].detach().cpu(),
        }


def reconstruct_prompt(tensors: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    """Rebuild `[P,256]` static prompt tokens from a compact artifact."""
    required = {
        "valid_text_tokens",
        "valid_text_positions",
        "text_mask",
        "geometry_tokens",
        "geometry_mask",
    }
    missing = required - tensors.keys()
    if missing:
        raise KeyError(f"memory payload is missing {sorted(missing)}")
    text_mask = tensors["text_mask"].bool()
    if text_mask.shape != (1, 32):
        raise ValueError("text_mask must be [1,32]")
    positions = tensors["valid_text_positions"].long()
    valid = tensors["valid_text_tokens"].float()
    if valid.shape != (positions.numel(), 256):
        raise ValueError("valid token shape differs from valid positions")
    dense = torch.zeros((32, 256), dtype=valid.dtype, device=valid.device)
    dense = dense.index_copy(0, positions.to(valid.device), valid)
    geometry = tensors["geometry_tokens"].float()
    geometry_mask = tensors["geometry_mask"].bool()
    if geometry.ndim != 3 or geometry.shape[0] != 1 or geometry.shape[-1] != 256:
        raise ValueError("geometry_tokens must be [1,G,256]")
    if geometry_mask.shape != geometry.shape[:2]:
        raise ValueError("geometry_mask must be [1,G]")
    return (
        torch.cat([dense, geometry[0]], dim=0),
        torch.cat([text_mask, geometry_mask], dim=1),
    )


@dataclass(frozen=True)
class MemoryArtifact:
    record_path: Path
    record: dict[str, Any]
    tensors: dict[str, Tensor]

    @property
    def memory_id(self) -> str:
        return str(self.record["memory_id"])

    @property
    def prompt(self) -> str:
        return str(self.record["prompt"])

    def prompt_tensors(self) -> tuple[Tensor, Tensor]:
        return reconstruct_prompt(self.tensors)


def save_memory_artifact(
    directory: str | Path,
    *,
    model: PrivateLinearFOMemory,
    prompt: str,
    sam_checkpoint: str | Path,
    model_resolution: int,
) -> Path:
    """Write a portable artifact without serializing machine-local paths."""
    from safetensors.torch import save_file

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    tensors = model.folded_tensors()
    tensor_path = root / "tokens.safetensors"
    save_file(tensors, str(tensor_path))
    memory_id = f"fo:{_slug(prompt)}:private_linear"
    record = {
        "schema_version": 1,
        "memory_id": memory_id,
        "prompt": prompt,
        "method": model.method_name,
        "sam_frozen": True,
        "model_resolution": int(model_resolution),
        "sam_checkpoint_file": Path(sam_checkpoint).name,
        "sam_checkpoint_sha256": file_sha256(sam_checkpoint),
        "tensor_file": tensor_path.name,
        "storage_format": "valid_text_slots_v1",
        "tensor_shapes": {name: list(value.shape) for name, value in tensors.items()},
    }
    record_path = root / "metadata.json"
    atomic_json(record_path, record)
    return record_path


def load_memory_artifact(
    record_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    checkpoint_sha256: str | None = None,
    model_resolution: int | None = None,
) -> MemoryArtifact:
    from safetensors.torch import load_file

    path = Path(record_path).resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("method") != PrivateLinearFOMemory.method_name:
        raise ValueError(f"unsupported FO Memory method: {record.get('method')!r}")
    if checkpoint_sha256 and record.get("sam_checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("FO Memory and runtime SAM3 checkpoint hashes differ")
    if model_resolution is not None and int(record.get("model_resolution", -1)) != int(
        model_resolution
    ):
        raise ValueError("FO Memory and runtime SAM3 resolutions differ")
    tensors = load_file(str(path.parent / str(record["tensor_file"])), device=str(device))
    actual_shapes = {name: list(value.shape) for name, value in tensors.items()}
    if actual_shapes != record.get("tensor_shapes"):
        raise ValueError("FO Memory tensor shapes differ from metadata")
    reconstruct_prompt(tensors)
    return MemoryArtifact(path, record, tensors)


def build_memory_index(records: list[str | Path], destination: str | Path) -> Path:
    """Build an exact prompt/memory-id lookup index for one or more artifacts."""
    rows = []
    prompts: set[str] = set()
    memory_ids: set[str] = set()
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    for value in records:
        artifact = load_memory_artifact(value)
        if artifact.prompt in prompts or artifact.memory_id in memory_ids:
            raise ValueError(f"duplicate prompt or memory id: {artifact.prompt}")
        prompts.add(artifact.prompt)
        memory_ids.add(artifact.memory_id)
        rows.append(
            {
                "memory_id": artifact.memory_id,
                "prompt": artifact.prompt,
                "record": os.path.relpath(artifact.record_path, output.parent.resolve()),
            }
        )
    atomic_json(output, {"schema_version": 1, "memories": rows})
    return output


def lookup_memory(index_path: str | Path, query: str) -> Path:
    """Resolve an exact prompt or memory ID; unknown prompts are rejected."""
    path = Path(index_path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        row
        for row in value.get("memories", [])
        if query in {str(row.get("prompt")), str(row.get("memory_id"))}
    ]
    if len(matches) != 1:
        raise KeyError(f"FO prompt is not uniquely registered: {query!r}")
    return (path.parent / str(matches[0]["record"])).resolve()


def _sigmoid_focal(logits: Tensor, targets: Tensor, alpha: float, gamma: float) -> Tensor:
    logits = logits.float()
    targets = targets.to(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = logits.sigmoid()
    p_t = probability * targets + (1 - probability) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t).pow(gamma) * bce).mean()


def _box_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, width, height = boxes.unbind(-1)
    return torch.stack(
        [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2], dim=-1
    )


def _generalized_iou(a: Tensor, b: Tensor) -> Tensor:
    a, b = _box_xyxy(a), _box_xyxy(b)
    intersection_min = torch.maximum(a[..., :2], b[..., :2])
    intersection_max = torch.minimum(a[..., 2:], b[..., 2:])
    intersection = (intersection_max - intersection_min).clamp_min(0).prod(-1)
    area_a = (a[..., 2:] - a[..., :2]).clamp_min(0).prod(-1)
    area_b = (b[..., 2:] - b[..., :2]).clamp_min(0).prod(-1)
    union = (area_a + area_b - intersection).clamp_min(1e-6)
    enclosing_min = torch.minimum(a[..., :2], b[..., :2])
    enclosing_max = torch.maximum(a[..., 2:], b[..., 2:])
    enclosing = (enclosing_max - enclosing_min).clamp_min(0).prod(-1).clamp_min(1e-6)
    return intersection / union - (enclosing - union) / enclosing


def fo_memory_loss(
    raw: dict[str, Tensor],
    *,
    presence: bool,
    target_mask: Tensor | None,
    target_box: Tensor | None,
    weights: dict[str, float],
) -> tuple[Tensor, dict[str, float]]:
    """One-query FO loss with one-target Hungarian-equivalent matching.

    With one target, Hungarian assignment reduces exactly to selecting the
    minimum classification + L1 + GIoU cost candidate. Absent rows supervise
    only the native SAM3 presence logit.
    """
    presence_logits = raw["presence_logit_dec"].reshape(-1)
    alpha = float(weights.get("focal_alpha", 0.25))
    gamma = float(weights.get("focal_gamma", 2.0))
    presence_target = torch.full_like(presence_logits, float(presence))
    presence_loss = _sigmoid_focal(
        presence_logits,
        presence_target,
        float(weights.get("presence_focal_alpha", 0.5)),
        gamma,
    )
    presence_weight = float(
        weights["presence_positive"] if presence else weights["presence_negative"]
    )
    total = presence_weight * presence_loss
    diagnostics = {"presence": float(presence_loss.detach())}
    if not presence:
        return total, diagnostics
    if target_mask is None or target_box is None:
        raise ValueError("positive FO sample requires mask and box targets")

    object_logits = raw["pred_object_logits_unfused"][0].reshape(-1)
    boxes = raw["pred_boxes"][0].reshape(-1, 4).float()
    target_box = target_box.reshape(1, 4).to(boxes)
    probability = object_logits.float().sigmoid()
    classification_cost = -probability
    l1_cost = torch.cdist(boxes, target_box, p=1)[:, 0]
    giou_cost = -_generalized_iou(boxes, target_box.expand_as(boxes))
    matched = int(
        torch.argmin(
            float(weights.get("matching_object_cost", 2.0)) * classification_cost
            + float(weights.get("matching_box_l1_cost", 5.0)) * l1_cost
            + float(weights.get("matching_box_giou_cost", 2.0)) * giou_cost
        )
    )

    object_targets = torch.zeros_like(object_logits)
    object_targets[matched] = 1
    object_loss = _sigmoid_focal(object_logits, object_targets, alpha, gamma)
    box_l1 = F.l1_loss(boxes[matched], target_box[0])
    box_giou = 1 - _generalized_iou(boxes[matched], target_box[0])

    masks = raw["pred_masks"][0]
    predicted_mask = masks[matched][None, None].float()
    target = target_mask.to(predicted_mask).reshape(1, 1, *target_mask.shape[-2:])
    target = F.interpolate(target, size=predicted_mask.shape[-2:], mode="nearest")
    probability_mask = predicted_mask.sigmoid()
    intersection = (probability_mask * target).sum()
    dice = 1 - (2 * intersection + 1) / (
        probability_mask.sum() + target.sum() + 1
    )
    pixel_bce = F.binary_cross_entropy_with_logits(predicted_mask, target)
    total = total + (
        float(weights["mask_dice"]) * dice
        + float(weights["mask_bce"]) * pixel_bce
        + float(weights["object"]) * object_loss
        + float(weights["box_l1"]) * box_l1
        + float(weights["box_giou"]) * box_giou
    )
    diagnostics.update(
        {
            "dice": float(dice.detach()),
            "mask_bce": float(pixel_bce.detach()),
            "object": float(object_loss.detach()),
            "box_l1": float(box_l1.detach()),
            "box_giou": float(box_giou.detach()),
            "matched_query": matched,
        }
    )
    return total, diagnostics

"""Fine-grained SAM3 Adapter Encoder (FSAE).

For each ordered target/context pair, FSAE applies Box ROIAlign and mask gating.
The target query q = W_t pool(F_t) + W_g g attends to a 4x4 context grid, where
g = [dx, dy, log(w_t/w_c), log(h_t/h_c), IoU, target-in-context]. The resulting
32D relation controls a zero-initialized, target-only residual channel gate.

DP emits 16 mask-aware keypoints (32D) plus box4 per object. ACT retains an
8x8 grid per object and projects each token from 256D to 512D.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import roi_align


@dataclass(frozen=True)
class AlignedObjects:
    features: Tensor  # [B,K,C,R,R]
    masks: Tensor  # [B,K,1,R,R]
    boxes: Tensor  # [B,K,4], normalized cxcywh
    full_masks: Tensor  # [B,K,H,W]


def mask_normalized_pool2d(features: Tensor, masks: Tensor, output_size: int) -> Tensor:
    masks = masks.to(features)
    numerator = F.adaptive_avg_pool2d(features * masks, output_size)
    denominator = F.adaptive_avg_pool2d(masks, output_size)
    return numerator / denominator.clamp_min(1e-6)


class ObjectROIAlign(nn.Module):
    """Apply aligned ROIAlign to a shared feature map and ordered object masks."""

    def __init__(self, roi_size: int = 16, mask_threshold: float = 0.3) -> None:
        super().__init__()
        self.roi_size = int(roi_size)
        self.mask_threshold = float(mask_threshold)

    @staticmethod
    def _roi_boxes(boxes: Tensor, feature_hw: tuple[int, int]) -> Tensor:
        batch_size, objects, _ = boxes.shape
        height, width = feature_hw
        cx, cy, box_w, box_h = boxes.unbind(-1)
        x1 = ((cx - box_w / 2) * width).clamp(0, max(width - 1, 0))
        y1 = ((cy - box_h / 2) * height).clamp(0, max(height - 1, 0))
        x2 = ((cx + box_w / 2) * width).clamp(1, width)
        y2 = ((cy + box_h / 2) * height).clamp(1, height)
        x2 = torch.maximum(x2, x1 + 1).clamp(max=width)
        y2 = torch.maximum(y2, y1 + 1).clamp(max=height)
        batch = torch.arange(batch_size, device=boxes.device, dtype=boxes.dtype)
        batch = batch[:, None].expand(batch_size, objects)
        return torch.stack([batch, x1, y1, x2, y2], dim=-1).reshape(-1, 5)

    def forward(self, feature: Tensor, masks: Tensor, boxes: Tensor) -> AlignedObjects:
        if feature.ndim != 4 or masks.ndim != 4 or boxes.ndim != 3:
            raise ValueError("feature/masks/boxes must be [B,C,H,W]/[B,K,H,W]/[B,K,4]")
        batch_size, channels, height, width = feature.shape
        if masks.shape[0] != batch_size or masks.shape[-2:] != (height, width):
            raise ValueError("masks must share feature batch and spatial dimensions")
        if boxes.shape != (batch_size, masks.shape[1], 4):
            raise ValueError("boxes must align with ordered object masks")
        boxes = boxes.to(feature).clamp(0, 1)
        masks = masks.to(feature).clamp(0, 1)
        rois = self._roi_boxes(boxes, (height, width))
        roi_features = roi_align(
            feature,
            rois,
            (self.roi_size, self.roi_size),
            spatial_scale=1.0,
            sampling_ratio=-1,
            aligned=True,
        )
        objects = masks.shape[1]
        flat_masks = masks.reshape(batch_size * objects, 1, height, width)
        mask_rois = rois.clone()
        mask_rois[:, 0] = torch.arange(
            batch_size * objects, device=feature.device, dtype=feature.dtype
        )
        roi_masks = roi_align(
            flat_masks,
            mask_rois,
            (self.roi_size, self.roi_size),
            spatial_scale=1.0,
            sampling_ratio=-1,
            aligned=True,
        ).clamp(0, 1)
        return AlignedObjects(
            roi_features.reshape(batch_size, objects, channels, self.roi_size, self.roi_size),
            roi_masks.reshape(batch_size, objects, 1, self.roi_size, self.roi_size),
            boxes,
            masks,
        )


class TargetContextRelationGate(nn.Module):
    """Single-query 32D context attention with target-only residual gating."""

    def __init__(
        self,
        channels: int = 256,
        relation_dim: int = 32,
        context_grid_size: int = 4,
        mask_threshold: float = 0.3,
        feature_hw: tuple[int, int] = (24, 24),
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.relation_dim = int(relation_dim)
        self.context_grid_size = int(context_grid_size)
        self.mask_threshold = float(mask_threshold)
        self.target_query = nn.Linear(channels, relation_dim)
        self.context_key_value = nn.Linear(channels, 2 * relation_dim)
        self.geometry_query = nn.Linear(6, relation_dim)
        self.target_channel_gate = nn.Linear(relation_dim, channels)
        nn.init.zeros_(self.target_channel_gate.weight)
        nn.init.zeros_(self.target_channel_gate.bias)
        height, width = feature_hw
        y, x = torch.meshgrid(
            (torch.arange(height, dtype=torch.float32) + 0.5) / height,
            (torch.arange(width, dtype=torch.float32) + 0.5) / width,
            indexing="ij",
        )
        self.register_buffer("x_grid", x, persistent=False)
        self.register_buffer("y_grid", y, persistent=False)

    def _hard(self, mask: Tensor) -> Tensor:
        return mask >= self.mask_threshold if self.mask_threshold > 0 else mask > 0

    def _geometry(self, value: AlignedObjects, valid: Tensor) -> Tensor:
        target, context = value.boxes[:, 0], value.boxes[:, 1]
        tcx, tcy, tw, th = target.unbind(-1)
        ccx, ccy, cw, ch = context.unbind(-1)
        epsilon = torch.finfo(value.boxes.dtype).eps
        dx = ((tcx - ccx) / cw.clamp_min(epsilon)).clamp(-4, 4)
        dy = ((tcy - ccy) / ch.clamp_min(epsilon)).clamp(-4, 4)
        log_w = torch.log(tw.clamp_min(epsilon) / cw.clamp_min(epsilon)).clamp(-4, 4)
        log_h = torch.log(th.clamp_min(epsilon) / ch.clamp_min(epsilon)).clamp(-4, 4)
        tx1, ty1, tx2, ty2 = tcx - tw / 2, tcy - th / 2, tcx + tw / 2, tcy + th / 2
        cx1, cy1, cx2, cy2 = ccx - cw / 2, ccy - ch / 2, ccx + cw / 2, ccy + ch / 2
        inter = (torch.minimum(tx2, cx2) - torch.maximum(tx1, cx1)).clamp_min(0)
        inter = inter * (torch.minimum(ty2, cy2) - torch.maximum(ty1, cy1)).clamp_min(0)
        iou = inter / (tw * th + cw * ch - inter).clamp_min(epsilon)
        target_mask = self._hard(value.full_masks[:, 0]).to(value.boxes)
        x = self.x_grid.to(value.boxes)
        y = self.y_grid.to(value.boxes)
        inside = (
            (x[None] >= cx1[:, None, None])
            & (x[None] <= cx2[:, None, None])
            & (y[None] >= cy1[:, None, None])
            & (y[None] <= cy2[:, None, None])
        ).to(value.boxes)
        target_inside = (target_mask * inside).sum((-2, -1)) / target_mask.sum(
            (-2, -1)
        ).clamp_min(1)
        geometry = torch.stack([dx, dy, log_w, log_h, iou, target_inside], dim=-1)
        return torch.where(valid[:, None], geometry, torch.zeros_like(geometry))

    def forward(self, value: AlignedObjects) -> Tensor:
        if value.features.shape[1] != 2:
            raise ValueError("TCRG requires exactly two slots: target then context")
        target_feature, context_feature = value.features[:, 0], value.features[:, 1]
        target_mask = self._hard(value.masks[:, 0]).to(target_feature)
        context_mask = self._hard(value.masks[:, 1]).to(context_feature)
        target_summary = mask_normalized_pool2d(target_feature, target_mask, 1).flatten(1)
        context_grid = mask_normalized_pool2d(
            context_feature, context_mask, self.context_grid_size
        )
        context_tokens = context_grid.flatten(2).transpose(1, 2)
        target_summary = F.layer_norm(target_summary, (self.channels,))
        context_tokens = F.layer_norm(context_tokens, (self.channels,))
        token_valid = F.adaptive_max_pool2d(
            context_mask, self.context_grid_size
        ).flatten(1).bool()
        valid = target_mask.flatten(1).bool().any(-1) & token_valid.any(-1)
        query = self.target_query(target_summary) + self.geometry_query(
            self._geometry(value, valid)
        )
        key, attended_value = self.context_key_value(context_tokens).chunk(2, -1)
        logits = torch.einsum("bd,bnd->bn", query, key) / math.sqrt(self.relation_dim)
        logits = logits.masked_fill(~token_valid, torch.finfo(logits.dtype).min)
        attention = F.softmax(logits, -1)
        attention = torch.where(valid[:, None], attention, torch.zeros_like(attention))
        relation = torch.einsum("bn,bnd->bd", attention, attended_value)
        scale = torch.tanh(self.target_channel_gate(relation))
        active = target_mask * valid[:, None, None, None].to(target_mask)
        enhanced_target = target_feature + target_feature * scale[:, :, None, None] * active
        return torch.stack([enhanced_target, context_feature], dim=1)


class MaskAwareSpatialSoftmax(nn.Module):
    def __init__(self, side: int) -> None:
        super().__init__()
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, side), torch.linspace(-1, 1, side), indexing="ij"
        )
        self.register_buffer("positions", torch.stack([x, y], -1).reshape(-1, 2))

    def forward(self, heatmaps: Tensor, masks: Tensor) -> Tensor:
        logits = heatmaps.flatten(2)
        valid = masks.flatten(2).bool()
        attention = F.softmax(
            logits.masked_fill(~valid, torch.finfo(logits.dtype).min), dim=-1
        )
        attention = torch.where(valid.any(-1, keepdim=True), attention, torch.zeros_like(attention))
        return attention @ self.positions.to(heatmaps)


def _position_2d(grid: int, dim: int) -> Tensor:
    if dim % 4:
        raise ValueError("ACT token dimension must be divisible by four")
    y, x = torch.meshgrid(torch.linspace(0, 1, grid), torch.linspace(0, 1, grid), indexing="ij")
    omega = torch.arange(dim // 4, dtype=torch.float32)
    omega = 1 / (10000 ** (omega / max(dim // 4 - 1, 1)))
    xp = x.reshape(-1, 1) * 2 * math.pi * omega
    yp = y.reshape(-1, 1) * 2 * math.pi * omega
    return torch.cat([xp.sin(), xp.cos(), yp.sin(), yp.cos()], -1)


class FSAE(nn.Module):
    """Shared public interface for DP conditions and ACT visual tokens."""

    def __init__(
        self,
        *,
        backend: str,
        num_cameras: int = 2,
        feature_channels: int = 256,
        feature_hw: tuple[int, int] = (24, 24),
        objects_per_camera: int = 2,
        roi_size: int = 16,
        mask_threshold: float = 0.3,
        relation_dim: int = 32,
        context_grid_size: int = 4,
        spatial_keypoints: int = 16,
        act_token_grid_size: int = 8,
        act_dim: int = 512,
    ) -> None:
        super().__init__()
        if backend not in {"dp", "act"}:
            raise ValueError("backend must be 'dp' or 'act'")
        if objects_per_camera != 2:
            raise ValueError("the release contract requires target/context slots")
        self.backend = backend
        self.num_cameras = int(num_cameras)
        self.feature_channels = int(feature_channels)
        self.feature_hw = tuple(map(int, feature_hw))
        self.objects_per_camera = int(objects_per_camera)
        self.mask_threshold = float(mask_threshold)
        self.roi = ObjectROIAlign(roi_size, mask_threshold)
        gate_count = self.num_cameras if backend == "dp" else 1
        self.relation_gates = nn.ModuleList(
            [
                TargetContextRelationGate(
                    feature_channels,
                    relation_dim,
                    context_grid_size,
                    mask_threshold,
                    self.feature_hw,
                )
                for _ in range(gate_count)
            ]
        )
        if backend == "dp":
            self.spatial_keypoints = int(spatial_keypoints)
            self.heatmaps = nn.ModuleList(
                [nn.Conv2d(feature_channels, spatial_keypoints, 1) for _ in range(num_cameras)]
            )
            self.spatial_pool = MaskAwareSpatialSoftmax(roi_size)
        else:
            self.act_token_grid_size = int(act_token_grid_size)
            self.act_dim = int(act_dim)
            self.feature_norm = nn.LayerNorm(feature_channels)
            self.feature_projection = nn.Linear(feature_channels, act_dim, bias=False)
            self.box_projection = nn.Sequential(
                nn.Linear(4, min(128, act_dim)), nn.ReLU(), nn.Linear(min(128, act_dim), act_dim)
            )
            self.camera_embedding = nn.Embedding(num_cameras, act_dim)
            self.role_embedding = nn.Embedding(objects_per_camera, act_dim)
            self.register_buffer(
                "local_position", _position_2d(act_token_grid_size, act_dim), persistent=True
            )

    @property
    def dp_visual_dim(self) -> int:
        return self.num_cameras * self.objects_per_camera * 36

    @property
    def act_visual_tokens(self) -> int:
        return self.num_cameras * self.objects_per_camera * self.act_token_grid_size**2

    def _validate(self, features: Tensor, masks: Tensor, boxes: Tensor) -> None:
        batch = features.shape[0]
        expected_features = (batch, self.num_cameras, self.feature_channels, *self.feature_hw)
        expected_masks = (batch, self.num_cameras, self.objects_per_camera, *self.feature_hw)
        expected_boxes = (batch, self.num_cameras, self.objects_per_camera, 4)
        if tuple(features.shape) != expected_features:
            raise ValueError(f"features shape {tuple(features.shape)} != {expected_features}")
        if tuple(masks.shape) != expected_masks or tuple(boxes.shape) != expected_boxes:
            raise ValueError("ordered mask/box shape differs from the FSAE contract")

    def forward(self, features: Tensor, masks: Tensor, boxes: Tensor) -> Tensor:
        self._validate(features, masks, boxes)
        camera_outputs: list[Tensor] = []
        for camera in range(self.num_cameras):
            aligned = self.roi(features[:, camera], masks[:, camera], boxes[:, camera])
            gate = self.relation_gates[camera if self.backend == "dp" else 0]
            enhanced = gate(aligned)
            hard_masks = (
                aligned.masks >= self.mask_threshold
                if self.mask_threshold > 0
                else aligned.masks > 0
            ).to(enhanced)
            batch_size = features.shape[0]
            if self.backend == "dp":
                flat_feature = enhanced.flatten(0, 1)
                flat_mask = hard_masks.flatten(0, 1)
                heatmaps = self.heatmaps[camera](flat_feature * flat_mask)
                spatial = self.spatial_pool(heatmaps, flat_mask).flatten(1)
                spatial = spatial.reshape(batch_size, self.objects_per_camera, -1)
                camera_outputs.append(torch.cat([spatial, aligned.boxes], -1))
            else:
                pooled = mask_normalized_pool2d(
                    enhanced.flatten(0, 1),
                    hard_masks.flatten(0, 1),
                    self.act_token_grid_size,
                ).reshape(
                    batch_size,
                    self.objects_per_camera,
                    self.feature_channels,
                    self.act_token_grid_size,
                    self.act_token_grid_size,
                )
                tokens = pooled.permute(0, 1, 3, 4, 2)
                tokens = self.feature_projection(self.feature_norm(tokens))
                tokens = tokens + self.box_projection(aligned.boxes)[:, :, None, None]
                tokens = tokens + self.camera_embedding.weight[camera].reshape(1, 1, 1, 1, -1)
                tokens = tokens + self.role_embedding.weight.reshape(
                    1, self.objects_per_camera, 1, 1, -1
                )
                tokens = tokens + self.local_position.reshape(
                    1, 1, self.act_token_grid_size, self.act_token_grid_size, -1
                )
                camera_outputs.append(tokens.reshape(batch_size, -1, self.act_dim))
        if self.backend == "dp":
            return torch.cat(camera_outputs, dim=1).flatten(1)
        return torch.cat(camera_outputs, dim=1)

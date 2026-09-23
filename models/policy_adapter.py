#!/usr/bin/env python
# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.
"""Self-contained DP and ACT heads driven by FSAE conditions.

Only the policy-critical neural blocks are retained from the project's LeRobot
integration. Dataset, robot, logging, Hub, and deployment machinery are not
part of this release.

Paper interfaces:
    c_DP = concat(FSAE_DP(H, M, B), state[t-1:t]) in R^164,
    eps_hat = UNet(a_tau, tau, c_DP),
    Z_ACT = [z, state, FSAE_ACT(H, M, B)] in R^(258 x 512).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .fsae import FSAE


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.register_buffer("device_anchor", torch.empty(0), persistent=False)

    def forward(self, timestep: Tensor | int) -> Tensor:
        value = torch.as_tensor(timestep, device=self.device_anchor.device)
        if value.ndim == 0:
            value = value[None]
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        frequencies = torch.exp(torch.arange(half, device=value.device) * -scale)
        phase = value.float()[:, None] * frequencies[None]
        return torch.cat([phase.sin(), phase.cos()], -1)


class Conv1dBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kernel: int, groups: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel, padding=kernel // 2),
            nn.GroupNorm(groups, output_dim),
            nn.Mish(),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.block(value)


class ConditionalResidualBlock1d(nn.Module):
    """Residual Conv1d block with scale-and-bias FiLM conditioning."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        condition_dim: int,
        kernel: int,
        groups: int,
        film_scale: bool,
    ) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.film_scale = bool(film_scale)
        self.conv1 = Conv1dBlock(input_dim, output_dim, kernel, groups)
        self.conv2 = Conv1dBlock(output_dim, output_dim, kernel, groups)
        film_dim = output_dim * (2 if film_scale else 1)
        self.condition = nn.Sequential(nn.Mish(), nn.Linear(condition_dim, film_dim))
        self.residual = nn.Conv1d(input_dim, output_dim, 1) if input_dim != output_dim else nn.Identity()

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        output = self.conv1(value)
        film = self.condition(condition).unsqueeze(-1)
        if self.film_scale:
            scale, bias = film.chunk(2, 1)
            output = scale * output + bias
        else:
            output = output + film
        return self.conv2(output) + self.residual(value)


class ConditionalUnet1d(nn.Module):
    """FiLM 1D U-Net used by the FOM-SAM Diffusion Policy."""

    def __init__(
        self,
        *,
        action_dim: int,
        global_condition_dim: int,
        time_dim: int = 128,
        down_dims: tuple[int, ...] = (512, 1024, 2048),
        kernel: int = 5,
        groups: int = 8,
        film_scale: bool = True,
    ) -> None:
        super().__init__()
        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )
        condition_dim = time_dim + global_condition_dim
        dimensions = (action_dim, *down_dims)
        self.down = nn.ModuleList()
        for index, (dim_in, dim_out) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            last = index == len(down_dims) - 1
            self.down.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1d(dim_in, dim_out, condition_dim, kernel, groups, film_scale),
                        ConditionalResidualBlock1d(dim_out, dim_out, condition_dim, kernel, groups, film_scale),
                        nn.Identity() if last else nn.Conv1d(dim_out, dim_out, 3, 2, 1),
                    ]
                )
            )
        self.middle = nn.ModuleList(
            [
                ConditionalResidualBlock1d(down_dims[-1], down_dims[-1], condition_dim, kernel, groups, film_scale),
                ConditionalResidualBlock1d(down_dims[-1], down_dims[-1], condition_dim, kernel, groups, film_scale),
            ]
        )
        self.up = nn.ModuleList()
        pairs = list(zip(dimensions[:-1], dimensions[1:]))
        for dim_out, dim_in in reversed(pairs[1:]):
            self.up.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1d(dim_in * 2, dim_out, condition_dim, kernel, groups, film_scale),
                        ConditionalResidualBlock1d(dim_out, dim_out, condition_dim, kernel, groups, film_scale),
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1),
                    ]
                )
            )
        self.output = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel, groups),
            nn.Conv1d(down_dims[0], action_dim, 1),
        )

    def forward(self, noisy_actions: Tensor, timestep: Tensor | int, condition: Tensor) -> Tensor:
        value = noisy_actions.permute(0, 2, 1)
        time = torch.as_tensor(timestep, device=value.device)
        if time.ndim == 0:
            time = time.repeat(value.shape[0])
        global_condition = torch.cat([self.time_encoder(time), condition], -1)
        skips: list[Tensor] = []
        for first, second, downsample in self.down:
            value = second(first(value, global_condition), global_condition)
            skips.append(value)
            value = downsample(value)
        for block in self.middle:
            value = block(value, global_condition)
        for first, second, upsample in self.up:
            skip = skips.pop()
            if value.shape[-1] != skip.shape[-1]:
                value = torch.nn.functional.interpolate(value, size=skip.shape[-1], mode="nearest")
            value = torch.cat([value, skip], 1)
            value = second(first(value, global_condition), global_condition)
            value = upsample(value)
        return self.output(value).permute(0, 2, 1)


class DiffusionPolicyAdapter(nn.Module):
    """FSAE DP condition (144 visual + 20 state) and FiLM 1D U-Net."""

    def __init__(
        self,
        fsae: FSAE,
        *,
        state_dim: int = 10,
        state_history: int = 2,
        action_dim: int = 10,
        time_dim: int = 128,
        down_dims: tuple[int, ...] = (512, 1024, 2048),
        kernel: int = 5,
        groups: int = 8,
        film_scale: bool = True,
    ) -> None:
        super().__init__()
        if fsae.backend != "dp":
            raise ValueError("DiffusionPolicyAdapter requires a DP FSAE")
        self.fsae = fsae
        self.state_dim = int(state_dim)
        self.state_history = int(state_history)
        global_dim = fsae.dp_visual_dim + state_dim * state_history
        self.unet = ConditionalUnet1d(
            action_dim=action_dim,
            global_condition_dim=global_dim,
            time_dim=time_dim,
            down_dims=down_dims,
            kernel=kernel,
            groups=groups,
            film_scale=film_scale,
        )

    def condition(self, features: Tensor, masks: Tensor, boxes: Tensor, state: Tensor) -> Tensor:
        expected = (features.shape[0], self.state_history, self.state_dim)
        if tuple(state.shape) != expected:
            raise ValueError(f"state shape {tuple(state.shape)} != {expected}")
        return torch.cat([self.fsae(features, masks, boxes), state.flatten(1)], -1)

    def forward(
        self,
        noisy_actions: Tensor,
        timestep: Tensor | int,
        features: Tensor,
        masks: Tensor,
        boxes: Tensor,
        state: Tensor,
    ) -> Tensor:
        return self.unet(noisy_actions, timestep, self.condition(features, masks, boxes, state))


def _sinusoidal_sequence(length: int, dim: int) -> Tensor:
    position = torch.arange(length, dtype=torch.float32)[:, None]
    frequency = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    output = torch.zeros(length, dim)
    output[:, 0::2] = torch.sin(position * frequency)
    output[:, 1::2] = torch.cos(position * frequency)
    return output


class ACTPolicyAdapter(nn.Module):
    """CVAE Action Chunking Transformer over 256 ordered FSAE visual tokens."""

    def __init__(
        self,
        fsae: FSAE,
        *,
        state_dim: int = 10,
        action_dim: int = 10,
        chunk_size: int = 30,
        dim_model: int = 512,
        heads: int = 8,
        encoder_layers: int = 4,
        decoder_layers: int = 1,
        feedforward_dim: int = 3200,
        latent_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if fsae.backend != "act" or fsae.act_dim != dim_model:
            raise ValueError("ACTPolicyAdapter requires an ACT FSAE with matching dim_model")
        self.fsae = fsae
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.latent_dim = int(latent_dim)
        self.state_projection = nn.Linear(state_dim, dim_model)
        self.action_projection = nn.Linear(action_dim, dim_model)
        self.vae_cls = nn.Parameter(torch.zeros(1, 1, dim_model))
        vae_layer = nn.TransformerEncoderLayer(
            dim_model, heads, feedforward_dim, dropout=float(dropout), batch_first=True
        )
        self.vae_encoder = nn.TransformerEncoder(vae_layer, encoder_layers)
        self.latent_parameters = nn.Linear(dim_model, 2 * latent_dim)
        self.latent_projection = nn.Linear(latent_dim, dim_model)
        policy_layer = nn.TransformerEncoderLayer(
            dim_model, heads, feedforward_dim, dropout=float(dropout), batch_first=True
        )
        self.policy_encoder = nn.TransformerEncoder(policy_layer, encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            dim_model, heads, feedforward_dim, dropout=float(dropout), batch_first=True
        )
        self.policy_decoder = nn.TransformerDecoder(decoder_layer, decoder_layers)
        self.action_queries = nn.Parameter(torch.zeros(1, chunk_size, dim_model))
        self.action_head = nn.Linear(dim_model, action_dim)
        self.register_buffer(
            "vae_position", _sinusoidal_sequence(chunk_size + 2, dim_model), persistent=True
        )
        nn.init.normal_(self.action_queries, std=0.02)

    def _latent(self, state: Tensor, actions: Tensor | None) -> tuple[Tensor, Tensor | None, Tensor | None]:
        batch = state.shape[0]
        if self.training and actions is not None:
            if tuple(actions.shape[1:]) != (self.chunk_size, self.action_dim):
                raise ValueError("ACT training actions do not match chunk/action dimensions")
            tokens = torch.cat(
                [
                    self.vae_cls.expand(batch, -1, -1),
                    self.state_projection(state[:, -1]).unsqueeze(1),
                    self.action_projection(actions),
                ],
                1,
            )
            encoded = self.vae_encoder(tokens + self.vae_position[None].to(tokens))[:, 0]
            mean, log_variance = self.latent_parameters(encoded).chunk(2, -1)
            latent = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
            return latent, mean, log_variance
        return torch.zeros(batch, self.latent_dim, device=state.device, dtype=state.dtype), None, None

    def condition_tokens(self, features: Tensor, masks: Tensor, boxes: Tensor, state: Tensor) -> Tensor:
        visual = self.fsae(features, masks, boxes)
        latent = torch.zeros(state.shape[0], self.latent_dim, device=state.device, dtype=state.dtype)
        return torch.cat(
            [self.latent_projection(latent).unsqueeze(1), self.state_projection(state[:, -1]).unsqueeze(1), visual],
            1,
        )

    def forward(
        self,
        features: Tensor,
        masks: Tensor,
        boxes: Tensor,
        state: Tensor,
        actions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        latent, mean, log_variance = self._latent(state, actions)
        visual = self.fsae(features, masks, boxes)
        encoder_input = torch.cat(
            [
                self.latent_projection(latent).unsqueeze(1),
                self.state_projection(state[:, -1]).unsqueeze(1),
                visual,
            ],
            1,
        )
        memory = self.policy_encoder(encoder_input)
        queries = self.action_queries.expand(state.shape[0], -1, -1)
        decoded = self.policy_decoder(queries, memory)
        return self.action_head(decoded), mean, log_variance


def act_loss(
    prediction: Tensor,
    target: Tensor,
    mean: Tensor | None,
    log_variance: Tensor | None,
    kl_weight: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    l1 = torch.nn.functional.l1_loss(prediction, target)
    if mean is None or log_variance is None:
        kl = l1.new_zeros(())
    else:
        kl = (-0.5 * (1 + log_variance - mean.square() - log_variance.exp())).sum(-1).mean()
    return l1 + float(kl_weight) * kl, {"l1": l1.detach(), "kl": kl.detach()}


def build_fsae(config: dict, backend: str) -> FSAE:
    """Construct FSAE from the public `policy.yaml` mapping."""
    value = config["fsae"]
    return FSAE(
        backend=backend,
        num_cameras=int(value["num_cameras"]),
        feature_channels=int(value["feature_channels"]),
        feature_hw=(int(value["feature_height"]), int(value["feature_width"])),
        objects_per_camera=int(value["objects_per_camera"]),
        roi_size=int(value["roi_size"]),
        mask_threshold=float(value["mask_threshold"]),
        relation_dim=int(value["relation_dim"]),
        context_grid_size=int(value["context_grid_size"]),
        spatial_keypoints=int(value["spatial_keypoints"]),
        act_token_grid_size=int(value["act_token_grid_size"]),
        act_dim=int(value["act_dim"]),
    )


def build_policy(config: dict) -> DiffusionPolicyAdapter | ACTPolicyAdapter:
    """Build the selected self-contained policy adapter from YAML values."""
    backend = str(config["active_backend"]).lower()
    fsae = build_fsae(config, backend)
    if backend == "dp":
        value = config["dp"]
        return DiffusionPolicyAdapter(
            fsae,
            state_dim=int(config["state_dim"]),
            state_history=int(config["state_history"]),
            action_dim=int(config["action_dim"]),
            time_dim=int(value["diffusion_step_dim"]),
            down_dims=tuple(map(int, value["down_dims"])),
            kernel=int(value["kernel_size"]),
            groups=int(value["groups"]),
            film_scale=bool(value["film_scale"]),
        )
    if backend == "act":
        value = config["act"]
        return ACTPolicyAdapter(
            fsae,
            state_dim=int(config["state_dim"]),
            action_dim=int(config["action_dim"]),
            chunk_size=int(value["chunk_size"]),
            dim_model=int(value["dim_model"]),
            heads=int(value["heads"]),
            encoder_layers=int(value["encoder_layers"]),
            decoder_layers=int(value["decoder_layers"]),
            feedforward_dim=int(value["feedforward_dim"]),
            latent_dim=int(value["latent_dim"]),
            dropout=float(value["dropout"]),
        )
    raise ValueError(f"unknown policy backend: {backend!r}")

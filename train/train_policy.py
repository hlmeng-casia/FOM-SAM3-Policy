#!/usr/bin/env python3
"""Train the FSAE adapter with the self-contained DP or ACT policy head."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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

from clean_release.datasets.policy_cache_dataset import PolicyCacheDataset  # noqa: E402
from clean_release.models.policy_adapter import act_loss, build_policy  # noqa: E402
from clean_release.utils.misc import atomic_json, load_config, resolve_path, set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to policy.yaml")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    set_seed(int(config["seed"]))
    backend = str(config["active_backend"]).lower()
    action_steps = int(config["dp"]["horizon"] if backend == "dp" else config["act"]["chunk_size"])
    fsae = config["fsae"]
    data = config["data"]
    dataset = PolicyCacheDataset(
        resolve_path(data["cache_root"], config_path),
        data["manifest"],
        num_cameras=int(fsae["num_cameras"]),
        objects_per_camera=int(fsae["objects_per_camera"]),
        action_steps=action_steps,
        feature_channels=int(fsae["feature_channels"]),
        feature_hw=(int(fsae["feature_height"]), int(fsae["feature_width"])),
        state_history=int(config["state_history"]),
        state_dim=int(config["state_dim"]),
        action_dim=int(config["action_dim"]),
    )
    train = config["train"]
    loader = DataLoader(
        dataset,
        batch_size=int(train["batch_size"]),
        shuffle=True,
        num_workers=int(train["num_workers"]),
        drop_last=False,
    )
    device = torch.device(str(config["device"]))
    policy = build_policy(config).to(device).train()
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(train["learning_rate"]),
        weight_decay=float(train["weight_decay"]),
    )
    noise_scheduler = None
    if backend == "dp":
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=int(config["dp"]["train_timesteps"]),
            beta_schedule=str(config["dp"]["beta_schedule"]),
            prediction_type="epsilon",
        )
    iterator = iter(loader)
    output_dir = resolve_path(config["output"]["run_dir"], config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_metrics: dict[str, float] = {}
    for step in tqdm(range(1, int(train["steps"]) + 1), desc=f"Policy/{backend}"):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        if backend == "dp":
            assert noise_scheduler is not None
            noise = torch.randn_like(batch["actions"])
            timesteps = torch.randint(
                0,
                int(config["dp"]["train_timesteps"]),
                (batch["actions"].shape[0],),
                device=device,
            )
            noisy = noise_scheduler.add_noise(batch["actions"], noise, timesteps)
            prediction = policy(
                noisy,
                timesteps,
                batch["features"],
                batch["masks"],
                batch["boxes"],
                batch["state"],
            )
            loss = torch.nn.functional.mse_loss(prediction, noise)
            last_metrics = {"epsilon_mse": float(loss.detach())}
        else:
            prediction, mean, log_variance = policy(
                batch["features"],
                batch["masks"],
                batch["boxes"],
                batch["state"],
                batch["actions"],
            )
            loss, metrics = act_loss(
                prediction,
                batch["actions"],
                mean,
                log_variance,
                float(config["act"]["kl_weight"]),
            )
            last_metrics = {name: float(value) for name, value in metrics.items()}
        loss.backward()
        optimizer.step()
        if step % int(train["save_every_steps"]) == 0 or step == int(train["steps"]):
            torch.save(
                {
                    "schema_version": 1,
                    "backend": backend,
                    "step": step,
                    "model": policy.state_dict(),
                },
                output_dir / f"step_{step:06d}.pt",
            )
    torch.save(
        {
            "schema_version": 1,
            "backend": backend,
            "step": int(train["steps"]),
            "model": policy.state_dict(),
        },
        output_dir / "last.pt",
    )
    atomic_json(
        output_dir / "train_summary.json",
        {"backend": backend, "steps": int(train["steps"]), "last_metrics": last_metrics},
    )
    print(f"Saved {backend.upper()} policy adapter to {output_dir / 'last.pt'}")


if __name__ == "__main__":
    main()

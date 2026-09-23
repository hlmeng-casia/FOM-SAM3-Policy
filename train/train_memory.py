#!/usr/bin/env python3
"""Train one Private Linear FO Memory from a small registration set."""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch
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

from clean_release.datasets.registration_dataset import RegistrationDataset  # noqa: E402
from clean_release.models.fo_memory import PrivateLinearFOMemory, fo_memory_loss  # noqa: E402
from clean_release.sam3.interface import FrozenSAM3  # noqa: E402
from clean_release.utils.misc import atomic_json, load_config, resolve_path, set_seed  # noqa: E402


def _target_box(sample, image, device: torch.device) -> torch.Tensor:
    if sample.box_xyxy is None:
        raise ValueError("positive registration row has no box")
    x1, y1, x2, y2 = sample.box_xyxy
    width, height = image.size
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(
            f"box_xyxy {sample.box_xyxy} is invalid for image size {(width, height)}"
        )
    return torch.tensor(
        [
            (x1 + x2) / (2 * width),
            (y1 + y2) / (2 * height),
            (x2 - x1) / width,
            (y2 - y1) / height,
        ],
        device=device,
        dtype=torch.float32,
    ).clamp(0, 1)


def _save_checkpoint(path: Path, model: PrivateLinearFOMemory, prompt: str, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "method": model.method_name,
            "prompt": prompt,
            "epoch": int(epoch),
            "state_dict": model.state_dict(),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to memory.yaml")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    set_seed(int(config["seed"]))
    data_root = resolve_path(config["data"]["root"], config_path)
    dataset = RegistrationDataset(data_root, config["data"]["annotations"])
    if int(config["train"]["batch_size"]) != 1:
        raise ValueError("the minimal release uses batch_size=1 for native SAM3 prompts")

    sam_config = config["sam3"]
    runner = FrozenSAM3(
        source=resolve_path(sam_config["source"], config_path),
        checkpoint=resolve_path(sam_config["checkpoint"], config_path),
        device=str(config["device"]),
        model_resolution=int(sam_config["model_resolution"]),
        use_fa3=bool(sam_config["use_fa3"]),
        compile_model=bool(sam_config["compile"]),
    )
    prompt_text = str(config["data"]["prompt"])
    positive = next(sample for sample in dataset.samples if sample.presence)
    native_tokens, native_mask = runner.native_prompt(positive.load_image(), prompt_text)
    if native_tokens.shape[0] <= 32:
        raise RuntimeError("official SAM3 prompt did not include a geometry token")
    model = PrivateLinearFOMemory(
        native_tokens[:32], native_mask[:32], native_tokens[32:], native_mask[32:]
    ).to(runner.device)
    method = config["method"]
    train = config["train"]
    optimizer = torch.optim.AdamW(
        model.optimizer_groups(
            float(method["token_lr"]),
            float(method["affine_lr"]),
            float(train["weight_decay"]),
        )
    )
    total_steps = int(train["epochs"]) * len(dataset)
    warmup = int(train["warmup_steps"])

    def schedule(step: int) -> float:
        current = max(step, 1)
        if current <= warmup:
            return current / max(warmup, 1)
        return math.sqrt(max(warmup, 1) / current)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    run_dir = resolve_path(config["output"]["run_dir"], config_path)
    checkpoint_path = run_dir / str(config["output"]["checkpoint"])
    use_amp = str(train["mixed_precision"]).lower() == "bfloat16"
    last_diagnostics: dict[str, float] = {}
    progress = tqdm(total=total_steps, desc="FO Memory")
    for epoch in range(1, int(train["epochs"]) + 1):
        model.train()
        for sample in dataset.samples:
            image = sample.load_image()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                prompt, prompt_mask = model()
                raw = runner.forward_train(image, prompt, prompt_mask)
                mask = sample.load_mask()
                target_mask = (
                    None
                    if mask is None
                    else torch.from_numpy(mask).to(device=runner.device, dtype=torch.float32)
                )
                target_box = None if not sample.presence else _target_box(sample, image, runner.device)
                loss, diagnostics = fo_memory_loss(
                    raw,
                    presence=sample.presence,
                    target_mask=target_mask,
                    target_box=target_box,
                    weights=train["loss"],
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train["gradient_clip_norm"]))
            optimizer.step()
            scheduler.step()
            last_diagnostics = {"total": float(loss.detach()), **diagnostics}
            progress.update(1)
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
        if epoch % int(train["save_every_epochs"]) == 0:
            _save_checkpoint(
                run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", model, prompt_text, epoch
            )
    progress.close()
    _save_checkpoint(checkpoint_path, model, prompt_text, int(train["epochs"]))
    atomic_json(
        run_dir / "train_summary.json",
        {
            "method": model.method_name,
            "prompt": prompt_text,
            "epochs": int(train["epochs"]),
            "registration_rows": len(dataset),
            "positive_rows": sum(sample.presence for sample in dataset.samples),
            "absent_rows": sum(not sample.presence for sample in dataset.samples),
            "last_diagnostics": last_diagnostics,
            "checkpoint": str(Path(config["output"]["checkpoint"])),
        },
    )
    runner.close()
    print(f"Saved train-time FO Memory checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()

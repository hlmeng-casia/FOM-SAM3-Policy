"""Frozen SAM3 prompt-token interface used by FOM-SAM3.

The official SAM3 package and checkpoint are external. This module only
adapts runtime resolution, injects exported non-spatial prompt tokens, and
exposes native detector outputs/FPN features without modifying upstream code.

Candidate acceptance follows the paper's frozen-head score
`s(q) = sigmoid(s_presence) * sigmoid(s_object(q))`.
"""

from __future__ import annotations

import importlib
import importlib.resources
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from ..models.fo_memory import MemoryArtifact
from ..utils.misc import file_sha256


def joint_candidate_scores(presence_logits: Tensor, object_logits: Tensor) -> Tensor:
    """Shared acceptance score: sigmoid(presence) * sigmoid(object)."""
    presence = presence_logits.float().sigmoid()
    objects = object_logits.float().sigmoid()
    while presence.ndim < objects.ndim:
        presence = presence.unsqueeze(-1)
    return presence * objects


def _activate_external_sam3(source: str | Path) -> None:
    """Make the configured external official SAM3 checkout authoritative."""
    source_root = Path(source).expanduser().resolve()
    package_root = source_root / "sam3"
    if not package_root.is_dir():
        raise RuntimeError(
            f"official SAM3 package was not found at {package_root}; clone/install SAM3 first"
        )
    loaded = sys.modules.get("sam3")
    loaded_file = getattr(loaded, "__file__", None) if loaded is not None else None
    if loaded_file is not None:
        try:
            Path(loaded_file).resolve().relative_to(source_root)
        except ValueError as error:
            raise RuntimeError(
                f"sam3 was already imported from a different checkout: {loaded_file}"
            ) from error
    source_value = str(source_root)
    sys.path[:] = [value for value in sys.path if value != source_value]
    sys.path.insert(0, source_value)
    importlib.invalidate_caches()


def _install_pkg_resources_compat() -> None:
    if "pkg_resources" in sys.modules:
        return
    compatibility = types.ModuleType("pkg_resources")
    compatibility.resource_filename = lambda package, resource: str(
        importlib.resources.files(package).joinpath(resource)
    )
    sys.modules["pkg_resources"] = compatibility


def adapt_sam3_resolution(model: Any, image_size: int, patch_size: int = 14) -> None:
    """Synchronize SAM3's derived 1008-resolution geometry with a smaller input."""
    if image_size <= 0 or image_size % patch_size:
        raise ValueError(f"image_size must be a positive multiple of {patch_size}")
    feature_size = image_size // patch_size
    rope_blocks = 0
    for module in model.modules():
        if getattr(module, "image_size", None) == 1008:
            module.image_size = image_size
        if getattr(module, "resolution", None) == 1008:
            module.resolution = image_size
        if getattr(module, "interpol_size", None) == [1152, 1152]:
            module.interpol_size = [feature_size * 16, feature_size * 16]
        if getattr(module, "sam_image_embedding_size", None) == 72:
            module.sam_image_embedding_size = feature_size
        if getattr(module, "low_res_mask_size", None) == 288:
            module.low_res_mask_size = feature_size * 4
        if getattr(module, "input_mask_size", None) == 1152:
            module.input_mask_size = feature_size * 16
        if module.__class__.__name__ == "PromptEncoder":
            if tuple(getattr(module, "image_embedding_size", ())) == (72, 72):
                module.image_embedding_size = (feature_size, feature_size)
            if tuple(getattr(module, "input_image_size", ())) == (1008, 1008):
                module.input_image_size = (image_size, image_size)
            if tuple(getattr(module, "mask_input_size", ())) == (288, 288):
                module.mask_input_size = (feature_size * 4, feature_size * 4)
        if (
            module.__class__.__name__ == "Attention"
            and getattr(module, "use_rope", False)
            and tuple(getattr(module, "input_size", ())) == (72, 72)
        ):
            module.input_size = (feature_size, feature_size)
            module.rope_interp = True
            module._setup_rope_freqs()
            device = next(module.parameters()).device
            module.freqs_cis = module.freqs_cis.to(device)
            if getattr(module, "use_rope_real", False):
                module.freqs_cis_real = module.freqs_cis_real.to(device)
                module.freqs_cis_imag = module.freqs_cis_imag.to(device)
            rope_blocks += 1
        if getattr(module, "_bb_feat_sizes", None) == [
            (288, 288),
            (144, 144),
            (72, 72),
        ]:
            module._bb_feat_sizes = [
                (feature_size * 4, feature_size * 4),
                (feature_size * 2, feature_size * 2),
                (feature_size, feature_size),
            ]
    if rope_blocks == 0:
        raise RuntimeError("SAM3 global RoPE blocks were not found for resolution adaptation")


@dataclass(frozen=True)
class LocalizationResult:
    present: bool
    score: float
    box_cxcywh: np.ndarray | None
    mask: np.ndarray | None
    query_index: int | None
    rejection_reason: str | None = None


class FrozenSAM3:
    """Official frozen SAM3 detector with external static prompt conditioning."""

    def __init__(
        self,
        *,
        source: str | Path,
        checkpoint: str | Path,
        device: str = "cuda",
        model_resolution: int = 504,
        use_fa3: bool = False,
        compile_model: bool = False,
    ) -> None:
        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("the official SAM3 detector currently requires a CUDA device")
        _activate_external_sam3(source)
        _install_pkg_resources_compat()
        try:
            builder = importlib.import_module("sam3.model_builder")
            processor_module = importlib.import_module("sam3.model.sam3_image_processor")
        except Exception as error:
            raise RuntimeError(
                "failed to import the external official SAM3 package; verify its dependencies"
            ) from error
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        self.checkpoint_sha256 = file_sha256(self.checkpoint)
        self.device = torch.device(device)
        self.model_resolution = int(model_resolution)
        predictor = builder.build_sam3_multiplex_video_predictor(
            checkpoint_path=str(self.checkpoint),
            compile=bool(compile_model),
            warm_up=False,
            use_fa3=bool(use_fa3),
            use_rope_real=True,
        )
        self.predictor = predictor
        adapt_sam3_resolution(predictor.model, self.model_resolution)
        self.detector = predictor.model.detector.eval()
        for parameter in self.detector.parameters():
            parameter.requires_grad_(False)
        # Private Linear training uses the native presence head as a separate
        # rejection objective. Preserve the current paper protocol by stopping
        # the joint candidate-score gradient at presence probability.
        self.detector.detach_presence_in_joint_score = True
        self.processor = processor_module.Sam3Processor(
            self.detector,
            resolution=self.model_resolution,
            device=str(self.device),
            confidence_threshold=0.0,
        )
        self.find_stage = self.processor.find_stage

    def close(self) -> None:
        """Release the external predictor before loading another resolution."""
        self.processor = None
        self.detector = None
        self.predictor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def encode_image(self, image: Image.Image) -> dict[str, Any]:
        with torch.no_grad():
            state = self.processor.set_image(image.convert("RGB"))
        return state["backbone_out"]

    def native_prompt(self, image: Image.Image, text: str) -> tuple[Tensor, Tensor]:
        """Return the native 32 text slots plus dummy geometry prompt."""
        backbone = self.encode_image(image)
        text_output = self.detector.backbone.forward_text([text], device=self.device)
        backbone.update(text_output)
        with torch.no_grad():
            prompt, mask, _ = self.detector._encode_prompt(
                backbone_out=backbone,
                find_input=self.find_stage,
                geometric_prompt=self.detector._get_dummy_prompt(),
                visual_prompt_embed=None,
                visual_prompt_mask=None,
                encode_text=True,
            )
        return prompt[:, 0].detach(), mask[0].detach().bool()

    def _forward_from_backbone(
        self, backbone: dict[str, Any], prompt: Tensor, prompt_mask: Tensor
    ) -> dict[str, Tensor]:
        if prompt.ndim == 2:
            prompt = prompt[:, None]
        if prompt_mask.ndim == 1:
            prompt_mask = prompt_mask[None]
        prompt = prompt.to(self.device)
        prompt_mask = prompt_mask.to(device=self.device, dtype=torch.bool)
        backbone, encoder, _ = self.detector._run_encoder(
            backbone_out=backbone,
            find_input=self.find_stage,
            prompt=prompt,
            prompt_mask=prompt_mask,
        )
        output: dict[str, Any] = {
            "encoder_hidden_states": encoder["encoder_hidden_states"],
            "prev_encoder_out": {"encoder_out": encoder, "backbone_out": backbone},
        }
        output, hidden = self.detector._run_decoder(
            memory=output["encoder_hidden_states"],
            pos_embed=encoder["pos_embed"],
            src_mask=encoder["padding_mask"],
            out=output,
            prompt=prompt,
            prompt_mask=prompt_mask,
            encoder_out=encoder,
        )
        if self.detector.use_dot_prod_scoring:
            object_logits = self.detector.dot_prod_scoring(hidden, prompt, prompt_mask)
        else:
            object_logits = self.detector.class_embed(hidden)
        apply_dac = self.detector.transformer.decoder.dac and self.detector.training
        count = hidden.size(2) // 2 if apply_dac else hidden.size(2)
        output["pred_object_logits_unfused"] = object_logits[-1, :, :count]
        image_ids = self.find_stage.img_ids
        if backbone.get("id_mapping") is not None:
            image_ids = backbone["id_mapping"][image_ids]
        self.detector._run_segmentation_heads(
            out=output,
            backbone_out=backbone,
            img_ids=image_ids,
            vis_feat_sizes=encoder["vis_feat_sizes"],
            encoder_hidden_states=output["encoder_hidden_states"],
            prompt=prompt,
            prompt_mask=prompt_mask,
            hs=hidden,
        )
        return output

    def forward_train(self, image: Image.Image, prompt: Tensor, prompt_mask: Tensor) -> dict[str, Tensor]:
        """Forward frozen SAM3 while retaining gradients only to prompt tensors."""
        return self._forward_from_backbone(self.encode_image(image), prompt, prompt_mask)

    @torch.inference_mode()
    def localize(
        self,
        image: Image.Image,
        artifact: MemoryArtifact,
        *,
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        mask_nms_iou: float = 0.8,
    ) -> LocalizationResult:
        """Localize one identity or explicitly reject it as absent."""
        if artifact.record.get("sam_checkpoint_sha256") != self.checkpoint_sha256:
            raise ValueError("FO Memory was built for a different SAM3 checkpoint")
        if int(artifact.record.get("model_resolution", -1)) != self.model_resolution:
            raise ValueError("FO Memory was built for a different detector resolution")
        prompt, prompt_mask = artifact.prompt_tensors()
        raw = self._forward_from_backbone(self.encode_image(image), prompt, prompt_mask)
        scores = joint_candidate_scores(
            raw["presence_logit_dec"], raw["pred_object_logits_unfused"]
        )[0].reshape(-1)
        order = torch.argsort(scores, descending=True)
        mask_logits = raw["pred_masks"][0]
        resized = F.interpolate(
            mask_logits[:, None].float(),
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        binary = resized.sigmoid() >= float(mask_threshold)
        kept: list[int] = []
        for candidate in order.tolist():
            if float(scores[candidate]) < float(score_threshold):
                break
            current = binary[candidate]
            if not bool(current.any()):
                continue
            duplicate = False
            for previous in kept:
                intersection = (current & binary[previous]).sum().float()
                union = (current | binary[previous]).sum().float().clamp_min(1)
                if float(intersection / union) >= float(mask_nms_iou):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
        if not kept:
            best = int(order[0])
            reason = "score_below_threshold" if float(scores[best]) < score_threshold else "empty_mask"
            return LocalizationResult(False, float(scores[best]), None, None, None, reason)
        best = kept[0]
        box = raw["pred_boxes"][0, best].detach().float().cpu().numpy()
        mask = binary[best].detach().cpu().numpy().astype(bool)
        return LocalizationResult(True, float(scores[best]), box, mask, best)

    @torch.inference_mode()
    def extract_fpn(self, image: Image.Image, level: int = 2) -> Tensor:
        """Return one frozen detector FPN feature as `[C,H,W]`."""
        backbone = self.encode_image(image)
        feature = backbone["backbone_fpn"][int(level)]
        value = feature.tensors if hasattr(feature, "tensors") else feature
        if value.ndim != 4 or value.shape[0] != 1:
            raise RuntimeError(f"unexpected SAM3 FPN shape: {tuple(value.shape)}")
        return value[0].detach()

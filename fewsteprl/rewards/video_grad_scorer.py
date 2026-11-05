"""Differentiable HPSv3 reward and VAE decode for the MeanFlow-OPSD / ReFL trainers.

These trainers need ``torch.autograd.grad(R, x)`` on decoded pixels, but the
scorers in ``video_multi_scorer.py`` run under ``no_grad`` and score from temp
files. HPSv3's ``reward()`` is ``prepare_batch`` (paths -> pixels + tokens)
followed by a differentiable model call, so ``HPSv3GradScorer`` rebuilds the image
half of ``prepare_batch`` in tensor ops and reuses the gradient-free text half.
VideoAlign (mq/ta) decodes video files internally and has no differentiable path.

The HF image processor resizes in uint8, which a differentiable path cannot do, so
this scorer's values differ slightly from the uint8 pipeline's; it is meant to
supply a gradient direction.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class HPSv3GradScorer(nn.Module):
    """HPSv3 as a differentiable function of pixels in ``[0, 1]``.

    Weights are frozen; gradients flow to the input only.
    """

    #: Must match ``HPSv3VideoScorer.GENERAL_PROMPT``.
    GENERAL_PROMPT = "A high-quality image"

    def __init__(self, device: torch.device, hpsv3_root: Optional[str] = None):
        super().__init__()
        # The shared loader applies the transformers compatibility shims.
        from fewsteprl.rewards.video_multi_scorer import _get_hpsv3_inferencer

        self.device = device
        self.inferencer = _get_hpsv3_inferencer(device, hpsv3_root)
        self.model = self.inferencer.model
        self.model.requires_grad_(False)
        self.model.eval()

        ip = self.inferencer.processor.image_processor
        self.patch_size = int(ip.patch_size)
        self.merge_size = int(ip.merge_size)
        self.temporal_patch_size = int(ip.temporal_patch_size)
        # Created directly on ``device``: nothing calls ``.to()`` on this module.
        self.register_buffer(
            "pix_mean",
            torch.tensor(ip.image_mean, dtype=torch.float32, device=device).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pix_std",
            torch.tensor(ip.image_std, dtype=torch.float32, device=device).view(1, 3, 1, 1),
            persistent=False,
        )
        # Gradient-free text half of the batch, cached per (prompt, H, W).
        self._template_cache: dict[tuple, dict[str, torch.Tensor]] = {}

    # --- batch construction ---

    def _template(self, prompt: str, h: int, w: int) -> dict[str, Any]:
        """Token ids and image grid for ``prompt`` at pixel size ``(h, w)``.

        Obtained by running the real ``prepare_batch`` once on a dummy image, so the
        chat template and ``image_grid_thw`` match HPSv3 exactly.
        """
        key = (prompt, h, w)
        hit = self._template_cache.get(key)
        if hit is not None:
            return hit

        import os
        import tempfile

        import numpy as np
        from PIL import Image

        tmpdir = tempfile.mkdtemp(prefix="hpsv3grad-")
        png = os.path.join(tmpdir, "probe.png")
        try:
            Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(png)
            batch = self.inferencer.prepare_batch([png], [prompt])
        finally:
            try:
                os.remove(png)
                os.rmdir(tmpdir)
            except OSError:
                pass

        grid = batch["image_grid_thw"]
        tpl = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "image_grid_thw": grid,
            "pv_dtype": batch["pixel_values"].dtype,
            # Resize target derived from the grid the processor chose.
            "rh": int(grid[0][1]) * self.patch_size,
            "rw": int(grid[0][2]) * self.patch_size,
            "gh": int(grid[0][1]),
            "gw": int(grid[0][2]),
        }
        self._template_cache[key] = tpl
        return tpl

    def _to_pixel_values(self, img01: torch.Tensor, tpl: dict[str, Any]) -> torch.Tensor:
        """``[3, H, W]`` in [0, 1] -> ``[n_patches, C*tp*ps*ps]``, differentiably.

        Reproduces ``Qwen2VLImageProcessor._preprocess``'s reshape/permute order.
        ``antialias=True`` keeps the float resize much closer to the uint8 one.
        """
        import torchvision.transforms.v2.functional as TVF

        ps, ms, tp = self.patch_size, self.merge_size, self.temporal_patch_size
        gh, gw = tpl["gh"], tpl["gw"]
        x = TVF.resize(
            img01.unsqueeze(0), [tpl["rh"], tpl["rw"]],
            interpolation=TVF.InterpolationMode.BICUBIC, antialias=True,
        )
        x = (x.clamp(0.0, 1.0) - self.pix_mean) / self.pix_std      # [1,3,rh,rw]
        x = x.squeeze(0).unsqueeze(0).repeat(tp, 1, 1, 1)           # temporal patch
        x = x.reshape(1, tp, 3, gh // ms, ms, ps, gw // ms, ms, ps)
        x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        return x.reshape(gh * gw, 3 * tp * ps * ps)

    # --- scoring ---

    def forward(
        self,
        images01: torch.Tensor,
        prompts: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """Differentiable reward ``[B]`` for ``[B, 3, H, W]`` pixels in ``[0, 1]``.

        ``prompts=None`` uses :attr:`GENERAL_PROMPT` (``hpsv3_general``); passing
        captions gives ``hpsv3_percentile``'s per-frame score (the top-30% frame
        reduction is left to the caller). Images are scored one at a time to bound
        peak memory.
        """
        if images01.ndim != 4 or images01.shape[1] != 3:
            raise ValueError(
                f"images01 must be [B, 3, H, W], got {tuple(images01.shape)}."
            )
        b, _, h, w = images01.shape
        if prompts is None:
            prompts = [self.GENERAL_PROMPT] * b
        if len(prompts) != b:
            raise ValueError(f"{len(prompts)} prompts for {b} images.")

        out = []
        for i in range(b):
            tpl = self._template(prompts[i], h, w)
            pv = self._to_pixel_values(images01[i], tpl).to(dtype=tpl["pv_dtype"])
            logits = self.model(
                return_dict=True,
                input_ids=tpl["input_ids"],
                attention_mask=tpl["attention_mask"],
                pixel_values=pv,
                image_grid_thw=tpl["image_grid_thw"],
            )["logits"]
            out.append(logits.float().reshape(-1)[0])
        return torch.stack(out)

    def to_device(self, device: torch.device) -> "HPSv3GradScorer":
        """Move the reward model (used to free GPU memory between phases)."""
        self.model.to(device)
        self.device = device
        if hasattr(self.inferencer, "device"):
            self.inferencer.device = device
        return self


def decode_wan_latents_grad(vae: nn.Module, latents: torch.Tensor) -> torch.Tensor:
    """Differentiable Wan VAE decode -> ``[B, F, 3, H, W]`` in ``[0, 1]``.

    Same arithmetic as ``fewsteprl.models.wan.decode_wan_latents``, which is
    ``no_grad``. Runs in fp32: a bf16 backward noticeably rotates the gradient
    direction, which is the only part of the gradient the targets use.
    """
    cfg = vae.config
    z_dim = int(getattr(cfg, "z_dim", latents.shape[2]))
    mean = getattr(cfg, "latents_mean", None)
    std = getattr(cfg, "latents_std", None)

    # [B, F, C, H, W] -> [B, C, F, H, W] for the VAE.
    x = latents.to(torch.float32).permute(0, 2, 1, 3, 4)
    if mean is not None and std is not None:
        m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, z_dim, 1, 1, 1)
        s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, z_dim, 1, 1, 1)
        x = x / s + m
    frames = vae.decode(x, return_dict=False)[0]          # [B, 3, F, H, W]
    frames = (frames / 2.0 + 0.5).clamp(0.0, 1.0)
    return frames.permute(0, 2, 1, 3, 4)                   # [B, F, 3, H, W]

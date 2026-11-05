"""Differentiable video-reward gradients for the OPSD / ReFL trainers (mixed into both).

OPSD and ReFL need ``autograd.grad(R, x)`` on decoded pixels and use only its
direction (targets move along a unit-normalized gradient). The stock Wan rewards
score from files under ``no_grad``, so this mixin decodes with grad through an
fp32 view of the causal 3D VAE and scores with the tensor-native
``HPSv3GradScorer`` (``fewsteprl/rewards/video_grad_scorer.py``). VideoAlign
(mq/ta) has no differentiable path.

A whole-clip decode with grad does not fit in GPU memory (the causal VAE keeps
the activations of the whole prefix alive), so the clip is decoded in windows of
``OPSD_CHUNK_LATENT_FRAMES`` latent frames and each window's graph is freed once
its gradient is taken. Because the VAE is causal, a window decoded alone is
treated as a clip start, so the windowed gradient misses the cross-window terms
of the joint-decode gradient. The reward model runs in fp32: a bf16 backward
noticeably rotates the gradient direction.

The scorer supplies a direction only. Reported rewards and advantages still come
from the base ``VideoMultiScorer`` (uint8 path), as for every other trainer.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


class OPSDGradMixin:
    """Supplies the differentiable decode / reward gradient for the OPSD trainers.

    Mixed in before the trainer class so these methods override the generic
    (image) implementations.
    """

    #: Latent frames decoded jointly per window in the deterministic sweep.
    #: Wider windows keep more cross-frame terms but cost far more memory.
    OPSD_CHUNK_LATENT_FRAMES: int = 1

    #: Draw one random window per reward-gradient call instead of sweeping the
    #: whole clip (off by default).
    OPSD_RANDOM_WINDOW: bool = False

    #: Window width when ``OPSD_RANDOM_WINDOW`` is on.
    OPSD_WINDOW_RANDOM: int = 2

    #: Run the reward model in fp32 rather than its shipped bf16; a bf16
    #: backward rotates the gradient direction while barely changing the value.
    #: Requires sdpa attention (FlashAttention is fp16/bf16 only).
    OPSD_SCORER_FP32: bool = True

    #: Offload saved activations to host RAM during the reward-gradient pass.
    #: Trades GPU memory for large host RAM and PCIe time; the host budget grows
    #: quickly with the window width and rank count.
    OPSD_ACTIVATION_OFFLOAD: bool = False

    #: Weights for ``opsd_reward_kind="mixed"`` (target direction only). The two
    #: HPSv3 reductions share one model:
    #:   general    -- fixed prompt, mean over frames
    #:   percentile -- the caption, mean of the top 30% of frames
    OPSD_MIXED_WEIGHTS: dict = {"hpsv3_general": 1.0, "hpsv3_percentile": 1.0}

    #: Fraction of frames ``hpsv3_percentile`` keeps, highest-scoring first
    #: (matches ``HPSv3VideoScorer``).
    OPSD_PERCENTILE_FRAC: float = 0.3

    _opsd_grad_scorer: Optional[Any] = None
    _opsd_vae32: Optional[Any] = None

    # ------------------------------------------------------------------

    def _opsd_grad_ctx(self):
        """Context the whole reward-gradient pass runs inside."""
        if not self.OPSD_ACTIVATION_OFFLOAD:
            return contextlib.nullcontext()
        from torch.autograd.graph import save_on_cpu

        # pin_memory=True raises `CUDA error: invalid argument` at this
        # activation volume.
        return save_on_cpu(pin_memory=False)

    def _opsd_reward_kinds(self) -> dict:
        """``{kind: weight}`` the gradient follows. One entry unless "mixed".

        Config-derived only, so it is identical (and identically ordered) on
        every rank.
        """
        kind = str(getattr(self._rl_cfg(), "opsd_reward_kind", "hpsv3_general"))
        if kind == "mixed":
            return dict(self.OPSD_MIXED_WEIGHTS)
        return {kind: 1.0}

    @staticmethod
    def _opsd_cast_scorer_fp32(sc: Any) -> str:
        """Cast the reward model to fp32, switching attention to sdpa first.

        FlashAttention accepts fp16/bf16 only, so the switch must precede the
        cast. It is applied to every submodule config, including Qwen2-VL's
        separate ``vision_config`` (the pixel gradient flows through the vision
        tower). Returns a dtype label to log; on failure the scorer stays bf16
        with a warning.
        """
        try:
            for m in sc.model.modules():
                cfg = getattr(m, "config", None)
                if cfg is not None and getattr(cfg, "_attn_implementation", None):
                    cfg._attn_implementation = "sdpa"
            cfg = getattr(sc.model, "config", None)
            if cfg is not None:
                cfg._attn_implementation = "sdpa"
                if getattr(cfg, "vision_config", None) is not None:
                    cfg.vision_config._attn_implementation = "sdpa"
            sc.model = sc.model.float()
            return "fp32+sdpa"
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[OPSD] fp32 cast of the reward model failed (%s); falling back "
                "to bf16; the bf16 reward gradient rotates the target direction "
                "noticeably.", e,
            )
            return "bf16 (fp32 cast FAILED)"

    def _ensure_grad_scorer(self) -> None:
        """Build the resident differentiable scorer and the fp32 VAE."""
        device = torch.device("cuda")
        if self._opsd_grad_scorer is None:
            from fewsteprl.rewards.video_grad_scorer import HPSv3GradScorer

            for kind in self._opsd_reward_kinds():
                if not kind.startswith("hpsv3"):
                    raise ValueError(
                        f"fewstep_rl.opsd_reward_kind yields {kind!r}, which has "
                        "no differentiable implementation on the wan branch. Only "
                        "HPSv3 can be differentiated here: VideoAlign's reward() "
                        "takes video_paths and decodes inside Qwen2-VL's video "
                        "pipeline, so there is no tensor seam. See "
                        "fewsteprl/rewards/video_grad_scorer.py."
                    )
            # One scorer instance serves both HPSv3 reductions.
            self._opsd_grad_scorer = HPSv3GradScorer(device)
            dt = "bf16"
            if self.OPSD_SCORER_FP32:
                dt = self._opsd_cast_scorer_fp32(self._opsd_grad_scorer)
            logger.info(
                "[OPSD] differentiable HPSv3 scorer built (VAE fp32, scorer %s); "
                "gradient follows %s",
                dt,
                ", ".join(f"{k}x{v:g}" for k, v in self._opsd_reward_kinds().items()),
            )
        if self._opsd_vae32 is None:
            # fp32 VAE for the gradient pass: a bf16 decode backward rotates the
            # gradient direction substantially on Wan.
            self._opsd_vae32 = self.vae.to(torch.float32)
            self._opsd_vae32.disable_tiling()   # tiling multiplies the backward
            logger.info("[OPSD] fp32 VAE view ready (tiling disabled)")

    def _opsd_score_frames(
        self, kind: str, imgs: torch.Tensor, prompt: str,
    ) -> torch.Tensor:
        """One clip's scalar reward under ``kind``, for ``imgs`` ``[f,3,H,W]``.

        Same two reductions as ``HPSv3VideoScorer``, without ``no_grad``:

          general    -- fixed prompt, mean over all frames.
          percentile -- the caption, mean of the top ``OPSD_PERCENTILE_FRAC``
                        of frames by score (only those frames get gradient).
        """
        sc = self._opsd_grad_scorer
        if kind == "hpsv3_general":
            # prompts=None selects the scorer's GENERAL_PROMPT.
            return sc(imgs, None).mean()
        if kind == "hpsv3_percentile":
            per_frame = sc(imgs, [prompt] * imgs.shape[0])
            k = max(1, int(per_frame.numel() * float(self.OPSD_PERCENTILE_FRAC)))
            return torch.topk(per_frame, k).values.mean()
        raise ValueError(f"unsupported differentiable reward kind {kind!r}")

    def _opsd_release_grad_memory(self) -> None:
        """Drop the resident scorer before the policy backward peaks."""
        if self._opsd_grad_scorer is not None:
            self._opsd_grad_scorer.to_device(torch.device("cpu"))
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------

    def _decode01_grad(self, latents: torch.Tensor) -> torch.Tensor:
        """Differentiable Wan VAE decode -> ``[B, F, 3, H, W]`` in ``[0, 1]``."""
        from fewsteprl.rewards.video_grad_scorer import decode_wan_latents_grad

        self._ensure_grad_scorer()
        return decode_wan_latents_grad(self._opsd_vae32, latents)

    def _reward_grad_chunked(
        self, latents: torch.Tensor, prompts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(reward_per_sample, dR/d(latents))``, computed chunk by chunk.

        Returns the gradient directly so each window's graph can be freed once
        its gradient is accumulated; peak memory is one window's regardless of
        clip length. Do not wrap the result in ``autograd.grad``.
        """
        self._ensure_grad_scorer()
        self._opsd_grad_scorer.to_device(torch.device("cuda"))
        kinds = self._opsd_reward_kinds()

        n_lat = latents.shape[1]
        grads = torch.zeros_like(latents)
        rewards = latents.new_zeros(latents.shape[0])

        # The window count depends only on shapes and class constants, never on
        # data, so every rank runs the same number of iterations.
        if self.OPSD_RANDOM_WINDOW:
            w = max(1, min(int(self.OPSD_WINDOW_RANDOM), n_lat))
            # Per-rank offset from the ambient (per-rank seeded) RNG; no
            # collective needed.
            i0 = int(torch.randint(0, max(1, n_lat - w + 1), (1,)).item())
            spans = [(i0, i0 + w)]
        else:
            step = max(1, int(self.OPSD_CHUNK_LATENT_FRAMES))
            spans = [(i, min(i + step, n_lat)) for i in range(0, n_lat, step)]

        for i, j in spans:
            with self._opsd_grad_ctx():
                xi = latents[:, i:j].detach().clone().requires_grad_(True)
                videos = self._decode01_grad(xi)           # [B, f, 3, H, W]
                per_sample = []
                for b in range(videos.shape[0]):
                    imgs = videos[b]                       # [f, 3, H, W]
                    p = str(prompts[b]) if prompts is not None else ""
                    # Weighted sum over reward kinds (one unless "mixed").
                    tot = None
                    for kind, w in kinds.items():
                        term = float(w) * self._opsd_score_frames(kind, imgs, p)
                        tot = term if tot is None else tot + term
                    per_sample.append(tot)
                r = torch.stack(per_sample)
            gi, = torch.autograd.grad(r.sum(), xi)
            grads[:, i:j] = gi
            rewards = rewards + r.detach()
            del xi, videos, r, gi

        # Mean over visited windows keeps the reward on a per-window scale.
        return rewards / len(spans), grads


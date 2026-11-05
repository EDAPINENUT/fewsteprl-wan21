"""Shared base for the x0-form RL trainer family on Wan2.1-T2V.

:class:`GeneralRLTrainer` implements the Wan engine (model setup, K-repeat
rollouts, reward/advantage pipeline, inner update loop, EMA/old-model decay,
checkpoint hooks) and the general x0-form loss::

    x_0_pred_u = x_t - t * u_theta(x_t, r, t)
    x_anchor_1 = x_t - t * u_old(x_t, r, t)
    x_anchor_2 = x_t - t * v_inst
    dev        = x_0_pred_u - x_anchor_1
    x_0_target = x_anchor_1 + omega * (x_0_rwd - x_anchor_2)
    loss       = weight * MSE(x_0_pred_u, sg(x_0_target))

Subclasses choose which instantaneous velocity ``v_inst`` is paired with ``x_t``
by overriding :meth:`_instantaneous_velocity_from_u` and the two anchor hooks.
Keeping the anchors separate makes ``dev`` free of the flow-map correction
``S = t01*(t-r)*dF/dt``, which only enters the detached ``x_anchor_2``; ``dev``
must use ``x_anchor_1``, never ``x_anchor_2``. :class:`AnchorInputs` packs and
validates the operands the anchor hooks may read.
"""

from __future__ import annotations

import copy
import logging
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers.training_utils import compute_density_for_timestep_sampling
from torch.utils.data import DataLoader

from fewsteprl.config import FewStepRLConfig
from fewsteprl.data.prompt_dataset import PromptDataset
from fewsteprl.models.wan import (
    decode_wan_latents,
    encode_prompts_wan,
    load_wan_models,
    predict_noise_wan,
)
from fewsteprl.parallel.utils import (
    ddp_wrap_model,
    fsdp_wrap_model,
    get_rank,
    get_transformer_wrap_policy,
    get_world_size,
    is_main_process,
)
from fewsteprl.rewards.video_multi_scorer import VideoMultiScorer
from fewsteprl.schedulers.flowmap_scheduler import FlowMapScheduler
from fewsteprl.trainers.base_trainer import BaseTrainer
from fewsteprl.trainers.rl_utils import DistributedKRepeatSampler, PerPromptStatTracker
from fewsteprl.utils.lora import setup_lora

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnchorInputs:
    """Operands the anchor hooks may read, validated once at construction.

    The x0-space operands (``x_t``, ``t_normed``, ``u_old``, ``u_theta``) must be
    fp32/fp64, detached, and ``t_normed`` must be broadcast to the latent rank: a
    bf16 operand would quantize the S-free deviation, and a ``[B]`` timestep would
    broadcast against the last (width) axis of the 5D latent instead of the batch
    axis. ``u_theta`` is detached so the anchors stay stop-grad; ``_rl_loss``
    builds the differentiable ``x_0_pred_u`` separately.

    ``prompt_embeds`` / ``pooled_embeds`` are unchecked: they feed the network in
    its own dtype. On Wan ``pooled_embeds`` is a ``[B, 1]`` zero placeholder (UMT5
    has no pooled projection).
    """

    x_t: torch.Tensor              #: noisy latent at t, [B, F, C, H, W]
    t_normed: torch.Tensor         #: t/T, broadcast to x_t's rank
    u_old: torch.Tensor            #: old_model average velocity u(x_t, r, t)
    u_theta: torch.Tensor          #: generator average velocity u(x_t, r, t), detached
    dF_dt: Optional[torch.Tensor]  #: shared central difference; None when r == t everywhere
    t_cd: torch.Tensor             #: raw timesteps, in the central difference's own units
    r_cd: Optional[torch.Tensor]   #: raw flow-map target timesteps, same units as t_cd
    prompt_embeds: torch.Tensor
    pooled_embeds: torch.Tensor

    def __post_init__(self) -> None:
        for name in ("x_t", "t_normed", "u_old", "u_theta"):
            operand = getattr(self, name)
            if operand.dtype not in (torch.float32, torch.float64):
                raise TypeError(
                    f"AnchorInputs.{name} is {operand.dtype}; the anchor algebra "
                    "runs in x0 space and needs the fp32/fp64 copy, not the "
                    "network's dtype."
                )
            if operand.requires_grad:
                raise RuntimeError(
                    f"AnchorInputs.{name} carries grad; the anchors are stop-grad "
                    "reference points and only x_0_pred_u may be differentiable."
                )
        if self.t_normed.ndim != self.x_t.ndim:
            raise ValueError(
                f"AnchorInputs.t_normed has ndim {self.t_normed.ndim}, expected "
                f"{self.x_t.ndim} to match x_t. Pass the broadcast timestep, not "
                "the [B] vector."
            )


class GeneralRLTrainer(BaseTrainer):
    """x0-form RL loss for Wan. Subclasses override only the velocity identity."""

    #: Set by subclasses; also saved in the checkpoint, since config.yaml does
    #: not record the ``--trainer`` registry key.
    TRAINER_KEY: str = "rl_general"

    #: Working dtype for the x0-space algebra (fp64 is useful for equivalence
    #: tests).
    _rl_work_dtype: torch.dtype = torch.float32

    #: Dtype the central-difference endpoints are cast to before subtraction in
    #: :meth:`_central_diff_dudt` (bf16 loses most of the quotient's precision).
    #: Hold it fixed across arms being compared. ``None`` keeps the network dtype.
    _cd_out_dtype: Optional[torch.dtype] = None

    #: Logged every step on every rank as ``rl_diagnostic/*``. The key set must be
    #: identical across ranks: ``BaseTrainer._reduce_metrics`` all-reduces per key
    #: and a mismatched key set hangs.
    _EXTRA_DIAGNOSTIC_KEYS: tuple[str, ...] = (
        "s_x0_abs_mean",        # |S_x0| = |x_anchor_1 - x_anchor_2|
        "s_rel_median",         # |S_x0| / |dev|
    )

    #: Logged every step on every rank as ``rl_loss/*``; same rank-uniformity
    #: requirement. ``_run_rl_epoch`` drops keys that are not registered here.
    #: The decoupled loss differs from the fused one by a cross term, so these are
    #: only comparable within an arm (see :meth:`decoupled_rl_loss`).
    _EXTRA_LOSS_KEYS: tuple[str, ...] = (
        "self_regression_loss",       # W * mean_sp[(P - x_anchor_1)^2]
        "reward_augmentation_loss",   # W * mean_sp[norm_grad^2]
        "loss_kl_unweighted",         # KL before t-weighting
        "kl_t_weight_mean",           # mean beta08 t-weight actually applied
        "kl_rebalance_mean",          # mean mode-rebalance multiplier (1.0 == inactive)
    )

    #: Start of the aux epoch's ``set_epoch`` stream, above any value the policy
    #: stream (``_rl_epoch * num_batches_per_epoch + i``) can reach, so the two
    #: never draw the same prompts.
    _AUX_SAMPLER_EPOCH_OFFSET: int = 1_000_000_000

    #: RNG stream offset for :meth:`_draw_training_s_given_t`, so ``s`` is not
    #: seeded identically to the ``(t, r)`` draw at the same coordinates. Large and
    #: coprime with the other seed multipliers. Declared on the base because the
    #: base method ``_make_training_tr_rng`` reads it.
    _S_RNG_STREAM_OFFSET: int = 2_654_435_761

    def __init__(self, config: FewStepRLConfig):
        super().__init__(config)
        self.model_type = str(self.config.model.model_type).lower()
        if self.model_type != "wan":
            raise ValueError(
                "The wan branch only supports model_type='wan'; "
                f"got {self.model_type!r}."
            )

        # Models (populated in setup_models)
        self.generator: Optional[nn.Module] = None
        self.generator_ema: Optional[nn.Module] = None
        self.old_model: Optional[nn.Module] = None
        self.ref_u_model: Optional[nn.Module] = None  # only built if kl_weight > 0
        self.text_encoders: list[nn.Module] = []
        self.tokenizers: list = []
        self.scheduler = None
        self.vae = None
        self.latent_channels: Optional[int] = None
        # Video latent geometry (5D). ``latent_size`` is kept only for inherited
        # BaseTrainer helpers.
        self.latent_frames: Optional[int] = None
        self.eval_latent_frames: Optional[int] = None
        self.latent_h: Optional[int] = None
        self.latent_w: Optional[int] = None
        self.latent_size: Optional[int] = None

        # Flow-map scheduler is initialized before LoRA injection.
        self.flowmap_scheduler: Optional[FlowMapScheduler] = None

        # Central difference computed once per ``_rl_loss`` step on old_model
        # (:meth:`_pre_compute_v_hook`), read by :meth:`_compute_x_anchor_2`,
        # cleared by :meth:`_post_compute_v_hook`.
        self._shared_dF_dt: Optional[torch.Tensor] = None

        # Reward scorers (lazy-initialized on first sampling phase / eval).
        self._reward_scorer: Optional[VideoMultiScorer] = None
        self._eval_video_scorer: Optional[VideoMultiScorer] = None
        self._stat_tracker: Optional[PerPromptStatTracker] = None

        # Epoch-based outer loop state.
        self._rl_epoch: int = 0
        self._rl_dataloader: Optional[DataLoader] = None
        self._rl_sampler: Optional[DistributedKRepeatSampler] = None
        self._rl_sample_iter = None

        # Auxiliary v-predictor (DMD-NFT). Built by setup_models only when
        # ``fewstep_rl.aux_v_enabled``; aux-path guards test ``is not None``.
        self.aux_v_model: Optional[nn.Module] = None
        self._aux_sample_iter = None
        self._aux_sample_round: int = 0
        # Lazily built by _evaluate (separate from the base image-scorer slot).
        self._eval_video_scorer: Optional[VideoMultiScorer] = None

    def _rl_cfg(self):
        return self.config.fewstep_rl

    def _kl_usage(self) -> tuple[bool, bool]:
        """Return ``(use_u_kl, use_v_kl)``.

        ``use_v_kl`` is always False on Wan: AnyFlow-Wan has no ``r``-free
        v-reference, and the config rejects ``use_ref_v_model``.
        """
        cfg = self._rl_cfg()
        kl_weight = float(cfg.kl_weight)
        use_u_kl = kl_weight > 0.0 and self.ref_u_model is not None
        return use_u_kl, False

    # --- Wan model setup ---

    def setup_models(self) -> None:
        """Load AnyFlow-Wan, inject LoRA, and build the policy/old/ref copies."""
        model_cfg = self.config.model
        dist_cfg = self.config.distributed

        logger.info("=" * 60)
        logger.info("Setting up Wan %s models", self.TRAINER_KEY)
        logger.info("=" * 60)

        components = load_wan_models(model_cfg)
        base_transformer = components["transformer"]
        self.text_encoders = components["text_encoders"]
        self.tokenizers = components["tokenizers"]
        self.scheduler = components["scheduler"]
        self.vae = components["vae"]

        if model_cfg.gradient_checkpointing:
            base_transformer.enable_gradient_checkpointing()

        self.generator = copy.deepcopy(base_transformer)
        # aux_v_model takes the pristine transformer (no LoRA, no flow-map
        # wrapper) as a plain v(x_t, t) predictor; self.generator is already an
        # independent copy, so no second deepcopy is needed.
        aux_v = base_transformer if bool(self._rl_cfg().aux_v_enabled) else None
        del base_transformer

        self._load_model_init_from_path(
            self.generator, model_cfg.generator_init_path, "generator",
        )

        # Wan is natively two-time; only the flow-map scheduler is needed.
        self._setup_scheduler()

        if model_cfg.generator_lora.enabled:
            setup_lora(self.generator, model_cfg.generator_lora)
            logger.info(
                "  Generator: LoRA injected (rank=%d)", model_cfg.generator_lora.rank,
            )
        else:
            self.generator.train()
            logger.info("  Generator: full-weight trainable")

        # Unfreeze delta_embedder (the r-conditioning MLP) after LoRA froze
        # everything. Persisted alongside LoRA by the checkpoint code.
        self._post_setup_lora()

        for enc in self.text_encoders:
            enc.requires_grad_(False)
            enc.eval()
        self.vae.requires_grad_(False)
        self.vae.eval()
        device = torch.device("cuda")
        self.vae.to(device)
        for enc in self.text_encoders:
            enc.to(device)

        # --- Video latent geometry ---
        self.latent_channels = self.generator.config.in_channels
        vae_t = int(getattr(self.vae.config, "scale_factor_temporal", 4))
        vae_s = int(getattr(self.vae.config, "scale_factor_spatial", 8))
        vcfg = self.config.video
        if (vcfg.num_frames - 1) % vae_t != 0:
            raise ValueError(
                f"video.num_frames-1 ({vcfg.num_frames - 1}) must be divisible by "
                f"vae temporal factor {vae_t} (e.g. 81 frames for factor 4)."
            )
        if vcfg.height % vae_s or vcfg.width % vae_s:
            raise ValueError(
                f"video.height/width ({vcfg.height}x{vcfg.width}) must be "
                f"divisible by vae spatial factor {vae_s}."
            )
        self.latent_frames = (vcfg.num_frames - 1) // vae_t + 1
        self.latent_h = vcfg.height // vae_s
        self.latent_w = vcfg.width // vae_s
        eval_nf = int(getattr(vcfg, "eval_num_frames", 0) or 0)
        if eval_nf > 0:
            if (eval_nf - 1) % vae_t != 0:
                raise ValueError(
                    f"video.eval_num_frames-1 ({eval_nf - 1}) must be divisible "
                    f"by vae temporal factor {vae_t}."
                )
            self.eval_latent_frames = (eval_nf - 1) // vae_t + 1
        else:
            self.eval_latent_frames = self.latent_frames
        # Benign value for any inherited reference to ``latent_size``.
        self.latent_size = self.latent_h

        if self.config.train.use_ema:
            self.generator_ema = copy.deepcopy(self.generator)
            self.generator_ema.requires_grad_(False)
            self.generator_ema.eval()

        self.old_model = copy.deepcopy(self.generator)
        self.old_model.requires_grad_(False)
        self.old_model.eval()

        # u-mode KL reference: a frozen snapshot of the generator, so both sides
        # of the divergence are average velocities and no S enters.
        if float(self._rl_cfg().kl_weight) > 0.0:
            self.ref_u_model = copy.deepcopy(self.generator)
            self.ref_u_model.requires_grad_(False)
            self.ref_u_model.eval()

        # aux_v_model: trainable v-predictor fitted online by
        # :meth:`_run_aux_epoch` with a flow-matching loss. Assigned before the
        # dtype cast and _wrap_models so it is cast, wrapped, and registered in
        # self.models (and hence checkpointed) like the other models.
        if aux_v is not None:
            self.aux_v_model = aux_v
            # Must load before _wrap_models; FSDP shards the module afterwards.
            self._load_model_init_from_path(
                self.aux_v_model, model_cfg.aux_v_init_path, "aux_v",
                subdirs=("aux_v_transformer", "transformer"),
            )
            self.aux_v_model.requires_grad_(True)
            self.aux_v_model.train()
            logger.info(
                "  aux_v_model: Wan v-predictor (trainable, init=%s)",
                model_cfg.aux_v_init_path or "pretrained",
            )

        _dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        _fsdp_dtype = _dtype_map[model_cfg.dtype]
        # FSDP needs a uniform dtype per flat param group, but the Wan
        # transformer keeps some modules in fp32 (``_keep_in_fp32_modules``), so
        # cast every model to the FSDP param dtype before wrapping. FP32LayerNorm
        # still computes in fp32 internally.
        for _m in (
            self.generator, self.generator_ema, self.old_model, self.ref_u_model,
            self.aux_v_model,
        ):
            if _m is not None:
                _m.to(_fsdp_dtype)

        self._wrap_models(dist_cfg)

        self.models = {"generator": self.generator}
        if self.generator_ema is not None:
            self.models["generator_ema"] = self.generator_ema
        if self.old_model is not None:
            self.models["old_model"] = self.old_model
        if self.ref_u_model is not None:
            self.models["ref_u_model"] = self.ref_u_model
        if self.aux_v_model is not None:
            self.models["aux_v_model"] = self.aux_v_model

        if self._rl_cfg().per_prompt_stat_tracking:
            self._stat_tracker = PerPromptStatTracker(
                global_std=self._rl_cfg().per_prompt_global_std,
            )

        if get_rank() == 0:
            logger.info(
                "[%s] latent geometry: frames=%d, C=%d, H=%d, W=%d",
                self.TRAINER_KEY, self.latent_frames, self.latent_channels,
                self.latent_h, self.latent_w,
            )
            logger.info(
                "[%s] cd_velocity_source=%s, work_dtype=%s",
                self.TRAINER_KEY, self._rl_cfg().cd_velocity_source,
                self._rl_work_dtype,
            )
        logger.info("Wan %s model setup complete", self.TRAINER_KEY)
        logger.info("=" * 60)

    def _setup_scheduler(self) -> None:
        """Build the flow-map scheduler (Wan is already two-time natively)."""
        self.flowmap_scheduler = FlowMapScheduler(
            num_train_timesteps=int(self.scheduler.config.num_train_timesteps),
            shift=float(getattr(self.scheduler.config, "shift", 5.0)),
            weight_type="uniform",
        )

    def _post_setup_lora(self) -> None:
        """Re-enable delta_embedder gradients after setup_lora freezes everything."""
        if not self.config.model.generator_lora.enabled:
            return
        for name, param in self.generator.named_parameters():
            if "delta_embedder" in name:
                param.requires_grad = True

    def _wrap_models(self, dist_cfg) -> None:
        """FSDP/DDP wrap on Wan transformer-block boundaries."""
        strategy = dist_cfg.strategy
        fsdp_precision = self.config.model.dtype

        if strategy == "fsdp":
            from fewsteprl.models.wan_transformer import WanTransformerBlock
            wrap_policy = get_transformer_wrap_policy(WanTransformerBlock)
            for name in (
                "generator", "generator_ema", "old_model", "ref_u_model",
                "aux_v_model",
            ):
                mod = getattr(self, name)
                if mod is not None:
                    setattr(
                        self, name,
                        fsdp_wrap_model(
                            mod,
                            sharding_strategy=dist_cfg.fsdp_sharding,
                            fsdp_precision=fsdp_precision,
                            auto_wrap_policy=wrap_policy,
                        ),
                    )
        elif strategy == "ddp":
            device = torch.device("cuda")
            has_lora = self.config.model.generator_lora.enabled
            ddp_find_unused_cfg = self.config.distributed.ddp_find_unused_parameters
            find_unused = has_lora if ddp_find_unused_cfg is None else bool(ddp_find_unused_cfg)
            self.generator = ddp_wrap_model(
                self.generator.to(device), find_unused_parameters=find_unused,
            )
            for name in ("generator_ema", "old_model", "ref_u_model"):
                mod = getattr(self, name)
                if mod is not None:
                    setattr(self, name, mod.to(device))
            # aux_v_model is trained, so it needs DDP to all-reduce its grads;
            # otherwise each rank's v-predictor (and x_anchor_2) would diverge.
            if self.aux_v_model is not None:
                self.aux_v_model = ddp_wrap_model(
                    self.aux_v_model.to(device), find_unused_parameters=False,
                )
        else:
            raise ValueError(f"Unknown distributed strategy: {strategy}")

    def setup_optimizers(self) -> None:
        """AdamW over the trainable generator params (LoRA + delta_embedder)."""
        solver_cfg = self.config.solver.generator
        params = [p for p in self.generator.parameters() if p.requires_grad]
        if not params:
            raise ValueError(
                "No trainable generator parameters: LoRA injection and the "
                "delta_embedder unfreeze both appear to have been skipped."
            )
        self.optimizers["generator"] = torch.optim.AdamW(
            params,
            lr=float(solver_cfg.lr),
            betas=(float(solver_cfg.beta1), float(solver_cfg.beta2)),
            eps=float(solver_cfg.eps),
            weight_decay=float(solver_cfg.weight_decay),
        )
        self.schedulers["generator"] = self.create_warmup_constant_scheduler(
            self.optimizers["generator"], int(solver_cfg.warmup_steps),
        )
        n_params = sum(p.numel() for p in params) / 1e6
        logger.info(
            "Optimizer created: generator (lr=%g, trainable_params=%.1f M)",
            float(solver_cfg.lr), n_params,
        )

        # Separate optimizer for the aux v-predictor (full fine-tune, no LoRA).
        if self.aux_v_model is not None:
            aux_cfg = self.config.solver.aux_v
            aux_params = [p for p in self.aux_v_model.parameters() if p.requires_grad]
            if not aux_params:
                raise ValueError(
                    "aux_v_model has no trainable parameters, so _run_aux_epoch "
                    "would take zero optimizer steps and x_anchor_2 would stay "
                    "pinned at the pretrained marginal forever."
                )
            self.optimizers["aux_v"] = torch.optim.AdamW(
                aux_params,
                lr=float(aux_cfg.lr),
                betas=(float(aux_cfg.beta1), float(aux_cfg.beta2)),
                eps=float(aux_cfg.eps),
                weight_decay=float(aux_cfg.weight_decay),
            )
            self.schedulers["aux_v"] = self.create_warmup_constant_scheduler(
                self.optimizers["aux_v"], int(aux_cfg.warmup_steps),
            )
            logger.info(
                "Optimizer created: aux_v (lr=%g, trainable_params=%.1f M)",
                float(aux_cfg.lr), sum(p.numel() for p in aux_params) / 1e6,
            )

    def _load_model_init_from_path(
        self,
        model: nn.Module,
        init_path: str,
        role_name: str,
        subdirs: tuple[str, ...] = ("transformer",),
    ) -> None:
        """Load role-specific init weights into ``model``, if a path is given.

        ``init_path`` may be a state-dict file or a model directory. ``subdirs``
        are probed in order when the directory itself has no ``config.json``
        (e.g. ``aux_v_transformer/`` then ``transformer/`` for the aux model).
        """
        if not init_path or not str(init_path).strip():
            return
        init_path = str(init_path).strip()
        if os.path.isfile(init_path):
            state_dict = torch.load(init_path, map_location="cpu", weights_only=False)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            logger.info(
                "  %s: loaded init checkpoint %s (keys=%d, missing=%d, unexpected=%d)",
                role_name, init_path, len(state_dict), len(missing), len(unexpected),
            )
            return
        from fewsteprl.utils.fast_init import fast_init
        model_dir = init_path
        if not os.path.exists(os.path.join(model_dir, "config.json")):
            for sub in subdirs:
                alt = os.path.join(model_dir, sub)
                if os.path.exists(os.path.join(alt, "config.json")):
                    model_dir = alt
                    break
            else:
                raise FileNotFoundError(
                    f"{role_name}_init_path has no config.json at {init_path}, "
                    f"nor in any of {list(subdirs)}"
                )
        with fast_init(torch.device("cpu")):
            loaded = model.__class__.from_pretrained(
                model_dir, torch_dtype=next(model.parameters()).dtype,
            )
        sd = loaded.state_dict()
        del loaded
        missing, unexpected = model.load_state_dict(sd, strict=False)
        logger.info(
            "  %s: loaded init transformer dir %s (keys=%d, missing=%d, unexpected=%d)",
            role_name, model_dir, len(sd), len(missing), len(unexpected),
        )

    def _predict_instantaneous_velocity(
        self,
        model: nn.Module,
        noisy_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timesteps: torch.Tensor,
        pooled_embeds: torch.Tensor,   # noqa: ARG002 - Wan has no pooled projection
        r_timesteps: torch.Tensor,
        guidance_scale: float = 1.0,
        model_u_pred: bool = False,
        dF_dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Instantaneous velocity ``v(x_t, t)`` from either kind of predictor.

        ``model_u_pred`` says what ``model`` emits:

        * ``False`` -- a v-predictor (``aux_v_model``), queried at ``r == t``
          (``u(x_t, t, t) = v(x_t, t)``); ``r_timesteps`` is not forwarded.
        * ``True`` -- a two-time flow map, queried at the sampled ``r`` and lifted
          to ``v`` with ``dF_dt``. ``dF_dt=None`` is accepted only when
          ``r == t`` everywhere.

        ``pooled_embeds`` is ignored (Wan has no pooled projection).
        """
        if model_u_pred:
            u = self._compute_u(
                model=model,
                x_t=noisy_latents,
                t_raw=timesteps,
                prompt_embeds=prompt_embeds,
                pooled_embeds=pooled_embeds,
                r_raw=r_timesteps,
                with_grad=False,
            )
            # Lift in the x0 work dtype; _s_term rejects bf16 operands.
            return self._instantaneous_velocity_from_u(
                u_old=u.to(self._rl_work_dtype),
                t_cd=timesteps,
                r_cd=r_timesteps,
                dF_dt=dF_dt,
            )

        if guidance_scale > 1.0:
            # The config pins guidance to 1.0 (AnyFlow-Wan is CFG-free); this
            # method cannot build uncond embeds, so refuse rather than ignore.
            raise ValueError(
                "_predict_instantaneous_velocity does not support CFG on the wan "
                f"branch (guidance_scale={guidance_scale}); AnyFlow-Wan is "
                "CFG-free and aux_v_model is trained on the conditional velocity "
                "only, so guiding here queries a field it has never seen."
            )
        # r == t: instantaneous velocity.
        return predict_noise_wan(
            model=model,
            noisy_latents=noisy_latents,
            text_embeddings=prompt_embeds,
            timesteps=timesteps,
            r_timesteps=timesteps,
            guidance_scale=1.0,
        )

    # --- Data ---

    def _setup_rl_dataloader(self) -> None:
        """Build the K-repeat prompt dataloader.

        Each iteration draws ``train.batch_size * world_size / K`` unique prompts
        globally, repeats each K times and shards across ranks; the K samples
        are regrouped via ``all_gather`` in :meth:`_compute_advantages`.
        """
        cfg = self._rl_cfg()
        K = int(cfg.num_image_per_prompt)
        per_rank_bs = int(self.config.train.batch_size)
        ws = get_world_size()
        if (per_rank_bs * ws) % K != 0:
            raise ValueError(
                f"train.batch_size * world_size ({per_rank_bs * ws}) must be "
                f"divisible by fewstep_rl.num_image_per_prompt={K} for K-repeat "
                "sampling."
            )

        dataset = PromptDataset(self.config.prompt_path, repeat=1)
        self._rl_sampler = DistributedKRepeatSampler(
            dataset_size=len(dataset),
            batch_size=per_rank_bs,
            k=K,
            num_replicas=ws,
            rank=get_rank(),
            seed=self.config.train.seed,
        )
        self._rl_dataloader = DataLoader(
            dataset,
            batch_sampler=self._rl_sampler,
            num_workers=0,
            collate_fn=lambda items: list(items),
        )
        self._rl_sampler.set_epoch(self._rl_epoch * int(cfg.num_batches_per_epoch))
        self._rl_sample_iter = iter(self._rl_dataloader)
        if is_main_process():
            logger.info(
                "RL dataloader: dataset_size=%d, K=%d, per_rank_bs=%d, "
                "num_batches_per_epoch=%d",
                len(dataset), K, per_rank_bs, cfg.num_batches_per_epoch,
            )

    # --- Network forwards ---

    def _predict_average_velocity(
        self,
        model: nn.Module,
        noisy_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timesteps: torch.Tensor,
        pooled_embeds: torch.Tensor,   # noqa: ARG002 - Wan has no pooled projection
        r_timesteps: torch.Tensor,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Flow-map average velocity ``u(x_t, r, t)`` -> ``[B, F, C, H, W]``.

        Timesteps are ``[B]``; ``predict_noise_wan`` expands them per frame.
        """
        return predict_noise_wan(
            model=model,
            noisy_latents=noisy_latents,
            text_embeddings=prompt_embeds,
            timesteps=timesteps,
            r_timesteps=r_timesteps,
            guidance_scale=guidance_scale,
        )

    def _encode_prompts(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """UMT5 sequence embeds plus a ``[B, 1]`` zero pooled placeholder.

        The placeholder keeps the pooled-embeds plumbing working; Wan ignores it.
        """
        device = torch.device("cuda")
        embeds, _ = encode_prompts_wan(
            prompts, self.text_encoders[0], self.tokenizers[0], device,
            max_sequence_length=int(self.config.video.max_sequence_length),
        )
        dummy_pooled = torch.zeros(embeds.shape[0], 1, device=device, dtype=embeds.dtype)
        return embeds, dummy_pooled

    def _get_last_rollout_timestep_pairs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(t_arr, r_arr)`` from the flow-map rollout, both length ``N``.

        ``flowmap_scheduler.timesteps`` has length ``N + 1`` after a rollout
        (``[t_0, ..., t_N=0]``); the jumps performed are ``(t_i, t_{i+1})``.
        """
        ts = self.flowmap_scheduler.timesteps.detach().to(dtype=torch.float32)
        return ts[:-1], ts[1:]

    # --- Rollout (5D flow-map Euler, CFG-free) ---

    @torch.no_grad()
    def _rollout_samples_ode(
        self,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        *,
        num_steps: int,
        guidance_scale: float,
        model: nn.Module,
        latent_frames: Optional[int] = None,
        return_intermediate: bool = False,
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """N-step flow-map Euler rollout (the inference path; CFG-free).

        Returns ``(x_0, x_r_all, x_t_all, end_ts, mid_ts, start_ts)``. With
        ``return_intermediate`` the trajectories are ``[B, num_steps, F, C, H, W]``
        and the timesteps ``[B, num_steps]``; otherwise all five are None.
        ``mid_ts`` (the flow-map jump target) equals ``end_ts`` on this path.
        """
        del guidance_scale  # Flow-map inference is CFG-free.
        device = prompt_embeds.device
        dtype = next(model.parameters()).dtype
        batch_size = prompt_embeds.shape[0]
        n_frames = self.latent_frames if latent_frames is None else int(latent_frames)

        self.flowmap_scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.flowmap_scheduler.timesteps  # [N+1]

        x_r_list: list[torch.Tensor] = []
        x_t_list: list[torch.Tensor] = []
        end_timestep_list: list[torch.Tensor] = []
        mid_timestep_list: list[torch.Tensor] = []
        start_timestep_list: list[torch.Tensor] = []

        x_t = torch.randn(
            batch_size, n_frames, self.latent_channels,
            self.latent_h, self.latent_w, device=device, dtype=dtype,
        )
        for i in range(num_steps):
            if return_intermediate:
                x_t_list.append(x_t.clone())
            t = timesteps[i].expand(batch_size).to(device=device, dtype=torch.float32)
            r = timesteps[i + 1].expand(batch_size).to(device=device, dtype=torch.float32)
            end_timestep_list.append(r)
            # s == r on this path: the ODE step lands exactly on its end point.
            mid_timestep_list.append(r)
            start_timestep_list.append(t)
            with self._autocast():
                u = self._predict_average_velocity(
                    model, x_t, prompt_embeds, t, pooled_embeds, r,
                    guidance_scale=1.0,
                )
            x_r = self.flowmap_scheduler.step(u, x_t, t, r)
            if return_intermediate:
                x_r_list.append(x_r.clone())
            x_t = x_r
        if not return_intermediate:
            return x_t, None, None, None, None, None
        # Keep the sample axis leading; _rl_sampling_phase concatenates on dim 0.
        return (
            x_t,
            torch.stack(x_r_list, dim=1),
            torch.stack(x_t_list, dim=1),
            torch.stack(end_timestep_list, dim=1),
            torch.stack(mid_timestep_list, dim=1),
            torch.stack(start_timestep_list, dim=1),
        )

    def _rollout_samples_sde(
        self,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        *,
        num_steps: int,
        guidance_scale: float,
        model: nn.Module,
        latent_frames: Optional[int] = None,
        return_intermediate: bool = False,
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """N-step stochastic (overshoot + renoise) flow-map rollout.

        Same grid as :meth:`_rollout_samples_ode`, but each step jumps from
        ``t_i`` to a random ``s <= t_{i+1}`` and renoises back up to ``t_{i+1}``;
        with ``s == t_{i+1}`` it reduces to the ODE rollout. Used by MeanFlowGRPO,
        whose policy ratio needs a stochastic behaviour policy.

        Same return contract as the ODE path; ``mid_ts`` is the drawn ``s``.
        CFG-free; ``guidance_scale`` is ignored.
        """
        del guidance_scale  # Flow-map inference is CFG-free.
        device = prompt_embeds.device
        dtype = next(model.parameters()).dtype
        batch_size = prompt_embeds.shape[0]
        n_frames = self.latent_frames if latent_frames is None else int(latent_frames)

        self.flowmap_scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.flowmap_scheduler.timesteps.to(device)  # [N+1], raw units
        T = float(self.flowmap_scheduler.config.num_train_timesteps)

        x_r_list: list[torch.Tensor] = []
        x_t_list: list[torch.Tensor] = []
        end_timestep_list: list[torch.Tensor] = []
        mid_timestep_list: list[torch.Tensor] = []
        start_timestep_list: list[torch.Tensor] = []

        x_t = torch.randn(
            batch_size, n_frames, self.latent_channels,
            self.latent_h, self.latent_w, device=device, dtype=dtype,
        )
        for i in range(num_steps):
            if return_intermediate:
                x_t_list.append(x_t.clone())
            t = timesteps[i].expand(batch_size).to(device=device, dtype=torch.float32)
            r = timesteps[i + 1].expand(batch_size).to(device=device, dtype=torch.float32)
            end_timestep_list.append(r)
            start_timestep_list.append(t)
            # One s per sample, uniform over ``timesteps[i + 1 :]`` (s <= r <= t);
            # on the last step s == r == 0 and the renoise is the identity.
            s_idx = torch.randint(i + 1, num_steps + 1, (batch_size,), device=device)
            s = timesteps[s_idx].to(dtype=torch.float32)  # [B]
            mid_timestep_list.append(s)
            with self._autocast():
                u = self._predict_average_velocity(
                    model, x_t, prompt_embeds, t, pooled_embeds, s,
                    guidance_scale=1.0,
                )
            x_s = self.flowmap_scheduler.step(u, x_t, t, s)
            # Renoise x_s from s to r on the linear path x_a = (1 - a) x_0 + a eps
            # (a = raw / T): alpha matches the x_0 coefficient, tau the noise
            # variance.
            bcast = (batch_size, *([1] * (x_s.ndim - 1)))
            r01 = (r / T).view(bcast)
            s01 = (s / T).view(bcast)
            alpha_r = (1.0 - r01) / (1.0 - s01)
            tau_r = torch.sqrt(torch.clamp(r01**2 - (alpha_r * s01) ** 2, min=0.0))
            eps = torch.randn(x_s.shape, device=device, dtype=torch.float32)
            x_r = (alpha_r * x_s.float() + tau_r * eps).to(dtype)
            if return_intermediate:
                x_r_list.append(x_r.clone())
            x_t = x_r
        if not return_intermediate:
            return x_t, None, None, None, None, None
        # Keep the sample axis leading; _rl_sampling_phase concatenates on dim 0.
        return (
            x_t,
            torch.stack(x_r_list, dim=1),
            torch.stack(x_t_list, dim=1),
            torch.stack(end_timestep_list, dim=1),
            torch.stack(mid_timestep_list, dim=1),
            torch.stack(start_timestep_list, dim=1),
        )

    @torch.no_grad()
    def _rollout_samples(
        self,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        *,
        num_steps: int,
        guidance_scale: float,
        model: nn.Module,
        latent_frames: Optional[int] = None,
    ) -> torch.Tensor:
        """Eval rollout returning ``x_0`` only.

        Always the deterministic ODE path; only the training rollout in
        :meth:`_one_sampling_batch` honours ``rollout_type``.
        """
        x_0, *_ = self._rollout_samples_ode(
            prompt_embeds, pooled_embeds, num_steps=num_steps,
            guidance_scale=guidance_scale, model=model,
            latent_frames=latent_frames, return_intermediate=False,
        )
        return x_0

    @torch.no_grad()
    def _decode_latents_to_tensor(self, latents: torch.Tensor) -> torch.Tensor:
        """Wan VAE decode -> ``[B, F, 3, H, W]`` in ``[0, 1]``."""
        return decode_wan_latents(self.vae, latents)

    # --- Central difference for dF/dt ---

    def _central_diff_dudt(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        v_dir: torch.Tensor,
        t_cd: torch.Tensor,
        r_cd: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Boundary-aware finite difference of ``u`` along ``v_dir`` w.r.t. ``t``.

        Perturbations to ``x_t`` are in [0, 1] units (``delta / T``), to ``t`` in
        raw timesteps, and the denominator is the actual span ``t_plus - t_minus``,
        so ``(t - r) * dF_dt`` is dimensionless. The endpoints are clamped to
        ``t_plus <= T`` and ``t_minus >= r`` (a flow map needs ``r <= t``).
        Called under :func:`torch.no_grad`.
        """
        T = float(self.scheduler.config.num_train_timesteps)
        eps_cd = float(self._rl_cfg().central_diff_epsilon)

        t_plus = (t_cd + eps_cd).clamp(max=T)
        # Holding r fixed requires every perturbed endpoint to satisfy r <= t.
        t_minus = torch.maximum((t_cd - eps_cd).clamp(min=0.0), r_cd)
        plus_delta = t_plus - t_cd
        minus_delta = t_cd - t_minus
        view_shape = (x_t.shape[0],) + (1,) * (x_t.ndim - 1)
        z_plus = x_t + v_dir * (plus_delta / T).view(view_shape).to(x_t.dtype)
        z_minus = x_t - v_dir * (minus_delta / T).view(view_shape).to(x_t.dtype)

        with self._autocast():
            f_plus = self._predict_average_velocity(
                model, z_plus, prompt_embeds, t_plus, pooled_embeds, r_cd,
                guidance_scale=1.0,
            )
            f_minus = self._predict_average_velocity(
                model, z_minus, prompt_embeds, t_minus, pooled_embeds, r_cd,
                guidance_scale=1.0,
            )
        # Promote before the subtraction (see ``_cd_out_dtype``).
        out_dtype = self._cd_out_dtype
        if out_dtype is not None:
            f_plus, f_minus = f_plus.to(out_dtype), f_minus.to(out_dtype)
        # Return 0 where the clamps collapse the span; (t - r) is 0 there too.
        denominator = (t_plus - t_minus).view(view_shape)
        valid = denominator > 0
        derivative = (f_plus - f_minus) / denominator.clamp_min(1e-6).to(
            f_plus.dtype
        )
        return torch.where(valid, derivative, torch.zeros_like(derivative))

    @torch.no_grad()
    def _compute_dF_dt_for_model(
        self,
        *,
        cd_model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        r_cd: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        x_0: torch.Tensor,
    ) -> torch.Tensor:
        """Run one central-difference dF/dt on ``cd_model``.

        ``v_dir`` is selected by ``fewstep_rl.cd_velocity_source``:

          - ``"noise_minus_x0"``: ``v_dir = eps - x_0`` with ``eps`` recovered
            from (x_t, x_0, t); model-independent.
          - ``"u_self"``: ``v_dir = cd_model(x_t, t, t, c).detach()`` (iMF,
            arXiv:2512.02012, Algorithm 2), conditional only.
        """
        cfg = self._rl_cfg()
        T = float(self.scheduler.config.num_train_timesteps)

        if cfg.cd_velocity_source == "noise_minus_x0":
            # eps = (x_t - (1 - t01) * x_0) / t01
            t01 = (t.to(x_0.dtype) / T).view(x_0.shape[0], *([1] * (x_0.ndim - 1)))
            eps = (x_t - (1.0 - t01) * x_0) / t01.clamp(min=1.0 / T)
            v_dir = eps - x_0
        elif cfg.cd_velocity_source == "u_self":
            with self._autocast():
                u_dir = self._predict_average_velocity(
                    cd_model, x_t, prompt_embeds, t, pooled_embeds, t,  # r = t
                    guidance_scale=1.0,
                )
            v_dir = u_dir.to(x_t.dtype)
        else:
            raise ValueError(
                f"Unknown fewstep_rl.cd_velocity_source={cfg.cd_velocity_source!r}; "
                "expected 'noise_minus_x0' or 'u_self'."
            )

        return self._central_diff_dudt(
            model=cd_model,
            x_t=x_t.detach(),
            v_dir=v_dir.detach(),
            t_cd=t,
            r_cd=r_cd,
            prompt_embeds=prompt_embeds,
            pooled_embeds=pooled_embeds,
        )

    def _pre_compute_v_hook(
        self,
        *,
        x_t: torch.Tensor,
        t_for_model: torch.Tensor,
        r_for_model: Optional[torch.Tensor],
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        x_0: torch.Tensor,
    ) -> None:
        """Populate ``self._shared_dF_dt`` for this step.

        The central difference runs FSDP forwards (collectives), so the decision
        to run it must be rank-uniform. Do not skip on data-dependent predicates
        such as ``r == t``: the three-mode sampler partitions by global index, so
        ranks would disagree and hang. Only the config-level ``diffusion_ratio``
        guard is safe; on the ``r == t`` slice S vanishes in ``_s_term`` anyway.
        """
        cfg = self._rl_cfg()
        if not cfg.share_cd_with_old:
            raise ValueError(
                f"{type(self).__name__} requires share_cd_with_old=True: the x0 "
                "form assumes ONE shared dF/dt so that x_anchor_1 is exactly "
                "S-free. Per-network CD makes S_theta != S_old (measured fp64 "
                "gradient rel 4.9e-1 on SD3.5)."
            )
        # Config-level, hence rank-uniform: every sample has r == t.
        if float(cfg.diffusion_ratio) >= 1.0 - 1e-8:
            return
        if self.old_model is None or r_for_model is None:
            return
        self._shared_dF_dt = self._compute_dF_dt_for_model(
            cd_model=self.old_model,
            x_t=x_t, t=t_for_model, r_cd=r_for_model,
            prompt_embeds=prompt_embeds, pooled_embeds=pooled_embeds, x_0=x_0,
        )

    def _post_compute_v_hook(self) -> None:
        self._shared_dF_dt = None

    # --- Velocity queries ---

    def _compute_u(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t_raw: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        r_raw: torch.Tensor,
        *,
        with_grad: bool,
    ) -> torch.Tensor:
        """Leading flow-map output ``u(x_t, r, t)``."""
        ctx = nullcontext() if with_grad else torch.no_grad()
        with ctx:
            with self._autocast():
                return self._predict_average_velocity(
                    model, x_t, prompt_embeds, t_raw, pooled_embeds, r_raw,
                    guidance_scale=1.0,
                )

    # --- Subclass hooks ---

    def _instantaneous_velocity_from_u(
        self,
        u_old: torch.Tensor,
        t_cd: torch.Tensor,
        r_cd: torch.Tensor,
        dF_dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Lift an average velocity ``u(x_t, r, t)`` to the instantaneous ``v(x_t, t)``.

        Contract: ``u_old`` is fp32/fp64 and the result has its dtype and shape
        (grad propagates iff ``u_old`` has it); ``t_cd`` / ``r_cd`` are raw
        timesteps; ``dF_dt`` is the shared central difference, ``None`` only when
        ``r == t`` everywhere.
        """
        raise NotImplementedError

    @torch.no_grad()
    def _compute_x_anchor_1(self, anchors: AnchorInputs) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def _compute_x_anchor_2(self, anchors: AnchorInputs) -> torch.Tensor:
        raise NotImplementedError

    # --- KL ---

    def compute_kl_term(
        self,
        x_t: torch.Tensor,
        t_for_model: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        r_for_model: Optional[torch.Tensor],
        shared_dF_dt: Optional[torch.Tensor],
        u_theta_f: torch.Tensor,
        spatial: tuple[int, ...],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """KL against the frozen reference; returns ``(kl_scalar, info)``.

        ``ref_u_model`` is a snapshot of the generator, so both sides are average
        velocities ``u(x_t, r, t)`` and the KL is a plain per-sample MSE on u (the
        S term would cancel). The v-mode reference is unavailable on Wan (see
        :meth:`_kl_usage`).
        """
        # Pre-seeded so the key set is identical on every rank and arm
        # (``_reduce_metrics`` hangs on mismatched keys).
        info: dict[str, float] = {
            "loss_kl_unweighted": 0.0,
            "kl_t_weight_mean": 0.0,
            "kl_rebalance_mean": 1.0,
        }

        wdt = self._rl_work_dtype
        kl_ps: Optional[torch.Tensor] = None
        use_u_kl, use_v_kl = self._kl_usage()
        if use_u_kl:
            u_ref = self._compute_u(
                self.ref_u_model, x_t, t_for_model, prompt_embeds, pooled_embeds,
                r_for_model, with_grad=False,
            ).detach()
            kl_ps = self.u_kl_loss(u_theta_f, u_ref.to(wdt), spatial)
        elif use_v_kl:  # pragma: no cover - unreachable on the wan branch
            raise RuntimeError(
                "The v-mode KL reference is unavailable on the wan branch; "
                "_kl_usage never returns use_v_kl=True."
            )

        # KL off: return a zero scalar; the caller adds it unconditionally.
        if kl_ps is None:
            zero = torch.zeros((), device=u_theta_f.device, dtype=u_theta_f.dtype)
            return zero, info
        return kl_ps.mean(), info

    # --- Loss kernels ---

    @staticmethod
    def _draw_eps(x_0: torch.Tensor) -> torch.Tensor:
        """Draw the forward-noising epsilon (separate so tests can pin it)."""
        return torch.randn_like(x_0)

    @staticmethod
    def _per_sample_mse(
        pred: torch.Tensor, target: torch.Tensor, spatial: tuple[int, ...],
    ) -> torch.Tensor:
        """``mean_spatial[(pred - target) ** 2]`` -> ``[B]`` (per-sample, unlike
        ``F.mse_loss``)."""
        return ((pred - target) ** 2).mean(dim=spatial)

    def self_regression(
        self,
        x_pred: torch.Tensor,
        x_reg: torch.Tensor,
        weight: torch.Tensor,
        spatial: tuple[int, ...],
    ) -> torch.Tensor:
        """``W * mean_sp[(x_pred - x_reg)^2]`` -> ``[B]``; gradient
        ``(2W/N) * (x_pred - x_reg)``."""
        return weight * self._per_sample_mse(x_pred, x_reg, spatial)

    def reward_augmentation(
        self,
        x_pred: torch.Tensor,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
        omega: torch.Tensor,
        weight: torch.Tensor,
        spatial: tuple[int, ...],
    ) -> torch.Tensor:
        """Injects the gradient ``-(2W/N) * omega * (x_pos - x_neg)`` -> ``[B]``.

        The target is ``x_pred.detach() + norm_grad`` with
        ``norm_grad = omega * (x_pos - x_neg)``, so the residual is ``-norm_grad``
        and the loss value is ``W * mean_sp[norm_grad ** 2]``. The sign matters
        and is invisible in the loss value: ``omega > 0`` must move ``x_pred``
        toward ``x_pos`` (the rewarded sample).
        """
        with torch.no_grad():
            norm_grad = (omega * (x_pos.double() - x_neg.double())).to(x_pred.dtype)
        x_target = (x_pred.detach() + norm_grad).to(x_pred.dtype)
        return weight * self._per_sample_mse(x_pred, x_target, spatial)

    def u_kl_loss(
        self, u_pred: torch.Tensor, u_ref: torch.Tensor, spatial: tuple[int, ...],
    ) -> torch.Tensor:
        """Per-sample squared divergence between ``u_pred`` and ``u_ref`` -> ``[B]``."""
        return self._per_sample_mse(u_pred, u_ref, spatial)

    @torch.no_grad()
    def decoupled_weights(
        self,
        x_0_pred_u: torch.Tensor,
        x_anchor_1: torch.Tensor,
        x_anchor_2: torch.Tensor,
        x_0_rwd: torch.Tensor,
        r_hat: torch.Tensor,
        beta: float,
        spatial: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r_broadcast = r_hat.view(-1, *([1] * (x_0_pred_u.ndim - 1))).double()
        # Probe direction from anchor_1 (S-free), center from anchor_2. Keep dev
        # in fp32 so it is exactly zero on the S == 0 slice.
        dev = x_0_pred_u.detach() - x_anchor_1
        probe_pos = x_anchor_2 + beta * dev
        probe_neg = x_anchor_2 - beta * dev
        w_pos = (
            (probe_pos.double() - x_0_rwd.double())
            .abs().mean(dim=spatial, keepdim=True).clip(min=1e-5)
        )
        w_neg = (
            (probe_neg.double() - x_0_rwd.double())
            .abs().mean(dim=spatial, keepdim=True).clip(min=1e-5)
        )
        a = r_broadcast / w_pos
        b = (1.0 - r_broadcast) / w_neg
        ab = a + b
        omega = (a - b) / (beta * ab)
        # Same weight W on both terms (see decoupled_rl_loss), flattened to [B];
        # omega keeps its broadcast shape since it scales a spatial direction.
        reg_weight = rwd_weight = (beta * ab).reshape(-1)
        return reg_weight, rwd_weight, omega

    def decoupled_rl_loss(
        self,
        x_0_pred_u: torch.Tensor,   # fp32, the only tensor carrying grad
        x_anchor_1: torch.Tensor,   # fp32 detached, S-free: direction origin + target base
        x_anchor_2: torch.Tensor,   # fp32 detached: probe center
        x_0_rwd: torch.Tensor,      # fp32 detached, the rewarded sample
        r_hat: torch.Tensor,
        beta: float,
        spatial: tuple[int, ...],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """The x0 policy loss, split into self-regression and reward terms.

        A single MSE against ``x_anchor_1 + omega * (x_0_rwd - x_anchor_2)`` has
        gradient::

            dL/dP = (2/N) * [ beta*(a+b) * (P - x_anchor_1)          # self-reg
                              - (a-b)    * (x_0_rwd - x_anchor_2) ]  # reward

        Both terms use weight ``W = beta*(a+b)`` (``W * omega == a - b``), which
        reproduces that gradient exactly; the weights are applied inside, so the
        caller must not reweight. Loss values exceed the fused loss by an
        indefinite-sign cross term, so compare gradients, not loss values.
        """
        with torch.no_grad():
            reg_weight, rwd_weight, omega = self.decoupled_weights(
                x_0_pred_u, x_anchor_1, x_anchor_2, x_0_rwd, r_hat, beta, spatial,
            )
        if x_anchor_1 is not None:
            self_reg_loss = self.self_regression(
                x_0_pred_u, x_anchor_1, reg_weight, spatial,
            )
        else:
            self_reg_loss = torch.zeros_like(reg_weight)
        if x_anchor_2 is not None:
            rwd_aug_loss = self.reward_augmentation(
                x_0_pred_u, x_0_rwd, x_anchor_2, omega, rwd_weight, spatial,
            )
        else:
            rwd_aug_loss = torch.zeros_like(rwd_weight)
        return self_reg_loss + rwd_aug_loss, {
            "affine_coeff_mean": float(reg_weight.mean().item()),
            "affine_adv_coeff_mean": float(omega.mean().item()),
            "self_regression_loss": float(self_reg_loss.detach().mean()),
            "reward_augmentation_loss": float(rwd_aug_loss.detach().mean()),
        }

    # --- Reward + advantage pipeline ---

    def _ensure_reward_scorer(self) -> None:
        if self._reward_scorer is not None:
            return
        cfg = self._rl_cfg()
        if not cfg.reward_fn:
            raise ValueError(
                f"{type(self).__name__} requires fewstep_rl.reward_fn with at "
                "least one weighted video reward (hpsv3_general / "
                "hpsv3_percentile / videoalign_mq / videoalign_ta)."
            )
        self._reward_scorer = VideoMultiScorer(
            device=torch.device("cpu"),
            score_dict=cfg.reward_fn,
            allow_unavailable=False,
            **(cfg.reward_model_paths or {}),
        )

    @staticmethod
    def _normalize_score_array(vals: Any) -> torch.Tensor:
        """Convert VideoMultiScorer per-reward output to a CPU fp32 tensor [B]."""
        if isinstance(vals, torch.Tensor):
            return vals.detach().cpu().float()
        if isinstance(vals, np.ndarray):
            return torch.from_numpy(vals).float()
        if isinstance(vals, list):
            return torch.tensor(
                [
                    float(v.detach().cpu().item()) if isinstance(v, torch.Tensor)
                    else float(v)
                    for v in vals
                ],
                dtype=torch.float32,
            )
        return torch.tensor(np.asarray(vals), dtype=torch.float32)

    @torch.no_grad()
    def _score_rewards(
        self, videos: torch.Tensor, prompts: list[str],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Score videos with the multi-reward backend.

        The reward models live on CPU between calls and move to GPU only for the
        forward, to save memory.
        """
        self._ensure_reward_scorer()
        device = torch.device("cuda")
        self._reward_scorer.to(device)
        try:
            score_details, _ = self._reward_scorer(
                videos, prompts, metadata=None, only_strict=False,
            )
            mean_vals = score_details["mean"]
            per_reward: dict[str, torch.Tensor] = {}
            for name in getattr(
                self._reward_scorer, "active_reward_names",
                list(self._rl_cfg().reward_fn.keys()),
            ):
                if name in score_details:
                    per_reward[name] = self._normalize_score_array(score_details[name])
        finally:
            self._reward_scorer.to(torch.device("cpu"))
            torch.cuda.empty_cache()

        mean_t = self._normalize_score_array(mean_vals).to(device=device)
        return mean_t, per_reward

    @torch.no_grad()
    def _gather_local_tensor(self, t: torch.Tensor) -> torch.Tensor:
        """All-gather a 1-D tensor across ranks; returns concatenated tensor."""
        ws = get_world_size()
        if not dist.is_initialized() or ws <= 1:
            return t.clone().to(device=torch.device("cuda"))
        t = t.contiguous().to(device=torch.device("cuda"))
        bucket = [torch.empty_like(t) for _ in range(ws)]
        dist.all_gather(bucket, t)
        return torch.cat(bucket, dim=0)

    @torch.no_grad()
    def _gather_prompts_and_rewards(
        self, prompts_local: list[str], rewards_local: torch.Tensor,
    ) -> tuple[list[str], torch.Tensor]:
        """All-gather prompts and rewards across ranks."""
        ws = get_world_size()
        if not dist.is_initialized() or ws <= 1:
            return list(prompts_local), rewards_local.clone()

        prompts_bucket: list[list[str] | None] = [None] * ws
        dist.all_gather_object(prompts_bucket, list(prompts_local))
        all_prompts: list[str] = []
        for chunk in prompts_bucket:
            if chunk:
                all_prompts.extend(chunk)

        rewards_local = rewards_local.contiguous()
        all_rewards = [torch.empty_like(rewards_local) for _ in range(ws)]
        dist.all_gather(all_rewards, rewards_local)
        all_rewards = torch.cat(all_rewards, dim=0)
        return all_prompts, all_rewards

    @torch.no_grad()
    def _compute_advantages(
        self, prompts_local: list[str], rewards_local: torch.Tensor,
    ) -> torch.Tensor:
        """Per-sample advantages with cross-rank per-prompt stat tracking.

        The K samples of a prompt are spread across ranks, so rewards are
        all-gathered before the per-prompt statistics are computed.
        """
        rank = get_rank()
        n_local = rewards_local.shape[0]

        all_prompts, all_rewards = self._gather_prompts_and_rewards(
            prompts_local, rewards_local,
        )

        if self._stat_tracker is None:
            # Without per-prompt tracking, fall back to global standardization.
            r = all_rewards.cpu().numpy().astype(np.float64)
            adv_all = (r - r.mean()) / (r.std() + 1e-8)
        else:
            adv_all = self._stat_tracker.update(
                all_prompts, all_rewards.cpu().numpy(),
            )
        adv_all_t = torch.as_tensor(
            adv_all, dtype=torch.float32, device=rewards_local.device,
        )
        # Slice this rank's contiguous block back out of the gathered order.
        start = rank * n_local
        return adv_all_t[start:start + n_local]

    @torch.no_grad()
    def _compute_advantages_weighted(
        self,
        prompts_local: list[str],
        rewards_local: torch.Tensor,
        per_reward_local: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Per-reward standardization, then a weighted sum of the advantages.

        Each reward is standardized within its prompt group so no reward
        dominates through its scale. Each reward needs a fresh
        ``PerPromptStatTracker`` (``update()`` accumulates per-prompt history, so
        a shared tracker would mix reward scales), and the weighted sum is
        re-standardized at the end.
        """
        cfg = self._rl_cfg()
        rank = get_rank()
        n_local = rewards_local.shape[0]
        weights: dict[str, float] = dict(cfg.reward_fn or {})

        # The weighted-sum reward is gathered only for the main tracker's stats.
        all_prompts, all_weighted = self._gather_prompts_and_rewards(
            prompts_local, rewards_local,
        )

        total_adv: np.ndarray | None = None
        for name, weight in weights.items():
            if name not in per_reward_local:
                continue
            r_local = per_reward_local[name].to(device=rewards_local.device).float()
            all_r = self._gather_local_tensor(r_local)
            r_np = all_r.cpu().numpy().astype(np.float64)
            if cfg.per_prompt_stat_tracking:
                tracker = PerPromptStatTracker(global_std=cfg.per_prompt_global_std)
                adv = np.asarray(tracker.update(all_prompts, r_np), dtype=np.float64)
            else:
                adv = (r_np - r_np.mean()) / (r_np.std() + 1e-4)
            adv = adv * float(weight)
            total_adv = adv if total_adv is None else total_adv + adv

        if total_adv is None:
            # No per-reward scores: fall back to the unweighted path.
            return self._compute_advantages(prompts_local, rewards_local)

        total_adv = (total_adv - total_adv.mean()) / (total_adv.std() + 1e-4)

        # Feed the main tracker so ``_collect_reward_stats`` stays valid (the
        # caller clears it afterwards).
        if self._stat_tracker is not None:
            self._stat_tracker.update(all_prompts, all_weighted.cpu().numpy())

        adv_local = np.ascontiguousarray(total_adv[rank * n_local:(rank + 1) * n_local])
        return torch.from_numpy(adv_local).to(
            device=rewards_local.device, dtype=torch.float32,
        )

    # --- Old-model decay ---

    def _current_decay(self) -> float:
        """DiffusionNFT decay schedule for ``old_model``, driven by
        ``self.global_step`` (inner gradient steps, not epochs)."""
        cfg = self._rl_cfg()
        step = int(self.global_step)
        if cfg.decay_type == 0:
            flat, uprate, uphold = 0, 0.0, 0.0
        elif cfg.decay_type == 1:
            flat, uprate, uphold = 0, 0.001, 0.5
        elif cfg.decay_type == 2:
            flat, uprate, uphold = 75, 0.0075, 0.999
        elif cfg.decay_type == 3:
            return 1.0
        else:
            raise ValueError(f"Unknown fewstep_rl.decay_type={cfg.decay_type}")
        if step < flat:
            return 0.0
        return min(float((step - flat) * uprate), float(uphold))

    def _current_ema_decay(self) -> float:
        """DiffusionNFT EMA warmup schedule for ``generator_ema``.

        ``decay(s) = min((1 + s) / (10 + s), target_decay)`` with
        ``s = self.global_step``.
        """
        s = int(self.global_step)
        target = float(self.config.train.ema_decay)
        return min((1.0 + float(s)) / (10.0 + float(s)), target)

    def _decay_old_model(self) -> float:
        cfg = self._rl_cfg()
        if cfg.rl_decay_interval <= 0:
            return 0.0
        if (self._rl_epoch + 1) % cfg.rl_decay_interval != 0:
            return 0.0
        decay = self._current_decay()
        # old <- decay * old + (1 - decay) * generator
        self.ema_update(self.generator, self.old_model, decay)
        return decay

    # --- Sampling phase ---

    @torch.no_grad()
    def _one_sampling_batch(
        self, prompts: list[str], return_rewards: bool = True,
        return_intermediate: bool = False,
    ) -> dict[str, Any]:
        """Roll out one K-repeat batch on this rank; return x_0, embeds, rewards.

        Samples with ``old_model`` (or ``generator_ema`` under
        ``rl_sample_use_ema``). ``return_rewards`` must be rank-uniform since
        the downstream reward/advantage path is collective.
        """
        cfg = self._rl_cfg()
        dtype = next(self.generator.parameters()).dtype

        with self._autocast():
            prompt_embeds, pooled_embeds = self._encode_prompts(prompts)
        prompt_embeds = prompt_embeds.to(dtype=dtype)
        pooled_embeds = pooled_embeds.to(
            dtype=dtype if pooled_embeds.is_floating_point() else pooled_embeds.dtype,
        )

        num_steps = int(cfg.sampling_num_steps)
        sampler_model = (
            self.generator_ema
            if (cfg.rl_sample_use_ema and self.generator_ema is not None)
            else self.old_model
        )
        sampler_model.eval()

        # Must be rank-uniform (config-level): the paths differ in FSDP forwards.
        rollout_type = str(getattr(cfg, "rollout_type", "ode"))
        if rollout_type == "ode":
            rollout_fn = self._rollout_samples_ode
        elif rollout_type == "sde":
            rollout_fn = self._rollout_samples_sde
        else:
            raise ValueError(
                f"rollout_type must be 'ode' or 'sde', got {rollout_type!r}."
            )
        x_0, x_r, x_t, ets, mts, sts = rollout_fn(
            prompt_embeds=prompt_embeds,
            pooled_embeds=pooled_embeds,
            num_steps=num_steps,
            guidance_scale=cfg.sampling_guidance_scale,
            model=sampler_model,
            return_intermediate=return_intermediate,
        )

        # Record the rollout's (t, r) jump schedule per sample.
        t_arr, r_arr = self._get_last_rollout_timestep_pairs()
        t_per_sample = t_arr.detach().cpu().repeat(x_0.shape[0], 1)
        r_per_sample = r_arr.detach().cpu().repeat(x_0.shape[0], 1)

        want_preview = int(cfg.log_sample_images) > 0
        videos = (
            self._decode_latents_to_tensor(x_0)
            if (return_rewards or want_preview) else None
        )
        if return_rewards:
            rewards, per_reward = self._score_rewards(videos, prompts)
            rewards = rewards.detach()
        else:
            rewards, per_reward = None, None

        return {
            "x_0": x_0.detach().cpu(),
            "prompt_embeds": prompt_embeds.detach().cpu(),
            "pooled_embeds": pooled_embeds.detach().cpu(),
            "timesteps": t_per_sample,
            "next_timesteps": r_per_sample,
            "prompts": list(prompts),
            "rewards": rewards,
            "per_reward": per_reward,
            "images_preview": (
                videos.detach().cpu() if (videos is not None and want_preview) else None
            ),
            "x_r": x_r.detach().cpu() if return_intermediate else None,
            "x_t": x_t.detach().cpu() if return_intermediate else None,
            "end_timesteps": ets.detach().cpu() if return_intermediate else None,
            "mid_timesteps": mts.detach().cpu() if return_intermediate else None,
            "start_timesteps": sts.detach().cpu() if return_intermediate else None,
        }

    def _rl_sampling_phase(
        self, return_rewards: bool = True, return_intermediate: bool = False,
    ) -> tuple[dict[str, Any], Optional[tuple[torch.Tensor, list[str]]]]:
        """Run ``num_batches_per_epoch`` sampling rounds + global advantage compute.

        ``return_rewards`` must be rank-uniform (``_compute_advantages``
        all-gathers).
        """
        cfg = self._rl_cfg()
        N = max(1, int(cfg.num_batches_per_epoch))
        self.generator.eval()

        batches: list[dict[str, Any]] = []
        first_preview = None
        for i in range(N):
            # Deterministic per-batch seeding, as in DiffusionNFT.
            sampler_epoch = self._rl_epoch * N + i
            self._rl_sampler.set_epoch(sampler_epoch)
            try:
                prompts = next(self._rl_sample_iter)
            except StopIteration:
                self._rl_sample_iter = iter(self._rl_dataloader)
                prompts = next(self._rl_sample_iter)
            assert isinstance(prompts, list) and len(prompts) > 0

            pack = self._one_sampling_batch(
                prompts, return_rewards=return_rewards,
                return_intermediate=return_intermediate,
            )
            batches.append(pack)
            if first_preview is None and pack["images_preview"] is not None:
                first_preview = (pack["images_preview"], pack["prompts"])

        # Concatenate per-rank slices across the sampling phase.
        x_0_all = torch.cat([b["x_0"] for b in batches], dim=0)
        emb_all = torch.cat([b["prompt_embeds"] for b in batches], dim=0)
        pool_all = torch.cat([b["pooled_embeds"] for b in batches], dim=0)
        ts_all = torch.cat([b["timesteps"] for b in batches], dim=0)
        nts_all = torch.cat([b["next_timesteps"] for b in batches], dim=0)

        # Only populated under ``return_intermediate``.
        intermediates: dict[str, Optional[torch.Tensor]] = {}
        for key in ("x_r", "x_t", "end_timesteps", "mid_timesteps", "start_timesteps"):
            intermediates[key] = (
                torch.cat([b[key] for b in batches], dim=0)
                if batches[0].get(key) is not None else None
            )

        prompts_all: list[str] = [p for b in batches for p in b["prompts"]]

        rewards_all: Optional[torch.Tensor] = None
        per_reward_all: Optional[dict[str, torch.Tensor]] = None
        advantages: Optional[torch.Tensor] = None
        reward_stats: Optional[dict[str, float]] = None
        if return_rewards:
            rewards_all = torch.cat([b["rewards"] for b in batches], dim=0)

            per_reward_all = {}
            for k in list((batches[0]["per_reward"] or {}).keys()):
                per_reward_all[k] = torch.cat(
                    [b["per_reward"][k] for b in batches], dim=0,
                )

            # The tracker is cleared after each epoch so normalization uses only
            # the current epoch's samples.
            if getattr(cfg, "weight_advantages", False):
                advantages = self._compute_advantages_weighted(
                    prompts_all, rewards_all, per_reward_all,
                )
            else:
                advantages = self._compute_advantages(prompts_all, rewards_all)
            reward_stats = self._collect_reward_stats(prompts_all, rewards_all)
            if self._stat_tracker is not None:
                self._stat_tracker.clear()

        sample_pack = {
            "x_0": x_0_all,
            "prompt_embeds": emb_all,
            "pooled_embeds": pool_all,
            "timesteps": ts_all,
            "next_timesteps": nts_all,
            "prompts": prompts_all,
            "rewards": rewards_all,
            "per_reward": per_reward_all,
            "advantages": advantages,
            "reward_stats": reward_stats,
            **intermediates,
        }
        return sample_pack, first_preview

    @torch.no_grad()
    def _collect_reward_stats(
        self, prompts_all: list[str], rewards_all: torch.Tensor,
    ) -> dict[str, float]:
        """Capture per-prompt summary stats; call before ``stat_tracker.clear()``.

        ``zero_std_ratio`` uses this rank's local slice, where many prompt groups
        have size 1, so a high value does not by itself mean collapsed advantages.
        """
        stats: dict[str, float] = {}
        if self._stat_tracker is not None:
            group_size, trained_prompt_num = self._stat_tracker.get_stats()
            stats["group_size"] = float(group_size)
            stats["trained_prompt_num"] = float(trained_prompt_num)
            for pct in (100, 75, 50, 25, 10):
                stats[f"mean_reward_{pct}"] = float(
                    self._stat_tracker.get_mean_of_top_rewards(pct)
                )
        prompts_arr = np.array(prompts_all)
        rewards_np = rewards_all.detach().cpu().numpy().astype(np.float64)
        unique_prompts, inverse_indices, counts = np.unique(
            prompts_arr, return_inverse=True, return_counts=True,
        )
        if len(unique_prompts) > 0 and rewards_np.size > 0:
            grouped = rewards_np[np.argsort(inverse_indices)]
            split_indices = np.cumsum(counts)[:-1]
            reward_groups = np.split(grouped, split_indices)
            prompt_stds = np.array([np.std(g) for g in reward_groups])
            zero_std_count = int(np.count_nonzero(prompt_stds == 0))
            stats["zero_std_ratio"] = float(zero_std_count / len(prompt_stds))
            stats["reward_std_mean"] = float(prompt_stds.mean())
        return stats

    # --- Training-time (t, r) sampling, decoupled from the rollout grid ---
    # Three-mode r sampling as in AnyFlow pretraining, with t clamped to
    # [rl_min_timestep, rl_max_timestep]. Each inner backward draws a fresh
    # (t, r) from a seeded per-step RNG; the mode partition is a deterministic
    # function of the global sample index, so no broadcast is needed.

    def _num_training_timesteps(self, n_ts_total: int) -> int:
        """Use fresh flow-map pairs, decoupled from the rollout grid."""
        n = int(self._rl_cfg().num_training_timesteps_per_sample)
        if n <= 0:
            raise ValueError(
                "fewstep_rl.num_training_timesteps_per_sample must be positive."
            )
        # One-time validation and config log.
        if not getattr(self, "_three_mode_log_emitted", False):
            cfg = self._rl_cfg()
            total_r = float(cfg.diffusion_ratio) + float(cfg.consistency_ratio)
            if not 0.0 <= total_r <= 1.0 + 1e-6:
                raise ValueError(
                    f"fewstep_rl.diffusion_ratio ({cfg.diffusion_ratio}) + "
                    f"consistency_ratio ({cfg.consistency_ratio}) must lie in "
                    f"[0, 1] (got {total_r:.4f})."
                )
            if get_rank() == 0:
                logger.info(
                    "[GeneralRL] (t, r) sampling: three-mode partition "
                    "(diffusion_ratio=%.3f, consistency_ratio=%.3f, "
                    "generic_ratio=%.3f), t clamp=[%d, %d], "
                    "iters/inner-batch=%d (decoupled from rollout N=%d), "
                    "trainer=%s.",
                    cfg.diffusion_ratio, cfg.consistency_ratio,
                    1.0 - total_r,
                    cfg.rl_min_timestep, cfg.rl_max_timestep,
                    n, n_ts_total, self.TRAINER_KEY,
                )
            self._three_mode_log_emitted = True
        return max(1, n)

    def _prepare_inner_epoch_tr_state(self, **_kwargs: Any) -> dict[str, Any]:
        """Fresh (t, r) is drawn per iteration -- no pre-shuffle needed."""
        return {}

    def _draw_training_tr(
        self,
        *,
        state: dict[str, Any],   # noqa: ARG002 - unused; we draw fresh
        chunk_start: int,
        chunk_end: int,
        j_idx: int,
        inner_epoch_idx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-iteration (t, r) draw: three-mode partition + t clamp."""
        chunk_size = max(0, chunk_end - chunk_start)
        if chunk_size == 0:
            empty = torch.empty(0, device=device, dtype=torch.float32)
            return empty, empty.clone()
        return self._sample_training_tr_three_mode(
            batch_size=chunk_size,
            chunk_start=chunk_start,
            j_idx=j_idx,
            inner_epoch_idx=inner_epoch_idx,
            device=device,
            dtype=dtype,
        )

    def _make_training_tr_rng(
        self,
        *,
        inner_epoch_idx: int,
        chunk_start: int,
        j_idx: int,
        device: torch.device,
        stream_offset: int = 0,
    ) -> torch.Generator:
        """Per-iteration seeded RNG.

        The seed mixes ``train.seed``, ``_rl_epoch``, ``global_step``,
        ``inner_epoch_idx``, ``chunk_start``, ``j_idx`` and the rank, so every
        draw within an optimizer cycle is distinct and ranks get different noise.
        """
        gen = torch.Generator(device=device)
        seed = (
            int(self.config.train.seed)
            + int(self._rl_epoch) * 1_000_003
            + int(self.global_step) * 10_007
            + int(inner_epoch_idx) * 101
            + int(chunk_start) * 17
            + int(j_idx) * 3
            + get_rank() * 7919
            + int(stream_offset)
        )
        gen.manual_seed(int(seed % (2**63 - 1)))
        return gen

    def _sample_training_tr_three_mode(
        self,
        *,
        batch_size: int,
        chunk_start: int,
        j_idx: int,
        inner_epoch_idx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw one (t, r) batch per three-mode partition + shift/clamp.

        1. ``t1, t2 ~ U(0, 1)`` per local sample (per-rank seeded).
        2. ``t01 = max(t1, t2)``, ``r01 = min(t1, t2)``.
        3. Global-index partition (``global_idx = rank * batch_size + b``):
            - first ``round(diffusion_ratio * global_bsz)``  : ``r01 = t01`` (r = t)
            - next  ``round(consistency_ratio * global_bsz)``: ``r01 = 0`` (r = 0)
            - rest                                           : keep ``r01``.
        4. Apply the scheduler shift to both endpoints, scale to raw timesteps.
        5. Clamp ``t`` to ``[rl_min_timestep, rl_max_timestep]``, clamp ``r`` to
           ``[0, t]``, then re-pin the partition equalities.
        """
        nc = self._rl_cfg()
        T = float(self.scheduler.config.num_train_timesteps)

        ws = max(1, get_world_size())
        rank = get_rank()
        global_start_idx = rank * batch_size
        global_bsz = ws * batch_size

        gen = self._make_training_tr_rng(
            inner_epoch_idx=inner_epoch_idx,
            chunk_start=chunk_start,
            j_idx=j_idx,
            device=device,
        )
        # Sample continuously in fp32: bf16 quantizes the (t, r) distribution.
        t1 = torch.rand(batch_size, dtype=torch.float32, device=device, generator=gen)
        t2 = torch.rand(batch_size, dtype=torch.float32, device=device, generator=gen)
        t01 = torch.maximum(t1, t2)
        r01 = torch.minimum(t1, t2)

        n_diffusion = round(float(nc.diffusion_ratio) * global_bsz)
        n_consistency = round(float(nc.consistency_ratio) * global_bsz)
        g_idx = torch.arange(
            global_start_idx, global_start_idx + batch_size,
            device=device, dtype=torch.long,
        )
        is_diffusion = g_idx < n_diffusion
        is_consistency = (g_idx >= n_diffusion) & (g_idx < n_diffusion + n_consistency)
        r01 = torch.where(is_diffusion, t01, r01)
        r01 = torch.where(is_consistency, torch.zeros_like(r01), r01)

        t_shifted = self.flowmap_scheduler.apply_shift(t01) * T
        r_shifted = self.flowmap_scheduler.apply_shift(r01) * T

        t_min = float(nc.rl_min_timestep)
        t_max = float(nc.rl_max_timestep)
        t_raw = t_shifted.clamp(min=t_min, max=t_max)
        r_raw = r_shifted.clamp(min=0.0).minimum(t_raw)
        r_raw = torch.where(is_diffusion, t_raw, r_raw)
        r_raw = torch.where(is_consistency, torch.zeros_like(r_raw), r_raw)

        return t_raw.to(dtype=torch.float32), r_raw.to(dtype=torch.float32)

    def _invert_flowmap_shift(self, sigmas: torch.Tensor) -> torch.Tensor:
        """Inverse of ``flowmap_scheduler.apply_shift`` on ``[0, 1]``.

        ``apply_shift`` is ``y = k s / (1 + (k - 1) s)``, a bijection on
        ``[0, 1]`` with inverse ``s = y / (k - (k - 1) y)``; the denominator is
        at least ``min(1, k) > 0`` there.
        """
        k = float(self.flowmap_scheduler.config.shift)
        if k == 1.0:
            return sigmas
        return sigmas / (k - (k - 1.0) * sigmas)

    def _draw_training_s_given_t(
        self,
        t_raw: torch.Tensor,
        *,
        state: dict[str, Any],   # noqa: ARG002 - unused; we draw fresh
        chunk_start: int,
        chunk_end: int,
        j_idx: int,
        inner_epoch_idx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Draw ``s`` given a fixed ``t``, with the same law as ``r`` given ``t``.

        For callers whose ``t`` comes from elsewhere (MeanFlowGRPO replays rollout
        transitions). Since ``r01 | t01 ~ U(0, t01)`` before the shift, ``t`` is
        mapped back through :meth:`_invert_flowmap_shift`, ``s01 = u * t01`` is
        drawn, and the shift is reapplied. The three-mode pins carry over
        (``s = t`` / ``s = 0``); ``t`` is not re-clamped. Only ``s <= t`` is
        enforced, not ``s <= r``; callers that renoise to ``r`` must bound ``s``
        themselves.

        Returns ``[chunk_size]`` float32 raw timesteps.
        """
        chunk_size = max(0, chunk_end - chunk_start)
        if chunk_size == 0:
            return torch.empty(0, device=device, dtype=torch.float32)
        if t_raw.shape != (chunk_size,):
            raise ValueError(
                f"t_raw has shape {tuple(t_raw.shape)}, expected ({chunk_size},) "
                f"from chunk [{chunk_start}, {chunk_end}). The global-index "
                "partition is keyed by position in the chunk, so a mismatch "
                "would assign modes to the wrong samples."
            )

        T = float(self.scheduler.config.num_train_timesteps)
        is_diffusion, is_consistency = self._three_mode_index_masks(
            batch_size=chunk_size, device=device,
        )
        gen = self._make_training_tr_rng(
            inner_epoch_idx=inner_epoch_idx,
            chunk_start=chunk_start,
            j_idx=j_idx,
            device=device,
            stream_offset=self._S_RNG_STREAM_OFFSET,
        )
        # Drawn in ``dtype``, then promoted so the shift algebra runs in fp32.
        u = torch.rand(
            chunk_size, dtype=dtype, device=device, generator=gen,
        ).to(torch.float32)

        t_f = t_raw.detach().to(device=device, dtype=torch.float32)
        # Clamp to [0, 1], the inverse's domain.
        t01 = self._invert_flowmap_shift((t_f / T).clamp(0.0, 1.0))
        s_shifted = self.flowmap_scheduler.apply_shift(u * t01) * T

        s_raw = s_shifted.clamp(min=0.0).minimum(t_f)
        s_raw = torch.where(is_diffusion, t_f, s_raw)
        s_raw = torch.where(is_consistency, torch.zeros_like(s_raw), s_raw)
        return s_raw.to(dtype=torch.float32)

    def _three_mode_index_masks(
        self, *, batch_size: int, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The three-mode partition as ``(is_diffusion, is_consistency)`` masks.

        Must stay identical to the partition in
        :meth:`_sample_training_tr_three_mode`.
        """
        nc = self._rl_cfg()
        ws = max(1, get_world_size())
        rank = get_rank()
        global_start_idx = rank * batch_size
        global_bsz = ws * batch_size

        n_diffusion = round(float(nc.diffusion_ratio) * global_bsz)
        n_consistency = round(float(nc.consistency_ratio) * global_bsz)
        g_idx = torch.arange(
            global_start_idx, global_start_idx + batch_size,
            device=device, dtype=torch.long,
        )
        is_diffusion = g_idx < n_diffusion
        is_consistency = (g_idx >= n_diffusion) & (g_idx < n_diffusion + n_consistency)
        return is_diffusion, is_consistency

    # --- Loss (abstract) and the inner update loop ---

    def _rl_loss(
        self,
        x_0: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        advantages: torch.Tensor,
        t_raw: torch.Tensor,
        r_raw: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Loss on one microbatch. Called by keyword from :meth:`_rl_train_phase`,
        so overrides must keep these parameter names."""
        raise NotImplementedError

    def _rl_train_phase(self, sample_pack: dict[str, Any]) -> dict[str, float]:
        """Run ``num_inner_epochs`` x ``num_batches`` x ``num_train_timesteps`` updates.

        Follows the DiffusionNFT inner loop::

            effective_grad_accum_steps = gradient_accumulation_steps * num_train_timesteps
            for inner_epoch in range(num_inner_epochs):
                perm = randperm(B_local)
                for k_batch in chunks(B_local, train_bs):
                    for j_idx in range(num_train_timesteps):
                        (t, r) <- _draw_training_tr(...)
                        loss = _rl_loss(...)
                        (loss / effective_grad_accum_steps).backward()
                        if accumulated % effective == 0:
                            clip + step + sched + zero_grad; ema; global_step += 1

        ``global_step`` increments only on optimizer steps, and
        ``num_train_timesteps`` is independent of the rollout NFE.
        """
        cfg = self._rl_cfg()
        device = torch.device("cuda")

        x_0 = sample_pack["x_0"].to(device)
        prompt_embeds = sample_pack["prompt_embeds"].to(device)
        pooled_embeds = sample_pack["pooled_embeds"].to(device)
        advantages = sample_pack["advantages"].to(device)
        timesteps_all = sample_pack["timesteps"].to(device=device, dtype=torch.float32)
        next_timesteps_all = sample_pack["next_timesteps"].to(
            device=device, dtype=torch.float32,
        )
        n_local, n_ts_total = timesteps_all.shape
        if n_local == 0:
            return {}

        n_train_ts = self._num_training_timesteps(n_ts_total)

        inner_bs = int(cfg.rl_inner_batch_size) if cfg.rl_inner_batch_size > 0 else n_local
        inner_bs = max(1, min(inner_bs, n_local))
        n_inner_epochs = max(1, int(cfg.num_inner_epochs))
        max_grad_norm = float(self.config.solver.generator.max_grad_norm)
        grad_accum = max(1, int(self.config.train.gradient_accumulation_steps))
        effective_grad_accum_steps = grad_accum * n_train_ts

        self.generator.train()
        # Release allocator blocks fragmented by the sampling phase; the backward
        # needs large contiguous FSDP buffers.
        torch.cuda.empty_cache()
        accumulator: dict[str, float] = {}
        n_backward = 0
        n_opt_steps = 0
        current_accumulated_steps = 0
        self.optimizers["generator"].zero_grad(set_to_none=True)
        last_grad_norm: float = 0.0

        dtype = next(self.generator.parameters()).dtype

        for inner_epoch_idx in range(n_inner_epochs):
            perm = torch.randperm(n_local, device=device)
            sx0 = x_0[perm]
            spe = prompt_embeds[perm]
            spp = pooled_embeds[perm]
            sadv = advantages[perm]
            tr_state = self._prepare_inner_epoch_tr_state(
                timesteps_all=timesteps_all,
                next_timesteps_all=next_timesteps_all,
                sample_perm=perm,
                n_ts_total=n_ts_total,
                n_train_ts=n_train_ts,
                device=device,
            )

            for start in range(0, n_local, inner_bs):
                end = min(start + inner_bs, n_local)
                x0_b = sx0[start:end]
                pe_b = spe[start:end]
                pp_b = spp[start:end]
                adv_b = sadv[start:end]
                for j_idx in range(n_train_ts):
                    t_raw, r_raw = self._draw_training_tr(
                        state=tr_state,
                        chunk_start=start,
                        chunk_end=end,
                        j_idx=j_idx,
                        inner_epoch_idx=inner_epoch_idx,
                        device=device,
                        dtype=dtype,
                    )
                    loss, info = self._rl_loss(
                        x_0=x0_b,
                        prompt_embeds=pe_b,
                        pooled_embeds=pp_b,
                        advantages=adv_b,
                        t_raw=t_raw,
                        r_raw=r_raw,
                    )
                    (loss / float(effective_grad_accum_steps)).backward()
                    current_accumulated_steps += 1
                    n_backward += 1

                    for k, v in info.items():
                        accumulator[k] = accumulator.get(k, 0.0) + float(v)

                    if current_accumulated_steps % effective_grad_accum_steps == 0:
                        last_grad_norm = self._compute_grad_norm(self.generator)
                        if max_grad_norm > 0:
                            self._clip_grad_norm(self.generator, max_grad_norm)
                        self.optimizers["generator"].step()
                        self.schedulers["generator"].step()
                        self.optimizers["generator"].zero_grad(set_to_none=True)

                        if (
                            self.generator_ema is not None
                            and self.global_step % self.config.train.ema_every == 0
                        ):
                            self.ema_update(
                                self.generator, self.generator_ema,
                                self._current_ema_decay(),
                            )

                        self.global_step += 1
                        n_opt_steps += 1
                        accumulator["grad_norm"] = (
                            accumulator.get("grad_norm", 0.0) + float(last_grad_norm)
                        )

        if n_backward == 0:
            return {}
        averaged = {k: v / n_backward for k, v in accumulator.items()}
        if n_opt_steps > 0:
            averaged["grad_norm"] = accumulator.get("grad_norm", 0.0) / n_opt_steps
        averaged["lr"] = self.schedulers["generator"].get_last_lr()[0]
        averaged["inner_backward_steps"] = float(n_backward)
        averaged["inner_opt_steps"] = float(n_opt_steps)
        averaged["num_train_timesteps"] = float(n_train_ts)
        averaged["effective_grad_accum_steps"] = float(effective_grad_accum_steps)
        return averaged

    # --- One outer RL epoch ---

    def _run_rl_epoch(
        self,
        sample_pack: Optional[dict[str, Any]] = None,
        preview: Optional[tuple[Any, list[str]]] = None,
    ) -> dict[str, dict[str, float]]:
        """One RL epoch: sampling + inner update + old_model decay.

        Returns a metrics dict whose top-level keys are wandb section names:

        - ``rl_reward``     -- reward summary + DiffusionNFT stat-tracker stats
        - ``rl_advantage``  -- advantage / r_hat distribution
        - ``rl_loss``       -- per-iteration policy / split / KL losses
        - ``rl_train``      -- optimization counters (lr, grad_norm, decay, ...)
        - ``rl_diagnostic`` -- sanity-check signals (x0 norm, S magnitude, ...)
        """
        cfg = self._rl_cfg()
        active = self._rl_epoch >= int(cfg.rl_start_epoch)

        if sample_pack is None:
            with self.timer.measure("rl_sampling"):
                sample_pack, preview = self._rl_sampling_phase()

        self.generator.train()

        # Report globally gathered rewards so every rank logs the same values.
        gathered_rewards_all = self._gather_local_tensor(sample_pack["rewards"])
        reward_section: dict[str, float] = {
            "reward_mean": float(gathered_rewards_all.mean().item()),
            "reward_std": float(gathered_rewards_all.std().item()),
            "reward_min": float(gathered_rewards_all.min().item()),
            "reward_max": float(gathered_rewards_all.max().item()),
        }
        for name, vals in (sample_pack.get("per_reward") or {}).items():
            gathered = self._gather_local_tensor(vals.to(sample_pack["rewards"].device))
            reward_section[f"{name}_mean"] = float(gathered.mean().item())
            reward_section[f"{name}_std"] = float(gathered.std().item())
            reward_section[f"{name}_min"] = float(gathered.min().item())
            reward_section[f"{name}_max"] = float(gathered.max().item())
        reward_section.update(sample_pack.get("reward_stats") or {})

        advantage_section: dict[str, float] = {
            "advantage_mean": float(sample_pack["advantages"].mean().item()),
            "advantage_std": float(sample_pack["advantages"].std().item()),
            "advantage_abs_mean": float(sample_pack["advantages"].abs().mean().item()),
        }

        loss_section: dict[str, float] = {}
        train_section: dict[str, float] = {}
        diagnostic_section: dict[str, float] = {
            "n_samples_local": float(sample_pack["x_0"].shape[0]),
        }

        if active:
            with self.timer.measure("rl_update"):
                upd = self._rl_train_phase(sample_pack)

            for k in ("loss_policy", "loss_unscaled", "loss_kl", "loss_total",
                      "affine_coeff_mean",
                      "affine_adv_coeff_mean") + self._EXTRA_LOSS_KEYS:
                if k in upd:
                    loss_section[k] = float(upd[k])
            for k in ("lr", "grad_norm", "inner_backward_steps", "inner_opt_steps",
                      "num_train_timesteps", "effective_grad_accum_steps", "t_mean"):
                if k in upd:
                    train_section[k] = float(upd[k])
            if "r_hat_mean" in upd:
                advantage_section["r_hat_mean"] = float(upd["r_hat_mean"])
            for k in ("x0_norm_mean", "old_deviate") + self._EXTRA_DIAGNOSTIC_KEYS:
                if k in upd:
                    diagnostic_section[k] = float(upd[k])

            with self.timer.measure("rl_old_decay"):
                decay_val = self._decay_old_model()
            train_section["old_decay"] = float(decay_val)

        metrics: dict[str, dict[str, float]] = {
            "rl_reward": reward_section,
            "rl_advantage": advantage_section,
        }
        if loss_section:
            metrics["rl_loss"] = loss_section
        if train_section:
            metrics["rl_train"] = train_section
        if diagnostic_section:
            metrics["rl_diagnostic"] = diagnostic_section

        self._log_sampling_preview(preview)
        return metrics

    # --- Auxiliary v-predictor loop (DMD-NFT) ---
    # Before each policy update, ``aux_v_model`` is fitted with a flow-matching
    # loss on the generator's rollouts, so ``v_fake`` estimates the marginal
    # ``E[eps - x_0 | x_t, t]`` of the rollout distribution (used for
    # ``x_anchor_2``). Every method is a no-op when ``aux_v_model is None``.

    @torch.no_grad()
    def _aux_sampling_phase(
        self, shared_pack: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Assemble ``max_aux_sample`` latents/GPU for the flow-matching loss.

        ``shared_pack`` is the policy epoch's rollout (same behaviour policy),
        reused to avoid a second rollout; only ``x_0`` and the embeddings are
        used. Any shortfall is topped up with extra rollouts (none are needed
        when ``max_aux_sample <= num_batches_per_epoch * train.batch_size``).
        All sizes are config-derived and hence rank-uniform, as required by the
        FSDP rollout.
        """
        cfg = self._rl_cfg()
        target = int(cfg.max_aux_sample)

        x0_parts: list[torch.Tensor] = []
        emb_parts: list[torch.Tensor] = []
        pool_parts: list[torch.Tensor] = []
        collected = 0

        if shared_pack is not None:
            avail = int(shared_pack["x_0"].shape[0])
            take = min(target, avail)
            if take > 0:
                if take < avail:
                    # Random subset rather than a prefix (the pack is ordered by
                    # sampling round). Indices may differ per rank; ``take`` may not.
                    idx = torch.randperm(
                        avail, device=shared_pack["x_0"].device,
                    )[:take]
                    x0_parts.append(shared_pack["x_0"][idx])
                    emb_parts.append(shared_pack["prompt_embeds"][idx])
                    pool_parts.append(shared_pack["pooled_embeds"][idx])
                else:
                    x0_parts.append(shared_pack["x_0"])
                    emb_parts.append(shared_pack["prompt_embeds"])
                    pool_parts.append(shared_pack["pooled_embeds"])
                collected = take

        if collected < target:
            if self._aux_sample_iter is None:
                self._aux_sample_iter = iter(self._rl_dataloader)
            if get_rank() == 0 and not getattr(self, "_aux_topup_log_emitted", False):
                logger.info(
                    "[%s] aux pack needs %d extra rollout sample(s)/GPU beyond "
                    "the shared policy pack (%d of %d reused). Set max_aux_sample "
                    "<= num_batches_per_epoch * train.batch_size to make the aux "
                    "epoch rollout-free.",
                    self.TRAINER_KEY, target - collected, collected, target,
                )
                self._aux_topup_log_emitted = True
            while collected < target:
                # Offset keeps top-up prompt batches disjoint from the RL rounds.
                self._rl_sampler.set_epoch(
                    self._AUX_SAMPLER_EPOCH_OFFSET + self._aux_sample_round
                )
                self._aux_sample_round += 1
                try:
                    prompts = next(self._aux_sample_iter)
                except StopIteration:
                    self._aux_sample_iter = iter(self._rl_dataloader)
                    prompts = next(self._aux_sample_iter)

                pack = self._one_sampling_batch(prompts, return_rewards=False)
                x0_parts.append(pack["x_0"])
                emb_parts.append(pack["prompt_embeds"])
                pool_parts.append(pack["pooled_embeds"])
                collected += pack["x_0"].shape[0]

        # Truncate to exactly ``max_aux_sample`` (rank-uniform). A single part is
        # aliased rather than copied; the aux path never writes to it.
        def _finalize(parts: list[torch.Tensor]) -> torch.Tensor:
            out = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
            return out if out.shape[0] == target else out[:target]

        return {
            "x_0": _finalize(x0_parts),
            "prompt_embeds": _finalize(emb_parts),
            "pooled_embeds": _finalize(pool_parts),
        }

    def _aux_fm_loss(
        self,
        x_0: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,   # noqa: ARG002 - Wan has no pooled projection
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Plain rectified-flow matching loss for :attr:`aux_v_model`.

        Logit-normal shifted ``t``, linear noising, ``target = eps - x_0``, no
        timestep weighting, reduced in fp32. The model is queried at ``r == t``
        without CFG, i.e. trained on the conditional ``v(x_t, t)``.
        """
        cfg = self._rl_cfg()
        device = x_0.device
        model_dtype = next(self.aux_v_model.parameters()).dtype
        T = float(self.scheduler.config.num_train_timesteps)
        batch_size = x_0.shape[0]

        t01 = compute_density_for_timestep_sampling(
            weighting_scheme="logit_normal",
            batch_size=batch_size, logit_mean=0.0, logit_std=1.0,
        ).to(device=device, dtype=model_dtype)
        timesteps = (self.flowmap_scheduler.apply_shift(t01) * T).to(device)
        # Same t window as the policy loss, where the aux model is queried.
        timesteps = timesteps.clamp(cfg.rl_min_timestep, cfg.rl_max_timestep)

        noise = self._draw_eps(x_0)
        noisy_latents = self.flowmap_scheduler.scale_noise(x_0, timesteps, noise)

        with self._autocast():
            velocity_pred = self._predict_instantaneous_velocity(
                model=self.aux_v_model,
                noisy_latents=noisy_latents,
                prompt_embeds=prompt_embeds,
                timesteps=timesteps,
                pooled_embeds=pooled_embeds,
                r_timesteps=timesteps,
                guidance_scale=1.0,
                model_u_pred=False,
            )
        target = noise - x_0
        per_sample = (velocity_pred.float() - target.float()) ** 2
        loss = per_sample.reshape(batch_size, -1).mean(dim=-1).mean()

        return loss, {
            "loss_fm": float(loss.detach()),
            "aux_t_mean": float(timesteps.mean().item()),
        }

    def _aux_train_phase(self, sample_pack: dict[str, Any]) -> dict[str, float]:
        """``renoise_per_epoch`` flow-matching updates over one aux sample pack.

        Each update re-noises latents from the pack with fresh ``t`` and ``eps``.
        The pack is consumed as a cycling permutation, so each completed sweep
        uses every sample exactly once. ``global_step``, the generator EMA and the
        old-model decay are keyed to generator steps and are not touched here.
        """
        cfg = self._rl_cfg()
        device = torch.device("cuda")

        x_0 = sample_pack["x_0"].to(device)
        prompt_embeds = sample_pack["prompt_embeds"].to(device)
        pooled_embeds = sample_pack["pooled_embeds"].to(device)
        n_local = x_0.shape[0]
        if n_local == 0:
            return {}

        # Not clamped to ``n_local``: t and eps are drawn per element, so repeated
        # latents in one microbatch still get independent noise.
        micro_bs = max(1, int(cfg.aux_v_batch_size) or n_local)
        accum = max(1, int(cfg.aux_v_gradient_accum))
        n_renoise = max(1, int(cfg.renoise_per_epoch))
        max_grad_norm = float(self.config.solver.aux_v.max_grad_norm)

        # One-time log of the effective reuse factor and step count.
        if not getattr(self, "_aux_align_log_emitted", False):
            if get_rank() == 0:
                sweep_len = -(-n_local // micro_bs)
                logger.info(
                    "[%s] aux fm loop: pack=%d/GPU, micro_bs=%d, renoise=%d, "
                    "accum=%d -> %d sweep(s) of %d evals, %.2f uses per sample, "
                    "%d opt step(s)/epoch, global batch/step=%d.",
                    self.TRAINER_KEY, n_local, micro_bs, n_renoise, accum,
                    n_renoise // sweep_len, sweep_len,
                    n_renoise * micro_bs / n_local, n_renoise // accum,
                    micro_bs * max(1, get_world_size()),
                )
            self._aux_align_log_emitted = True

        opt = self.optimizers["aux_v"]
        sched = self.schedulers["aux_v"]
        opt.zero_grad(set_to_none=True)

        accumulator: dict[str, float] = {}
        n_backward = 0
        n_opt_steps = 0
        order = torch.randperm(n_local, device=device)
        cursor = 0

        for _ in range(n_renoise):
            # Draw exactly ``micro_bs`` indices, wrapping into a fresh permutation
            # when needed, so no short microbatch skews the gradient weighting.
            picks: list[torch.Tensor] = []
            remaining = micro_bs
            while remaining > 0:
                if cursor >= n_local:
                    order = torch.randperm(n_local, device=device)
                    cursor = 0
                take = min(remaining, n_local - cursor)
                picks.append(order[cursor:cursor + take])
                cursor += take
                remaining -= take
            idx = picks[0] if len(picks) == 1 else torch.cat(picks)

            loss, info = self._aux_fm_loss(
                x_0=x_0[idx],
                prompt_embeds=prompt_embeds[idx],
                pooled_embeds=pooled_embeds[idx],
            )
            (loss / float(accum)).backward()
            n_backward += 1
            for k, v in info.items():
                accumulator[k] = accumulator.get(k, 0.0) + float(v)

            if n_backward % accum == 0:
                grad_norm = self._compute_grad_norm(self.aux_v_model)
                if max_grad_norm > 0:
                    self._clip_grad_norm(self.aux_v_model, max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                n_opt_steps += 1
                accumulator["grad_norm"] = (
                    accumulator.get("grad_norm", 0.0) + float(grad_norm)
                )

        # Discard a trailing partial accumulation window rather than step on it.
        if n_backward % accum != 0:
            opt.zero_grad(set_to_none=True)

        averaged = {k: v / n_backward for k, v in accumulator.items()}
        if n_opt_steps > 0:
            averaged["grad_norm"] = accumulator.get("grad_norm", 0.0) / n_opt_steps
        averaged["lr"] = sched.get_last_lr()[0]
        averaged["aux_backward_steps"] = float(n_backward)
        averaged["aux_opt_steps"] = float(n_opt_steps)
        averaged["aux_micro_batch_size"] = float(micro_bs)
        return averaged

    def _run_aux_epoch(
        self, shared_pack: Optional[dict[str, Any]] = None,
    ) -> dict[str, dict[str, float]]:
        """Auxiliary-network epoch (DMD-NFT); must run before the policy update.

        ``shared_pack`` should be the pack the policy update trains on, so
        ``_rl_loss`` reads an ``aux_v_model`` fitted to those samples; ``None``
        rolls out from scratch. Returns ``{}`` when there is no ``aux_v_model``.
        Both gates are config-level and hence rank-uniform (FSDP forwards follow).
        """
        if self.aux_v_model is None:
            return {}
        cfg = self._rl_cfg()
        if self._rl_epoch < int(cfg.rl_start_epoch):
            return {}

        # Swap train/eval modes for the aux phase and restore them afterwards.
        self.generator.eval()
        self.aux_v_model.train()
        try:
            with self.timer.measure("aux_sampling"):
                sample_pack = self._aux_sampling_phase(shared_pack)
            with self.timer.measure("aux_update"):
                upd = self._aux_train_phase(sample_pack)
        finally:
            self.aux_v_model.eval()
            self.generator.train()

        if not upd:
            return {}
        train_section = {
            k: float(upd[k])
            for k in ("loss_fm", "lr", "grad_norm",
                      "aux_backward_steps", "aux_opt_steps")
            if k in upd
        }
        diagnostic_section = {
            k: float(upd[k])
            for k in ("aux_t_mean", "aux_micro_batch_size")
            if k in upd
        }
        diagnostic_section["aux_n_samples_local"] = float(
            sample_pack["x_0"].shape[0]
        )
        return {"aux_train": train_section, "aux_diagnostic": diagnostic_section}

    # --- Main training loop (epoch-based, as in DiffusionNFT) ---

    def _train_loop(self) -> None:
        """Epoch-based loop following DiffusionNFT's outer-loop structure."""
        cfg = self._rl_cfg()
        train_cfg = self.config.train

        self._setup_rl_dataloader()
        self._restore_rng_state()

        if is_main_process():
            logger.info(
                "Starting RL training (%s): num_epochs=%d, "
                "num_batches_per_epoch=%d, num_inner_epochs=%d, K=%d, "
                "per_rank_bs=%d",
                self.TRAINER_KEY, cfg.num_epochs, cfg.num_batches_per_epoch,
                cfg.num_inner_epochs, cfg.num_image_per_prompt,
                train_cfg.batch_size,
            )

        while self._rl_epoch < int(cfg.num_epochs):
            eval_cfg = self.config.eval
            if (
                eval_cfg.enabled
                and eval_cfg.eval_interval > 0
                and self._rl_epoch % eval_cfg.eval_interval == 0
            ):
                with self.timer.measure("eval"):
                    self._evaluate()

            if (
                train_cfg.save_interval > 0
                and self._rl_epoch % train_cfg.save_interval == 0
                and self._rl_epoch > 0
            ):
                self._save_checkpoint()

            with self.timer.measure("epoch"):
                with self.timer.measure("rl_sampling"):
                    # One rollout per epoch, shared by the aux loop and the policy
                    # update. ``reuse_x_t`` also returns the intermediate
                    # trajectory, which MeanFlowGRPO replays.
                    sample_pack, preview = self._rl_sampling_phase(
                        return_intermediate=bool(getattr(cfg, "reuse_x_t", False)),
                    )
                # Aux epoch first: _rl_loss reads aux_v_model fitted to this pack.
                # Returns ``{}`` on arms without an aux model.
                aux_metrics = self._run_aux_epoch(sample_pack)
                metrics = self._run_rl_epoch(sample_pack, preview)
                for section, vals in (aux_metrics or {}).items():
                    metrics.setdefault(section, {}).update(vals)

            if self._rl_epoch % train_cfg.log_interval == 0:
                self._log_step(metrics)

            self._rl_epoch += 1

        logger.info(
            "RL training complete: epoch=%d, global_step=%d",
            self._rl_epoch, self.global_step,
        )
        self._save_checkpoint()
        self._save_final_transformer()

    # --- Gradient-norm helpers ---

    @staticmethod
    def _compute_grad_norm(model: nn.Module) -> float:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if isinstance(model, FSDP):
            total = torch.zeros((), device=torch.cuda.current_device())
            for p in model.parameters():
                if p.grad is not None:
                    total = total + p.grad.detach().float().norm() ** 2
            total = total.clone()
            if dist.is_initialized():
                dist.all_reduce(total, op=dist.ReduceOp.SUM)
            return float(total.sqrt().item())
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += float(p.grad.detach().float().norm().item()) ** 2
        return total ** 0.5

    @staticmethod
    def _clip_grad_norm(model: nn.Module, max_norm: float) -> None:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if isinstance(model, FSDP):
            model.clip_grad_norm_(max_norm)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    # --- Checkpoint hooks ---

    #: Optimizers whose state is not checkpointed. ``aux_v`` is a full fine-tune,
    #: so its AdamW moments are large; the aux weights are saved, and on resume
    #: the aux optimizer restarts from zero momentum (``load_checkpoint`` skips
    #: missing optimizer files).
    _CHECKPOINT_SKIP_OPTIMIZERS: frozenset[str] = frozenset({"aux_v"})

    def _save_checkpoint(self) -> None:
        """Save, minus the optimizer states in :attr:`_CHECKPOINT_SKIP_OPTIMIZERS`.

        Temporarily swaps ``self.optimizers`` for a filtered dict around the
        ``super()`` call; the original is restored in ``finally``.
        """
        skipped = [k for k in self.optimizers if k in self._CHECKPOINT_SKIP_OPTIMIZERS]
        if not skipped:
            super()._save_checkpoint()
            return

        full = self.optimizers
        try:
            self.optimizers = {
                k: v for k, v in full.items()
                if k not in self._CHECKPOINT_SKIP_OPTIMIZERS
            }
            super()._save_checkpoint()
        finally:
            self.optimizers = full
        if get_rank() == 0:
            logger.info(
                "  (optimizer state for %s omitted by design -- re-fitted every "
                "epoch, ~5.3 GB/checkpoint saved)", ", ".join(sorted(skipped)),
            )

    def _get_extra_checkpoint_state(self) -> dict[str, Any]:
        """Save the outer-loop counter, ``trainer_key``, and the stat tracker."""
        extra: dict[str, Any] = {
            "rl_epoch": self._rl_epoch,
            "trainer_key": self.TRAINER_KEY,
        }
        if self._stat_tracker is not None:
            extra["rl_stat_tracker"] = {
                "rank": get_rank(),
                "state": self._stat_tracker.state_dict(),
            }
        return extra

    def _restore_extra_checkpoint_state(self, extra: dict[str, Any]) -> None:
        saved_key = extra.get("trainer_key")
        if saved_key is not None and saved_key != self.TRAINER_KEY:
            raise ValueError(
                f"Checkpoint was written by trainer {saved_key!r} but is being "
                f"resumed into {self.TRAINER_KEY!r}. The two arms have different "
                "loss algebra; resuming across them silently mixes runs."
            )
        self._rl_epoch = int(extra.get("rl_epoch", 0))
        if self._stat_tracker is not None:
            st = extra.get("rl_stat_tracker")
            if st:
                self._stat_tracker.load_state_dict(st.get("state", {}))

    def _get_lora_model_names(self) -> set[str]:
        names = super()._get_lora_model_names()
        if self.config.model.generator_lora.enabled:
            if self.old_model is not None:
                names.add("old_model")
            if self.ref_u_model is not None:
                names.add("ref_u_model")
        return names

    def _get_lora_configs(self) -> dict[str, Any]:
        cfgs = super()._get_lora_configs()
        if self.config.model.generator_lora.enabled:
            if self.old_model is not None:
                cfgs["old_model"] = self.config.model.generator_lora
            if self.ref_u_model is not None:
                cfgs["ref_u_model"] = self.config.model.generator_lora
        return cfgs

    # --- Video evaluation ---

    @torch.no_grad()
    def _evaluate(self) -> None:
        """Stream test-set generation and scoring without retaining all videos.

        Every rank runs the same number of batches (``max_batches``), padding
        with a repeated prompt when its share runs out: the rollout issues FSDP
        collectives, so a rank that stopped early would hang the others.
        """
        eval_cfg = self.config.eval
        if not eval_cfg.reward_fn:
            return
        rank = get_rank()
        world_size = get_world_size()
        logger.info(
            "[Eval] Wan video eval at epoch %d (rank=%d)", self._rl_epoch, rank,
        )
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(),
        }
        try:
            if self._eval_video_scorer is None:
                self._eval_video_scorer = VideoMultiScorer(
                    device=torch.device("cpu"),
                    score_dict=eval_cfg.reward_fn,
                    allow_unavailable=True,
                    **(self._rl_cfg().reward_model_paths or {}),
                )
            scorer = self._eval_video_scorer
            if not scorer.active_reward_names:
                logger.warning("[Eval] No available video rewards; skipping.")
                return
            eval_seed = int(eval_cfg.eval_seed) + rank
            random.seed(eval_seed)
            np.random.seed(eval_seed)
            torch.manual_seed(eval_seed)
            torch.cuda.manual_seed_all(eval_seed)
            gen_model = self.models.get("generator_ema") or self.models.get("generator")
            was_training = gen_model.training
            gen_model.eval()
            all_prompts = self._get_cached_eval_prompts()
            rank_prompts = all_prompts[rank::world_size]
            batch_size = max(1, int(eval_cfg.eval_batch_size))
            max_per_rank = (len(all_prompts) + world_size - 1) // world_size
            max_batches = (max_per_rank + batch_size - 1) // batch_size
            sums: dict[str, float] = {}
            num_local = 0
            media: list[torch.Tensor] = []
            media_prompts: list[str] = []
            scorer.to(torch.device("cuda"))
            try:
                for batch_index in range(max_batches):
                    start_index = batch_index * batch_size
                    end_index = min(start_index + batch_size, len(rank_prompts))
                    if start_index < len(rank_prompts):
                        batch_prompts = rank_prompts[start_index:end_index]
                        actual_count = end_index - start_index
                        if len(batch_prompts) < batch_size:
                            batch_prompts = batch_prompts + [batch_prompts[-1]] * (
                                batch_size - len(batch_prompts)
                            )
                    else:
                        fallback = rank_prompts[0] if rank_prompts else all_prompts[0]
                        batch_prompts = [fallback] * batch_size
                        actual_count = 0
                    embeds, dummy = self._encode_prompts(batch_prompts)
                    latents = self._rollout_samples(
                        embeds,
                        dummy,
                        num_steps=int(eval_cfg.eval_num_steps),
                        guidance_scale=1.0,
                        model=gen_model,
                        latent_frames=self.eval_latent_frames,
                    )
                    videos = self._decode_latents_to_tensor(latents)
                    if actual_count:
                        valid_videos = videos[:actual_count]
                        valid_prompts = batch_prompts[:actual_count]
                        details, _ = scorer(
                            valid_videos, valid_prompts,
                            metadata=None, only_strict=False,
                        )
                        for name in (*scorer.active_reward_names, "mean"):
                            if name in details:
                                values = np.asarray(details[name], dtype=np.float64)
                                sums[name] = sums.get(name, 0.0) + float(values.sum())
                        if rank == 0 and len(media) < int(eval_cfg.num_media_images):
                            keep = min(
                                actual_count,
                                int(eval_cfg.num_media_images) - len(media),
                            )
                            media.extend(valid_videos[:keep].detach().cpu())
                            media_prompts.extend(valid_prompts[:keep])
                        num_local += actual_count
                    del embeds, dummy, latents, videos
                    torch.cuda.empty_cache()
            finally:
                scorer.to(torch.device("cpu"))
                torch.cuda.empty_cache()
            local_metrics = {
                name: value / num_local
                for name, value in sums.items()
                if num_local > 0
            }
            metrics = self._aggregate_eval_metrics_across_ranks(
                local_metrics, num_local,
            )
            if was_training:
                gen_model.train()
            if self.wandb_logger:
                self.wandb_logger.log(metrics, step=self.global_step, section="eval")
                media_tensor = torch.stack(media) if media else None
                self._log_eval_videos(media_tensor, media_prompts)
            if is_main_process():
                values = " | ".join(
                    f"{name}={value:.4g}" for name, value in metrics.items()
                )
                logger.info(
                    "[Eval] epoch=%d step=%d | %s",
                    self._rl_epoch, self.global_step, values,
                )
            self._maybe_run_vbench(gen_model)
        finally:
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.random.set_rng_state(rng_state["torch_cpu"])
            torch.cuda.set_rng_state(rng_state["torch_cuda"])
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

    def _log_eval_videos(
        self, videos_tensor: Optional[torch.Tensor], prompts_local: list[str],
    ) -> None:
        """Log a few eval videos to wandb as mp4 at the configured ``video.fps``."""
        if (
            self.wandb_logger is None
            or not is_main_process()
            or self.config.eval.num_media_images <= 0
            or videos_tensor is None
            or videos_tensor.shape[0] == 0
        ):
            return
        try:
            n = min(int(self.config.eval.num_media_images), videos_tensor.shape[0])
            self.wandb_logger.log_videos(
                videos=[videos_tensor[i] for i in range(n)],
                captions=prompts_local[:n],
                step=self.global_step,
                key="media/eval_video",
                fps=int(self.config.video.fps),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[Eval] video logging failed: {e}")

    def _log_sampling_preview(self, preview) -> None:
        """Log rollout previews as mp4 (``fewstep_rl.log_sample_images > 0``).

        ``preview`` is ``(videos[B, F, 3, H, W], prompts)`` from the first
        sampling batch.
        """
        cfg = self._rl_cfg()
        if (
            cfg.log_sample_images <= 0
            or not is_main_process()
            or self.wandb_logger is None
            or preview is None
        ):
            return
        vids, prompts_preview = preview
        n = min(int(cfg.log_sample_images), vids.shape[0])
        try:
            self.wandb_logger.log_videos(
                videos=[vids[i] for i in range(n)],
                captions=prompts_preview[:n],
                step=self.global_step,
                key="media/rl_samples",
                fps=int(self.config.video.fps),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[RL] sample video logging failed: {e}")

    @torch.no_grad()
    def _maybe_run_vbench(self, gen_model: nn.Module) -> None:
        vb = getattr(self.config.eval, "vbench", None)
        if vb is None or not vb.enabled:
            return
        every = int(getattr(vb, "eval_every_epochs", 0))
        evaluation_only = bool(getattr(self, "_evaluation_only", False))
        if not evaluation_only and (every <= 0 or (self._rl_epoch % every != 0)):
            return
        try:
            from fewsteprl.eval.vbench_eval import run_vbench_eval
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[VBench] not available ({e}); skipping.")
            return
        run_vbench_eval(self, gen_model)

    @property
    def _nft_epoch(self) -> int:
        """Alias of ``_rl_epoch``; ``fewsteprl/eval/vbench_eval.py`` reads
        ``trainer._nft_epoch``."""
        return self._rl_epoch

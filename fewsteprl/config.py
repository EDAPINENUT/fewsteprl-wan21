"""Strict configuration schema for Wan2.1 MeanFlowNFT training."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import yaml


def _apply_flat_overrides(data: dict, overrides: dict[str, Any]) -> dict:
    for key, value in overrides.items():
        target = data
        parts = key.split(".")
        for part in parts[:-1]:
            child = target.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(
                    f"Cannot apply override {key!r}: {part!r} is not a mapping."
                )
            target = child
        target[parts[-1]] = value
    return data


def _dataclass_from_dict(cls, data: Any):
    if not isinstance(data, dict):
        return data
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown {cls.__name__} field(s): {', '.join(unknown)}"
        )
    kwargs = {}
    for item in fields(cls):
        if item.name not in data:
            continue
        value = data[item.name]
        item_type = item.type
        if isinstance(item_type, str):
            item_type = eval(item_type, globals())
        if isinstance(value, dict) and hasattr(item_type, "__dataclass_fields__"):
            value = _dataclass_from_dict(item_type, value)
        kwargs[item.name] = value
    return cls(**kwargs)


@dataclass
class LoRAConfig:
    enabled: bool = True
    rank: int = 32
    lora_alpha: int = 64
    target_modules: list[str] = field(
        default_factory=lambda: [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "net.0.proj",
            "net.2",
        ]
    )
    init_lora_weights: str = "gaussian"
    load_path: str = ""
    pre_merge_paths: list[str] = field(default_factory=list)
    merge_before_training: bool = False

    def __post_init__(self) -> None:
        if self.rank < 1 or self.lora_alpha < 1:
            raise ValueError("LoRA rank and alpha must be positive.")
        if self.enabled and not self.target_modules:
            raise ValueError("LoRA target_modules cannot be empty.")


@dataclass
class ModelConfig:
    pretrained_path: str = "nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers"
    generator_init_path: str = ""
    #: Resume point for ``aux_v_model`` (empty = pretrained weights). Probes
    #: ``aux_v_transformer/`` then ``transformer/``, so a run's ``output_dir``
    #: works directly.
    aux_v_init_path: str = ""
    model_type: str = "wan"
    gradient_checkpointing: bool = True
    dtype: str = "bf16"
    generator_lora: LoRAConfig = field(default_factory=LoRAConfig)

    def __post_init__(self) -> None:
        if self.model_type.lower() != "wan":
            raise ValueError("The wan branch only supports model_type='wan'.")
        self.model_type = "wan"
        self.dtype = self.dtype.lower()
        if self.dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError("model.dtype must be bf16, fp16, or fp32.")
        if not str(self.pretrained_path).strip():
            raise ValueError("model.pretrained_path is required.")


@dataclass
class VideoConfig:
    num_frames: int = 81
    eval_num_frames: int = 81
    height: int = 480
    width: int = 832
    fps: int = 16
    max_sequence_length: int = 512

    def __post_init__(self) -> None:
        for name in ("num_frames", "eval_num_frames"):
            value = int(getattr(self, name))
            if value < 1 or (value - 1) % 4 != 0:
                raise ValueError(f"video.{name} must satisfy (value - 1) % 4 == 0.")
        if self.height < 8 or self.width < 8:
            raise ValueError("video height and width must be positive.")
        if self.height % 8 or self.width % 8:
            raise ValueError("video height and width must be divisible by 8.")
        if self.fps < 1 or self.max_sequence_length < 1:
            raise ValueError("video fps and max_sequence_length must be positive.")


@dataclass
class MeanFlowNFTRLConfig:
    num_epochs: int = 10000
    num_batches_per_epoch: int = 2
    num_inner_epochs: int = 1
    nft_inner_batch_size: int = 0
    nft_start_epoch: int = 0
    num_image_per_prompt: int = 16
    nft_sample_use_ema: bool = False
    sampling_num_steps: int = 4
    sampling_guidance_scale: float = 1.0
    per_prompt_stat_tracking: bool = True
    per_prompt_global_std: bool = True
    weight_advantages: bool = True
    adv_clip_max: float = 5.0
    beta: float = 0.1
    kl_weight: float = 0.0001
    decay_type: int = 1
    nft_decay_interval: int = 1
    reward_fn: dict[str, float] = field(
        default_factory=lambda: {
            "hpsv3_general": 0.5,
            "hpsv3_percentile": 0.5,
            "videoalign_mq": 0.5,
            "videoalign_ta": 1.0,
        }
    )
    reward_model_paths: dict[str, str] = field(default_factory=dict)
    log_sample_images: int = 0
    nft_velocity_mode: str = "meanflow_v"
    central_diff_epsilon: float = 5.0
    cd_velocity_source: str = "noise_minus_x0"
    share_cd_with_old: bool = True
    num_training_timesteps_per_sample: int = 4
    nft_min_timestep: int = 0
    nft_max_timestep: int = 1000
    diffusion_ratio: float = 0.5
    consistency_ratio: float = 0.25

    def __post_init__(self) -> None:
        if self.sampling_num_steps < 1:
            raise ValueError("meanflow_nft.sampling_num_steps must be positive.")
        if self.sampling_guidance_scale != 1.0:
            raise ValueError("Wan AnyFlow sampling is CFG-free; use guidance 1.0.")
        if self.num_training_timesteps_per_sample < 1:
            raise ValueError(
                "meanflow_nft.num_training_timesteps_per_sample must be positive."
            )
        if self.nft_velocity_mode != "meanflow_v":
            raise ValueError(
                "The release supports only meanflow_nft.nft_velocity_mode='meanflow_v'."
            )
        if self.cd_velocity_source not in {"noise_minus_x0", "u_self"}:
            raise ValueError(
                "meanflow_nft.cd_velocity_source must be noise_minus_x0 or u_self."
            )
        if self.central_diff_epsilon <= 0:
            raise ValueError("meanflow_nft.central_diff_epsilon must be positive.")
        if self.nft_min_timestep > self.nft_max_timestep:
            raise ValueError(
                "meanflow_nft.nft_min_timestep cannot exceed nft_max_timestep."
            )
        ratio = self.diffusion_ratio + self.consistency_ratio
        if (
            self.diffusion_ratio < 0
            or self.consistency_ratio < 0
            or ratio > 1.0 + 1e-6
        ):
            raise ValueError(
                "meanflow_nft diffusion_ratio and consistency_ratio must be "
                "non-negative and sum to at most 1."
            )
        if self.adv_clip_max <= 0 or self.beta <= 0:
            raise ValueError("meanflow_nft adv_clip_max and beta must be positive.")
        valid_rewards = {
            "hpsv3_general",
            "hpsv3_percentile",
            "videoalign_mq",
            "videoalign_ta",
        }
        unsupported = sorted(set(self.reward_fn) - valid_rewards)
        if unsupported:
            raise ValueError(f"Unsupported Wan reward(s): {unsupported}")


@dataclass
class GeneratorSolverConfig:
    lr: float = 3e-6
    warmup_steps: int = 0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0


@dataclass
class AuxSolverConfig:
    """Optimizer for ``aux_v_model``, the online-trained v-predictor.

    Kept separate from ``generator``: the aux network trains with a plain
    flow-matching loss and takes ``renoise_per_epoch / aux_v_gradient_accum``
    steps per epoch, so its learning rate is tuned independently (two-timescale).
    """

    lr: float = 1e-5
    warmup_steps: int = 0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0


@dataclass
class SolverConfig:
    generator: GeneratorSolverConfig = field(default_factory=GeneratorSolverConfig)
    #: Only read when ``fewstep_rl.aux_v_enabled`` (dmdnft).
    aux_v: AuxSolverConfig = field(default_factory=AuxSolverConfig)


@dataclass
class TrainConfig:
    total_steps: int = 0
    save_interval: int = 50
    log_interval: int = 1
    seed: int = 42
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    autocast_dtype: str = "bf16"
    resume_from: str = ""
    use_ema: bool = True
    ema_decay: float = 0.9
    ema_every: int = 1


@dataclass
class DistributedConfig:
    strategy: str = "hsdp"
    fsdp_sharding: str = "hybrid"
    ddp_find_unused_parameters: bool | None = None

    def __post_init__(self) -> None:
        self.strategy = self.strategy.lower()
        self.fsdp_sharding = self.fsdp_sharding.lower()
        if self.strategy == "hsdp":
            self.strategy = "fsdp"
            self.fsdp_sharding = "hybrid"
        if self.strategy not in {"fsdp", "ddp"}:
            raise ValueError("distributed.strategy must be fsdp, hsdp, or ddp.")


@dataclass
class VBenchConfig:
    enabled: bool = False
    aug_info_json: str = "dataset/vbench/VBench_aug_full_info.json"
    full_info_json: str = "dataset/vbench/VBench_full_info.json"
    num_samples_per_prompt: int = 5
    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    cache_dir: str = "models/vbench"
    output_subdir: str = "vbench"
    eval_every_epochs: int = 0

    def __post_init__(self) -> None:
        if self.num_samples_per_prompt < 1 or self.num_inference_steps < 1:
            raise ValueError("VBench sample and inference counts must be positive.")
        if self.guidance_scale != 1.0:
            raise ValueError("Wan AnyFlow VBench generation is CFG-free.")


@dataclass
class EvalConfig:
    enabled: bool = True
    eval_interval: int = 50
    eval_prompt_source: str = "dataset"
    eval_prompt_path: str = ""
    dataset: str = "wan_dancegrpo"
    dataset_path: str = "dataset"
    reward_fn: dict[str, float] = field(
        default_factory=lambda: {
            "hpsv3_general": 0.5,
            "hpsv3_percentile": 0.5,
            "videoalign_mq": 0.5,
            "videoalign_ta": 1.0,
        }
    )
    reward_dataset_map: dict[str, str] = field(default_factory=dict)
    reward_ckpt_path: str = ""
    eval_num_steps: int = 4
    eval_guidance_scale: float = 1.0
    eval_batch_size: int = 1
    num_media_images: int = 4
    eval_seed: int = 12345
    vbench: VBenchConfig = field(default_factory=VBenchConfig)


@dataclass
class LoggingConfig:
    project: str = "fewsteprl-wan21"
    run_name: str = "wan2.1-t2v-1.3b-meanflow-nft"
    entity: str = ""
    tags: list[str] = field(
        default_factory=lambda: [
            "wan2.1",
            "video",
            "anyflow",
            "meanflow-nft",
            "480p",
            "81f",
        ]
    )
    enabled: bool = True


@dataclass
class MeanFlowNFTConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    meanflow_nft: MeanFlowNFTRLConfig = field(default_factory=MeanFlowNFTRLConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    prompt_path: str = "dataset/wan_dancegrpo"
    output_dir: str = "./outputs/meanflow_nft"

    @classmethod
    def from_yaml(
        cls,
        path: str,
        overrides: dict[str, Any] | None = None,
    ) -> "MeanFlowNFTConfig":
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise TypeError(f"Configuration must be a YAML mapping: {path}")
        if overrides:
            raw = _apply_flat_overrides(raw, overrides)
        return _dataclass_from_dict(cls, raw)

    def to_dict(self) -> dict:
        return asdict(self)

    def save_yaml(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)


# --- Few-step RL (x0-form trainer family) ---
# Separate from MeanFlowNFTConfig above: the x0-form splits the two anchors so
# S = (t-r)*dF/dt only appears inside a detached anchor (see
# fewsteprl/trainers/rl_general_base.py), so its knobs do not mean the same thing.


@dataclass
class FewstepRLTrainConfig:
    """Few-step RL configuration (the ``fewstep_rl`` block)."""

    # Epoch loop and sampling.
    num_epochs: int = 1000
    num_batches_per_epoch: int = 1
    num_inner_epochs: int = 1
    rl_inner_batch_size: int = 0
    rl_start_epoch: int = 0
    num_image_per_prompt: int = 4
    rl_sample_use_ema: bool = False
    sampling_num_steps: int = 4
    sampling_guidance_scale: float = 1.0

    # Advantage and policy loss.
    per_prompt_stat_tracking: bool = True
    per_prompt_global_std: bool = True
    weight_advantages: bool = False
    adv_clip_max: float = 5.0
    beta: float = 0.1
    kl_weight: float = 0.0

    # Old-policy update.
    decay_type: int = 1
    rl_decay_interval: int = 1

    # Reward scoring and logging.
    reward_fn: dict[str, float] = field(default_factory=dict)
    reward_model_paths: dict[str, str] = field(default_factory=dict)
    log_sample_images: int = 0

    # Central difference for dF/dt.
    central_diff_epsilon: float = 5.0
    cd_velocity_source: str = "noise_minus_x0"
    #: Must be True: ``x_anchor_2`` reuses the single dF/dt computed on
    #: ``old_model``; per-network CDs would give the anchors different S.
    share_cd_with_old: bool = True

    #: OPSD only: skip the central difference and treat dF/dt as 0, saving two
    #: ``old_model`` forwards per call while keeping r randomised. S cancels in
    #: the differentiable channel (``x_0_pred_v - x_anchor_1``) but not in the
    #: stop-grad coefficients (trust-region budget, W, reward force), so this is
    #: a slightly different objective. Other trainers raise if it is set.
    disable_cd: bool = False

    # Fresh (t, r) pairs drawn for every inner update.
    num_training_timesteps_per_sample: int = 4
    rl_min_timestep: int = 0
    rl_max_timestep: int = 1000
    diffusion_ratio: float = 0.25
    consistency_ratio: float = 0.25

    # Ref for KL.
    #: Must be False on Wan: a v-mode KL reference needs a transformer without
    #: the r input. Only the u-mode KL (frozen generator snapshot) is available.
    use_ref_v_model: bool = False
    guidance_scale: float = 1.0

    #: "ode" (default) or "sde" (meanflowgrpo; requires ``reuse_x_t``).
    rollout_type: str = "ode"

    # --- AWM / GRPO (meanflowawm, meanflowgrpo) ---
    #: PPO clip half-width on the likelihood ratio. Small because
    #: ``decoupled_weights`` builds log-rho from a spatial-mean MSE, i.e. a
    #: per-dimension log-ratio (the joint ratio would overflow).
    clip_range: float = 1e-3

    #: Also return the rollout's intermediate ``(x_t, t, r)`` states. Needed only
    #: by meanflowgrpo; meanflowawm re-noises ``x_0`` at a fresh ``(t, r)``.
    reuse_x_t: bool = False

    # --- Auxiliary v-predictor (dmdnft / dmdawm) ---
    #: Build and train ``aux_v_model``; dmdnft reads ``x_anchor_2`` from it.
    aux_v_enabled: bool = False
    #: Rollout samples per GPU in the aux pack (0 = ``renoise_per_epoch *
    #: aux_v_batch_size``). At or below ``num_batches_per_epoch *
    #: train.batch_size`` the policy epoch's pack is reused (no extra rollouts).
    max_aux_sample: int = 0
    #: Flow-matching passes over the pack per epoch, each with fresh ``(t, eps)``.
    renoise_per_epoch: int = 10
    #: Backward passes per aux optimizer step.
    aux_v_gradient_accum: int = 1
    #: Microbatch per flow-matching evaluation; 0 = the whole pack.
    aux_v_batch_size: int = 0

    # --- OPSD / ReFL (unread unless ``opsd_enabled``) ---
    #: Master switch; False leaves this block unread and unvalidated.
    opsd_enabled: bool = False

    #: Trust-region radius factor rho: targets stay within
    #: ``||y_pm - y_0|| <= rho * ||y_0||``.
    opsd_rho: float = 0.10

    #: Query noise level sigma*. Not on Wan's 4-step rollout grid, so
    #: ``opsd_state_mode="forward"`` forward-noises to this exact sigma.
    opsd_query_sigma: float = 0.278

    #: M_tgt: normalized reward-gradient steps per target. The first gradient is
    #: shared by the ascent and descent legs, so this costs ``2*M - 1`` reward
    #: backwards (each expensive on Wan: full clip through VAE + reward).
    opsd_n_ascent: int = 2

    #: eta_tgt, the target-step multiplier. At <= 1 the trust-region projection
    #: is inactive in exact arithmetic.
    opsd_eta_tgt: float = 1.0

    #: Differentiable reward followed by the target ascent (direction only;
    #: ``reward_fn`` still sets the advantages). Only HPSv3 is differentiable;
    #: VideoAlign scores from video files.
    opsd_reward_kind: str = "hpsv3_general"

    #: Samples per reward-gradient chunk; bounds peak memory of the VAE + reward
    #: backward.
    opsd_target_chunk: int = 1

    #: eps_g: gradient-normalization stabilizer (paper App. C.1).
    opsd_eps_grad: float = 1e-12

    #: eps_gamma: floor on the adaptive normalizers gamma_pm.
    opsd_eps_gamma: float = 1e-5

    #: Dual-negative branch. False = positive-branch only (the paper's ablation).
    opsd_dual_neg: bool = True

    #: Query-state provenance.
    #:   "forward" -- forward-noise the rollout endpoint to the exact
    #:                ``opsd_query_sigma``. Needs neither reuse_x_t nor SDE.
    #:   "rollout" -- the rollout's own state at the nearest on-grid sigma.
    #:                Requires reuse_x_t=True.
    opsd_state_mode: str = "forward"

    #: How the query sigma is drawn:
    #:   "fixed"     -- every sample queried at exactly ``opsd_query_sigma``.
    #:   "lognormal" -- per-sample logit-normal draw pushed through the
    #:                  scheduler shift, centred on ``opsd_query_sigma``.
    opsd_sigma_mode: str = "fixed"

    #: Only read when ``opsd_sigma_mode == "lognormal"``. logit_mean =
    #: logit(inverse_shift(0.278)), so it must change with the scheduler shift.
    opsd_logit_mean: float = -2.053
    opsd_logit_std: float = 0.6

    #: Where ``r`` comes from in the v-form trainers: "zero" (r = 0) or
    #: "given_t" (``_draw_training_s_given_t``). "given_t" requires the v-form
    #: clean-output lift, under which y_theta is invariant to r.
    opsd_r_mode: str = "zero"

    #: ReFL only: weight the single ascent square by the normalized advantage
    #: (False = advantage-blind).
    refl_use_advantage: bool = False

    #: Target direction: "grad" (the method), or the controls "rand",
    #: "residual", "noop".
    opsd_dir_mode: str = "grad"

    #: Root for reward checkpoints; empty uses each scorer's own default.
    reward_ckpt_path: str = ""

    # --- GRPO (meanflowgrpo) ---
    #: Noise scale of flow-GRPO's Euler-Maruyama scoring kernel:
    #: ``std_dev_t = noise_level * sqrt(sigma / (1 - sigma))``.
    grpo_noise_level: float = 0.7

    def __post_init__(self) -> None:
        if self.sampling_num_steps < 1:
            raise ValueError("fewstep_rl.sampling_num_steps must be positive.")
        if self.sampling_guidance_scale != 1.0 or self.guidance_scale != 1.0:
            raise ValueError(
                "Wan AnyFlow sampling is CFG-free; both guidance scales must be 1.0."
            )
        if self.num_training_timesteps_per_sample < 1:
            raise ValueError(
                "fewstep_rl.num_training_timesteps_per_sample must be positive."
            )
        if self.cd_velocity_source not in {"noise_minus_x0", "u_self"}:
            raise ValueError(
                "fewstep_rl.cd_velocity_source must be noise_minus_x0 or u_self."
            )
        if self.central_diff_epsilon <= 0:
            raise ValueError("fewstep_rl.central_diff_epsilon must be positive.")
        if not self.share_cd_with_old:
            raise ValueError(
                "fewstep_rl.share_cd_with_old must be True: the x0-form anchors "
                "read one shared dF/dt computed on old_model."
            )
        if self.rl_min_timestep > self.rl_max_timestep:
            raise ValueError(
                "fewstep_rl.rl_min_timestep cannot exceed rl_max_timestep."
            )
        ratio = self.diffusion_ratio + self.consistency_ratio
        if self.diffusion_ratio < 0 or self.consistency_ratio < 0 or ratio > 1.0 + 1e-6:
            raise ValueError(
                "fewstep_rl diffusion_ratio and consistency_ratio must be "
                "non-negative and sum to at most 1."
            )
        if self.adv_clip_max <= 0 or self.beta <= 0:
            raise ValueError("fewstep_rl adv_clip_max and beta must be positive.")
        if self.rollout_type not in {"ode", "sde"}:
            raise ValueError(
                "fewstep_rl.rollout_type must be 'ode' or 'sde'; got "
                f"{self.rollout_type!r}."
            )
        # An SDE rollout only pays off if its intermediate states are kept for
        # GRPO's policy ratio.
        if self.rollout_type == "sde" and not self.reuse_x_t:
            raise ValueError(
                "fewstep_rl.rollout_type='sde' with reuse_x_t=false: the SDE "
                "rollout exists to supply replayable transitions, and without "
                "reuse_x_t the intermediate states are discarded, so this costs "
                "the extra noise draws and buys nothing. Set reuse_x_t=true "
                "(the GRPO arm needs both) or go back to 'ode'."
            )
        if self.use_ref_v_model:
            raise ValueError(
                "fewstep_rl.use_ref_v_model is unsupported on the wan branch: "
                "AnyFlow-Wan has no pristine instantaneous-velocity reference "
                "(its two-time conditioning is baked in). Use the u-mode KL."
            )
        if self.clip_range <= 0:
            raise ValueError("fewstep_rl.clip_range must be positive.")
        # ``disable_cd`` relies on the OPSD loss's S cancellation; in the other
        # trainers S enters the target directly.
        if self.disable_cd and not self.opsd_enabled:
            raise ValueError(
                "fewstep_rl.disable_cd is OPSD-specific (the cd term cancels "
                "in the OPSD loss's differentiable channel) but "
                "opsd_enabled=false. Drop disable_cd, or use "
                "diffusion_ratio=1.0 if you want an r == t run without the "
                "central difference on a non-OPSD arm."
            )
        if self.opsd_enabled:
            if self.opsd_rho <= 0:
                raise ValueError("fewstep_rl.opsd_rho must be positive.")
            if not 0.0 < self.opsd_query_sigma < 1.0:
                raise ValueError(
                    "fewstep_rl.opsd_query_sigma must be in (0, 1); got "
                    f"{self.opsd_query_sigma!r}."
                )
            if self.opsd_n_ascent < 1:
                raise ValueError("fewstep_rl.opsd_n_ascent must be at least 1.")
            if self.opsd_target_chunk < 1:
                raise ValueError("fewstep_rl.opsd_target_chunk must be at least 1.")
            if self.opsd_state_mode not in {"forward", "rollout"}:
                raise ValueError(
                    "fewstep_rl.opsd_state_mode must be 'forward' or 'rollout'; "
                    f"got {self.opsd_state_mode!r}."
                )
            if self.opsd_state_mode == "rollout" and not self.reuse_x_t:
                raise ValueError(
                    "fewstep_rl.opsd_state_mode='rollout' reads the rollout's own "
                    "intermediate states, which are only returned under "
                    "reuse_x_t=true."
                )
            if self.opsd_dir_mode not in {"grad", "rand", "residual", "noop"}:
                raise ValueError(
                    "fewstep_rl.opsd_dir_mode must be grad/rand/residual/noop; "
                    f"got {self.opsd_dir_mode!r}."
                )
            # Only HPSv3 is differentiable (VideoAlign scores from video files),
            # so "mixed" spans the two HPSv3 reductions only.
            if not (
                self.opsd_reward_kind.startswith("hpsv3")
                or self.opsd_reward_kind == "mixed"
            ):
                raise ValueError(
                    f"fewstep_rl.opsd_reward_kind={self.opsd_reward_kind!r} is not "
                    "differentiable on the wan branch; only hpsv3_* and 'mixed' "
                    "(= the two hpsv3 reductions) are. See "
                    "fewsteprl/rewards/video_grad_scorer.py."
                )
            if self.opsd_sigma_mode not in {"fixed", "lognormal"}:
                raise ValueError(
                    "fewstep_rl.opsd_sigma_mode must be 'fixed' or 'lognormal'; "
                    f"got {self.opsd_sigma_mode!r}."
                )
            if self.opsd_logit_std <= 0.0:
                raise ValueError(
                    "fewstep_rl.opsd_logit_std must be positive; got "
                    f"{self.opsd_logit_std!r}."
                )
            if self.opsd_r_mode not in {"zero", "given_t"}:
                raise ValueError(
                    "fewstep_rl.opsd_r_mode must be 'zero' or 'given_t'; got "
                    f"{self.opsd_r_mode!r}."
                )
        # std_dev_t already diverges as sigma -> 1; values past ~3 are typos.
        if not 0.0 < self.grpo_noise_level <= 3.0:
            raise ValueError(
                "fewstep_rl.grpo_noise_level must be in (0, 3]; got "
                f"{self.grpo_noise_level!r}."
            )
        if self.renoise_per_epoch < 1:
            raise ValueError("fewstep_rl.renoise_per_epoch must be at least 1.")
        if self.aux_v_gradient_accum < 1:
            raise ValueError("fewstep_rl.aux_v_gradient_accum must be at least 1.")
        if self.aux_v_batch_size < 0:
            raise ValueError(
                "fewstep_rl.aux_v_batch_size must be >= 0 (0 means the whole pack)."
            )
        if self.max_aux_sample < 0:
            raise ValueError(
                "fewstep_rl.max_aux_sample must be >= 0 (0 derives it from "
                "renoise_per_epoch * aux_v_batch_size)."
            )
        # Gated on aux_v_enabled: both fields default to 0 and derive from each
        # other, so an ungated derivation would reject every non-aux config.
        if self.aux_v_enabled:
            if self.max_aux_sample == 0:
                # Resolved here so every consumer (and the saved config.yaml)
                # sees one concrete number.
                if self.aux_v_batch_size < 1:
                    raise ValueError(
                        "fewstep_rl.max_aux_sample=0 derives the pack size from "
                        "aux_v_batch_size, but aux_v_batch_size=0 derives the "
                        "microbatch from the pack size. Set one of the two to a "
                        "positive value."
                    )
                self.max_aux_sample = self.renoise_per_epoch * self.aux_v_batch_size
            # The aux loop cycles a permutation of the pack; require at least one
            # full sweep so no rollout is wasted.
            micro = self.aux_v_batch_size or self.max_aux_sample
            sweep_len = (self.max_aux_sample + micro - 1) // micro
            if self.renoise_per_epoch < sweep_len:
                raise ValueError(
                    "fewstep_rl.renoise_per_epoch="
                    f"{self.renoise_per_epoch} is too small to consume "
                    f"max_aux_sample={self.max_aux_sample} at a microbatch of "
                    f"{micro}: {sweep_len} evaluations are needed for every "
                    "rolled-out sample to be trained on at least once. Raise "
                    "renoise_per_epoch to at least that, or lower max_aux_sample."
                )
            if self.renoise_per_epoch < self.aux_v_gradient_accum:
                raise ValueError(
                    "fewstep_rl.renoise_per_epoch="
                    f"{self.renoise_per_epoch} is below aux_v_gradient_accum="
                    f"{self.aux_v_gradient_accum}, so the accumulation window "
                    "never fills and the aux model would take ZERO optimizer "
                    "steps per epoch."
                )
        valid_rewards = {
            "hpsv3_general",
            "hpsv3_percentile",
            "videoalign_mq",
            "videoalign_ta",
        }
        unsupported = sorted(set(self.reward_fn) - valid_rewards)
        if unsupported:
            raise ValueError(f"Unsupported Wan reward(s): {unsupported}")


@dataclass
class FewStepRLConfig:
    """Top-level config for the x0-form few-step RL trainer family."""

    model: ModelConfig = field(default_factory=ModelConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    fewstep_rl: FewstepRLTrainConfig = field(default_factory=FewstepRLTrainConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    prompt_path: str = "dataset/wan_dancegrpo"
    output_dir: str = "./outputs/meanflownft"

    @classmethod
    def from_yaml(
        cls,
        path: str,
        overrides: dict[str, Any] | None = None,
    ) -> "FewStepRLConfig":
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise TypeError(f"Configuration must be a YAML mapping: {path}")
        if overrides:
            raw = _apply_flat_overrides(raw, overrides)
        return _dataclass_from_dict(cls, raw)

    def to_dict(self) -> dict:
        return asdict(self)

    def save_yaml(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)

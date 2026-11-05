"""Base trainer: distributed setup, training-loop skeleton, checkpointing, logging.

Handles FSDP/DDP setup, gradient accumulation, checkpoint save/resume, EMA
updates, wandb logging and seeding. Subclasses implement ``setup_models()``,
``setup_optimizers()`` and ``train_step(batch) -> dict``. The RL trainers derive
from it via ``GeneralRLTrainer`` (``rl_general_base.py``).
"""

from __future__ import annotations

import json
import logging
import os
import random
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from fewsteprl.config import MeanFlowNFTConfig
from fewsteprl.parallel.utils import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    get_world_size,
)
from fewsteprl.utils.checkpoint import save_checkpoint, load_checkpoint
from fewsteprl.utils.logging import (
    WandbLogger,
    Timer,
    get_gpu_memory_gb,
    setup_logging,
)

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Abstract base trainer with distributed training infrastructure.

    Args:
        config: The root MeanFlowNFT configuration.
    """

    _AUTOCAST_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}

    def __init__(self, config: MeanFlowNFTConfig):
        self.config = config
        self.global_step = 0
        self.models: dict[str, torch.nn.Module] = {}
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.schedulers: dict[str, Any] = {}
        self.timer = Timer()
        self.wandb_logger: WandbLogger | None = None
        self._autocast_dtype = self._AUTOCAST_DTYPE.get(config.train.autocast_dtype)
        # Evaluation state (lazy-initialized on first _evaluate() call).
        self._eval_scorer = None
        self._eval_prompts: list[str] | None = None
        # key: "__default__" or "dataset::<name>" -> list[str]
        self._eval_prompts_cache: dict[str, list[str]] = {}
        # key: "__default__" or "dataset::<name>" -> list[dict] | None
        self._eval_metadata_cache: dict[str, list[dict[str, Any]] | None] = {}
        # key: (scorer_attr_name, reward_name) -> MultiScorer({reward_name: weight})
        self._eval_single_reward_scorers: dict[tuple[str, str], Any] = {}
        # key: dataset name -> (images [N,C,H,W] on CPU, prompt list)
        self._eval_media_groups: dict[str, tuple[torch.Tensor, list[str]]] = {}
        # When True, _evaluate() keeps the scorer on GPU after scoring.
        self._defer_scorer_offload = False

    def _autocast(self):
        """Return a torch.autocast context manager using the configured precision."""
        return torch.autocast(
            device_type="cuda",
            dtype=self._autocast_dtype,
            enabled=self._autocast_dtype is not None,
        )

    # ------------------------------------------------------------------
    # Lifecycle methods
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Main training entry point. Sets up everything and runs the loop."""
        self._setup_environment()
        try:
            self.setup_models()
            self.setup_optimizers()
            self._setup_logging()
            self._maybe_resume()
            self._train_loop()
        finally:
            self._cleanup()

    #: ``train.resume_from`` value that evaluates the pre-RL policy (the LoRA is
    #: zero-initialised, so this is the AnyFlow base model). An empty
    #: ``resume_from`` still raises in evaluation-only mode.
    EVAL_INIT_SENTINEL: str = "ANYFLOW_INIT"

    def evaluate(self) -> None:
        """Load a checkpoint and run reward/VBench evaluation once."""
        resume = self.config.train.resume_from
        if resume == self.EVAL_INIT_SENTINEL:
            logger.info(
                "[Eval] train.resume_from=%s -- evaluating the PRE-RL policy "
                "(zero-init LoRA, i.e. the AnyFlow backbone). No checkpoint is "
                "loaded.", self.EVAL_INIT_SENTINEL,
            )
            self.config.train.resume_from = ""
        elif not resume:
            raise ValueError(
                "Evaluation-only mode requires train.resume_from=<checkpoint-dir>, "
                f"or train.resume_from={self.EVAL_INIT_SENTINEL} to evaluate the "
                "pre-RL policy."
            )
        self._evaluation_only = True
        self._setup_environment()
        try:
            self.setup_models()
            self.setup_optimizers()
            self._setup_logging()
            self._maybe_resume()
            self._evaluate()
        finally:
            self._cleanup()

    def _setup_environment(self) -> None:
        """Initialize distributed backend, set seeds, and prepare output dir."""
        setup_distributed()
        setup_logging()
        self._set_seed(self.config.train.seed)

        # Append a timestamp for run isolation.
        if is_main_process():
            date_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.config.output_dir = os.path.join(self.config.output_dir, date_suffix)
        # Broadcast output_dir to all ranks so they agree on the path
        if dist.is_initialized() and get_world_size() > 1:
            output_dir_list = [self.config.output_dir]
            dist.broadcast_object_list(output_dir_list, src=0)
            self.config.output_dir = output_dir_list[0]

        logger.info(
            f"Environment ready: rank={get_rank()}, world_size={get_world_size()}, "
            f"device=cuda:{torch.cuda.current_device()}, output_dir={self.config.output_dir}"
        )

    def _set_seed(self, seed: int) -> None:
        """Seed all RNGs (offset by rank) and enable TF32.

        Set ``TORCH_DETERMINISTIC=1`` for fully deterministic (slower) kernels.
        """
        seed = seed + get_rank()  # Different seed per rank for data diversity
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if os.environ.get("TORCH_DETERMINISTIC", "0") == "1":
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
            logger.info("Full deterministic mode enabled (TORCH_DETERMINISTIC=1)")

    def _setup_logging(self) -> None:
        """Initialize wandb logger on the main process."""
        log_cfg = self.config.logging
        self.wandb_logger = WandbLogger(
            project=log_cfg.project,
            run_name=log_cfg.run_name,
            entity=log_cfg.entity,
            config=self.config.to_dict(),
            tags=log_cfg.tags,
            enabled=log_cfg.enabled,
            output_dir=self.config.output_dir,
        )
        # Save config YAML alongside checkpoints for reproducibility
        if is_main_process():
            os.makedirs(self.config.output_dir, exist_ok=True)
            self.config.save_yaml(
                os.path.join(self.config.output_dir, "config.yaml")
            )

    def _maybe_resume(self) -> None:
        """Resume from checkpoint if configured.

        Restores model weights, optimizer and LR scheduler states, and the global
        step. RNG states are restored later in ``_train_loop()``, after the
        dataloader is created and fast-forwarded (both consume RNG).

        Do not override; use ``_restore_extra_checkpoint_state()`` and
        ``_post_load_checkpoint()`` instead.
        """
        self._resume_rng_state = None  # Will be set if resuming
        resume_path = self.config.train.resume_from
        if resume_path:
            extra = load_checkpoint(
                resume_path,
                models=self.models,
                optimizers=self.optimizers,
            )
            self.global_step = extra.get("step", 0)

            scheduler_states = extra.get("schedulers", {})
            for name, sched in self.schedulers.items():
                if name in scheduler_states:
                    sched.load_state_dict(scheduler_states[name])
                    logger.info(f"  Restored LR scheduler: {name}")

            # Save RNG state for deferred restoration (after dataloader setup)
            rng_state = extra.get("rng_state", None)
            if rng_state:
                self._resume_rng_state = rng_state
                logger.info("  RNG states loaded (will restore after dataloader setup)")

            self._restore_extra_checkpoint_state(extra)

            ckpt_dir = resume_path if os.path.isdir(resume_path) else os.path.dirname(resume_path)
            self._post_load_checkpoint(ckpt_dir, extra)

            logger.info(f"Resumed training from step {self.global_step}")

    def _post_load_checkpoint(self, ckpt_dir: str, extra: dict) -> None:
        """Hook after the main checkpoint is loaded (e.g. per-rank buffers)."""
        pass

    def _restore_rng_state(self) -> None:
        """Restore RNG states saved during resume (after dataloader fast-forward)."""
        rng_state = self._resume_rng_state
        if rng_state is None:
            return

        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.random.set_rng_state(rng_state["torch_cpu"])
        # Restore per-rank CUDA RNG state
        if "torch_cuda_all_ranks" in rng_state:
            rank = get_rank()
            all_states = rng_state["torch_cuda_all_ranks"]
            if rank < len(all_states):
                torch.cuda.set_rng_state(all_states[rank])
                logger.info(f"  Restored CUDA RNG state for rank {rank}")
            else:
                logger.warning(
                    f"  CUDA RNG state not found for rank {rank} "
                    f"(saved {len(all_states)} ranks), using default"
                )
        elif "torch_cuda" in rng_state:
            # Legacy format: single CUDA state (only correct for rank 0)
            torch.cuda.set_rng_state(rng_state["torch_cuda"])
        logger.info("  Restored RNG states")
        self._resume_rng_state = None  # Clear after restoration

    def _cleanup(self) -> None:
        """Clean up distributed resources and finalize logging."""
        if self.wandb_logger:
            self.wandb_logger.finish()
        cleanup_distributed()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _log_step(self, metrics: dict[str, Any]) -> None:
        """Average ``{section: {name: value}}`` metrics across ranks and log them."""
        if dist.is_initialized() and get_world_size() > 1:
            metrics = self._reduce_metrics(metrics)

        profile_metrics = {
            "gpu_mem_gb": get_gpu_memory_gb(),
        }
        profile_metrics.update(self.timer.get_metrics())

        all_sections = dict(metrics)
        all_sections["profile"] = profile_metrics

        if self.wandb_logger:
            self.wandb_logger.log_multi_section(all_sections, step=self.global_step)

        if is_main_process():
            parts = [f"step={self.global_step}"]
            for section, section_metrics in all_sections.items():
                for k, v in section_metrics.items():
                    if isinstance(v, float):
                        parts.append(f"{section}/{k}={v:.4g}")
                    else:
                        parts.append(f"{section}/{k}={v}")
            logger.info(" | ".join(parts))

    def _save_checkpoint(self) -> None:
        """Save a fully resumable checkpoint.

        RNG states are captured before and restored after saving, since saving can
        consume RNG (e.g. by instantiating a model). Do not override; use
        ``_get_extra_checkpoint_state()`` and ``_post_save_checkpoint()``.
        """
        scheduler_states = {
            name: sched.state_dict() for name, sched in self.schedulers.items()
        }
        # CUDA RNG states differ per rank; gather all so each rank restores its own.
        local_cuda_rng = torch.cuda.get_rng_state()
        if dist.is_initialized() and get_world_size() > 1:
            # NCCL only supports GPU tensors, so move to CUDA for all_gather
            local_rng_cuda = local_cuda_rng.to(torch.cuda.current_device())
            all_cuda_rng = [torch.empty_like(local_rng_cuda) for _ in range(get_world_size())]
            dist.all_gather(all_cuda_rng, local_rng_cuda)
            all_cuda_rng = [t.cpu() for t in all_cuda_rng]  # move back to CPU for saving
        else:
            all_cuda_rng = [local_cuda_rng]

        # Capture all RNG states BEFORE save (save may consume RNG)
        saved_python_rng = random.getstate()
        saved_numpy_rng = np.random.get_state()
        saved_torch_cpu_rng = torch.random.get_rng_state()

        rng_state = {
            "python": saved_python_rng,
            "numpy": saved_numpy_rng,
            "torch_cpu": saved_torch_cpu_rng,
            "torch_cuda_all_ranks": all_cuda_rng,
        }

        # Only LoRA params are saved for LoRA models.
        lora_models = self._get_lora_model_names()
        lora_configs = self._get_lora_configs()

        extra_state = {
            "step": self.global_step,
            "schedulers": scheduler_states,
            "rng_state": rng_state,
        }
        extra_state.update(self._get_extra_checkpoint_state())

        save_checkpoint(
            output_dir=self.config.output_dir,
            step=self.global_step,
            models=self.models,
            optimizers=self.optimizers,
            extra_state=extra_state,
            pretrained_path=self.config.model.pretrained_path,
            lora_models=lora_models,
            lora_configs=lora_configs,
        )

        self._post_save_checkpoint()

        # Training continues identically whether or not a checkpoint was saved.
        random.setstate(saved_python_rng)
        np.random.set_state(saved_numpy_rng)
        torch.random.set_rng_state(saved_torch_cpu_rng)
        torch.cuda.set_rng_state(local_cuda_rng)

    def _post_save_checkpoint(self) -> None:
        """Hook after the main checkpoint is saved, before RNG restore.

        The checkpoint directory is ``<output_dir>/checkpoint-<global_step>``.
        """
        pass

    def _save_final_transformer(self) -> None:
        """Save the final generator (EMA if available) to ``output_dir/transformer``.

        Full-weight models are saved in Diffusers format; LoRA models in peft
        format, loadable via ``pipe.load_lora_weights()``.
        """
        from fewsteprl.utils.checkpoint import (
            _gather_fsdp_state_dict,
            _save_diffusers_transformer,
            save_lora_peft_format,
        )

        generator = self.models.get("generator_ema") or self.models.get("generator")
        if generator is None:
            return

        source_name = "generator_ema" if "generator_ema" in self.models else "generator"
        lora_models = self._get_lora_model_names()
        lora_configs = self._get_lora_configs()

        # Gather full state dict (all ranks participate)
        state_dict = _gather_fsdp_state_dict(generator)

        if is_main_process():
            transformer_dir = os.path.join(self.config.output_dir, "transformer")
            if source_name in lora_models:
                # LoRA mode: save in peft format
                from fewsteprl.utils.checkpoint import _resolve_lora_config
                lora_cfg = _resolve_lora_config(source_name, lora_configs)
                if lora_cfg is not None:
                    save_lora_peft_format(state_dict, lora_cfg, transformer_dir)
                    logger.info(
                        f"Final {source_name} LoRA saved in peft format: {transformer_dir}"
                    )
                else:
                    logger.warning(
                        f"No LoRAConfig for {source_name}, skipping final peft save"
                    )
            else:
                # Full weight mode: save in diffusers format
                _save_diffusers_transformer(
                    state_dict=state_dict,
                    config_path=self.config.model.pretrained_path,
                    save_dir=transformer_dir,
                )
                logger.info(f"Final {source_name} saved: {transformer_dir}")

        if dist.is_initialized():
            from fewsteprl.parallel.utils import barrier
            barrier()

    def _get_lora_model_names(self) -> set[str]:
        """Return the names of models trained with LoRA."""
        lora_models = set()
        model_cfg = self.config.model
        if hasattr(model_cfg, "generator_lora") and model_cfg.generator_lora.enabled:
            lora_models.add("generator")
            if "generator_ema" in self.models:
                lora_models.add("generator_ema")
        if hasattr(model_cfg, "fake_score_lora") and model_cfg.fake_score_lora.enabled:
            lora_models.add("fake_score_net")
        return lora_models

    def _get_lora_configs(self) -> dict[str, Any]:
        """Map model name -> LoRAConfig for LoRA-enabled models (for peft saving)."""
        lora_configs = {}
        model_cfg = self.config.model
        if hasattr(model_cfg, "generator_lora") and model_cfg.generator_lora.enabled:
            lora_configs["generator"] = model_cfg.generator_lora
        if hasattr(model_cfg, "fake_score_lora") and model_cfg.fake_score_lora.enabled:
            lora_configs["fake_score_net"] = model_cfg.fake_score_lora
        return lora_configs

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _load_eval_prompts_and_metadata(
        self,
        dataset_override: str | None = None,
    ) -> tuple[list[str], list[dict[str, Any]] | None]:
        """Load evaluation prompts and optional metadata from file or dataset directory."""
        eval_cfg = self.config.eval

        use_dataset = dataset_override is not None or eval_cfg.eval_prompt_source == "dataset"
        if use_dataset:
            # Load from the configured dataset directory.
            dataset_name = dataset_override if dataset_override is not None else eval_cfg.dataset
            dataset_dir = os.path.join(eval_cfg.dataset_path, dataset_name)
            # Try common prompt file names (eval uses test split)
            for fname in [
                "test.txt",
                "test.jsonl",
                "test_metadata.jsonl",
                "prompts.json",
                "prompts.txt",
                "metadata.jsonl",
            ]:
                fpath = os.path.join(dataset_dir, fname)
                if os.path.exists(fpath):
                    return self._read_prompt_file_with_metadata(fpath)
            raise FileNotFoundError(
                f"No prompt file found in dataset directory: {dataset_dir}"
            )
        else:
            # Load from explicit prompt file path
            return self._read_prompt_file_with_metadata(eval_cfg.eval_prompt_path)

    @staticmethod
    def _read_prompt_file_with_metadata(
        path: str,
    ) -> tuple[list[str], list[dict[str, Any]] | None]:
        """Read prompts and optional metadata from a JSON/JSONL/text file."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], str):
                    return data, None
                elif len(data) > 0 and isinstance(data[0], dict):
                    prompts = [item.get("prompt", item.get("text", "")) for item in data]
                    return prompts, data
            raise ValueError(f"Cannot parse prompts from JSON: {path}")
        elif ext == ".jsonl":
            prompts = []
            metadata: list[dict[str, Any]] = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        if isinstance(item, dict):
                            prompts.append(item.get("prompt", item.get("text", "")))
                            metadata.append(item)
                        else:
                            prompts.append(str(item))
                            metadata.append({"text": str(item)})
            return prompts, metadata
        else:
            # Plain text: one prompt per line
            with open(path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()], None

    @staticmethod
    def _cache_key_for_dataset(dataset_override: str | None = None) -> str:
        return "__default__" if dataset_override is None else f"dataset::{dataset_override}"

    def _get_cached_eval_prompts(self, dataset_override: str | None = None) -> list[str]:
        """Get cached eval prompts for default or reward-specific dataset."""
        cache_key = self._cache_key_for_dataset(dataset_override=dataset_override)
        prompts = self._eval_prompts_cache.get(cache_key)
        if prompts is None:
            prompts, metadata = self._load_eval_prompts_and_metadata(
                dataset_override=dataset_override
            )
            self._eval_prompts_cache[cache_key] = prompts
            self._eval_metadata_cache[cache_key] = metadata
            if dataset_override is None:
                self._eval_prompts = prompts
            source = (
                f"dataset={dataset_override}"
                if dataset_override is not None
                else f"{self.config.eval.eval_prompt_source}"
            )
            logger.info(f"[Eval] Loaded {len(prompts)} prompts ({source})")
        return prompts

    @staticmethod
    def _aggregate_eval_metrics_across_ranks(
        eval_metrics: dict[str, float],
        num_local: int,
    ) -> dict[str, float]:
        """Sample-weighted cross-rank average; tolerates ranks missing some keys."""
        world_size = get_world_size()
        if not dist.is_initialized() or world_size <= 1:
            return eval_metrics

        device = torch.device("cuda", torch.cuda.current_device())
        # Ensure every rank iterates the same key set to avoid collective mismatch.
        local_keys = list(eval_metrics.keys())
        gathered_keys: list[list[str] | None] = [None] * world_size
        dist.all_gather_object(gathered_keys, local_keys)
        all_keys = sorted({k for keys in gathered_keys if keys is not None for k in keys})
        aggregated = {}
        for k in all_keys:
            has_local = k in eval_metrics
            local_num = torch.tensor(
                float(eval_metrics[k] * num_local) if has_local else 0.0,
                device=device,
            )
            local_den = torch.tensor(
                float(num_local) if has_local else 0.0,
                device=device,
            )
            dist.all_reduce(local_num, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_den, op=dist.ReduceOp.SUM)
            if local_den.item() > 0:
                aggregated[k] = local_num.item() / local_den.item()
        return aggregated

    # ------------------------------------------------------------------
    # Metric aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _reduce_metrics(
        metrics: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """All-reduce-average float metrics in ``{section: {name: value}}``.

        One collective per float key, so every rank must pass the same key set.
        Non-float values are kept as the local value.
        """
        world_size = get_world_size()
        reduced = {}
        for section, section_metrics in metrics.items():
            reduced_section = {}
            for k, v in section_metrics.items():
                if isinstance(v, float):
                    tensor = torch.tensor(v, device="cuda")
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                    reduced_section[k] = tensor.item() / world_size
                else:
                    reduced_section[k] = v
            reduced[section] = reduced_section
        return reduced

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def ema_update(
        src_model: torch.nn.Module,
        tgt_model: torch.nn.Module,
        decay: float,
    ) -> None:
        """EMA update ``tgt = decay * tgt + (1 - decay) * src`` (src cast to fp32)."""
        for src_param, tgt_param in zip(src_model.parameters(), tgt_model.parameters()):
            tgt_param.data.mul_(decay).add_(
                src_param.data.to(torch.float32), alpha=1.0 - decay
            )

    # ------------------------------------------------------------------
    # LR Scheduler helpers
    # ------------------------------------------------------------------

    @staticmethod
    def create_warmup_constant_scheduler(
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """LambdaLR with linear warmup over ``warmup_steps``, then constant."""
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # Abstract methods (subclass must implement)
    # ------------------------------------------------------------------

    @abstractmethod
    def setup_models(self) -> None:
        """Load, configure and wrap all models; populate ``self.models``."""
        ...

    @abstractmethod
    def setup_optimizers(self) -> None:
        """Populate ``self.optimizers`` and ``self.schedulers``."""
        ...


"""Checkpoint utilities for fewsteprl.

A checkpoint holds model weights (adapter-only for LoRA models), optimizer state,
scheduler/RNG metadata, and an inference-ready ``transformer/`` directory.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, TYPE_CHECKING

import torch
from safetensors.torch import save_file as safetensors_save_file
from safetensors.torch import load_file as safetensors_load_file
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)

if TYPE_CHECKING:
    from fewsteprl.config import LoRAConfig

from fewsteprl.parallel.utils import is_main_process, barrier

logger = logging.getLogger(__name__)


def _gather_fsdp_state_dict(model: torch.nn.Module) -> dict:
    """Full state dict; FSDP models are gathered to rank 0 on CPU (collective)."""
    if isinstance(model, FSDP):
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            return model.state_dict()
    else:
        return model.state_dict()


def _filter_lora_state_dict(state_dict: dict) -> dict:
    """Keep only trainable adapter weights: LoRA ``lora_A`` / ``lora_B`` keys and
    the flow-map ``delta_embedder.*`` tensors (trained in full, outside LoRA)."""
    return {
        k: v for k, v in state_dict.items()
        if ("lora_A" in k) or ("lora_B" in k) or ("delta_embedder" in k)
    }


def _is_lora_checkpoint(state_dict: dict) -> bool:
    """True if every key is a LoRA or ``delta_embedder`` key (adapter-only checkpoint)."""
    if not state_dict:
        return False
    return all(
        ("lora_A" in k) or ("lora_B" in k) or ("delta_embedder" in k)
        for k in state_dict.keys()
    )


def save_lora_peft_format(
    state_dict: dict,
    lora_config: LoRAConfig,
    save_dir: str,
) -> None:
    """Save adapter weights in peft format (``adapter_config.json`` +
    ``adapter_model.safetensors``).

    ``state_dict`` is the full model state dict; it is filtered to adapter keys
    (LoRA plus ``delta_embedder``).
    """
    os.makedirs(save_dir, exist_ok=True)

    lora_state = _filter_lora_state_dict(state_dict)

    safetensors_save_file(lora_state, os.path.join(save_dir, "adapter_model.safetensors"))

    adapter_config = {
        "r": lora_config.rank,
        "lora_alpha": lora_config.lora_alpha,
        "target_modules": list(lora_config.target_modules),
        "lora_dropout": 0.0,
        "bias": "none",
        "peft_type": "LORA",
        "task_type": None,
        "init_lora_weights": lora_config.init_lora_weights,
        "use_rslora": False,
    }
    with open(os.path.join(save_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)

    logger.info(f"  LoRA saved in peft format: {save_dir} ({len(lora_state)} keys)")


def load_lora_peft_format(
    model: torch.nn.Module,
    lora_dir: str,
    map_location: str = "cpu",
) -> None:
    """Load adapter weights from a peft-format directory (``strict=False``).

    ``model`` must already have its LoRA adapters injected.
    """
    safetensors_path = os.path.join(lora_dir, "adapter_model.safetensors")
    if os.path.exists(safetensors_path):
        state_dict = safetensors_load_file(safetensors_path, device=map_location)
    else:
        # Fall back to a .pt adapter file.
        pt_path = os.path.join(lora_dir, "adapter_model.pt")
        if os.path.exists(pt_path):
            state_dict = torch.load(pt_path, map_location=map_location, weights_only=False)
        else:
            raise FileNotFoundError(
                f"No adapter_model.safetensors or adapter_model.pt found in {lora_dir}"
            )
    if isinstance(model, FSDP):
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    logger.info(f"  Loaded LoRA weights from peft format: {lora_dir} ({len(state_dict)} keys)")


def is_peft_lora_dir(path: str) -> bool:
    """True if ``path`` is a directory with ``adapter_model.safetensors`` or
    ``adapter_config.json``."""
    return os.path.isdir(path) and (
        os.path.exists(os.path.join(path, "adapter_model.safetensors"))
        or os.path.exists(os.path.join(path, "adapter_config.json"))
    )


def resolve_meanflow_nft_adapter(path: str) -> str:
    """Resolve a trained Wan adapter file, or find one inside a checkpoint directory
    (EMA generator preferred)."""
    path = os.path.abspath(os.path.expanduser(str(path)))
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Stage-3 checkpoint not found: {path}")
    candidates = (
        os.path.join(path, "generator_ema.pt"),
        os.path.join(path, "transformer", "adapter_model.safetensors"),
        os.path.join(path, "adapter_model.safetensors"),
        os.path.join(path, "generator.pt"),
    )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "No generator_ema.pt, generator.pt, or adapter_model.safetensors "
        f"under {path}"
    )


def load_wan_meanflow_nft_adapter(
    model: torch.nn.Module,
    path: str,
    lora_config: "LoRAConfig",
) -> dict[str, int | str]:
    """Load the composite Wan adapter (LoRA plus full delta embedder)."""
    from dataclasses import replace

    from fewsteprl.utils.lora import setup_lora

    resolved = resolve_meanflow_nft_adapter(path)
    clean_config = replace(
        lora_config,
        load_path="",
        pre_merge_paths=[],
        merge_before_training=False,
    )
    setup_lora(model, clean_config)
    if resolved.endswith(".safetensors"):
        state_dict = safetensors_load_file(resolved, device="cpu")
    else:
        state_dict = torch.load(
            resolved,
            map_location="cpu",
            weights_only=False,
        )
    if not isinstance(state_dict, dict):
        raise TypeError(f"Adapter must contain a state dict: {resolved}")
    lora_keys = [
        key
        for key in state_dict
        if "lora_A" in key or "lora_B" in key
    ]
    delta_keys = [key for key in state_dict if "delta_embedder" in key]
    if not lora_keys:
        raise ValueError(f"No LoRA tensors found in {resolved}")
    if not delta_keys:
        raise ValueError(
            "Wan Stage-3 checkpoints must include delta_embedder tensors."
        )
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            "Unexpected Stage-3 checkpoint keys: "
            f"{incompatible.unexpected_keys[:20]}"
        )
    model.requires_grad_(False)
    model.eval()
    logger.info(
        "Loaded Wan MeanFlowNFT adapter: %s (%d LoRA, %d delta tensors)",
        resolved,
        len(lora_keys),
        len(delta_keys),
    )
    return {
        "path": resolved,
        "lora_tensors": len(lora_keys),
        "delta_tensors": len(delta_keys),
    }


def _save_diffusers_transformer(
    state_dict: dict,
    config_path: str,
    save_dir: str,
) -> None:
    """Save a full (unsharded) transformer state_dict in diffusers format.

    Builds a fresh unsharded model from ``<config_path>/transformer/config.json``,
    loads the weights and calls ``save_pretrained``.
    """
    import diffusers
    import json
    from fewsteprl.utils.fast_init import fast_init

    config_json = os.path.join(config_path, "transformer", "config.json")
    if not os.path.exists(config_json):
        logger.warning(
            f"Cannot save diffusers format: config not found at {config_json}. "
            "Skipping save_pretrained."
        )
        return

    # fast_init avoids consuming RNG state (this runs inside the RNG save/restore).
    with open(config_json, "r") as f:
        config = json.load(f)
    class_name = config.get(
        "_class_name",
        "WanAnyFlowTransformer3DModel",
    )
    model_cls = getattr(diffusers, class_name, None)
    if model_cls is None and class_name in {
        "AnyFlowTransformer3DModel",
        "FAR_Wan_Transformer3DModel",
        "WanAnyFlowTransformer3DModel",
    }:
        from fewsteprl.models.wan_transformer import (
            WanAnyFlowTransformer3DModel,
        )

        model_cls = WanAnyFlowTransformer3DModel
    if model_cls is None:
        logger.warning(
            f"Cannot save diffusers format: unsupported transformer class '{class_name}' "
            f"from {config_json}. Skipping save_pretrained."
        )
        return

    with fast_init(torch.device("cpu")):
        model = model_cls(**{k: v for k, v in config.items() if not k.startswith("_")})
    model.load_state_dict(state_dict)
    model.save_pretrained(save_dir)


def _resolve_lora_config(
    model_name: str,
    lora_configs: dict[str, "LoRAConfig"],
) -> "LoRAConfig | None":
    """LoRAConfig for ``model_name`` (``generator_ema`` shares the generator's), or None."""
    if model_name in lora_configs:
        return lora_configs[model_name]
    if model_name == "generator_ema" and "generator" in lora_configs:
        return lora_configs["generator"]
    return None


def save_checkpoint(
    output_dir: str,
    step: int,
    models: dict[str, torch.nn.Module],
    optimizers: dict[str, torch.optim.Optimizer],
    extra_state: dict[str, Any] | None = None,
    pretrained_path: str = "",
    lora_models: set[str] | None = None,
    lora_configs: dict[str, "LoRAConfig"] | None = None,
) -> None:
    """Save a training checkpoint to ``<output_dir>/checkpoint-<step>/``.

    Writes ``{name}.pt`` per model (adapter keys only for ``lora_models``),
    ``optimizer_{name}.pt``, ``meta.pt`` (step plus ``extra_state``), and a
    ``transformer/`` folder from the EMA generator if present: peft format for LoRA
    models (needs ``lora_configs``), diffusers format otherwise (needs
    ``pretrained_path``). Must be called on all ranks (FSDP gathers); only rank 0
    writes.
    """
    if lora_models is None:
        lora_models = set()
    if lora_configs is None:
        lora_configs = {}

    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")

    model_states = {}
    for name, model in models.items():
        model_states[name] = _gather_fsdp_state_dict(model)

    # Under FSDP, optimizer.state_dict() is rank-local; full_optim_state_dict()
    # gathers the full state to rank 0.
    optimizer_states = {}
    for name, optimizer in optimizers.items():
        # The matching model is named ``name`` or ``name`` plus a suffix (e.g. "_net").
        fsdp_model = None
        for model_name, model in models.items():
            if isinstance(model, FSDP) and (
                model_name == name or model_name.startswith(name)
            ):
                fsdp_model = model
                break
        if fsdp_model is not None:
            optimizer_states[name] = FSDP.full_optim_state_dict(fsdp_model, optimizer)
        else:
            optimizer_states[name] = optimizer.state_dict()

    if is_main_process():
        os.makedirs(ckpt_dir, exist_ok=True)

        # The transformer/ folder uses the EMA generator when available.
        gen_source_name = None
        if "generator_ema" in model_states:
            gen_source_name = "generator_ema"
        elif "generator" in model_states:
            gen_source_name = "generator"

        has_lora_generator = any(
            name in lora_models
            for name in ["generator", "generator_ema"]
            if name in model_states
        )

        # 1. transformer/ folder for inference.
        if gen_source_name:
            transformer_dir = os.path.join(ckpt_dir, "transformer")
            if has_lora_generator:
                lora_cfg = _resolve_lora_config(gen_source_name, lora_configs)
                if lora_cfg is not None:
                    save_lora_peft_format(
                        model_states[gen_source_name], lora_cfg, transformer_dir,
                    )
                    logger.info(
                        f"  {gen_source_name} LoRA saved in peft format: {transformer_dir}"
                    )
                else:
                    logger.warning(
                        f"  No LoRAConfig for {gen_source_name}, "
                        f"skipping transformer/ peft save"
                    )
            elif pretrained_path:
                _save_diffusers_transformer(
                    state_dict=model_states[gen_source_name],
                    config_path=pretrained_path,
                    save_dir=transformer_dir,
                )
                logger.info(
                    f"  {gen_source_name} saved in diffusers format: {transformer_dir}"
                )

        # 2. Per-model .pt files for resume (adapter keys only for LoRA models).
        for name, state_dict in model_states.items():
            if name in lora_models:
                lora_state = _filter_lora_state_dict(state_dict)
                path = os.path.join(ckpt_dir, f"{name}.pt")
                torch.save(lora_state, path)
                logger.info(
                    f"  Model saved (LoRA resume): {path} ({len(lora_state)} keys)"
                )
            else:
                path = os.path.join(ckpt_dir, f"{name}.pt")
                torch.save(state_dict, path)
                logger.info(f"  Model saved: {path}")

        # 3. Optimizers.
        for name, state_dict in optimizer_states.items():
            path = os.path.join(ckpt_dir, f"optimizer_{name}.pt")
            torch.save(state_dict, path)
            logger.info(f"  Optimizer saved: {path}")

        # 4. Meta state (step, schedulers, RNG).
        meta = {"step": step}
        if extra_state:
            meta.update(extra_state)
        meta_path = os.path.join(ckpt_dir, "meta.pt")
        torch.save(meta, meta_path)
        logger.info(f"  Meta saved: {meta_path}")

        logger.info(f"Checkpoint saved: {ckpt_dir} (step={step})")

    barrier()


def load_checkpoint(
    ckpt_path: str,
    models: dict[str, torch.nn.Module],
    optimizers: dict[str, torch.optim.Optimizer] | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load a training checkpoint into ``models`` / ``optimizers``.

    Accepts a split-format directory (with ``meta.pt``) or a single-file
    checkpoint (``checkpoint.pt``). Returns the extra state (``step``,
    ``schedulers``, ``rng_state``, ...).
    """
    if os.path.isdir(ckpt_path):
        meta_path = os.path.join(ckpt_path, "meta.pt")
        if os.path.exists(meta_path):
            return _load_split_checkpoint(ckpt_path, models, optimizers, map_location)
        legacy_path = os.path.join(ckpt_path, "checkpoint.pt")
        if os.path.exists(legacy_path):
            return _load_legacy_checkpoint(legacy_path, models, optimizers, map_location)
        raise FileNotFoundError(
            f"Checkpoint directory {ckpt_path} contains neither meta.pt nor checkpoint.pt"
        )
    elif os.path.isfile(ckpt_path):
        return _load_legacy_checkpoint(ckpt_path, models, optimizers, map_location)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")


def _load_split_checkpoint(
    ckpt_dir: str,
    models: dict[str, torch.nn.Module],
    optimizers: dict[str, torch.optim.Optimizer] | None,
    map_location: str,
) -> dict[str, Any]:
    """Load from the split checkpoint format."""
    logger.info(f"Loading split checkpoint from: {ckpt_dir}")

    for name, model in models.items():
        path = os.path.join(ckpt_dir, f"{name}.pt")
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=map_location, weights_only=False)
            # Adapter-only checkpoints are loaded non-strictly.
            is_lora = _is_lora_checkpoint(state_dict)
            if isinstance(model, FSDP):
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                    model.load_state_dict(state_dict, strict=not is_lora)
            else:
                model.load_state_dict(state_dict, strict=not is_lora)
            suffix = " (LoRA partial)" if is_lora else ""
            logger.info(f"  Loaded model: {name}{suffix}")
        else:
            logger.warning(f"  Model file not found: {path}, skipping")

    # Saved optimizer states are full (gathered); under FSDP they are scattered back
    # to shards with optim_state_dict_to_load().
    if optimizers:
        for name, optimizer in optimizers.items():
            path = os.path.join(ckpt_dir, f"optimizer_{name}.pt")
            if os.path.exists(path):
                full_osd = torch.load(path, map_location=map_location, weights_only=False)
                fsdp_model = None
                for model_name, model in models.items():
                    if isinstance(model, FSDP) and (
                        model_name == name or model_name.startswith(name)
                    ):
                        fsdp_model = model
                        break
                if fsdp_model is not None:
                    sharded_osd = FSDP.optim_state_dict_to_load(
                        fsdp_model, optimizer, full_osd,
                    )
                    optimizer.load_state_dict(sharded_osd)
                else:
                    optimizer.load_state_dict(full_osd)
                logger.info(f"  Loaded optimizer: {name}")
            else:
                logger.warning(f"  Optimizer file not found: {path}, skipping")

    meta_path = os.path.join(ckpt_dir, "meta.pt")
    meta = torch.load(meta_path, map_location=map_location, weights_only=False)
    logger.info(f"  Resumed from step {meta.get('step', 'unknown')}")
    return meta


def _load_legacy_checkpoint(
    ckpt_path: str,
    models: dict[str, torch.nn.Module],
    optimizers: dict[str, torch.optim.Optimizer] | None,
    map_location: str,
) -> dict[str, Any]:
    """Load from the single-file checkpoint format."""
    logger.info(f"Loading legacy checkpoint from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    for name, model in models.items():
        key = f"model_{name}"
        if key in state:
            if isinstance(model, FSDP):
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                    model.load_state_dict(state[key])
            else:
                model.load_state_dict(state[key])
            logger.info(f"  Loaded model: {name}")
        else:
            logger.warning(f"  Model '{name}' not found in checkpoint, skipping")

    if optimizers:
        for name, optimizer in optimizers.items():
            key = f"optimizer_{name}"
            if key in state:
                optimizer.load_state_dict(state[key])
                logger.info(f"  Loaded optimizer: {name}")
            else:
                logger.warning(f"  Optimizer '{name}' not found in checkpoint, skipping")

    extra = {k: v for k, v in state.items()
             if not k.startswith("model_") and not k.startswith("optimizer_")}
    logger.info(f"  Resumed from step {extra.get('step', 'unknown')}")
    return extra

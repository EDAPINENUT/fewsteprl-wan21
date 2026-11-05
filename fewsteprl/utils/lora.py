"""LoRA injection utilities for fewsteprl.

``setup_lora(model, lora_config)`` injects the adapters, optionally merges earlier
LoRA stages into the base weights, freezes everything else and returns the
trainable parameter names.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from fewsteprl.config import LoRAConfig

logger = logging.getLogger(__name__)


def inject_lora(model: nn.Module, lora_config: LoRAConfig) -> nn.Module:
    """Inject a LoRA adapter in place (``add_adapter`` if available, else peft's
    ``inject_adapter_in_model``) and optionally load or merge existing LoRA weights.
    """
    from peft import LoraConfig as PeftLoraConfig, inject_adapter_in_model

    peft_config = PeftLoraConfig(
        r=lora_config.rank,
        lora_alpha=lora_config.lora_alpha,
        init_lora_weights=lora_config.init_lora_weights,
        target_modules=lora_config.target_modules,
        use_rslora=False,
    )

    # --- pre_merge_paths: fold earlier LoRA stages into the base weights ---
    # Each path is loaded into the same adapter (same peft_config) and merged;
    # the adapter is then zeroed so the next merge does not re-add the delta.
    if lora_config.pre_merge_paths:
        from peft.tuners.tuners_utils import BaseTunerLayer
        if hasattr(model, "add_adapter") and callable(model.add_adapter):
            model.add_adapter(peft_config)
        else:
            inject_adapter_in_model(peft_config, model)
        for idx, pre_path in enumerate(lora_config.pre_merge_paths):
            state_dict = torch.load(pre_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            for module in model.modules():
                if isinstance(module, BaseTunerLayer):
                    module.merge()
            # Zero the adapter, then unmerge() (a no-op on the weights) to clear the
            # merged flag and re-enable the adapter.
            for name, param in model.named_parameters():
                if "lora_B" in name:
                    nn.init.zeros_(param.data)
                elif "lora_A" in name:
                    nn.init.zeros_(param.data)
            for module in model.modules():
                if isinstance(module, BaseTunerLayer) and module.merged:
                    module.unmerge()
            logger.info(
                f"Pre-merge LoRA[{idx}] folded into base: {pre_path}"
            )

    # --- merge_before_training: merge load_path into the base, train a fresh LoRA ---
    if lora_config.merge_before_training and lora_config.load_path:
        from peft.tuners.tuners_utils import BaseTunerLayer

        # 1. Inject the adapter (unless pre_merge_paths already did).
        adapter_already_injected = any(
            isinstance(m, BaseTunerLayer) for m in model.modules()
        )
        if not adapter_already_injected:
            if hasattr(model, "add_adapter") and callable(model.add_adapter):
                model.add_adapter(peft_config)
            else:
                inject_adapter_in_model(peft_config, model)

        # 2. Load the LoRA weights.
        state_dict = torch.load(lora_config.load_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded pretrained LoRA weights from {lora_config.load_path}")

        # 3. Merge into the base weights.
        merged_count = 0
        for module in model.modules():
            if isinstance(module, BaseTunerLayer):
                module.merge()
                merged_count += 1
        logger.info(f"Merged {merged_count} LoRA layers into base model weights")

        # 4. Re-initialize the adapter in place (lora_B = 0). delete_adapters()
        # cannot be used: it unmerges first, undoing step 3.
        reinit_count = 0
        for name, param in model.named_parameters():
            if "lora_B" in name:
                nn.init.zeros_(param.data)
                reinit_count += 1
            elif "lora_A" in name:
                if lora_config.init_lora_weights == "gaussian":
                    nn.init.normal_(param.data)
                else:
                    nn.init.kaiming_uniform_(param.data, a=5 ** 0.5)
                reinit_count += 1
        logger.info(f"Re-initialized {reinit_count} LoRA params for fresh training")

        # 5. Clear the merged flag so the adapter runs in forward; with lora_B = 0,
        # unmerge() leaves the base weights unchanged.
        for module in model.modules():
            if isinstance(module, BaseTunerLayer) and module.merged:
                module.unmerge()
        logger.info("Cleared merged flag — fresh LoRA adapter is now active")

        return model

    # --- normal path (adapter may already be injected by pre_merge_paths) ---
    from peft.tuners.tuners_utils import BaseTunerLayer
    adapter_already_injected = any(
        isinstance(m, BaseTunerLayer) for m in model.modules()
    )
    if not adapter_already_injected:
        if hasattr(model, "add_adapter") and callable(model.add_adapter):
            model.add_adapter(peft_config)
        else:
            logger.info("add_adapter() not found, fallback to inject_adapter_in_model()")
            inject_adapter_in_model(peft_config, model)

    # load_path initializes a new run from existing LoRA weights. This runs before
    # resuming, so train.resume_from (if set) overrides it.
    if lora_config.load_path:
        state_dict = torch.load(lora_config.load_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded pretrained LoRA weights from {lora_config.load_path}")
    elif adapter_already_injected:
        # pre_merge_paths zeroed lora_A as well; re-init it (lora_B stays 0).
        for name, param in model.named_parameters():
            if "lora_A" in name:
                if lora_config.init_lora_weights == "gaussian":
                    nn.init.normal_(param.data)
                else:
                    nn.init.kaiming_uniform_(param.data, a=5 ** 0.5)

    return model


def filter_lora_params(model: nn.Module, target_modules: list[str]) -> list[str]:
    """Names of the LoRA A/B parameters on ``target_modules`` (e.g. ``["attn.to_q"]``)."""
    lora_names = []
    for name, _ in model.named_parameters():
        for module in target_modules:
            if f"{module}.lora_A" in name or f"{module}.lora_B" in name:
                lora_names.append(name)
                break
    return lora_names


def setup_lora(model: nn.Module, lora_config: LoRAConfig) -> list[str]:
    """Inject LoRA, freeze all non-LoRA parameters, and return the LoRA parameter
    names."""
    inject_lora(model, lora_config)
    trainable_names = filter_lora_params(model, lora_config.target_modules)

    trainable_set = set(trainable_names)
    for name, param in model.named_parameters():
        param.requires_grad = name in trainable_set

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"LoRA injected: {num_trainable:,} trainable / {num_total:,} total params "
        f"({100 * num_trainable / num_total:.2f}%), rank={lora_config.rank}"
    )

    return trainable_names

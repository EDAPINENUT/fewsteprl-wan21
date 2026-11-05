"""FewStepRL training entry point for Wan2.1-T2V-1.3B (AnyFlow flow map).

Every trainer reads the ``fewstep_rl`` config section and is launched by one
of the ``scripts/train/<arm>.sh`` wrappers, which pin the trainer key:

- ``meanflow_nft``: MeanFlowNFT (self-induced anchor).
- ``nft_rl``: S-term ablation of MeanFlowNFT (no central difference).
- ``dmdnft_rl``: anchor velocity from an online-trained aux v-predictor.
- ``dmdawm_rl``: AWM ratio with no central difference + aux v-predictor.
- ``meanflowgrpo_rl``: MeanFlowGRPO on replayed SDE rollout transitions.
- ``meanflowawm_rl``: MeanFlowAWM.
- ``meanflowopsd_rl``: DiffusionOPSD under MeanFlow (v-form).
- ``meanflowrefl_rl``: the ReFL corner of few-step OPSD.
"""

import argparse
import yaml

from fewsteprl.config import FewStepRLConfig
from fewsteprl.trainers.dmdawm import DMDAWMTrainer
from fewsteprl.trainers.dmdnft import DMDNFTTrainer
from fewsteprl.trainers.meanflowawm import MeanFlowAWMTrainer
from fewsteprl.trainers.meanflowgrpo import MeanFlowGRPOTrainer
from fewsteprl.trainers.meanflownft import MeanFlowNFTTrainer, NFTTrainer
from fewsteprl.trainers.meanflowopsd import MeanFlowOPSDTrainer
from fewsteprl.trainers.meanflowrefl import MeanFlowReFLTrainer


TRAINER_REGISTRY = {
    # Decoupled loss: the fused MSE is split into self_regression +
    # reward_augmentation.
    "meanflow_nft": MeanFlowNFTTrainer,
    # x_anchor_2's instantaneous velocity comes from the online-trained
    # aux_v_model rather than from old_model's central difference.
    # Requires fewstep_rl.aux_v_enabled.
    "dmdnft_rl": DMDNFTTrainer,
    # dmdawm: no central difference; aux_v_model is the reward baseline.
    # Requires fewstep_rl.aux_v_enabled.
    "dmdawm_rl": DMDAWMTrainer,
    "meanflowgrpo_rl": MeanFlowGRPOTrainer,
    "meanflowawm_rl": MeanFlowAWMTrainer,
    # Both require fewstep_rl.opsd_enabled.
    "meanflowopsd_rl": MeanFlowOPSDTrainer,
    "meanflowrefl_rl": MeanFlowReFLTrainer,
    # S-term ablation of meanflow_nft: the flow map's average velocity is used
    # as the instantaneous one, and the central difference is never computed.
    "nft_rl": NFTTrainer,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FewStepRL Wan2.1 Training")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--trainer",
        type=str,
        required=True,
        choices=list(TRAINER_REGISTRY.keys()),
        help="Trainer key (pinned by scripts/train/<arm>.sh)",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides in key=value format, e.g. train.seed=123",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load train.resume_from and run test-set/VBench evaluation once.",
    )
    return parser.parse_args()


def parse_overrides(override_list: list[str]) -> dict:
    """Parse CLI overrides such as ``train.seed=123`` into a dictionary."""
    overrides = {}
    for item in override_list:
        if "=" not in item:
            raise ValueError(f"Invalid override format: {item}. Expected key=value.")
        key, raw_value = item.split("=", 1)
        value = "" if raw_value == "" else yaml.safe_load(raw_value)
        overrides[key] = value
    return overrides


def main():
    args = parse_args()

    overrides = parse_overrides(args.override) if args.override else None
    config = FewStepRLConfig.from_yaml(args.config, overrides=overrides)

    trainer = TRAINER_REGISTRY[args.trainer](config)
    if args.eval_only:
        trainer.evaluate()
    else:
        trainer.train()


if __name__ == "__main__":
    main()

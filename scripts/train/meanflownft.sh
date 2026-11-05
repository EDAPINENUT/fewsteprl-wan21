#!/usr/bin/env bash
# Launch the `meanflownft` experiment.
#
# MeanFlowNFT: x_anchor_2 is self-induced from old_model's average
# velocity plus the central-difference S-term; decoupled NFT loss.
#
# Script name, config filename and checkpoint directory all use the SAME
# experiment name `meanflownft`.
#
#   config     : scripts/train/configs/meanflownft.yaml
#   trainer    : meanflow_nft
#   released   : checkpoints/meanflownft/checkpoint-250/generator_ema.pt
#   default    : 4 nodes x 8 GPUs = 32 GPUs (the released run)
#
# Usage (run on every node; NODE_RANK=0..NNODES-1, MASTER_ADDR=<rank-0 host>):
#   NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash scripts/train/meanflownft.sh [GPUS_PER_NODE] [OVERRIDES...]
#   NNODES=1 bash scripts/train/meanflownft.sh 8 fewstep_rl.num_epochs=2      # 1-node smoke
#
# TRAINER_KEY is a CLI argument and is NOT written into config.yaml -- reading
# the config alone cannot tell you which algorithm ran. That is why it is
# pinned here rather than left to a default.

set -euo pipefail

ARM='meanflownft'
TRAINER='meanflow_nft'
PORT=29711

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"

NGPU="${1:-8}"; shift || true

export TRAINER_KEY="${TRAINER}"
export CONFIG="scripts/train/configs/${ARM}.yaml"
export NNODES="${NNODES:-4}"
export MASTER_PORT="${MASTER_PORT:-${PORT}}"

echo "[${ARM}] trainer=${TRAINER} config=${CONFIG} nodes=${NNODES} gpus/node=${NGPU} port=${MASTER_PORT}"
exec bash scripts/train/_launcher.sh "${NGPU}" "$@"

#!/usr/bin/env bash
# Launch the `meanflowawm` experiment.
#
# MeanFlowAWM: advantage-weighted x0 matching with a PPO trust region.
#
# Script name, config filename and checkpoint directory all use the SAME
# experiment name `meanflowawm`.
#
#   config     : scripts/train/configs/meanflowawm.yaml
#   trainer    : meanflowawm_rl
#   released   : checkpoints/meanflowawm/checkpoint-250/generator_ema.pt
#   default    : 4 nodes x 8 GPUs = 32 GPUs (the released run)
#
# Usage (run on every node; NODE_RANK=0..NNODES-1, MASTER_ADDR=<rank-0 host>):
#   NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash scripts/train/meanflowawm.sh [GPUS_PER_NODE] [OVERRIDES...]
#   NNODES=1 bash scripts/train/meanflowawm.sh 8 fewstep_rl.num_epochs=2      # 1-node smoke
#
# TRAINER_KEY is a CLI argument and is NOT written into config.yaml -- reading
# the config alone cannot tell you which algorithm ran. That is why it is
# pinned here rather than left to a default.

set -euo pipefail

TRIAL='meanflowawm'
TRAINER='meanflowawm_rl'
PORT=29716

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"

NGPU="${1:-8}"; shift || true

export TRAINER_KEY="${TRAINER}"
export CONFIG="scripts/train/configs/${TRIAL}.yaml"
export NNODES="${NNODES:-4}"
export MASTER_PORT="${MASTER_PORT:-${PORT}}"

echo "[${TRIAL}] trainer=${TRAINER} config=${CONFIG} nodes=${NNODES} gpus/node=${NGPU} port=${MASTER_PORT}"
exec bash scripts/train/_launcher.sh "${NGPU}" "$@"

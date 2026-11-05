#!/usr/bin/env bash
# Launch the `meanflowopsd` experiment.
#
# MeanFlow-OPSD (v-form): lognormal query sigma, r drawn given t, and
# fewstep_rl.disable_cd=true; reward direction from the differentiable hpsv3_general.
#
# Script name, config filename and checkpoint directory all use the SAME
# experiment name `meanflowopsd`.
#
#   config     : scripts/train/configs/meanflowopsd.yaml
#   trainer    : meanflowopsd_rl
#   released   : checkpoints/meanflowopsd/checkpoint-250/generator_ema.pt
#   default    : 5 nodes x 8 GPUs = 40 GPUs (the released run)
#
# Usage (run on every node; NODE_RANK=0..NNODES-1, MASTER_ADDR=<rank-0 host>):
#   NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash scripts/train/meanflowopsd.sh [GPUS_PER_NODE] [OVERRIDES...]
#   NNODES=1 bash scripts/train/meanflowopsd.sh 8 fewstep_rl.num_epochs=2      # 1-node smoke
#
# TRAINER_KEY is a CLI argument and is NOT written into config.yaml -- reading
# the config alone cannot tell you which algorithm ran. That is why it is
# pinned here rather than left to a default.

set -euo pipefail

ARM='meanflowopsd'
TRAINER='meanflowopsd_rl'
PORT=29717

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"

NGPU="${1:-8}"; shift || true

export TRAINER_KEY="${TRAINER}"
export CONFIG="scripts/train/configs/${ARM}.yaml"
export NNODES="${NNODES:-5}"
export MASTER_PORT="${MASTER_PORT:-${PORT}}"

echo "[${ARM}] trainer=${TRAINER} config=${CONFIG} nodes=${NNODES} gpus/node=${NGPU} port=${MASTER_PORT}"
exec bash scripts/train/_launcher.sh "${NGPU}" "$@"

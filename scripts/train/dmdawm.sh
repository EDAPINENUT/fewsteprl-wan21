#!/usr/bin/env bash
# Launch the `dmdawm` experiment.
#
# DMD-AWM: AWM ratio against old_model with no central difference
# (x_anchor_2 = x_t - t * u_old) and the aux v-predictor as the reward baseline.
#
# Script name, config filename and checkpoint directory all use the SAME
# experiment name `dmdawm`.
#
#   config     : scripts/train/configs/dmdawm.yaml
#   trainer    : dmdawm_rl
#   released   : checkpoints/dmdawm/checkpoint-250/generator_ema.pt
#   default    : 6 nodes x 8 GPUs = 48 GPUs (the released run)
#
# Usage (run on every node; NODE_RANK=0..NNODES-1, MASTER_ADDR=<rank-0 host>):
#   NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash scripts/train/dmdawm.sh [GPUS_PER_NODE] [OVERRIDES...]
#   NNODES=1 bash scripts/train/dmdawm.sh 8 fewstep_rl.num_epochs=2      # 1-node smoke
#
# TRAINER_KEY is a CLI argument and is NOT written into config.yaml -- reading
# the config alone cannot tell you which algorithm ran. That is why it is
# pinned here rather than left to a default.

set -euo pipefail

TRIAL='dmdawm'
TRAINER='dmdawm_rl'
PORT=29714

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"

NGPU="${1:-8}"; shift || true

export TRAINER_KEY="${TRAINER}"
export CONFIG="scripts/train/configs/${TRIAL}.yaml"
export NNODES="${NNODES:-6}"
export MASTER_PORT="${MASTER_PORT:-${PORT}}"

echo "[${TRIAL}] trainer=${TRAINER} config=${CONFIG} nodes=${NNODES} gpus/node=${NGPU} port=${MASTER_PORT}"
exec bash scripts/train/_launcher.sh "${NGPU}" "$@"

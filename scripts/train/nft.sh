#!/usr/bin/env bash
# Launch the `nft` experiment.
#
# S-term ablation of meanflownft: the flow map's average velocity is used as the
# instantaneous one and the central difference is never computed.
#
# Script name and config filename use the SAME experiment name `nft`.
#
#   config     : scripts/train/configs/nft.yaml
#   trainer    : nft_rl
#   released   : none (ablation)
#   default    : 4 nodes x 8 GPUs = 32 GPUs (same as meanflownft)
#
# Usage (run on every node; NODE_RANK=0..NNODES-1, MASTER_ADDR=<rank-0 host>):
#   NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash scripts/train/nft.sh [GPUS_PER_NODE] [OVERRIDES...]
#   NNODES=1 bash scripts/train/nft.sh 8 fewstep_rl.num_epochs=2      # 1-node smoke
#
# TRAINER_KEY is a CLI argument and is NOT written into config.yaml -- reading
# the config alone cannot tell you which algorithm ran. That is why it is
# pinned here rather than left to a default.

set -euo pipefail

ARM='nft'
TRAINER='nft_rl'
PORT=29712

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"

NGPU="${1:-8}"; shift || true

export TRAINER_KEY="${TRAINER}"
export CONFIG="scripts/train/configs/${ARM}.yaml"
export NNODES="${NNODES:-4}"
export MASTER_PORT="${MASTER_PORT:-${PORT}}"

echo "[${ARM}] trainer=${TRAINER} config=${CONFIG} nodes=${NNODES} gpus/node=${NGPU} port=${MASTER_PORT}"
exec bash scripts/train/_launcher.sh "${NGPU}" "$@"

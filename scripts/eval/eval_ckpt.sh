#!/usr/bin/env bash
# Sample + score one checkpoint.
#
# Usage:
#   bash scripts/eval/eval_ckpt.sh <TARGET> [NUM_GPUS]
#
#   TARGET   experiment name under checkpoints/, `base`, or a generator_ema.pt path
#
# Steps:
#   1. 256 wan_dancegrpo test prompts -> hpsv3_general, hpsv3_percentile,
#      videoalign_mq, videoalign_ta (the training rewards)
#   2. with VBENCH=1: 946 x 5 VBench videos -> 16 dimensions + aggregate
#
# Environment:
#   VBENCH=1         also run VBench (generation here, scoring with $VBENCH_PYTHON)
#   VBENCH_PYTHON    interpreter of the separate VBench scoring env (required for VBENCH=1)
#   MAX_PROMPTS, TAG, FORCE, OUT_ROOT   passed through to sample.sh
#
# Reward scoring runs on the first visible GPU; VBench scoring uses all
# NUM_GPUS GPUs (one dimension per GPU).

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"

TARGET="${1:?usage: eval_ckpt.sh <TARGET> [NUM_GPUS]}"
NGPU="${2:-1}"
OUT_ROOT="${OUT_ROOT:-outputs/eval}"
export OUT_ROOT

# Resolve the tag and mode sample.sh will use, so the scorers read the same directory.
if [[ "${TARGET}" == "base" ]]; then
    MODE=anyflow; TAG="${TAG:-base}"
elif [[ -d "checkpoints/${TARGET}" ]]; then
    MODE=meanflow_nft; TAG="${TAG:-${TARGET}}"
else
    CKPT="$(cd "$(dirname "${TARGET}")" && pwd)/$(basename "${TARGET}")"
    MODE=meanflow_nft
    TAG="${TAG:-$(basename "$(dirname "$(dirname "${CKPT}")")")_$(basename "$(dirname "${CKPT}")")}"
fi
export TAG

bash scripts/eval/sample.sh "${TARGET}" wan_dancegrpo "${NGPU}"
python3 scripts/eval/score.py --sample_dir "${OUT_ROOT}/${TAG}/wan_dancegrpo/${MODE}"

if [[ "${VBENCH:-0}" == "1" ]]; then
    [[ -n "${VBENCH_PYTHON:-}" ]] || { echo "ERROR: VBENCH=1 needs VBENCH_PYTHON (see scripts/eval/README.md)" >&2; exit 1; }
    bash scripts/eval/sample.sh "${TARGET}" vbench "${NGPU}"
    "${VBENCH_PYTHON}" scripts/eval/vbench_score.py \
        --sample_dir "${OUT_ROOT}/${TAG}/vbench/${MODE}" --gpus "$(seq -s ' ' 0 $((NGPU - 1)))"
fi

echo "[eval] ${TAG} done; summaries under ${OUT_ROOT}/${TAG}/"

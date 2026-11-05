#!/usr/bin/env bash
# Evaluate the pre-RL baseline and every released checkpoint, then aggregate.
#
# Usage:
#   bash scripts/eval/eval_all.sh [NUM_GPUS] [TARGET ...]
#
# With no TARGETs: base + every checkpoints/<arm>. Honours the same environment
# as eval_ckpt.sh (VBENCH, VBENCH_PYTHON, MAX_PROMPTS, FORCE, OUT_ROOT).
# Finished sample / score sets are skipped, so the script can be re-run after
# an interruption.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"

NGPU="${1:-1}"
shift || true

if [[ $# -gt 0 ]]; then
    TARGETS=("$@")
else
    TARGETS=(base)
    for d in checkpoints/*/; do
        TARGETS+=("$(basename "${d}")")
    done
fi

for t in "${TARGETS[@]}"; do
    echo "================ ${t} ================"
    bash scripts/eval/eval_ckpt.sh "${t}" "${NGPU}"
done

python3 scripts/eval/collect.py --out_root "${OUT_ROOT:-outputs/eval}"

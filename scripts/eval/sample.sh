#!/usr/bin/env bash
# Generate eval videos for one checkpoint on one prompt set.
#
# Usage:
#   bash scripts/eval/sample.sh <TARGET> <PROMPT_SET> [NUM_GPUS] [EXTRA_OVERRIDES...]
#
#   TARGET      an experiment name under checkpoints/ (e.g. meanflownft),
#               `base` for the pre-RL AnyFlow model, or a path to any
#               Stage-3 generator_ema.pt
#   PROMPT_SET  wan_dancegrpo   dataset/wan_dancegrpo/test.txt (256 prompts)
#               vbench          946 VBench prompts x 5 samples (4730 videos)
#   NUM_GPUS    default 1; >1 shards prompts over GPUs with torchrun
#
# Output: outputs/eval/<TAG>/<PROMPT_SET>/<mode>/{00000.mp4, ..., metadata.json}
# (<mode> is meanflow_nft, or anyflow for `base`). For vbench the videos are
# also linked as vbench_samples/{prompt_en}-{k}.mp4.
#
# Environment:
#   MAX_PROMPTS                 0 (default) = the whole prompt set
#   TAG                         override the output tag
#   FORCE=1                     regenerate even if metadata.json is complete
#   FEWSTEPRL_ANYFLOW_WAN_PATH  AnyFlow pipeline (default models/AnyFlow-Wan2.1-T2V-1.3B-Diffusers)
#   OUT_ROOT                    default outputs/eval

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"

TARGET="${1:?usage: sample.sh <TARGET> <PROMPT_SET> [NUM_GPUS] [EXTRA_OVERRIDES...]}"
PROMPT_SET="${2:?usage: sample.sh <TARGET> <PROMPT_SET> [NUM_GPUS] [EXTRA_OVERRIDES...]}"
NGPU="${3:-1}"
EXTRA=("${@:4}")

CONFIG="scripts/eval/configs/sample_wan.yaml"
OUT_ROOT="${OUT_ROOT:-outputs/eval}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
ANYFLOW_PATH="${FEWSTEPRL_ANYFLOW_WAN_PATH:-models/AnyFlow-Wan2.1-T2V-1.3B-Diffusers}"

fail() { echo "ERROR: $*" >&2; exit 1; }

# --- Resolve TARGET -> (mode, checkpoint, tag) --------------------------------
MODE_ARGS=()
if [[ "${TARGET}" == "base" ]]; then
    MODE=anyflow; TAG="${TAG:-base}"
    MODE_ARGS=("mode=anyflow" "meanflow_nft_path=")
elif [[ -f "${TARGET}" ]]; then
    CKPT="$(cd "$(dirname "${TARGET}")" && pwd)/$(basename "${TARGET}")"
    MODE=meanflow_nft
    TAG="${TAG:-$(basename "$(dirname "$(dirname "${CKPT}")")")_$(basename "$(dirname "${CKPT}")")}"
    MODE_ARGS=("mode=meanflow_nft" "meanflow_nft_path=${CKPT}")
elif [[ -d "checkpoints/${TARGET}" ]]; then
    mapfile -t found < <(ls checkpoints/"${TARGET}"/checkpoint-*/generator_ema.pt 2>/dev/null)
    [[ ${#found[@]} -eq 1 ]] || fail "expected exactly one checkpoints/${TARGET}/checkpoint-*/generator_ema.pt, found ${#found[@]}"
    MODE=meanflow_nft; TAG="${TAG:-${TARGET}}"
    MODE_ARGS=("mode=meanflow_nft" "meanflow_nft_path=${REPO}/${found[0]}")
else
    fail "TARGET '${TARGET}' is not 'base', a file, or a directory under checkpoints/"
fi

SET_DIR="${OUT_ROOT}/${TAG}/${PROMPT_SET}"
SAMPLE_DIR="${SET_DIR}/${MODE}"

# --- Resolve PROMPT_SET --------------------------------------------------------
case "${PROMPT_SET}" in
    wan_dancegrpo)
        PROMPT_FILE="dataset/wan_dancegrpo/test.txt"; SEED=12345
        N_EXPECTED="$(grep -cve '^\s*$' "${PROMPT_FILE}")" ;;
    vbench)
        # Seed 0 makes each video's seed its global sample index, as in the
        # training-time VBench evaluator.
        PROMPT_FILE="${SET_DIR}/vbench_prompts.json"; SEED=0
        python3 scripts/eval/vbench_prompts.py expand --out "${PROMPT_FILE}" >/dev/null
        N_EXPECTED="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "${PROMPT_FILE}")" ;;
    *) fail "unknown prompt set '${PROMPT_SET}' (wan_dancegrpo | vbench)" ;;
esac
if [[ "${MAX_PROMPTS}" -gt 0 && "${MAX_PROMPTS}" -lt "${N_EXPECTED}" ]]; then
    N_EXPECTED="${MAX_PROMPTS}"
fi

# --- Skip if already complete -------------------------------------------------
if [[ -z "${FORCE:-}" && -f "${SAMPLE_DIR}/metadata.json" ]]; then
    N_DONE="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["videos"]))' "${SAMPLE_DIR}/metadata.json")"
    if [[ "${N_DONE}" -ge "${N_EXPECTED}" ]]; then
        echo "[sample] skip ${TAG}/${PROMPT_SET}: ${N_DONE} videos already in ${SAMPLE_DIR}"
        exit 0
    fi
fi

# --- Preflight ----------------------------------------------------------------
for sub in transformer vae text_encoder tokenizer; do
    [[ -d "${ANYFLOW_PATH}/${sub}" ]] || fail "AnyFlow-Wan pipeline incomplete at ${ANYFLOW_PATH} (bash download_scripts/download_assets.sh)"
done

export HF_HOME="${HF_HOME:-${REPO}/models/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "${SET_DIR}"
OVERRIDES=(
    "anyflow_pretrained_path=${ANYFLOW_PATH}"
    "${MODE_ARGS[@]}"
    "prompt_file=${PROMPT_FILE}"
    "max_prompts=${MAX_PROMPTS}"
    "seed=${SEED}"
    "output_dir=${SET_DIR}"
    "${EXTRA[@]}"
)

echo "[sample] ${TAG} / ${PROMPT_SET}: ${N_EXPECTED} videos on ${NGPU} GPU(s) -> ${SAMPLE_DIR}"
if [[ "${NGPU}" -gt 1 ]]; then
    "${TORCHRUN_BIN:-torchrun}" --standalone --nproc_per_node="${NGPU}" \
        inference.py "${CONFIG}" --override "${OVERRIDES[@]}"
else
    python3 inference.py "${CONFIG}" --override "${OVERRIDES[@]}"
fi

if [[ "${PROMPT_SET}" == "vbench" && "${MAX_PROMPTS}" -eq 0 ]]; then
    python3 scripts/eval/vbench_prompts.py link --sample_dir "${SAMPLE_DIR}"
fi

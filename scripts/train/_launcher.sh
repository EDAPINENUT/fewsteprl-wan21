#!/bin/bash
# Shared launcher for every Wan2.1 arm. Not meant to be called directly: the
# scripts/train/<arm>.sh wrappers export CONFIG / TRAINER_KEY / NNODES and exec
# this file. Run the same command on every node with NODE_RANK=0..NNODES-1 and
# MASTER_ADDR set to the rank-0 host.
#
# Prerequisites (see README.md, "Assets"):
#   bash download_scripts/download_assets.sh
#
# Environment overrides:
#   FEWSTEPRL_ANYFLOW_WAN_PATH   AnyFlow-Wan2.1-T2V-1.3B diffusers pipeline dir
#   FEWSTEPRL_VIDEOREWARD_PATH   VideoReward checkpoint dir (videoalign_mq / _ta)
#   FEWSTEPRL_VIDEOALIGN_PATH    VideoAlign source dir (default: third_party/VideoAlign)
#   HF_HOME                      HF cache holding HPSv3 and the Qwen2-VL bases
#   EVAL_ONLY=1                  pass --eval-only (needs train.resume_from=<full ckpt dir>)
#   NNODES NODE_RANK MASTER_ADDR MASTER_PORT TORCHRUN_BIN

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

NUM_GPUS="${1:-8}"
EXTRA_ARGS=("${@:2}")
set --

fail() { echo "ERROR: $*" >&2; exit 1; }

[[ -n "${CONFIG:-}" ]] || fail "CONFIG is not set; launch through scripts/train/<arm>.sh"
[[ -n "${TRAINER_KEY:-}" ]] || fail "TRAINER_KEY is not set; launch through scripts/train/<arm>.sh"

NNODES="${NNODES:-1}"
[[ "${NNODES}" =~ ^[1-9][0-9]*$ ]] || fail "NNODES must be a positive integer; got ${NNODES}."
if [[ "${NNODES}" -eq 1 ]]; then
    NODE_RANK="${NODE_RANK:-0}"
    MASTER_ADDR="${MASTER_ADDR:-localhost}"
fi
[[ -n "${NODE_RANK:-}" && -n "${MASTER_ADDR:-}" ]] \
    || fail "set NODE_RANK (0..$((NNODES - 1))) and MASTER_ADDR on every node."
(( NODE_RANK >= 0 && NODE_RANK < NNODES )) || fail "NODE_RANK must be in [0, $((NNODES - 1))]; got ${NODE_RANK}."
MASTER_PORT="${MASTER_PORT:-29500}"

# --- Local asset defaults (populated by download_scripts/download_assets.sh) --
ANYFLOW_PATH="${FEWSTEPRL_ANYFLOW_WAN_PATH:-${REPO_ROOT}/models/AnyFlow-Wan2.1-T2V-1.3B-Diffusers}"
VIDEOREWARD_PATH="${FEWSTEPRL_VIDEOREWARD_PATH:-${REPO_ROOT}/models/VideoReward}"
VIDEOALIGN_PATH="${FEWSTEPRL_VIDEOALIGN_PATH:-${REPO_ROOT}/third_party/VideoAlign}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/models/hf_cache}"

# --- Preflight: fail loudly BEFORE torchrun spawns the ranks -----------------
for sub in transformer vae text_encoder tokenizer; do
    [[ -d "${ANYFLOW_PATH}/${sub}" ]] \
        || fail "AnyFlow-Wan pipeline incomplete (no ${sub}/) at ${ANYFLOW_PATH}.
  Run: bash download_scripts/download_assets.sh"
done
[[ -f "${VIDEOREWARD_PATH}/model_config.json" ]] && ls "${VIDEOREWARD_PATH}"/checkpoint-*/model.pth >/dev/null 2>&1 \
    || fail "VideoReward checkpoint not found at ${VIDEOREWARD_PATH} (videoalign_mq / videoalign_ta).
  Run: bash download_scripts/download_assets.sh"
[[ -f "${VIDEOALIGN_PATH}/inference.py" ]] || fail "VideoAlign source not found at ${VIDEOALIGN_PATH}"
for repo in MizzenAI--HPSv3 Qwen--Qwen2-VL-7B-Instruct Qwen--Qwen2-VL-2B-Instruct; do
    [[ -d "${HF_HOME}/hub/models--${repo}" ]] \
        || fail "${repo/--//} not in ${HF_HOME}/hub (HPSv3 / VideoAlign reward models).
  Run: bash download_scripts/download_assets.sh"
done

# Config sanity + the K-repeat constraint: batch_size * world_size % K == 0.
python3 - "${CONFIG}" "${NUM_GPUS}" "${NNODES}" <<'PYEOF' || exit 1
import sys, yaml
cfg_path, num_gpus, nnodes = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cfg = yaml.safe_load(open(cfg_path))
if "fewstep_rl" not in cfg:
    print(f"ERROR: {cfg_path} has no 'fewstep_rl' section.", file=sys.stderr)
    sys.exit(1)
fs = cfg["fewstep_rl"]
bs, K = int(cfg["train"]["batch_size"]), int(fs["num_image_per_prompt"])
total = bs * num_gpus * nnodes
if total % K:
    print(f"ERROR: train.batch_size * total_gpus = {bs} * {num_gpus * nnodes} = {total} "
          f"is not divisible by num_image_per_prompt = {K}.", file=sys.stderr)
    sys.exit(1)
print(f"  Preflight OK: {total} samples/round / K={K} = {total // K} prompts/round")
print("  Train reward: " + " + ".join(f"{k} {v}" for k, v in fs.get("reward_fn", {}).items()))
PYEOF

# Validate the key against the registry here, before torchrun.
python3 - "${TRAINER_KEY}" <<'PYEOF' || exit 1
import sys
sys.path.insert(0, ".")
from main import TRAINER_REGISTRY
key = sys.argv[1]
if key not in TRAINER_REGISTRY:
    print(f"ERROR: TRAINER_KEY={key!r} is not in main.py's TRAINER_REGISTRY.\n"
          f"       Valid keys: {sorted(TRAINER_REGISTRY)}", file=sys.stderr)
    sys.exit(1)
print(f"  Preflight OK: trainer {key} -> {TRAINER_REGISTRY[key].__module__}")
PYEOF

# --- Distributed / NCCL env ---------------------------------------------------
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export FEWSTEPRL_DIST_TIMEOUT_HOURS="${FEWSTEPRL_DIST_TIMEOUT_HOURS:-6}"
# Without network / a wandb api_key, an unset WANDB_MODE makes rank 0 raise
# inside wandb.init while the other ranks block in NCCL: the job hangs instead
# of failing. Export WANDB_MODE=online if you are logged in.
export WANDB_MODE="${WANDB_MODE:-offline}"

MODE_ARGS=()
[[ "${EVAL_ONLY:-0}" == "1" ]] && MODE_ARGS=(--eval-only)

echo "=============================================="
echo "FewStepRL Wan2.1 -- ${TRAINER_KEY}"
echo "  Nodes:       ${NNODES} (this node: ${NODE_RANK})"
echo "  GPUs/node:   ${NUM_GPUS}    Total: $((NUM_GPUS * NNODES))"
echo "  Master:      ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Config:      ${CONFIG}"
echo "  AnyFlow:     ${ANYFLOW_PATH}"
echo "  Mode:        ${MODE_ARGS[*]:-train}"
echo "  Time:        $(date)"
echo "=============================================="

"${TORCHRUN_BIN:-torchrun}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${NUM_GPUS}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    main.py \
    "${CONFIG}" \
    --trainer "${TRAINER_KEY}" \
    "${MODE_ARGS[@]}" \
    --override \
    model.pretrained_path="${ANYFLOW_PATH}" \
    fewstep_rl.reward_model_paths.videoalign_dir="${VIDEOALIGN_PATH}" \
    fewstep_rl.reward_model_paths.videoalign_ckpt="${VIDEOREWARD_PATH}" \
    "${EXTRA_ARGS[@]}"

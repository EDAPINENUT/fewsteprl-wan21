#!/bin/bash
# Download every external asset needed to train (scripts/train) and evaluate
# (scripts/eval). Needs internet; run once, then everything runs offline.
#
#   1. AnyFlow-Wan2.1-T2V-1.3B-Diffusers full pipeline  (~26 GB)  the policy base
#   2. VideoReward + Qwen2-VL-2B base         (~5 + 4.4 GB)      videoalign_mq / _ta
#   3. HPSv3 head + Qwen2-VL-7B base          (~17 + 17 GB)      hpsv3_general / _percentile
#   4. bert-base-uncased                      (~0.4 GB)          VBench scoring (GRiT, tag2text)
#   5. The 11 VBench metric checkpoints       (~13 GB)           eval only, WITH_VBENCH=1 (default)
#
# The VideoAlign source is vendored in third_party/VideoAlign; nothing to fetch.
#
# Usage:
#   bash download_scripts/download_assets.sh
#   WITH_VBENCH=0 bash download_scripts/download_assets.sh      # training only (~69 GB)
#
# Override locations with FEWSTEPRL_ANYFLOW_WAN_PATH / FEWSTEPRL_VIDEOREWARD_PATH /
# VBENCH_CACHE_DIR / HF_HOME (the train and eval scripts read the same
# variables). Behind a proxy, export http_proxy / https_proxy as usual.
# Resumable: `hf download` and `wget -c` skip complete files.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ANYFLOW_DIR="${FEWSTEPRL_ANYFLOW_WAN_PATH:-${REPO_ROOT}/models/AnyFlow-Wan2.1-T2V-1.3B-Diffusers}"
VIDEOREWARD_DIR="${FEWSTEPRL_VIDEOREWARD_PATH:-${REPO_ROOT}/models/VideoReward}"
VBENCH_DIR="${VBENCH_CACHE_DIR:-${REPO_ROOT}/models/vbench}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/models/hf_cache}"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
WITH_VBENCH="${WITH_VBENCH:-1}"

mkdir -p "${ANYFLOW_DIR}" "${VIDEOREWARD_DIR}" "${HF_HOME}" "${VBENCH_DIR}"

if command -v hf >/dev/null 2>&1; then
    HF=(hf)
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF=(huggingface-cli)
else
    echo "ERROR: install the HF CLI first:  pip install -U 'huggingface_hub[cli]'" >&2
    exit 1
fi

echo "=============================================="
echo "FewStepRL-Wan2.1 asset download"
echo "  AnyFlow-Wan:  ${ANYFLOW_DIR}"
echo "  VideoReward:  ${VIDEOREWARD_DIR}"
echo "  VBench cache: ${VBENCH_DIR}  (WITH_VBENCH=${WITH_VBENCH})"
echo "  HF_HOME:      ${HF_HOME}"
echo "=============================================="

if ! curl -fsS --max-time 30 -o /dev/null https://huggingface.co; then
    echo "ERROR: cannot reach huggingface.co (set http_proxy / https_proxy if you need a proxy)." >&2
    exit 1
fi

# Fetch a plain URL to an exact path (VBench looks for exact filenames).
fetch() {
    local url="$1" dest="$2" label="${3:-$(basename "$2")}"
    if [[ -s "${dest}" ]]; then
        printf '  skip (present)  %s\n' "${label}"; return 0
    fi
    printf '  -> %s\n' "${label}"
    mkdir -p "$(dirname "${dest}")"
    if wget -q -c -O "${dest}.part" "${url}"; then
        mv -f "${dest}.part" "${dest}"
    else
        rm -f "${dest}.part"; echo "     FAILED: ${url}"; return 1
    fi
}

# --- 1. AnyFlow-Wan policy (ungated; non-commercial license) -------------------
echo "[1/5] nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers"
"${HF[@]}" download nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers --local-dir "${ANYFLOW_DIR}" \
    || echo "  WARNING: AnyFlow download reported an error."

# --- 2. VideoReward (videoalign_mq / videoalign_ta) ------------------------------
# KwaiVGI/VideoReward was renamed to KlingTeam/VideoReward. Its base model is
# read from model_config.json at runtime and must be in HF_HOME.
echo "[2/5] KlingTeam/VideoReward + Qwen/Qwen2-VL-2B-Instruct"
"${HF[@]}" download KlingTeam/VideoReward --local-dir "${VIDEOREWARD_DIR}" \
    || echo "  WARNING: VideoReward download reported an error."
"${HF[@]}" download Qwen/Qwen2-VL-2B-Instruct >/dev/null \
    || echo "  WARNING: Qwen2-VL-2B download reported an error."

# --- 3. HPSv3 (hpsv3_general / hpsv3_percentile) ---------------------------------
# HPSv3RewardInferencer resolves MizzenAI/HPSv3 through hf_hub_download and its
# packaged HPSv3_7B.yaml names Qwen/Qwen2-VL-7B-Instruct as the base.
echo "[3/5] MizzenAI/HPSv3 + Qwen/Qwen2-VL-7B-Instruct"
"${HF[@]}" download MizzenAI/HPSv3 HPSv3.safetensors config.json >/dev/null \
    || echo "  WARNING: HPSv3 download reported an error."
"${HF[@]}" download Qwen/Qwen2-VL-7B-Instruct >/dev/null \
    || echo "  WARNING: Qwen2-VL-7B download reported an error."

# --- 4. bert-base-uncased (VBench GRiT + tag2text; the scene dimension loads the weights) --
echo "[4/5] bert-base-uncased"
"${HF[@]}" download bert-base-uncased \
    config.json tokenizer.json tokenizer_config.json vocab.txt model.safetensors >/dev/null \
    || echo "  WARNING: bert-base-uncased download reported an error."

# --- 5. VBench metric checkpoints ------------------------------------------------
# VBench is called with local=True, so every weight must already sit at the
# exact path below. GRiT and RAFT come from mirrors because their official
# sources are no longer reachable.
if [[ "${WITH_VBENCH}" == "1" ]]; then
    echo "[5/5] VBench metric checkpoints"
    H="https://huggingface.co"
    fetch "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt" \
          "${VBENCH_DIR}/clip_model/ViT-B-32.pt"
    fetch "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt" \
          "${VBENCH_DIR}/clip_model/ViT-L-14.pt"
    fetch "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_l_14_linear.pth" \
          "${VBENCH_DIR}/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth"
    fetch "${H}/OpenGVLab/VBench_Used_Models/resolve/main/l16_ptk710_ftk710_ftk400_f16_res224.pth" \
          "${VBENCH_DIR}/umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth"
    fetch "${H}/lalala125/AMT/resolve/main/amt-s.pth" "${VBENCH_DIR}/amt_model/amt-s.pth"
    fetch "${H}/ddrfan/RAFT/resolve/main/raft-things.pth" "${VBENCH_DIR}/raft_model/models/raft-things.pth"
    fetch "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth" \
          "${VBENCH_DIR}/dino_model/dino_vitbase16_pretrain.pth"
    if [[ ! -d "${VBENCH_DIR}/dino_model/facebookresearch_dino_main/.git" ]]; then
        git clone -q https://github.com/facebookresearch/dino "${VBENCH_DIR}/dino_model/facebookresearch_dino_main" \
            || echo "     FAILED: dino clone"
    fi
    fetch "https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth" \
          "${VBENCH_DIR}/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth"
    fetch "${H}/OpenGVLab/VBench_Used_Models/resolve/main/grit_b_densecap_objectdet.pth" \
          "${VBENCH_DIR}/grit_model/grit_b_densecap_objectdet.pth"
    fetch "${H}/spaces/xinyu1205/recognize-anything/resolve/main/tag2text_swin_14m.pth" \
          "${VBENCH_DIR}/caption_model/tag2text_swin_14m.pth"
    fetch "${H}/OpenGVLab/VBench_Used_Models/resolve/main/ViClip-InternVid-10M-FLT.pth" \
          "${VBENCH_DIR}/ViCLIP/ViClip-InternVid-10M-FLT.pth"
else
    echo "[5/5] VBench: SKIPPED (WITH_VBENCH=1 to fetch)"
fi

# --- Verify ----------------------------------------------------------------------
echo "---------- verification ----------"
status=0
check() {  # check <path> <label>
    if [[ -e "$1" ]]; then printf '  OK    %s\n' "$2"; else printf '  MISS  %s (%s)\n' "$2" "$1"; status=1; fi
}
for sub in transformer vae text_encoder tokenizer; do
    check "${ANYFLOW_DIR}/${sub}" "AnyFlow-Wan ${sub}/"
done
check "${VIDEOREWARD_DIR}/model_config.json" "VideoReward model_config.json"
ls "${VIDEOREWARD_DIR}"/checkpoint-*/model.pth >/dev/null 2>&1 \
    && echo "  OK    VideoReward checkpoint-*/model.pth" \
    || { echo "  MISS  VideoReward checkpoint-*/model.pth"; status=1; }
check "${REPO_ROOT}/third_party/VideoAlign/inference.py" "VideoAlign source (vendored)"
for repo in MizzenAI--HPSv3 Qwen--Qwen2-VL-7B-Instruct Qwen--Qwen2-VL-2B-Instruct bert-base-uncased; do
    check "${HF_HOME}/hub/models--${repo}" "HF cache ${repo/--//}"
done
if [[ "${WITH_VBENCH}" == "1" ]]; then
    for rel in clip_model/ViT-B-32.pt clip_model/ViT-L-14.pt \
               aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth \
               umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth amt_model/amt-s.pth \
               raft_model/models/raft-things.pth dino_model/dino_vitbase16_pretrain.pth \
               dino_model/facebookresearch_dino_main pyiqa_model/musiq_spaq_ckpt-358bb6af.pth \
               grit_model/grit_b_densecap_objectdet.pth caption_model/tag2text_swin_14m.pth \
               ViCLIP/ViClip-InternVid-10M-FLT.pth; do
        check "${VBENCH_DIR}/${rel}" "vbench/${rel}"
    done
fi

if [[ "${status}" -eq 0 ]]; then
    echo "All requested assets are present."
    echo "  Train: NODE_RANK=<i> MASTER_ADDR=<rank-0 host> bash scripts/train/meanflownft.sh 8"
    echo "  Eval:  bash scripts/eval/eval_ckpt.sh meanflownft 8"
fi
exit "${status}"

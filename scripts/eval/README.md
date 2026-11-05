# Evaluation: sampling, reward scoring, VBench

Generate videos from a released (or your own) Stage-3 checkpoint and score them
with the four training rewards and, optionally, VBench. Run every command from
the repository root.

```
scripts/eval/
  configs/sample_wan.yaml   sampling protocol (4-step flow map, 81 frames, 480x832)
  sample.sh        <target> <prompt_set> [gpus]   -> outputs/eval/<tag>/<set>/<mode>/*.mp4 + metadata.json
  score.py         --sample_dir ...               -> score_<metric>.json + summary.json
  vbench_prompts.py  expand / link                VBench prompt list and VBench filenames
  vbench_score.py  --sample_dir ... --gpus ...    -> vbench_info/{dim}_eval_results.json + vbench_summary.json
  setup_vbench_env.sh                             fixes for the separate VBench scoring env
  eval_ckpt.sh     <target> [gpus]                sample + score one checkpoint
  eval_all.sh      [gpus] [targets...]            base + every checkpoints/<trial>, then collect
  collect.py       [--out_root outputs/eval]      -> outputs/eval/summary.{csv,md}
```

## 1. Environments

Training / sampling / reward scoring (Python 3.10, CUDA 12.4, PyTorch 2.6):

```bash
pip install -r requirements.txt
pip install hpsv3==1.0.0 --no-deps
pip install trl==0.12.2 --no-deps
pip install vbench==0.1.5 --no-deps
```

`hpsv3`, `trl` (used by VideoAlign) and `vbench` are installed without their
dependency metadata on purpose: it pins older Transformers releases that break
the Wan model path. `fewsteprl/rewards/video_multi_scorer.py` carries the
compatibility shims for Transformers 4.57 (Qwen2-VL key layout, `embed_tokens`
alias, `image_utils.VideoInput` alias).

VBench scoring needs its own environment, a copy of the training env with a
few fixes (detectron2 built from source, two import patches, a `torch.load`
default shim). Create it once:

```bash
conda create -n vbench-scoring --clone <training-env>
conda activate vbench-scoring
bash scripts/eval/setup_vbench_env.sh      # ends with "16/16 dimensions importable"
```

and pass its interpreter as `VBENCH_PYTHON`.

## 2. Models and scorers

`bash download_scripts/download_assets.sh` fetches everything below into the
layout the code expects (`WITH_VBENCH=0` skips the VBench weights).

| what | source | local path |
|---|---|---|
| AnyFlow-Wan2.1-T2V-1.3B (policy base, non-commercial license) | [nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers) | `models/AnyFlow-Wan2.1-T2V-1.3B-Diffusers/` |
| Stage-3 checkpoints | Hugging Face (coming soon) | `checkpoints/<trial>/checkpoint-250/generator_ema.pt` |
| **hpsv3_general / hpsv3_percentile** | [MizzenAI/HPSv3](https://huggingface.co/MizzenAI/HPSv3) `HPSv3.safetensors` on [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) | `models/hf_cache/` |
| **videoalign_mq / videoalign_ta** | [KlingTeam/VideoReward](https://huggingface.co/KlingTeam/VideoReward) on [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct); code from [ModelTC/VideoAlign](https://github.com/ModelTC/VideoAlign) @ `aba26b65` | `models/VideoReward/`, `models/hf_cache/`, `third_party/VideoAlign/` (vendored) |
| VBench tokenizer | [bert-base-uncased](https://huggingface.co/bert-base-uncased) (GRiT, tag2text) | `models/hf_cache/` |

VBench metric checkpoints (`models/vbench/`, `VBENCH_CACHE_DIR`). VBench is
called with `local=True`, so each file must already sit at this exact path:

| file | dimensions | source |
|---|---|---|
| `clip_model/ViT-B-32.pt` | background_consistency, appearance_style | [OpenAI CLIP](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt) |
| `clip_model/ViT-L-14.pt` | aesthetic_quality | [OpenAI CLIP](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) |
| `aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth` | aesthetic_quality | [LAION-AI/aesthetic-predictor](https://github.com/LAION-AI/aesthetic-predictor) |
| `umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth` | human_action | [OpenGVLab/VBench_Used_Models](https://huggingface.co/OpenGVLab/VBench_Used_Models) |
| `amt_model/amt-s.pth` | motion_smoothness | [lalala125/AMT](https://huggingface.co/lalala125/AMT) |
| `raft_model/models/raft-things.pth` | dynamic_degree | [ddrfan/RAFT](https://huggingface.co/ddrfan/RAFT) (mirror; the official Dropbox zip hangs) |
| `dino_model/dino_vitbase16_pretrain.pth` + `dino_model/facebookresearch_dino_main/` | subject_consistency | [facebookresearch/dino](https://github.com/facebookresearch/dino) |
| `pyiqa_model/musiq_spaq_ckpt-358bb6af.pth` | imaging_quality | [IQA-PyTorch releases](https://github.com/chaofengc/IQA-PyTorch/releases/tag/v0.1-weights) |
| `grit_model/grit_b_densecap_objectdet.pth` | object_class, multiple_objects, color, spatial_relationship | [OpenGVLab/VBench_Used_Models](https://huggingface.co/OpenGVLab/VBench_Used_Models) (mirror; the official Azure blob is closed) |
| `caption_model/tag2text_swin_14m.pth` | scene | [xinyu1205/recognize-anything](https://huggingface.co/spaces/xinyu1205/recognize-anything) |
| `ViCLIP/ViClip-InternVid-10M-FLT.pth` | temporal_style, overall_consistency | [OpenGVLab/VBench_Used_Models](https://huggingface.co/OpenGVLab/VBench_Used_Models) |

Environment variables (all optional; defaults are the paths above):
`FEWSTEPRL_ANYFLOW_WAN_PATH`, `FEWSTEPRL_VIDEOREWARD_PATH`,
`FEWSTEPRL_VIDEOALIGN_PATH`, `VBENCH_CACHE_DIR`, `HF_HOME`. The scripts set
`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`, so everything loads from local
files.

## 3. Protocol

Sampling matches the eval block of every training config: the AnyFlow
pipeline with the Stage-3 LoRA (`mode: meanflow_nft`), **4-step flow map,
CFG-free, 81 frames at 480x832, 16 fps**. `base` uses `mode: anyflow`, the
pre-RL model every trial starts from. Each video's seed is `seed + prompt index`,
so results do not depend on the number of GPUs.

| prompt set | videos | scored with |
|---|---|---|
| `wan_dancegrpo` (`dataset/wan_dancegrpo/test.txt`) | 256 | hpsv3_general, hpsv3_percentile, videoalign_mq, videoalign_ta |
| `vbench` (`dataset/vbench/VBench_aug_full_info.json`) | 946 x 5 = 4730 | 16 VBench dimensions -> quality / semantic / overall |

VBench videos are generated from `aug_prompt_en` and linked as
`{prompt_en}-{k}.mp4`, the filenames VBench matches on.

## 4. Usage

```bash
# training rewards for one released checkpoint, 8 GPUs for sampling
bash scripts/eval/eval_ckpt.sh meanflownft 8

# plus VBench (generation ~4730 videos, then 16 dimensions over 8 GPUs)
VBENCH=1 VBENCH_PYTHON=/path/to/vbench-scoring/bin/python bash scripts/eval/eval_ckpt.sh meanflownft 8

# the pre-RL baseline
bash scripts/eval/eval_ckpt.sh base 8

# everything + summary table
bash scripts/eval/eval_all.sh 8

# your own checkpoint from a training run
bash scripts/eval/eval_ckpt.sh outputs/meanflownft/<timestamp>/checkpoint-100/generator_ema.pt 8

# the steps by hand
bash scripts/eval/sample.sh meanflownft wan_dancegrpo 8
python3 scripts/eval/score.py --sample_dir outputs/eval/meanflownft/wan_dancegrpo/meanflow_nft
bash scripts/eval/sample.sh meanflownft vbench 8
/path/to/vbench-scoring/bin/python scripts/eval/vbench_score.py \
    --sample_dir outputs/eval/meanflownft/vbench/meanflow_nft --gpus "0 1 2 3 4 5 6 7"
```

Quick smoke test: `MAX_PROMPTS=8 OUT_ROOT=outputs/eval_smoke bash scripts/eval/eval_ckpt.sh meanflownft 1`.
Finished sample sets and scored metrics / dimensions are skipped on re-runs;
set `FORCE=1` (sampling) or pass `--force` (`score.py`) to redo them.

## 5. Reading the numbers

- **MQ / TA are z-scores.** VideoAlign is called with `use_norm=True`, which
  standardises against VideoReward's training statistics. `score.py` reports
  both the z-score mean (what training logs, and what `mean` sums) and the raw
  value `z * std + mean` (MQ 1.1646 / 1.3811, TA 2.8105 / 2.5121).
- **Eight VBench dimensions are rescaled before aggregation** (`_NORM_RANGES`
  in `fewsteprl/eval/vbench_eval.py`): subject_consistency, motion_smoothness,
  temporal_flickering, background_consistency, scene, appearance_style,
  temporal_style, overall_consistency are mapped through fixed ranges;
  `overall = 0.8 * quality + 0.2 * semantic`.
- **The training rewards and VBench disagree.** The four rewards are the
  training objective and cannot separate quality gains from reward hacking;
  VBench is held out. Compare `dynamic_degree` in particular (see
  `checkpoints/README.md`).
- **Offline re-sampling is not the training-time eval.** The archived numbers in
  `checkpoints/README.md` were produced inside training with its own seeding;
  this pipeline draws its own samples, so expect small differences and compare
  checkpoints against each other under the same protocol.

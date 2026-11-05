# Released checkpoints

One Stage-3 LoRA per main experiment, all at epoch 250 (the `nft` ablation has
no released checkpoint). Every experiment name
matches its launcher `scripts/train/<trial>.sh` and config
`scripts/train/configs/<trial>.yaml`.

```
checkpoints/<trial>/
  config.yaml                         # the config the run saved (paths made relative)
  checkpoint-250/generator_ema.pt     # EMA generator: LoRA A/B + condition_embedder.delta_embedder
```

The `generator_ema.pt` weights are not in this repository; they will be
released on Hugging Face soon. `generator_ema.pt` is a plain tensor state
dict. It is applied on top of the full
[nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers)
pipeline: `inference.py` with `mode: meanflow_nft` and
`meanflow_nft_path=<generator_ema.pt>` does this, and `scripts/eval/sample.sh
<trial>` wraps it. The AnyFlow-Wan license restricts the model and derivative
weights to non-commercial use.

## Training-reward evaluation (256 held-out prompts, iteration 250; paper Tab. 8)

`dataset/wan_dancegrpo/test.txt`, 4-step flow map, 81 frames, 480x832, seed
12345, logged inside training. `mean` is the logged sum of the four rewards
with MQ / TA as z-scores. MQ / TA below are converted back to the raw scale
with VideoReward's own statistics (`raw = z * std + mean`; MQ 1.1646 / 1.3811,
TA 2.8105 / 2.5121, from `inference_config` in
`models/VideoReward/model_config.json`). This is the
training objective, so it does not separate quality gains from reward hacking.

| trial | trainer | mean | HPSv3-G | HPSv3-P | MQ (raw) | TA (raw) | source run |
|---|---|---|---|---|---|---|---|
| step 0 (AnyFlow, pre-RL) | -- | 16.54 | 6.42 | 9.43 | 2.64 | 1.85 | |
| meanflownft | `meanflow_nft` | 16.80 | 6.53 | 9.51 | 2.76 | 1.80 | `wan_rpd_mfnft_32gpu/20260921_203542` |
| dmdnft | `dmdnft_rl` | 17.93 | 7.01 | 10.01 | 3.06 | 1.68 | `wan_dmdnft_32gpu/20260925_151719` |
| dmdawm | `dmdawm_rl` | 16.78 | 6.51 | 9.53 | 2.74 | 1.82 | `wan_dmdawm_nocd_32gpu/20261008_212312` |
| fewstepgrpo | `meanflowgrpo_rl` | 15.89 | 6.24 | 9.38 | 2.40 | 1.23 | `wan_grpo_32gpu/20260928_150048` |
| meanflowawm | `meanflowawm_rl` | 16.66 | 6.49 | 9.46 | 2.68 | 1.84 | `wan_awm_32gpu/20260924_205834` |
| meanflowopsd | `meanflowopsd_rl` | 16.26 | 6.30 | 9.32 | 2.54 | 1.91 | `wan_opsd_v_32gpu/20261007_163726` |
| reflmx | `meanflowrefl_rl` | 16.46 | 6.39 | 9.38 | 2.61 | 1.89 | `wan_reflmx_32gpu/20261005_234849` |

## VBench (held out, checkpoint-250; paper Tab. 8 and Tab. S2)

946 GPT-augmented prompts x 5 samples, 16 dimensions, aggregated with AnyFlow's
weights (eight dimensions are rescaled before aggregation, see
`fewsteprl/eval/vbench_eval.py`). Scores in %.

| trial | overall | quality | semantic | dynamic_degree |
|---|---|---|---|---|
| step 0 (AnyFlow, pre-RL) | 83.68 | 85.36 | 76.98 | -- |
| meanflownft | 83.62 | 85.34 | 76.75 | 58.06 |
| dmdnft | 83.38 | 85.07 | 76.63 | 53.33 |
| dmdawm | 83.64 | 85.35 | 76.76 | 58.61 |
| fewstepgrpo | 83.62 | 85.27 | 76.98 | 59.62 |
| meanflowawm | 83.70 | 85.36 | 77.04 | 58.89 |
| meanflowopsd | 83.60 | 85.37 | 76.53 | 59.44 |
| reflmx | 83.69 | 85.40 | 76.85 | 59.17 |

The two benchmarks disagree: VBench overall spans only 0.32 points across the
trials, while the training reward spans 2.04 (paper Tab. S2). `dmdnft` has the
highest training reward and the lowest VBench overall and `dynamic_degree`;
`fewstepgrpo` has the lowest training reward and the highest `dynamic_degree`.

## Notes

- `dmdawm` is the no-central-difference DMD-AWM (`x_anchor_2 = x_t - t * u_old`).
- `meanflowopsd` and `reflmx` take their reward direction from the two
  differentiable HPSv3 terms only (96% of the aggregate; VideoAlign MQ and TA
  have no gradient), computed in fp32 with the `sdpa` backend and with the clip
  decoded one latent frame at a time. Per-frame decoding deviates the gradient
  from a joint decode by 79.2 degrees on average, so these two rows do not use
  the gradient of the full reward (paper App. D.5). The reported reward values
  use the full four-term scorer and are unaffected.
- `dmdnft` was trained in two legs (0-50, then resumed to 250); its
  `config.yaml` is the second leg's with `train.resume_from` cleared.
- GPU counts as in the paper (Tab. 8): 32 for the five group-advantage trials,
  48 for `meanflowopsd` and `reflmx`, with the same per-rank batch, so the
  48-GPU runs use 1.5x as many rollouts per iteration (3072 vs. 2048).

`SHA256SUMS` lists every weight file in this directory.

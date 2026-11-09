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

The `generator_ema.pt` weights are on Hugging Face,
[Delcher/fewsteprl-wan21](https://huggingface.co/Delcher/fewsteprl-wan21). Download and verify them with

```bash
hf download Delcher/fewsteprl-wan21 --local-dir checkpoints --include "*.pt"
cd checkpoints && shasum -a 256 -c SHA256SUMS
```

`generator_ema.pt` is a plain tensor state dict. It is applied on top of the full
[nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers)
pipeline: `inference.py` with `mode: meanflow_nft` and
`meanflow_nft_path=<generator_ema.pt>` does this, and `scripts/eval/sample.sh
<trial>` wraps it. The AnyFlow-Wan license restricts the model and derivative
weights to non-commercial use.

## Training-reward evaluation (256 held-out prompts, iteration 250; paper Tab. 8)

`dataset/wan_dancegrpo/test.txt`, 4-step flow map, 81 frames, 480x832, seed
12345. `mean` is the sum of the four rewards
with MQ / TA as z-scores. MQ / TA below are converted back to the raw scale
with VideoReward's own statistics (`raw = z * std + mean`; MQ 1.1646 / 1.3811,
TA 2.8105 / 2.5121, from `inference_config` in
`models/VideoReward/model_config.json`). This is the
training objective, so it does not separate quality gains from reward hacking.

| trial | trainer | mean | HPSv3-G | HPSv3-P | MQ (raw) | TA (raw) |
|---|---|---|---|---|---|---|
| step 0 (AnyFlow, pre-RL) | -- | 16.540 | 6.420 | 9.430 | 2.640 | 1.850 |
| meanflownft | `meanflow_nft` | 16.800 | 6.530 | 9.510 | 2.760 | 1.800 |
| dmdnft | `dmdnft_rl` | 17.930 | 7.010 | 10.010 | 3.060 | 1.680 |
| dmdawm | `dmdawm_rl` | 16.780 | 6.510 | 9.530 | 2.740 | 1.820 |
| fewstepgrpo | `meanflowgrpo_rl` | 15.890 | 6.240 | 9.380 | 2.400 | 1.230 |
| meanflowawm | `meanflowawm_rl` | 16.660 | 6.490 | 9.460 | 2.680 | 1.840 |
| meanflowopsd | `meanflowopsd_rl` | 16.260 | 6.300 | 9.320 | 2.540 | 1.910 |
| reflmx | `meanflowrefl_rl` | 16.460 | 6.390 | 9.380 | 2.610 | 1.890 |

## VBench (held out, checkpoint-250; paper Tab. 8 and Tab. S2)

946 GPT-augmented prompts x 5 samples, 16 dimensions, aggregated with AnyFlow's
weights (eight dimensions are rescaled before aggregation, see
`fewsteprl/eval/vbench_eval.py`). Scores in %.

| trial | overall | quality | semantic | dynamic_degree |
|---|---|---|---|---|
| step 0 (AnyFlow, pre-RL) | 83.680 | 85.360 | 76.980 | -- |
| meanflownft | 83.620 | 85.340 | 76.750 | 58.060 |
| dmdnft | 83.380 | 85.070 | 76.630 | 53.330 |
| dmdawm | 83.640 | 85.350 | 76.760 | 58.610 |
| fewstepgrpo | 83.620 | 85.270 | 76.980 | 59.620 |
| meanflowawm | 83.700 | 85.360 | 77.040 | 58.890 |
| meanflowopsd | 83.600 | 85.370 | 76.530 | 59.440 |
| reflmx | 83.690 | 85.400 | 76.850 | 59.170 |

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

`SHA256SUMS` lists every weight file in this directory.

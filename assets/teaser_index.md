# Teaser index (`assets/teaser.*`)

Clips row by row (4 per row). All: 4 steps, no CFG, 81 frames, 480×832, seed 0. Reproduce from the repository root with

```bash
python inference.py scripts/eval/configs/sample_wan.yaml --prompt "<prompt>" \
    --override seed=0 mode=meanflow_nft meanflow_nft_path=<checkpoint>
```

Prompts were curated; for these prompts the seed was fixed to 0 (not picked per clip).

| # | trainer | checkpoint | prompt |
|---|---|---|---|
| 1 | ReFL | `checkpoints/reflmx/checkpoint-250/generator_ema.pt` | A majestic dragon with shimmering scales flying over snowy mountain peaks at sunrise, wings beating slowly, epic cinematic aerial shot. |
| 2 | MeanFlowNFT | `checkpoints/meanflownft/checkpoint-250/generator_ema.pt` | A phoenix made of fire spreading its wings and rising into the night sky, embers and sparks swirling around it, cinematic slow motion. |
| 3 | FewStepOPSD | `checkpoints/meanflowopsd/checkpoint-250/generator_ema.pt` | A giant whale swimming slowly through pink sunset clouds above a small seaside town, dreamy surreal cinematic shot. |
| 4 | DMDNFT | `checkpoints/dmdnft/checkpoint-250/generator_ema.pt` | Thousands of glowing paper lanterns floating up into the night sky over a calm lake, reflections on the water, magical atmosphere. |
| 5 | MeanFlowAWM | `checkpoints/meanflowawm/checkpoint-250/generator_ema.pt` | Vivid green and purple northern lights dancing over a snowy mountain lake, stars twinkling, time-lapse. |
| 6 | DMDAWM | `checkpoints/dmdawm/checkpoint-250/generator_ema.pt` | An astronaut floating slowly in front of a colorful nebula, stars drifting, the visor reflecting galaxies, cinematic space shot. |
| 7 | FewStepGRPO | `checkpoints/fewstepgrpo/checkpoint-250/generator_ema.pt` | Hundreds of blue morpho butterflies fluttering through a sunlit rainforest, light rays through the canopy, magical slow motion. |
| 8 | ReFL | `checkpoints/reflmx/checkpoint-250/generator_ema.pt` | A dancer in a flowing red dress spinning on a sand dune at sunset, the fabric swirling in the wind, cinematic slow motion. |
| 9 | FewStepOPSD | `checkpoints/meanflowopsd/checkpoint-250/generator_ema.pt` | Cherry blossom petals falling on an old Kyoto street at dusk with glowing lanterns, people walking with umbrellas, cinematic. |
| 10 | MeanFlowNFT | `checkpoints/meanflownft/checkpoint-250/generator_ema.pt` | A glowing ice castle on a frozen lake under the aurora, snowflakes falling, the camera slowly pushing in, fantasy cinematic. |
| 11 | DMDNFT | `checkpoints/dmdnft/checkpoint-250/generator_ema.pt` | A colorful octopus gliding gracefully over a vibrant coral reef, tentacles flowing, sunlight beams underwater. |
| 12 | MeanFlowAWM | `checkpoints/meanflowawm/checkpoint-250/generator_ema.pt` | An erupting volcano at night with glowing lava fountains and rivers of lava flowing down the slope, dramatic nature footage. |

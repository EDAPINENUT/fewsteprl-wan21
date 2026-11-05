# VideoAlign (vendored subset)

Source: [ModelTC/VideoAlign](https://github.com/ModelTC/VideoAlign), commit
`aba26b658fec7d9fd30c295187b548ea673c8769`, MIT license (see `LICENSE`).

Only `inference.py` and the sibling modules it imports are included
(`data.py`, `prompt_template.py`, `train_reward.py`, `trainer.py`, `utils.py`,
`vision_process.py`), unmodified. `fewsteprl/rewards/video_multi_scorer.py`
loads `inference.py` by path to build the VideoReward MQ / TA scorers; the
checkpoint itself (`KlingTeam/VideoReward`) is downloaded by
`download_scripts/download_assets.sh` into `models/VideoReward/`.

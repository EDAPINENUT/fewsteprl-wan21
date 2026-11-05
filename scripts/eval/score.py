#!/usr/bin/env python3
"""Score a directory of videos written by scripts/eval/sample.sh with the training rewards.

    python3 scripts/eval/score.py --sample_dir outputs/eval/meanflownft/wan_dancegrpo/meanflow_nft

Reads ``<sample_dir>/metadata.json`` (file + prompt per video) and scores every
video with one ``fewsteprl.rewards.video_multi_scorer.VideoMultiScorer`` per
metric (hpsv3_general, hpsv3_percentile, videoalign_mq, videoalign_ta) -- the
same scorers and settings the training-time eval uses. Writes, per metric,

    <sample_dir>/score_<metric>.json   {"metric", "n", "mean", "scores": [...]}

and rebuilds ``<sample_dir>/summary.json`` from every ``score_*.json`` present.
VideoAlign MQ / TA are reported as z-scores (as logged in training) and also on
the raw scale, ``raw = z * std + mean`` with the statistics stored in the
VideoReward checkpoint's ``model_config.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
METRICS = ("hpsv3_general", "hpsv3_percentile", "videoalign_mq", "videoalign_ta")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sample_dir", required=True)
    p.add_argument("--metrics", default=",".join(METRICS),
                   help=f"comma-separated subset of {','.join(METRICS)}")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true",
                   help="rescore metrics that already have a score_<metric>.json")
    return p.parse_args()


def setup_env() -> None:
    os.environ.setdefault("HF_HOME", str(REPO / "models" / "hf_cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def videoalign_paths() -> dict:
    return {
        "videoalign_dir": os.environ.get("FEWSTEPRL_VIDEOALIGN_PATH", str(REPO / "third_party" / "VideoAlign")),
        "videoalign_ckpt": os.environ.get("FEWSTEPRL_VIDEOREWARD_PATH", str(REPO / "models" / "VideoReward")),
    }


def raw_stats(videoreward_dir: str) -> dict:
    cfg = json.loads((Path(videoreward_dir) / "model_config.json").read_text())["inference_config"]
    return {"videoalign_mq": (cfg["MQ_mean"], cfg["MQ_std"]),
            "videoalign_ta": (cfg["TA_mean"], cfg["TA_std"])}


def load_video(path: Path):
    import imageio.v2 as imageio
    import numpy as np
    return np.stack(list(imageio.get_reader(str(path), format="FFMPEG")), axis=0)  # [F, H, W, 3] uint8


def score_metric(metric: str, rows: list[dict], sample_dir: Path, device: str, batch_size: int) -> list[float]:
    import numpy as np
    import torch
    from fewsteprl.rewards.video_multi_scorer import VideoMultiScorer

    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    scorer = VideoMultiScorer(dev, {metric: 1.0}, allow_unavailable=False, **videoalign_paths())
    values: list[float] = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        videos = np.stack([load_video(sample_dir / r["file"]) for r in chunk], axis=0)
        details, _ = scorer(videos, [r["prompt"] for r in chunk])
        values.extend(float(x) for x in np.asarray(details[metric]).reshape(-1))
    del scorer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return values


def rebuild_summary(sample_dir: Path) -> dict:
    summary = {"sample_dir": str(sample_dir), "metrics": {}}
    for f in sorted(sample_dir.glob("score_*.json")):
        d = json.loads(f.read_text())
        entry = {"mean": d["mean"], "n": d["n"]}
        if "mean_raw" in d:
            entry["mean_raw"] = d["mean_raw"]
        summary["metrics"][d["metric"]] = entry
    (sample_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    setup_env()
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown = sorted(set(metrics) - set(METRICS))
    if unknown:
        sys.exit(f"ERROR: unsupported metric(s) {unknown}; supported: {list(METRICS)}")

    sample_dir = Path(args.sample_dir).resolve()
    meta = sample_dir / "metadata.json"
    if not meta.is_file():
        sys.exit(f"ERROR: {meta} not found -- run scripts/eval/sample.sh first")
    rows = sorted(json.loads(meta.read_text())["videos"], key=lambda r: r["index"])
    stats = raw_stats(videoalign_paths()["videoalign_ckpt"]) if any(m.startswith("videoalign") for m in metrics) else {}

    for metric in metrics:
        out = sample_dir / f"score_{metric}.json"
        if out.is_file() and not args.force and json.loads(out.read_text()).get("n") == len(rows):
            print(f"[score] skip {metric}: {out} exists")
            continue
        values = score_metric(metric, rows, sample_dir, args.device, args.batch_size)
        if len(values) != len(rows):
            sys.exit(f"ERROR: {metric} returned {len(values)} scores for {len(rows)} videos")
        rec = {"metric": metric, "n": len(values), "mean": sum(values) / len(values)}
        if metric in stats:
            mu, sd = stats[metric]
            rec["mean_raw"] = rec["mean"] * sd + mu
        rec["scores"] = [{"index": r["index"], "file": r["file"], "score": v} for r, v in zip(rows, values)]
        out.write_text(json.dumps(rec, indent=1))
        print(f"[score] {metric:17s} n={rec['n']:4d} mean={rec['mean']:.4f}"
              + (f" raw={rec['mean_raw']:.4f}" if "mean_raw" in rec else ""))

    print(json.dumps(rebuild_summary(sample_dir)["metrics"], indent=2))


if __name__ == "__main__":
    main()

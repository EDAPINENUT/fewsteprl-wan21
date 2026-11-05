#!/usr/bin/env python3
"""Score generated VBench videos, one dimension per single-GPU process, then aggregate.

    # all 16 dimensions spread over the listed GPUs, then aggregate
    python3 scripts/eval/vbench_score.py --sample_dir outputs/eval/<tag>/vbench/<mode> --gpus "0 1 2 3 4 5 6 7"

    # one dimension / aggregate only
    python3 scripts/eval/vbench_score.py --sample_dir ... --dimension subject_consistency
    python3 scripts/eval/vbench_score.py --sample_dir ... --aggregate

Run this in the separate VBench scoring environment (see scripts/eval/README.md).
Scoring is kept out of the generation / reward process on purpose: VideoAlign
puts a top-level ``utils`` module into ``sys.modules`` that shadows DINO's, and a
multi-rank scoring job would sit in a barrier while the slowest dimension runs.
Each dimension writes ``<sample_dir>/vbench_info/{dim}_eval_results.json`` and
is skipped if that file exists, so the run is resumable. The aggregate uses the
same normalisation and weights as the training-time evaluator.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from fewsteprl.eval.vbench_eval import _DIMENSIONS, _aggregate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger("vbench_score")
FULL_INFO = REPO / "dataset" / "vbench" / "VBench_full_info.json"


def score_one(samples: Path, info_dir: Path, dimension: str) -> int:
    cached = info_dir / f"{dimension}_eval_results.json"
    if cached.exists():
        logger.info("%s: already scored, skipping", dimension)
        return 0
    import torch
    from vbench import VBench

    evaluator = VBench(torch.device("cuda"), str(FULL_INFO), str(info_dir))
    evaluator.evaluate(videos_path=str(samples), name=dimension, dimension_list=[dimension], local=True)
    if not cached.exists():
        logger.error("%s: evaluate() returned but wrote no %s", dimension, cached)
        return 1
    logger.info("%s: done", dimension)
    return 0


def aggregate(info_dir: Path) -> int:
    info, missing = {}, []
    for dim in _DIMENSIONS:
        cached = info_dir / f"{dim}_eval_results.json"
        if not cached.exists():
            missing.append(dim)
            continue
        raw = json.loads(cached.read_text())[dim]
        info[dim] = float(raw[0] if isinstance(raw, (list, tuple)) else raw)  # [avg, per-video]
    if missing:
        # The aggregate divisors are fixed (6.5 / 9.0), so a missing dimension
        # would silently lower the score instead of being excluded.
        logger.warning("missing %d dimension(s): %s -- aggregate omitted", len(missing), ", ".join(missing))
        print(json.dumps(info, indent=2, sort_keys=True))
        return 1
    out = _aggregate(info)
    (info_dir / "vbench_summary.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def run_all(sample_dir: Path, gpus: list[str]) -> int:
    """One process per dimension, at most one per GPU at a time."""
    queue = [d for d in _DIMENSIONS if not (sample_dir / "vbench_info" / f"{d}_eval_results.json").exists()]
    running: dict[str, tuple[str, subprocess.Popen]] = {}
    failed = []
    log_dir = sample_dir / "vbench_info" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    while queue or running:
        for gpu in gpus:
            if gpu not in running and queue:
                dim = queue.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
                fh = open(log_dir / f"{dim}.log", "w")
                proc = subprocess.Popen(
                    [sys.executable, __file__, "--sample_dir", str(sample_dir), "--dimension", dim],
                    env=env, stdout=fh, stderr=subprocess.STDOUT)
                running[gpu] = (dim, proc)
                logger.info("GPU %s: %s", gpu, dim)
        for gpu, (dim, proc) in list(running.items()):
            if proc.poll() is not None:
                if proc.returncode:
                    failed.append(dim)
                    logger.error("%s failed (rc=%d), see %s", dim, proc.returncode, log_dir / f"{dim}.log")
                del running[gpu]
        time.sleep(5)
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_dir", required=True, help="sample.sh output dir (contains vbench_samples/)")
    ap.add_argument("--dimension", default="", help="score a single dimension")
    ap.add_argument("--aggregate", action="store_true", help="only combine existing per-dimension results")
    ap.add_argument("--gpus", default="0", help='GPU ids for the all-dimensions mode, e.g. "0 1 2 3"')
    ap.add_argument("--cache_dir", default=os.environ.get("VBENCH_CACHE_DIR", str(REPO / "models" / "vbench")),
                    help="VBench metric checkpoints (download_scripts/download_assets.sh)")
    args = ap.parse_args()

    os.environ["VBENCH_CACHE_DIR"] = args.cache_dir
    sample_dir = Path(args.sample_dir).resolve()
    samples, info_dir = sample_dir / "vbench_samples", sample_dir / "vbench_info"
    info_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate:
        return aggregate(info_dir)
    if not samples.is_dir():
        ap.error(f"{samples} not found -- run scripts/eval/sample.sh <target> vbench first")
    if args.dimension:
        if args.dimension not in _DIMENSIONS:
            ap.error(f"unknown dimension {args.dimension!r}; expected one of {_DIMENSIONS}")
        return score_one(samples, info_dir, args.dimension)
    rc = run_all(sample_dir, args.gpus.split())
    return aggregate(info_dir) or rc


if __name__ == "__main__":
    raise SystemExit(main())

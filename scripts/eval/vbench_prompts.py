#!/usr/bin/env python3
"""Expand the VBench prompt list and link generated videos to VBench filenames.

    # 946 prompts x 5 samples -> a prompt file inference.py can read
    python3 scripts/eval/vbench_prompts.py expand --out outputs/eval/vbench_prompts.json

    # after sampling: <sample_dir>/vbench_samples/{prompt_en}-{k}.mp4 -> <sample_dir>/<index>.mp4
    python3 scripts/eval/vbench_prompts.py link --sample_dir outputs/eval/<tag>/vbench/<mode>

Generation uses the GPT-augmented ``aug_prompt_en``; VBench matches videos by
the short ``prompt_en`` in the filename. Two ``prompt_en`` strings occur twice in
the info file, so their later samples overwrite the earlier ones, exactly as in
the training-time evaluator (``fewsteprl/eval/vbench_eval.py``).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AUG_INFO = REPO / "dataset" / "vbench" / "VBench_aug_full_info.json"


def expand(num_samples: int) -> list[dict]:
    items = []
    for entry in json.loads(AUG_INFO.read_text()):
        for k in range(num_samples):
            items.append({
                "prompt": entry.get("aug_prompt_en", entry["prompt_en"]),
                "prompt_en": entry["prompt_en"],
                "video_name": f"{entry['prompt_en']}-{k}.mp4",
            })
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("expand")
    e.add_argument("--out", required=True)
    e.add_argument("--num_samples", type=int, default=5)
    l = sub.add_parser("link")
    l.add_argument("--sample_dir", required=True)
    l.add_argument("--num_samples", type=int, default=5)
    args = ap.parse_args()

    items = expand(args.num_samples)
    if args.cmd == "expand":
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(items, ensure_ascii=False, indent=1))
        print(f"wrote {len(items)} prompts to {args.out}")
        return

    sample_dir = Path(args.sample_dir).resolve()
    videos = json.loads((sample_dir / "metadata.json").read_text())["videos"]
    if len(videos) != len(items):
        raise SystemExit(f"{sample_dir} has {len(videos)} videos, expected {len(items)}")
    out = sample_dir / "vbench_samples"
    out.mkdir(exist_ok=True)
    for rec in videos:
        link = out / items[rec["index"]]["video_name"]
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(sample_dir / rec["file"], link)
    print(f"linked {len(videos)} videos into {out}")


if __name__ == "__main__":
    main()

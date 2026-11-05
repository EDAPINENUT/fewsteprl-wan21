#!/usr/bin/env python3
"""Aggregate scores under outputs/eval into one table.

    python3 scripts/eval/collect.py [--out_root outputs/eval]

Writes ``<out_root>/summary.csv`` and ``<out_root>/summary.md`` with one row per
tag: the four training rewards on the wan_dancegrpo test prompts (``mean`` is
the sum with MQ / TA as z-scores, as logged in training; MQ / TA are also shown
on the raw scale) and, where scored, the VBench aggregate.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REWARDS = ("hpsv3_general", "hpsv3_percentile", "videoalign_mq", "videoalign_ta")
VBENCH = ("overall_score", "quality_score", "semantic_score", "dynamic_degree")
HEADER = ["tag", "mean", "hpsv3_general", "hpsv3_percentile", "mq_raw", "ta_raw",
          "vbench_overall", "vbench_quality", "vbench_semantic", "vbench_dynamic_degree"]


def first(paths):
    return next((p for p in paths if p.is_file()), None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="outputs/eval")
    root = Path(ap.parse_args().out_root)
    if not root.is_dir():
        raise SystemExit(f"{root} does not exist -- run scripts/eval/eval_ckpt.sh first")

    rows = []
    for tag_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        row = {"tag": tag_dir.name}
        z, raw = {}, {}
        for m in REWARDS:
            f = first(sorted(tag_dir.glob(f"wan_dancegrpo/*/score_{m}.json")))
            if f:
                d = json.loads(f.read_text())
                z[m], raw[m] = d["mean"], d.get("mean_raw")
        if len(z) == len(REWARDS):
            row["mean"] = sum(z.values())
        row["hpsv3_general"] = z.get("hpsv3_general")
        row["hpsv3_percentile"] = z.get("hpsv3_percentile")
        row["mq_raw"] = raw.get("videoalign_mq")
        row["ta_raw"] = raw.get("videoalign_ta")
        vb = first(sorted(tag_dir.glob("vbench/*/vbench_info/vbench_summary.json")))
        if vb:
            v = json.loads(vb.read_text())
            for k in VBENCH:
                row["vbench_" + k.replace("_score", "")] = 100.0 * v[k]
        if len(row) > 1:
            rows.append(row)

    rows.sort(key=lambda r: (r["tag"] != "base", r["tag"]))

    def fmt(v):
        return "" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))

    with (root / "summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for r in rows:
            w.writerow([fmt(r.get(h)) for h in HEADER])
    md = ["| " + " | ".join(HEADER) + " |", "|" + "---|" * len(HEADER)]
    md += ["| " + " | ".join(fmt(r.get(h)) or "--" for h in HEADER) + " |" for r in rows]
    (root / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"wrote {root / 'summary.csv'} and {root / 'summary.md'}")


if __name__ == "__main__":
    main()

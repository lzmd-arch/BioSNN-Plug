"""A3 补种子 1、2（判据见 `prereg-a3-amend1-seeds.md`）。只跑 taps=28。"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from research.brainregions.run_a3 import run_once

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "sweep_results" / "brainregions_a3"

TAPS = 28
SEEDS = (1, 2)
#: seed 0 的 k=28 实测值（`sweep_results/brainregions_a3/a3.json`）。
SEED0_ACCURACY = 0.917
#: A1 的均匀臂就是 k=1，三种子的实测值。
TAPS1_BY_SEED = {0: 0.759, 1: 0.762, 2: 0.763}
MIN_ACCURACY = 0.90
MIN_GAIN = 0.05
MIN_MEDIAN_GAIN = 0.10


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A3 补种子 1、2")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="a3_seeds")
    args = parser.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows: dict[str, dict] = {}
    for seed in SEEDS:
        result = run_once(TAPS, device, seed)
        rows[str(seed)] = {
            key: round(value, 4) if key != "elapsed_s" else value for key, value in result.items()
        }
        print(f"[seed {seed}] {rows[str(seed)]}", flush=True)

    gains = {}
    for seed in (0, *SEEDS):
        accuracy = SEED0_ACCURACY if seed == 0 else rows[str(seed)]["accuracy"]
        gains[str(seed)] = round(accuracy - TAPS1_BY_SEED[seed], 4)
    median_gain = round(statistics.median(gains.values()), 4)

    criteria = {
        "s0_both_learned": all(rows[str(seed)]["accuracy"] >= MIN_ACCURACY for seed in SEEDS),
        "s1_every_seed_wins": all(gain >= MIN_GAIN for gain in gains.values()),
        "s2_median_gain": median_gain >= MIN_MEDIAN_GAIN,
        "gains": gains,
        "median_gain": median_gain,
    }
    out = {
        "kind": "line A v3 amendment 1: extra seeds for taps=28",
        "taps": TAPS,
        "seeds": list(SEEDS),
        "taps1_by_seed": TAPS1_BY_SEED,
        "rows": rows,
        "criteria": criteria,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + chr(10), encoding="utf-8", newline=chr(10)
    )
    print(f"三种子增益 {gains}，中位 {median_gain:+.4f}")
    print(
        f"[S-0] {'过' if criteria['s0_both_learned'] else '不过'} | "
        f"[S-1] {'过' if criteria['s1_every_seed_wins'] else '不过'} | "
        f"[S-2] {'过' if criteria['s2_median_gain'] else '不过'}"
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

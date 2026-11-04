"""A7：补 (decay × k) 的两格，直接检验"每个神经元几个数"（判据见 `prereg-a7-k-ladder.md`）。

已有 k=1 / k=28 在 decay 0.95 与 0.99 下的四个点，本轮补 k=7 的两格。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from research.brainregions.run_a6_decay_control import N_REC, run_arm

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "sweep_results" / "brainregions_a7"

TAPS = 7
DECAYS = (0.99, 0.95)
#: 已有四点（只读，不重跑）。
KNOWN = {
    "0.95": {"1": 0.338, "28": 0.759},
    "0.99": {"1": 0.425, "28": 0.710},
}
FLOOR = 0.20
MAIN_SHARE = 0.70
SATURATION = 0.05
SECOND_MIN = 0.70 * (0.759 - 0.338)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A7：k 阶梯")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="a7")
    args = parser.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = {str(decay): run_arm(TAPS, device, decay) for decay in DECAYS}
    print(f"[A7] {rows}", flush=True)

    main_arm = rows["0.99"]["accuracy"]
    second_arm = rows["0.95"]["accuracy"]
    low, high = KNOWN["0.99"]["1"], KNOWN["0.99"]["28"]
    share = (main_arm - low) / (high - low)
    criteria = {
        "l0_both_learn": min(main_arm, second_arm) > FLOOR,
        "l1_main_share": share >= MAIN_SHARE,
        "l2_saturated": high - main_arm <= SATURATION,
        "l3_other_decay": (second_arm - KNOWN["0.95"]["1"]) >= SECOND_MIN,
        "share_1_to_7": round(share, 4),
    }
    out = {
        "kind": "line A v7: k ladder at two readout decays (SHD)",
        "n_rec": N_REC,
        "taps": TAPS,
        "epochs": 10,
        "seed": 0,
        "arms": rows,
        "known": KNOWN,
        "criteria": criteria,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        f"[L-0] {'过' if criteria['l0_both_learn'] else '不过'} | "
        f"[L-1] 1→7 占 {share:.2%}（门槛 {MAIN_SHARE:.0%}）-> "
        f"{'过' if criteria['l1_main_share'] else '不过'} | "
        f"[L-2] 7→28 差 {high - main_arm:+.4f} -> {'过' if criteria['l2_saturated'] else '不过'} | "
        f"[L-3] {'过' if criteria['l3_other_decay'] else '不过'}"
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

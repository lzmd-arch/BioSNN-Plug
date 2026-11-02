"""E1 v4：pilot / confirm 划分（判据见 prereg-e1-v4.md）。

pilot 半（迷宫 0–99）只标定量级；confirm 半（100–199）只用来判。
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .grid import make_maze
from .run_e0_e1 import SEEDS
from .run_e1_v3 import E1_SIZE, evaluate

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "sweep_results" / "sos_e1_v4"

PILOT = list(range(0, 100))
CONFIRM = list(range(100, 200))

GAP_RATIO = 0.70
CV_RATIO = 0.80
RHO_MIN = 0.30
COVERAGE_MIN = 4


def summarize(rows: list[dict]) -> dict:
    keys = ("gap", "spearman", "variance_cv", "hit_model", "hit_marginal")
    return {key: statistics.median(row[key] for row in rows) for key in keys} | {
        "coverage_seeds": sum(1 for row in rows if row["coverage_ok"])
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="脉冲 SoS 的 E1 v4（pilot/confirm）")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="e1_v4")
    args = parser.parse_args(argv)

    pilot_grids = [make_maze(seed, size=E1_SIZE) for seed in PILOT]
    confirm_grids = [make_maze(seed, size=E1_SIZE) for seed in CONFIRM]

    pilot_rows = [evaluate(seed * 2, pilot_grids) for seed in SEEDS]
    confirm_rows = [evaluate(seed * 2 + 1, confirm_grids) for seed in SEEDS]
    pilot = summarize(pilot_rows)
    confirm = summarize(confirm_rows)

    gap_threshold = GAP_RATIO * pilot["gap"]
    cv_threshold = CV_RATIO * pilot["variance_cv"]
    passed = (
        pilot["gap"] > 0
        and confirm["gap"] >= gap_threshold
        and confirm["spearman"] >= RHO_MIN
        and confirm["coverage_seeds"] >= COVERAGE_MIN
        and confirm["variance_cv"] >= cv_threshold
    )

    result = {
        "kind": "spiking sos e1 v4 (pilot/confirm)",
        "size": E1_SIZE,
        "pilot": {**pilot, "rows": pilot_rows},
        "confirm": {**confirm, "rows": confirm_rows},
        "thresholds": {"gap": gap_threshold, "cv": cv_threshold, "rho": RHO_MIN},
        "p1_ready": passed,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )

    print(
        f"[pilot]   gap {pilot['gap']:+.3f} | ρ {pilot['spearman']:.3f} | "
        f"CV {pilot['variance_cv']:.3f} | 命中 {pilot['hit_model']:.3f} vs 边际 {pilot['hit_marginal']:.3f}"
    )
    print(
        f"[confirm] gap {confirm['gap']:+.3f} | ρ {confirm['spearman']:.3f} | "
        f"CV {confirm['variance_cv']:.3f} | 覆盖 {confirm['coverage_seeds']}/5 | "
        f"命中 {confirm['hit_model']:.3f} vs 边际 {confirm['hit_marginal']:.3f}"
    )
    print(
        f"[门槛] gap ≥ {gap_threshold:+.3f}（pilot×{GAP_RATIO}） | CV ≥ {cv_threshold:.3f}"
        f"（pilot×{CV_RATIO}） | ρ ≥ {RHO_MIN} | 覆盖 ≥ {COVERAGE_MIN}"
    )
    print(f"[E1] P1 {'过' if passed else '不过'}")
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

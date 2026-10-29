"""D9（对比约束）的判据对账：从三个臂的 JSON 直接算两条数值判据。

预注册见 `sweep_results/rstdp_spiking_contrastive_prereg.md`，判据是：

1. 末次评测中位 ≥ **30** 步（对照 9.5）；
2. 末期极角网格上 `mean Δ` 的极差 ≥ **0.5** 个计数；
3. 极角斜率 > 0 的种子数 ≥ **7/10**（`slope = P(left|θ=+0.2) − P(left|θ=−0.2)`）。

用法::

    python scripts/analyze_d9.py A.json B.json C.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

__all__ = ["analyse_arm", "main"]

LAST_MIN = 30.0
SPREAD_MIN = 0.5
SLOPE_SEEDS_MIN = 7


def _probe_values(probe: dict, prefix: str) -> list[float]:
    return [probe[f"{prefix}_a{index}"] for index in range(21)]


def analyse_arm(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_seed = payload["per_seed"]
    lasts = [seed["last_steps"] for seed in per_seed if seed.get("last_steps") is not None]
    spreads = []
    slopes = []
    for seed in per_seed:
        probes = seed.get("probes") or []
        if not probes:
            continue
        last = probes[-1]
        deltas = _probe_values(last, "delta_mean")
        spreads.append(max(deltas) - min(deltas))
        pl_left = _probe_values(last, "p_left")
        slopes.append(pl_left[-1] - pl_left[0])
    return {
        "path": str(path),
        "gamma": payload.get("config", {}).get("contrastive_gamma"),
        "seeds": len(per_seed),
        "last_median": statistics.median(lasts) if lasts else None,
        "last_values": lasts,
        "spread_median": statistics.median(spreads) if spreads else None,
        "spread_values": spreads,
        "slope_positive_seeds": sum(1 for value in slopes if value > 0),
        "slope_values": slopes,
    }


def _verdict(row: dict) -> dict:
    last_ok = row["last_median"] is not None and row["last_median"] >= LAST_MIN
    spread_ok = row["spread_median"] is not None and row["spread_median"] >= SPREAD_MIN
    slope_ok = row["slope_positive_seeds"] >= SLOPE_SEEDS_MIN
    return {
        "criterion_1_last": last_ok,
        "criterion_2_spread": spread_ok,
        "criterion_3_slope": slope_ok,
        "ready": bool(last_ok and spread_ok and slope_ok),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D9 判据对账")
    parser.add_argument("arms", nargs="+", type=Path, help="每个臂的 summary JSON")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = []
    for path in args.arms:
        row = analyse_arm(path)
        row["verdict"] = _verdict(row)
        rows.append(row)
        verdict = row["verdict"]
        print(
            f"{path.name}: gamma={row['gamma']} "
            f"末次中位 {row['last_median']:.2f} ({'过' if verdict['criterion_1_last'] else '未过'})、"
            f"Δ 极差中位 {row['spread_median']:.2f} ({'过' if verdict['criterion_2_spread'] else '未过'})、"
            f"slope>0 {row['slope_positive_seeds']}/{row['seeds']} "
            f"({'过' if verdict['criterion_3_slope'] else '未过'}) → "
            f"{'ready' if verdict['ready'] else 'not ready'}"
        )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"arms": rows}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

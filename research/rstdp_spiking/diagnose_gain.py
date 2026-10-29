"""D11：计数读出对差驱动的增益标定（判据见 sweep_results/rstdp_spiking_readout_gain_prereg.md）。

不训练。用**构造的**公共列 `c` 与差方向 `δ`（相对幅度 ρ 已知），在 21 点极角网格上量
`Δ = counts_left − counts_right` 的极差，扫 threshold / n_refractory / baseline_period / steps。
回答的是：计数读出要多大的差驱动才分得开——不是「训练能不能学会」。
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path

import torch

from research.common.provenance import collect
from research.components.cartpole_encoding import encode_state, population_centers
from research.rstdp_spiking.neurons import LIFLayer, LIFState
from research.rstdp_spiking.spike_encoding import spike_encode

OUT = Path(__file__).resolve().parents[2] / "sweep_results" / "rstdp_spiking_readout_gain"

ANGLES = [round(-0.2 + 0.02 * index, 2) for index in range(21)]
REPEATS = 20
N_FEATURES = 64
SIGMA = 1.0
HALF_WIDTHS = (2.4, 3.0, 0.21, 3.0)

#: 目标相对幅度（ρ = 差驱动极差 ÷ 平均列驱动）。0 是空对照。
RHOS = (0.0, 0.05, 0.10)

#: 标定网格（预注册写死）。
THRESHOLDS = (1.0, 0.5, 2.0)
REFRACTORIES = (2, 0, 1)
BASELINE_PERIODS = (5, None, 10)
STEPS = (25, 50)

#: 判据阈值（改这里必须改预注册文件）。
GAIN_MIN = 2.0
NULL_MAX = 0.5
#: 反退化带：窗口 25 步时每列平均计数应落在 2–15；窗口加倍时按比例放宽。
COUNT_BAND = (2.0, 15.0)


def _grid_states() -> list[torch.Tensor]:
    states = []
    for angle in ANGLES:
        state = torch.zeros(4)
        state[2] = angle
        states.append(state)
    return states


def _common_column(n_features: int, seed: int) -> torch.Tensor:
    """与 ``LIFLayer`` 初始化同式的随机列，再按列 L1 归一化到 1（与 Actor 同口径）。"""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    bound = 1.0 / math.sqrt(n_features)
    column = torch.empty(n_features).uniform_(-bound, bound, generator=generator)
    return column / column.abs().sum().clamp(min=1e-8)


def _scale_delta(
    column: torch.Tensor,
    direction: torch.Tensor,
    rho: float,
    centers: torch.Tensor,
) -> torch.Tensor:
    """把差方向缩放到「网格上的相对幅度 = ρ」：`spread(δ·p) / mean|c·p| = ρ`。"""
    if rho == 0.0:
        return torch.zeros_like(direction)
    projections = []
    drives = []
    for state in _grid_states():
        features = encode_state(state, centers, SIGMA)
        projections.append(float(direction @ features))
        drives.append(abs(float(column @ features)))
    spread = max(projections) - min(projections)
    if spread <= 1e-12:
        raise ValueError("差方向在这组状态上的投影为常数，无法标定相对幅度。")
    return direction * (rho * statistics.fmean(drives) / spread)


@torch.no_grad()
def _measure(
    weight: torch.Tensor,
    centers: torch.Tensor,
    *,
    threshold: float,
    n_refractory: int,
    baseline_period: int | None,
    steps: int,
    seed: int,
) -> dict:
    layer = LIFLayer(
        N_FEATURES,
        2,
        tau=20.0,
        threshold=threshold,
        n_refractory=n_refractory,
        baseline_period=baseline_period,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    layer.weight.data.copy_(weight)
    deltas = []
    column_counts = [0.0, 0.0]
    empty_windows = 0
    for angle in ANGLES:
        state = torch.zeros(4)
        state[2] = angle
        features = encode_state(state, centers, SIGMA)
        generator = torch.Generator(device="cpu").manual_seed(
            seed * 100_000 + 2 * 100 + round(angle * 100)
        )
        per_column = [0.0, 0.0]
        for _ in range(REPEATS):
            pre = spike_encode(features, steps, generator=generator).unsqueeze(1)
            spike_state = LIFState.zeros(1, 2, device=pre.device)
            total = torch.zeros(2)
            for t in range(steps):
                spikes, spike_state = layer.step(pre[t], spike_state)
                total += spikes.squeeze(0)
            if float(total.sum()) == 0.0:
                empty_windows += 1
            per_column[0] += float(total[0])
            per_column[1] += float(total[1])
        column_counts[0] += per_column[0] / REPEATS
        column_counts[1] += per_column[1] / REPEATS
        deltas.append((per_column[0] - per_column[1]) / REPEATS)
    n_points = len(ANGLES) * REPEATS
    band = (
        COUNT_BAND[0] * (steps / 25.0),
        COUNT_BAND[1] * (steps / 25.0),
    )
    means = [value / len(ANGLES) for value in column_counts]
    return {
        "delta_range": max(deltas) - min(deltas),
        "column_counts": means,
        "in_band": all(band[0] <= value <= band[1] for value in means),
        "zero_fraction": empty_windows / n_points,
        "constant_points": sum(1 for value in deltas if abs(value) < 1e-9),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D11 计数读出增益标定")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    seeds = _parse_seeds(args.seeds)

    rows = []
    for seed in seeds:
        centers = population_centers(
            N_FEATURES,
            half_widths=HALF_WIDTHS,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        )
        column = _common_column(N_FEATURES, seed)
        raw_direction = torch.empty(N_FEATURES).normal_(
            generator=torch.Generator(device="cpu").manual_seed(seed * 7919 + 3)
        )
        raw_direction = raw_direction / raw_direction.norm()
        for threshold, n_refractory, baseline_period, steps in itertools.product(
            THRESHOLDS, REFRACTORIES, BASELINE_PERIODS, STEPS
        ):
            per_rho = {}
            for rho in RHOS:
                delta = _scale_delta(column, raw_direction, rho, centers)
                weight = torch.stack([column + delta / 2, column - delta / 2], dim=1)
                per_rho[str(rho)] = _measure(
                    weight,
                    centers,
                    threshold=threshold,
                    n_refractory=n_refractory,
                    baseline_period=baseline_period,
                    steps=steps,
                    seed=seed,
                )
            signal = per_rho["0.1"]
            usable = (
                signal["delta_range"] >= GAIN_MIN
                and signal["in_band"]
                and per_rho["0.0"]["delta_range"] <= NULL_MAX
            )
            rows.append(
                {
                    "seed": seed,
                    "threshold": threshold,
                    "n_refractory": n_refractory,
                    "baseline_period": baseline_period,
                    "steps": steps,
                    "rho": per_rho,
                    "usable": usable,
                }
            )
        usable_here = sum(1 for row in rows if row["seed"] == seed and row["usable"])
        print(f"seed {seed}: {usable_here} usable configs", flush=True)

    # 逐配置汇总：5 个种子全可用才算这个配置可用（不挑种子）。
    by_config: dict[tuple, dict] = {}
    for row in rows:
        key = (row["threshold"], row["n_refractory"], row["baseline_period"], row["steps"])
        entry = by_config.setdefault(key, {"seeds_usable": 0, "ranges": [], "counts": []})
        entry["seeds_usable"] += int(row["usable"])
        entry["ranges"].append(row["rho"]["0.1"]["delta_range"])
        entry["counts"].append(row["rho"]["0.1"]["column_counts"])
    configs = [
        {
            "threshold": key[0],
            "n_refractory": key[1],
            "baseline_period": key[2],
            "steps": key[3],
            "seeds_usable": value["seeds_usable"],
            "median_delta_range_rho_0.10": statistics.median(value["ranges"]),
        }
        for key, value in by_config.items()
    ]
    configs.sort(
        key=lambda item: (
            -item["seeds_usable"],
            item["steps"],
            -item["median_delta_range_rho_0.10"],
        )
    )
    recommended = next((config for config in configs if config["seeds_usable"] == len(seeds)), None)
    summary = {
        "kind": "calibration: count-readout gain vs relative differential drive",
        "n_features": N_FEATURES,
        "angles": ANGLES,
        "rhos": list(RHOS),
        "thresholds": list(THRESHOLDS),
        "refractories": list(REFRACTORIES),
        "baseline_periods": list(BASELINE_PERIODS),
        "steps": list(STEPS),
        "criteria": {
            "gain_min": GAIN_MIN,
            "null_max": NULL_MAX,
            "count_band": list(COUNT_BAND),
        },
        "configs": configs,
        "recommended": recommended,
        "readout_gain_calibrated": recommended is not None,
        "seeds": rows,
        "repro": collect(
            "W3 脉冲化：D11 计数读出增益标定",
            seeds=",".join(str(seed) for seed in seeds),
            notes=[
                "构造差方向（ρ=0 空对照 / 0.05 / 0.10），21 点网格，每点 20 窗口",
                "网格：threshold × n_refractory × baseline_period × steps",
            ],
            device=torch.device("cpu"),
        ).render_block(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "calibrated": summary["readout_gain_calibrated"],
                "recommended": recommended,
                "top": configs[:5],
            },
            ensure_ascii=False,
        )
    )
    return 0


def _parse_seeds(text: str) -> list[int]:
    text = text.strip()
    if "-" in text and "," not in text:
        low, high = text.split("-")
        return list(range(int(low), int(high) + 1))
    return [int(part) for part in text.split(",") if part]


if __name__ == "__main__":
    raise SystemExit(main())

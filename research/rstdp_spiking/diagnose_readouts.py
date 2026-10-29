"""D13：换读出——计数不行，低通率与亚阈值膜电位行不行（判据见 rstdp_spiking_readouts_prereg.md）。

构造与网格沿用 D12；**同一次窗口模拟**里同时算四种读出，保证可比：

* `count`：窗内脉冲个数；
* `rate`：窗内低通发放率（0.9 衰减），取窗末值；
* `potential_mean`：窗内膜电位均值；
* `potential_final`：窗末膜电位。

可分辨度 `S = 网格上 mean Δ 的极差 ÷ 单窗口 Δ 标准差`（与单位无关）。
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from research.common.provenance import collect
from research.components.cartpole_encoding import encode_state, population_centers
from research.rstdp_spiking.diagnose_gain import (
    ANGLES,
    HALF_WIDTHS,
    N_FEATURES,
    SIGMA,
    _grid_states,
)
from research.rstdp_spiking.diagnose_gain_scale import _build, _patterns
from research.rstdp_spiking.neurons import LIFLayer, LIFState
from research.rstdp_spiking.spike_encoding import spike_encode

OUT = Path(__file__).resolve().parents[2] / "sweep_results" / "rstdp_spiking_readouts"

#: 预注册网格（与 D12 同构）。
SIGN_MODES = ("signed", "aligned")
WEIGHT_NORMS = (1.0, 3.0)
LAMBDAS = (0.0, 0.05, 0.10, 0.20, 0.30)

#: 现状档。
THRESHOLD = 1.0
N_REFRACTORY = 2
BASELINE_PERIOD = 5
STEPS = 25
REPEATS = 20
RATE_DECAY = 0.9

#: 判据阈值（改这里必须改预注册文件）。
DECISION_LAMBDA = 0.10
S_MIN = 2.0
NULL_MAX = 0.5
READOUTS = ("count", "rate", "potential_mean", "potential_final")


@torch.no_grad()
def _simulate(
    weight: torch.Tensor,
    features: torch.Tensor,
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    """跑一个窗口，返回四种读出（各 `(2,)`）。"""
    layer = LIFLayer(
        N_FEATURES,
        2,
        tau=20.0,
        threshold=THRESHOLD,
        n_refractory=N_REFRACTORY,
        baseline_period=BASELINE_PERIOD,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    layer.weight.data.copy_(weight)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pre = spike_encode(features, STEPS, generator=generator).unsqueeze(1)
    state = LIFState.zeros(1, 2, device=pre.device)
    counts = torch.zeros(2)
    running = torch.zeros(2)
    potential_sum = torch.zeros(2)
    for step in range(STEPS):
        spikes, state = layer.step(pre[step], state)
        spikes = spikes.squeeze(0)
        counts += spikes
        running = RATE_DECAY * running + spikes
        potential_sum += state.v.squeeze(0)
    return {
        "count": counts,
        "rate": running,
        "potential_mean": potential_sum / STEPS,
        "potential_final": state.v.squeeze(0).clone(),
    }


@torch.no_grad()
def _measure_config(
    weight: torch.Tensor,
    centers: torch.Tensor,
    *,
    seed: int,
) -> dict[str, dict]:
    """21 点网格上逐点采 20 窗口，按读出汇总差值与可分辨度。"""
    per_readout = {name: {"means": [], "sds": []} for name in READOUTS}
    for state_tensor in _grid_states():
        features = encode_state(state_tensor, centers, SIGMA)
        point_seed = seed * 100_000 + 2 * 100 + round(float(state_tensor[2]) * 100)
        windows = {name: [] for name in READOUTS}
        for repeat in range(REPEATS):
            readout = _simulate(weight, features, seed=point_seed + repeat)
            for name in READOUTS:
                value = readout[name]
                windows[name].append(float(value[0] - value[1]))
        for name in READOUTS:
            values = windows[name]
            mean = statistics.fmean(values)
            sd = statistics.pstdev(values)
            per_readout[name]["means"].append(mean)
            per_readout[name]["sds"].append(sd)
    result = {}
    for name in READOUTS:
        means = per_readout[name]["means"]
        spread = max(means) - min(means)
        sd = statistics.median(per_readout[name]["sds"])
        # 整窗静默/常数化时 sd→0：可分辨度按 0 记，不给「除以零得无穷」的假绿。
        separability = spread / sd if sd > 1e-9 else 0.0
        result[name] = {
            "spread": spread,
            "window_sd_median": sd,
            "separability": separability,
            "delta_means": means,
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D13 换读出诊断")
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
        signed_m, signed_d, aligned_m, aligned_d = _patterns(seed)
        for sign_mode in SIGN_MODES:
            common, differential = (
                (signed_m, signed_d) if sign_mode == "signed" else (aligned_m, aligned_d)
            )
            for weight_norm in WEIGHT_NORMS:
                per_lambda = {}
                for lam in LAMBDAS:
                    weight = _build(common, differential, weight_norm=weight_norm, lam=lam)
                    per_lambda[f"{lam:.2f}"] = _measure_config(weight, centers, seed=seed)
                usable = {}
                for name in READOUTS:
                    decision = per_lambda[f"{DECISION_LAMBDA:.2f}"][name]["separability"]
                    null = per_lambda["0.00"][name]["separability"]
                    usable[name] = decision >= S_MIN and null <= NULL_MAX
                rows.append(
                    {
                        "seed": seed,
                        "sign_mode": sign_mode,
                        "weight_norm": weight_norm,
                        "per_lambda": per_lambda,
                        "usable": usable,
                    }
                )
        counts = {
            name: sum(1 for row in rows if row["seed"] == seed and row["usable"][name])
            for name in READOUTS
        }
        print(f"seed {seed}: usable {counts}", flush=True)

    by_config: dict[tuple, dict] = {}
    for row in rows:
        key = (row["sign_mode"], row["weight_norm"])
        entry = by_config.setdefault(
            key, {"seeds_usable": dict.fromkeys(READOUTS, 0), "separabilities": {}}
        )
        for name in READOUTS:
            entry["seeds_usable"][name] += int(row["usable"][name])
            entry["separabilities"].setdefault(name, []).append(
                row["per_lambda"][f"{DECISION_LAMBDA:.2f}"][name]["separability"]
            )
    configs = [
        {
            "sign_mode": key[0],
            "weight_norm": key[1],
            "seeds_usable": value["seeds_usable"],
            "median_separability_lambda_0.10": {
                name: statistics.median(values) for name, values in value["separabilities"].items()
            },
        }
        for key, value in by_config.items()
    ]
    candidates = ("rate", "potential_mean", "potential_final")
    winners = [
        (name, config)
        for config in configs
        for name in candidates
        if config["seeds_usable"][name] >= max(4, len(seeds) - 1)
    ]
    winners.sort(key=lambda item: -item[1]["median_separability_lambda_0.10"][item[0]])
    summary = {
        "kind": "diagnostic: which readout can express a state-varying preference",
        "n_features": N_FEATURES,
        "angles": ANGLES,
        "fixed": {
            "threshold": THRESHOLD,
            "n_refractory": N_REFRACTORY,
            "baseline_period": BASELINE_PERIOD,
            "steps": STEPS,
            "repeats": REPEATS,
            "rate_decay": RATE_DECAY,
        },
        "grid": {
            "sign_modes": list(SIGN_MODES),
            "weight_norms": list(WEIGHT_NORMS),
            "lambdas": list(LAMBDAS),
        },
        "criteria": {
            "decision_lambda": DECISION_LAMBDA,
            "separability_min": S_MIN,
            "null_max": NULL_MAX,
        },
        "configs": configs,
        "winners": [
            {
                "readout": name,
                "sign_mode": config["sign_mode"],
                "weight_norm": config["weight_norm"],
            }
            for name, config in winners
        ],
        "alternative_readout_found": bool(winners),
        "seeds": rows,
        "repro": collect(
            "W3 脉冲化：D13 四种读出的可分辨度",
            seeds=",".join(str(seed) for seed in seeds),
            notes=[
                "构造同 D12：w = weight_norm·((1−λ)·m ± λ·d)；21 点网格、每点 20 窗口",
                "四种读出（count / rate / potential_mean / potential_final）共用同一次模拟",
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
                "found": summary["alternative_readout_found"],
                "winners": summary["winners"],
                "configs": configs,
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

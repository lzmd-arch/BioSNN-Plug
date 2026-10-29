"""D15：首发放潜伏期读出（判据见 rstdp_spiking_latency_prereg.md）。

构造与网格沿用 D12/D13；λ 取**真实幅度**（0.01–0.10，D10 量到训练后是 0.5–7.3%）。
潜伏期读出：窗内第一次发放的时刻；**任一列整窗未发放 ⇒ 该窗口删失**（不计入，比例必须报）。
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

OUT = Path(__file__).resolve().parents[2] / "sweep_results" / "rstdp_spiking_latency"

LAMBDAS = (0.0, 0.01, 0.03, 0.05, 0.10)
THRESHOLD = 1.0
N_REFRACTORY = 2
BASELINE_PERIOD = 5
WEIGHT_NORM = 1.0
STEPS = 25
WINDOWS = 32

#: 判据（改这里必须改预注册文件）。
DECISION_LAMBDA = 0.05
S_MIN = 2.0
NULL_MAX = 0.5
CENSORED_MAX = 0.20


@torch.no_grad()
def _window(
    weight: torch.Tensor,
    features: torch.Tensor,
    *,
    seed: int,
) -> tuple[float | None, float | None, torch.Tensor]:
    """跑一个窗口，返回（两列各自的首发放时刻或 None，两列计数）。"""
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
    first: list[float | None] = [None, None]
    for step in range(STEPS):
        spikes, state = layer.step(pre[step], state)
        spikes = spikes.squeeze(0)
        counts += spikes
        for column in (0, 1):
            if first[column] is None and float(spikes[column]) > 0:
                first[column] = float(step)
    return first[0], first[1], counts


@torch.no_grad()
def _measure(
    weight: torch.Tensor,
    centers: torch.Tensor,
    *,
    seed: int,
) -> dict:
    latency_means: list[float] = []
    latency_sds: list[float] = []
    count_means: list[float] = []
    count_sds: list[float] = []
    censored = 0
    total = 0
    column_counts = [0.0, 0.0]
    for state_tensor in _grid_states():
        features = encode_state(state_tensor, centers, SIGMA)
        point_seed = seed * 1_000_000 + round(float(state_tensor[2]) * 100)
        latencies: list[float] = []
        count_deltas: list[float] = []
        for index in range(WINDOWS):
            first_left, first_right, counts = _window(
                weight, features, seed=point_seed + index * 7919
            )
            total += 1
            count_deltas.append(float(counts[0] - counts[1]))
            column_counts[0] += float(counts[0])
            column_counts[1] += float(counts[1])
            if first_left is None or first_right is None:
                censored += 1
                continue
            latencies.append(first_left - first_right)
        if latencies:
            latency_means.append(statistics.fmean(latencies))
            latency_sds.append(statistics.pstdev(latencies))
        count_means.append(statistics.fmean(count_deltas))
        count_sds.append(statistics.pstdev(count_deltas))
    latency_spread = (max(latency_means) - min(latency_means)) if latency_means else 0.0
    latency_sd = statistics.median(latency_sds) if latency_sds else 0.0
    count_spread = max(count_means) - min(count_means)
    count_sd = statistics.median(count_sds) if count_sds else 0.0
    return {
        "latency_spread": latency_spread,
        "latency_sd": latency_sd,
        "latency_separability": latency_spread / latency_sd if latency_sd > 1e-9 else 0.0,
        "count_spread": count_spread,
        "count_sd": count_sd,
        "count_separability": count_spread / count_sd if count_sd > 1e-9 else 0.0,
        "censored_fraction": censored / total if total else 1.0,
        "column_counts_mean": [value / total for value in column_counts],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D15 首发放潜伏期读出")
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
        signed_m, signed_d, _, _ = _patterns(seed)
        per_lambda = {}
        for lam in LAMBDAS:
            weight = _build(signed_m, signed_d, weight_norm=WEIGHT_NORM, lam=lam)
            per_lambda[f"{lam:.2f}"] = _measure(weight, centers, seed=seed)
        decision = per_lambda[f"{DECISION_LAMBDA:.2f}"]
        null = per_lambda["0.00"]
        rows.append(
            {
                "seed": seed,
                "per_lambda": per_lambda,
                "latency_ok": decision["latency_separability"] >= S_MIN,
                "censored_ok": decision["censored_fraction"] <= CENSORED_MAX,
                "null_ok": null["latency_separability"] <= NULL_MAX
                and null["count_separability"] <= NULL_MAX,
            }
        )
        print(
            f"seed {seed}: latency S {decision['latency_separability']:.2f}, "
            f"count S {decision['count_separability']:.2f}, "
            f"censored {decision['censored_fraction']:.1%}",
            flush=True,
        )

    passes = sum(1 for row in rows if row["latency_ok"])
    censored_ok = all(row["censored_ok"] for row in rows)
    null_ok = all(row["null_ok"] for row in rows)
    viable = passes >= max(4, len(seeds) - 1) and censored_ok and null_ok
    summary = {
        "kind": "diagnostic: first-spike latency readout at realistic differential amplitudes",
        "n_features": N_FEATURES,
        "angles": ANGLES,
        "fixed": {
            "threshold": THRESHOLD,
            "n_refractory": N_REFRACTORY,
            "baseline_period": BASELINE_PERIOD,
            "weight_norm": WEIGHT_NORM,
            "steps": STEPS,
            "windows": WINDOWS,
            "sign_mode": "signed",
        },
        "lambdas": list(LAMBDAS),
        "criteria": {
            "decision_lambda": DECISION_LAMBDA,
            "separability_min": S_MIN,
            "null_max": NULL_MAX,
            "censored_max": CENSORED_MAX,
        },
        "seeds_latency_above": passes,
        "censored_ok": censored_ok,
        "null_ok": null_ok,
        "latency_readout_viable": viable,
        "seeds": rows,
        "repro": collect(
            "W3 脉冲化：D15 首发放潜伏期读出",
            seeds=",".join(str(seed) for seed in seeds),
            notes=[
                "构造同 D12（signed, weight_norm=1）；λ 取真实幅度 0/0.01/0.03/0.05/0.10",
                "读出：首发放时刻差（整窗未发放的窗口删失并报比例）；对照：硬计数",
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
                "viable": viable,
                "seeds_latency_above": passes,
                "censored_ok": censored_ok,
                "null_ok": null_ok,
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

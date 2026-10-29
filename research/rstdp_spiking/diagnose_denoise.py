"""D14：降噪的代价曲线（判据见 sweep_results/rstdp_spiking_denoise_prereg.md）。

同一构造探针（`w = weight_norm·((1−λ)·m ± λ·d)`），扫每决策平均窗口数 K、窗口步数 T、
编码方式（伯努利 vs 确定性）。所有臂**用同一把尺子**：`S_ref = 网格上 mean Δ 极差 ÷ sd_ref`，
`sd_ref` 是 λ=0.05、K=1、T=25、伯努利档的单窗口 Δ 标准差的中位——避免
「确定性编码自身 sd=0 ⇒ 除以零得无穷」的假绿。
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

OUT = Path(__file__).resolve().parents[2] / "sweep_results" / "rstdp_spiking_denoise"

LAMBDAS = (0.05, 0.10, 0.30)
KS = (1, 2, 4, 8)
STEPS = (25, 50)
ENCODINGS = ("bernoulli", "deterministic")

THRESHOLD = 1.0
N_REFRACTORY = 2
BASELINE_PERIOD = 5
WEIGHT_NORM = 1.0
#: 每个网格点采多少个独立窗口（再按 K 分组算决策）。
WINDOWS = 32

#: 判据（改这里必须改预注册文件）。
DECISION_LAMBDA = 0.30  # 修正 1：0.05 处确定性档 spread 恒为 0（量化吞信号）
S_REF_MIN = 2.0
NULL_MAX = 0.5


@torch.no_grad()
def _window_delta(
    weight: torch.Tensor,
    features: torch.Tensor,
    *,
    steps: int,
    encoding: str,
    seed: int,
) -> float:
    """跑一个窗口，返回 `counts_left − counts_right`。"""
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
    if encoding == "bernoulli":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        pre = spike_encode(features, steps, generator=generator).unsqueeze(1)
    elif encoding == "deterministic":
        pre = features.unsqueeze(0).unsqueeze(1).repeat(steps, 1, 1)
    else:
        raise ValueError(f"未知 encoding：{encoding!r}。")
    state = LIFState.zeros(1, 2, device=pre.device)
    counts = torch.zeros(2)
    for step in range(steps):
        spikes, state = layer.step(pre[step], state)
        counts += spikes.squeeze(0)
    return float(counts[0] - counts[1])


def _decision_stats(values: list[float], k: int) -> tuple[float, float]:
    """把窗口序列按 K 分组求平均，返回（决策均值, 决策间标准差）。"""
    groups = [values[index : index + k] for index in range(0, len(values) - k + 1, k)]
    means = [statistics.fmean(group) for group in groups]
    return statistics.fmean(means), statistics.pstdev(means)


@torch.no_grad()
def _measure(
    weight: torch.Tensor,
    centers: torch.Tensor,
    *,
    steps: int,
    encoding: str,
    seed: int,
) -> dict:
    """一个（T, 编码）档：每点采 WINDOWS 个窗口，返回各 K 的决策均值与标准差。"""
    per_k = {k: {"means": [], "sds": []} for k in KS}
    for state_tensor in _grid_states():
        features = encode_state(state_tensor, centers, SIGMA)
        point_seed = seed * 1_000_000 + steps * 1_000 + round(float(state_tensor[2]) * 100)
        values = [
            _window_delta(
                weight,
                features,
                steps=steps,
                encoding=encoding,
                seed=point_seed + index * 7919,
            )
            for index in range(WINDOWS)
        ]
        for k in KS:
            mean, sd = _decision_stats(values, k)
            per_k[k]["means"].append(mean)
            per_k[k]["sds"].append(sd)
    return {
        k: {
            "spread": max(entry["means"]) - min(entry["means"]),
            "decision_sd_median": statistics.median(entry["sds"]),
        }
        for k, entry in per_k.items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D14 降噪代价曲线")
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
        per_lambda: dict[str, dict] = {}
        for lam in LAMBDAS:
            weight = _build(signed_m, signed_d, weight_norm=WEIGHT_NORM, lam=lam)
            per_config = {}
            for steps in STEPS:
                for encoding in ENCODINGS:
                    per_config[f"{steps}|{encoding}"] = _measure(
                        weight, centers, steps=steps, encoding=encoding, seed=seed
                    )
            per_lambda[f"{lam:.2f}"] = per_config
        # 尺子：λ=0.05、K=1、T=25、伯努利的单窗口 Δ 标准差（网格各点中位）。
        reference = per_lambda[f"{DECISION_LAMBDA:.2f}"]["25|bernoulli"][1]["decision_sd_median"]
        quantisation = {
            key: per_k[1]["spread"] for key, per_k in per_lambda[f"{DECISION_LAMBDA:.2f}"].items()
        }
        quantisation_low = {key: per_k[1]["spread"] for key, per_k in per_lambda["0.05"].items()}
        arms = []
        for key, per_k in per_lambda[f"{DECISION_LAMBDA:.2f}"].items():
            steps = int(key.split("|")[0])
            encoding = key.split("|")[1]
            for k in KS:
                spread = per_k[k]["spread"]
                decision_sd = per_k[k]["decision_sd_median"]
                noise_free = decision_sd <= 1e-12
                arms.append(
                    {
                        "lambda": DECISION_LAMBDA,
                        "steps": steps,
                        "encoding": encoding,
                        "k": k,
                        "cost": k * steps,
                        "spread": spread,
                        "decision_sd": decision_sd,
                        "s_ref": spread / reference if reference > 1e-12 else 0.0,
                        # 修正 2：判据用每臂自己的决策噪声；自身无噪声的档单列，不给 snr。
                        "snr": None if noise_free else spread / decision_sd,
                        "noise_free": noise_free,
                        "no_signal": spread <= 1e-12,
                    }
                )
        # 空对照：λ=0 的权重下，所有 K/T/编码里 spread ÷ reference 的最大值。
        null_weight = _build(signed_m, signed_d, weight_norm=WEIGHT_NORM, lam=0.0)
        null_worst = 0.0
        for steps in STEPS:
            for encoding in ENCODINGS:
                measured = _measure(null_weight, centers, steps=steps, encoding=encoding, seed=seed)
                null_worst = max(null_worst, max(entry["spread"] for entry in measured.values()))
        rows.append(
            {
                "seed": seed,
                "reference_sd": reference,
                "arms": arms,
                "null_s_ref": null_worst / reference if reference > 1e-12 else 0.0,
                "deterministic_k1_spread_at_decision_lambda": quantisation,
                "deterministic_k1_spread_at_lambda_0.05": quantisation_low,
            }
        )
        snrs = [arm["snr"] for arm in arms if arm["snr"] is not None]
        best = max(snrs) if snrs else float("nan")
        print(
            f"seed {seed}: reference sd {reference:.4f}, best snr {best:.2f}",
            flush=True,
        )

    combos: dict[tuple, list[dict]] = {}
    for row in rows:
        for arm in row["arms"]:
            combos.setdefault((arm["steps"], arm["encoding"], arm["k"]), []).append(arm)
    table = []
    for key, entries in combos.items():
        snrs = [entry["snr"] for entry in entries if entry["snr"] is not None]
        table.append(
            {
                "steps": key[0],
                "encoding": key[1],
                "k": key[2],
                "cost": key[0] * key[2],
                "median_snr": statistics.median(snrs) if snrs else None,
                "median_s_ref": statistics.median([entry["s_ref"] for entry in entries]),
                "median_spread": statistics.median([entry["spread"] for entry in entries]),
                "seeds_above": sum(1 for value in snrs if value >= S_REF_MIN),
                "noise_free_seeds": sum(1 for entry in entries if entry["noise_free"]),
            }
        )
    table.sort(key=lambda item: (item["cost"], item["encoding"] != "bernoulli", item["k"]))
    null_ok = all(row["null_s_ref"] <= NULL_MAX for row in rows)
    usable = (
        [
            item
            for item in table
            if item["median_snr"] is not None and item["seeds_above"] == len(seeds)
        ]
        if null_ok
        else []
    )
    deterministic_free = next(
        (
            item
            for item in table
            if item["encoding"] == "deterministic" and item["steps"] == 25 and item["k"] == 1
        ),
        None,
    )
    summary = {
        "kind": "diagnostic: cost of denoising a spiking decision",
        "n_features": N_FEATURES,
        "angles": ANGLES,
        "grid": {
            "lambdas": list(LAMBDAS),
            "decision_lambda": DECISION_LAMBDA,
            "prereg_amends": [
                "amend1: decision lambda 0.05 -> 0.30 (quantisation kills the signal at 0.05)"
            ],
            "ks": list(KS),
            "steps": list(STEPS),
            "encodings": list(ENCODINGS),
            "windows_per_point": WINDOWS,
            "fixed": {
                "threshold": THRESHOLD,
                "n_refractory": N_REFRACTORY,
                "baseline_period": BASELINE_PERIOD,
                "weight_norm": WEIGHT_NORM,
                "sign_mode": "signed",
            },
        },
        "criteria": {"snr_min": S_REF_MIN, "null_max": NULL_MAX},
        "prereg_amends_metric": "amend2: snr uses each arm's own decision sd",
        "combos": table,
        "recommended": usable[0] if usable else None,
        "denoise_path_found": bool(usable),
        "null_ok": null_ok,
        "deterministic_k1_t25": deterministic_free,
        "noise_is_encoding_only": bool(
            deterministic_free and deterministic_free["median_s_ref"] >= S_REF_MIN
        ),
        "seeds": rows,
        "repro": collect(
            "W3 脉冲化：D14 降噪代价曲线",
            seeds=",".join(str(seed) for seed in seeds),
            notes=[
                "构造同 D12（signed, weight_norm=1）；λ=0.05 判据、0.10 对照",
                "K∈{1,2,4,8} 窗平均 × T∈{25,50} × 编码∈{bernoulli, deterministic}；每点 32 窗",
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
                "found": summary["denoise_path_found"],
                "recommended": summary["recommended"],
                "noise_is_encoding_only": summary["noise_is_encoding_only"],
                "combos": table,
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

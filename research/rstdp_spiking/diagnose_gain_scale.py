"""D12：把净驱动抬起来，计数读出分得开吗（判据见 rstdp_spiking_readout_gain_scale_prereg.md）。

与 D11 同一构造探针，改为**固定 L1 预算 + 差分占比 λ**：

    w_{0,1} = weight_norm · ((1−λ)·m ± λ·d)

总驱动由 `weight_norm` 定，差方向占预算的比例由 λ 定——不再靠「把尺度越调越大」表达差方向
（那条路会被逐列 L1 归一化与平均绝对值分母一起吃掉，实测封顶在 0.08/0.03）。
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from research.common.provenance import collect
from research.components.cartpole_encoding import population_centers
from research.rstdp_spiking.diagnose_gain import (
    ANGLES,
    HALF_WIDTHS,
    N_FEATURES,
    _common_column,
    _measure,
)

OUT = Path(__file__).resolve().parents[2] / "sweep_results" / "rstdp_spiking_readout_gain_scale"

#: 预注册网格。
SIGN_MODES = ("signed", "aligned")
WEIGHT_NORMS = (1.0, 3.0, 5.0)
LAMBDAS = (0.0, 0.05, 0.10, 0.20, 0.30)

#: 与 D11 相同、本轮不扫的现状档。
THRESHOLD = 1.0
N_REFRACTORY = 2
BASELINE_PERIOD = 5
STEPS = 25

#: 判据阈值（改这里必须改预注册文件）。
DECISION_LAMBDA = 0.10
GAIN_MIN = 2.0
NULL_MAX = 0.5
COUNT_BAND = (2.0, 15.0)


def _patterns(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (signed_m, signed_d, aligned_m, aligned_d)——四者都按 L1 归一化到 1。"""
    column = _common_column(N_FEATURES, seed)
    direction = torch.empty(N_FEATURES).normal_(
        generator=torch.Generator(device="cpu").manual_seed(seed * 7919 + 3)
    )
    signed_m = column / column.abs().sum().clamp(min=1e-8)
    signed_d = direction / direction.abs().sum().clamp(min=1e-8)
    aligned_m = column.abs()
    aligned_m = aligned_m / aligned_m.abs().sum().clamp(min=1e-8)
    aligned_d = direction * column.sign()
    aligned_d = aligned_d / aligned_d.abs().sum().clamp(min=1e-8)
    return signed_m, signed_d, aligned_m, aligned_d


def _build(
    common: torch.Tensor,
    differential: torch.Tensor,
    *,
    weight_norm: float,
    lam: float,
) -> torch.Tensor:
    """固定预算构造：`weight_norm · ((1−λ)·m ± λ·d)`。"""
    return weight_norm * torch.stack(
        [(1.0 - lam) * common + lam * differential, (1.0 - lam) * common - lam * differential],
        dim=1,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D12 计数读出增益：预算与差分占比")
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
                    record = _measure(
                        weight,
                        centers,
                        threshold=THRESHOLD,
                        n_refractory=N_REFRACTORY,
                        baseline_period=BASELINE_PERIOD,
                        steps=STEPS,
                        seed=seed,
                    )
                    per_lambda[f"{lam:.2f}"] = record
                null_range = per_lambda["0.00"]["delta_range"]
                decision = per_lambda[f"{DECISION_LAMBDA:.2f}"]
                usable = (
                    decision["delta_range"] >= GAIN_MIN
                    and decision["in_band"]
                    and null_range <= NULL_MAX
                )
                first_lambda = next(
                    (
                        float(key)
                        for key, record in sorted(per_lambda.items())
                        if record["delta_range"] >= GAIN_MIN and record["in_band"]
                    ),
                    None,
                )
                rows.append(
                    {
                        "seed": seed,
                        "sign_mode": sign_mode,
                        "weight_norm": weight_norm,
                        "per_lambda": per_lambda,
                        "usable": usable,
                        "first_lambda_reaching_gain": first_lambda,
                    }
                )
        count = sum(1 for row in rows if row["seed"] == seed and row["usable"])
        print(f"seed {seed}: {count} usable configs", flush=True)

    by_config: dict[tuple, dict] = {}
    for row in rows:
        key = (row["sign_mode"], row["weight_norm"])
        entry = by_config.setdefault(
            key, {"seeds_usable": 0, "ranges": [], "first_lambdas": [], "counts": []}
        )
        entry["seeds_usable"] += int(row["usable"])
        entry["ranges"].append(row["per_lambda"][f"{DECISION_LAMBDA:.2f}"]["delta_range"])
        entry["counts"].append(row["per_lambda"][f"{DECISION_LAMBDA:.2f}"]["column_counts"])
        if row["first_lambda_reaching_gain"] is not None:
            entry["first_lambdas"].append(row["first_lambda_reaching_gain"])
    configs = [
        {
            "sign_mode": key[0],
            "weight_norm": key[1],
            "seeds_usable": value["seeds_usable"],
            "median_delta_range_lambda_0.10": statistics.median(value["ranges"]),
            "median_first_lambda_reaching_gain": (
                statistics.median(value["first_lambdas"]) if value["first_lambdas"] else None
            ),
            "seeds_reaching_gain": len(value["first_lambdas"]),
            "median_column_counts_lambda_0.10": [
                statistics.median([counts[index] for counts in value["counts"]]) for index in (0, 1)
            ],
        }
        for key, value in by_config.items()
    ]
    configs.sort(key=lambda item: (-item["seeds_usable"], item["weight_norm"]))
    recommended = next(
        (
            entry
            for entry in sorted(
                (item for item in configs if item["seeds_usable"] == len(seeds)),
                key=lambda item: (item["sign_mode"] != "signed", item["weight_norm"]),
            )
        ),
        None,
    )
    summary = {
        "kind": "calibration: count-readout gain vs per-column budget and differential share",
        "n_features": N_FEATURES,
        "angles": ANGLES,
        "grid": {
            "sign_modes": list(SIGN_MODES),
            "weight_norms": list(WEIGHT_NORMS),
            "lambdas": list(LAMBDAS),
            "fixed": {
                "threshold": THRESHOLD,
                "n_refractory": N_REFRACTORY,
                "baseline_period": BASELINE_PERIOD,
                "steps": STEPS,
            },
        },
        "criteria": {
            "decision_lambda": DECISION_LAMBDA,
            "gain_min": GAIN_MIN,
            "null_max": NULL_MAX,
            "count_band": list(COUNT_BAND),
        },
        "configs": configs,
        "recommended": recommended,
        "readout_gain_calibrated": recommended is not None,
        "seeds": rows,
        "repro": collect(
            "W3 脉冲化：D12 计数读出增益（预算与差分占比）",
            seeds=",".join(str(seed) for seed in seeds),
            notes=[
                "构造 w = weight_norm·((1−λ)·m ± λ·d)，21 点网格、每点 20 窗口",
                "网格：sign_mode × weight_norm × λ(0/0.05/0.10/0.20/0.30)",
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

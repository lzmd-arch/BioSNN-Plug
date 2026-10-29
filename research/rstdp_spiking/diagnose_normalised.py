"""D18：按轴归一化编码后重跑 D4/D8/D9 的核心探针（判据见 rstdp_spiking_normalised_prereg.md）。

先做**校准步**（不训练）：用 D17 的指标在 σ' ∈ {0.2, 0.35, 0.5} 里选一个让极角可分辨的值；
再用同一套 D6-B 超参训练 10 个种子，测极角 slope、末期 Δ 极差与末次评测。
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from research.common.provenance import collect
from research.components.cartpole_encoding import (
    CARTPOLE_HALF_WIDTHS,
    encode_state,
    population_centers,
)
from research.rstdp_spiking.cartpole import train

OUT = Path(__file__).resolve().parents[2] / "sweep_results" / "rstdp_spiking_normalised"

ANGLES = [round(-0.2 + 0.02 * index, 2) for index in range(21)]
PROBE_ANGLES = tuple((angle, f"a{index}") for index, angle in enumerate(ANGLES))
N_FEATURES = 64
SIGMA_CANDIDATES = (0.2, 0.35, 0.5)
UNIT_WIDTHS = (1.0, 1.0, 1.0, 1.0)
CALIBRATION_CENTERS = 5

#: 校准判据（改这里必须改预注册文件与修正 1）。
#: 目标：让 θ 的敏感度**匹配 D17 里三条已经工作正常的轴**（rel 中位 1.34），而不是"低就好"。
REL_TARGET = 1.34
REL_MIN = 0.30
COS_MAX = 0.95

#: 判据（改这里必须改预注册文件）。
SLOPE_SEEDS_MIN = 7
SPREAD_MEDIAN_MIN = 0.5
LAST_MEDIAN_MIN = 30.0


def _theta_sensitivity(sigma: float, centers_seed: int) -> tuple[float, float]:
    """归一化编码在 θ 上的（相对灵敏度, 端到端余弦）。"""
    centers = population_centers(
        N_FEATURES,
        half_widths=UNIT_WIDTHS,
        generator=torch.Generator(device="cpu").manual_seed(centers_seed),
    )
    half = CARTPOLE_HALF_WIDTHS[2]
    features = []
    for point in range(21):
        value = -half + (2 * half) * point / 20
        state = torch.tensor([0.0, 0.0, value / half, 0.0])
        features.append(encode_state(state, centers, sigma))
    middle = features[10]
    mass = float(middle.norm(p=1))
    rel = max(float((value - middle).norm(p=1)) for value in features) / max(mass, 1e-12)
    cosine = float(
        (features[0] @ features[-1]) / (features[0].norm() * features[-1].norm()).clamp(min=1e-12)
    )
    return rel, cosine


def calibrate() -> tuple[float | None, list[dict]]:
    """选 σ'：`rel` 最接近 `REL_TARGET`（匹配 D17 里正常工作的三条轴），且余弦足够小。"""
    rows = []
    for sigma in SIGMA_CANDIDATES:
        values = [_theta_sensitivity(sigma, seed) for seed in range(CALIBRATION_CENTERS)]
        rels = [value[0] for value in values]
        cosines = [value[1] for value in values]
        rel_median = statistics.median(rels)
        cosine_median = statistics.median(cosines)
        ok = (
            rel_median >= REL_MIN
            and cosine_median <= COS_MAX
            and all(cosine <= COS_MAX for cosine in cosines)
        )
        rows.append(
            {
                "sigma": sigma,
                "rel_median": rel_median,
                "cosine_median": cosine_median,
                "distance_to_target": abs(rel_median - REL_TARGET),
                "passes": ok,
            }
        )
    candidates = [row for row in rows if row["passes"]]
    if not candidates:
        return None, rows
    chosen = min(candidates, key=lambda row: row["distance_to_target"])
    return float(chosen["sigma"]), rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D18 归一化编码重跑核心探针")
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    seeds = _parse_seeds(args.seeds)

    sigma, calibration = calibrate()
    for row in calibration:
        print(
            f"校准 σ'={row['sigma']}: rel {row['rel_median']:.3f} cos {row['cosine_median']:.4f} "
            f"{'通过' if row['passes'] else '不过'}",
            flush=True,
        )
    if sigma is None:
        summary = {
            "kind": "D18 normalised encoding",
            "calibration": calibration,
            "sigma": None,
            "error": "没有任何候选 σ' 满足校准判据——按预注册不训练、不改判据",
        }
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        print(json.dumps({"error": summary["error"]}, ensure_ascii=False))
        return 1

    scale = torch.tensor(CARTPOLE_HALF_WIDTHS)
    rows = []
    for seed in seeds:
        result = train(
            seed=seed,
            episodes=args.episodes,
            eval_every=25,
            eval_episodes=10,
            n_features=N_FEATURES,
            sigma=sigma,
            steps=25,
            learning_rate=1e-3,
            baseline_decay=0.05,
            critic_kind="pop",
            critic_rate=0.1,
            gamma=0.99,
            pop_post="gradient",
            warmup=10,
            device=torch.device("cpu"),
            trace_rule="count-aligned",
            spiking_temperature=True,
            logit_scale_start=0.1,
            logit_scale_end=0.02,
            probe_every=25,
            probe_angles=PROBE_ANGLES,
            keep_actor=True,
            actor_after="final",
            scale=scale,
            half_widths=UNIT_WIDTHS,
        )
        last_probe = result.probes[-1] if result.probes else None
        if last_probe is None:
            raise SystemExit("探针为空：probe_every/probe_angles 没生效。")
        deltas = [last_probe[f"delta_mean_a{index}"] for index in range(len(ANGLES))]
        left = [last_probe[f"p_left_a{index}"] for index in range(len(ANGLES))]
        slope = left[-1] - left[0]
        spread = max(deltas) - min(deltas)
        rows.append(
            {
                "seed": seed,
                "slope": slope,
                "delta_spread": spread,
                "last_steps": result.eval_scores[-1]["mean_steps"] if result.eval_scores else None,
                "best_steps": result.best_steps,
            }
        )
        print(
            f"seed {seed}: slope {slope:+.2f}，Δ 极差 {spread:.2f}，"
            f"末次 {rows[-1]['last_steps']}，best {result.best_steps:.1f}",
            flush=True,
        )

    slopes = sum(1 for row in rows if row["slope"] > 0)
    spreads = [row["delta_spread"] for row in rows]
    lasts = [row["last_steps"] for row in rows if row["last_steps"] is not None]
    summary = {
        "kind": "D18: normalised per-axis encoding, re-running the D4/D8/D9 probes",
        "calibration": calibration,
        "sigma": sigma,
        "scale": list(CARTPOLE_HALF_WIDTHS),
        "half_widths": list(UNIT_WIDTHS),
        "seeds": rows,
        "slope_positive_seeds": slopes,
        "delta_spread_median": statistics.median(spreads),
        "last_median": statistics.median(lasts) if lasts else None,
        "criteria": {
            "slope_positive_min": SLOPE_SEEDS_MIN,
            "spread_median_min": SPREAD_MEDIAN_MIN,
            "last_median_min": LAST_MEDIAN_MIN,
        },
        "d18_ready": (
            slopes >= SLOPE_SEEDS_MIN
            and statistics.median(spreads) >= SPREAD_MEDIAN_MIN
            and bool(lasts)
            and statistics.median(lasts) >= LAST_MEDIAN_MIN
        ),
        "repro": collect(
            "W3 脉冲化：D18 归一化编码重跑核心探针",
            seeds=",".join(str(seed) for seed in seeds),
            notes=[
                f"状态按物理半宽归一化，中心在单位立方体上，σ'={sigma}",
                "其余与 D6-B 完全同参（count-aligned、温度 0.1→0.02、800 回合）",
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
                "sigma": sigma,
                "slope_positive": slopes,
                "spread_median": round(summary["delta_spread_median"], 2),
                "last_median": summary["last_median"],
                "ready": summary["d18_ready"],
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

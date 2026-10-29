"""D17：编码器能不能分辨极角（判据见 rstdp_spiking_encoder_prereg.md）。

对每个状态维度各取 21 点网格，量三种特征空间指标（相对灵敏度、端到端余弦、单步灵敏度），
比较脉冲线（`CARTPOLE_HALF_WIDTHS`、σ=1）与速率线（半宽 1.5、σ=0.5）两种编码。
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

OUT = Path(__file__).resolve().parents[2] / "sweep_results" / "rstdp_spiking_encoder"

#: 四个维度的物理范围（CartPole）。
AXES = (
    ("x", 0, 2.4),
    ("xdot", 1, 3.0),
    ("theta", 2, 0.2095),
    ("thetadot", 3, 3.0),
)
POINTS = 21

#: 两种编码的参数（与两条线各自的历史写法一致）。
SPIKING = {"half_widths": CARTPOLE_HALF_WIDTHS, "sigma": 1.0, "label": "spiking"}
RATE = {"half_widths": (1.5, 1.5, 1.5, 1.5), "sigma": 0.5, "label": "rate"}

#: 判据（改这里必须改预注册文件）。
SPIKING_REL_MAX = 0.10
SPIKING_COS_MIN = 0.99
RATE_REL_MIN = 0.50
RATE_COS_MAX = 0.90
CENTERS_MIN = 4


def _grid(index: int, half: float) -> list[torch.Tensor]:
    states = []
    for point in range(POINTS):
        value = -half + (2 * half) * point / (POINTS - 1)
        state = torch.zeros(4)
        state[index] = value
        states.append(state)
    return states


def _metrics(features: list[torch.Tensor]) -> dict:
    middle = features[len(features) // 2]
    mass = float(middle.norm(p=1))
    rel = max(float((value - middle).norm(p=1)) for value in features) / max(mass, 1e-12)
    cosine = float(
        (features[0] @ features[-1]) / (features[0].norm() * features[-1].norm()).clamp(min=1e-12)
    )
    steps = [
        float((features[index + 1] - features[index]).norm(p=1)) / max(mass, 1e-12)
        for index in range(len(features) - 1)
    ]
    return {"rel": rel, "cosine": cosine, "max_step": max(steps)}


def _measure(centers_seed: int, spec: dict) -> dict[str, dict]:
    centers = population_centers(
        64,
        half_widths=spec["half_widths"],
        generator=torch.Generator(device="cpu").manual_seed(centers_seed),
    )
    result = {}
    for name, index, half in AXES:
        features = [encode_state(state, centers, spec["sigma"]) for state in _grid(index, half)]
        result[name] = _metrics(features)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D17 编码器对极角的分辨力")
    parser.add_argument("--centers", default="0-4")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    seeds = _parse_seeds(args.centers)

    rows = []
    for seed in seeds:
        spiking = _measure(seed, SPIKING)
        rate = _measure(seed, RATE)
        spiking_ok = (
            spiking["theta"]["rel"] < SPIKING_REL_MAX
            and spiking["theta"]["cosine"] > SPIKING_COS_MIN
        )
        rate_ok = rate["theta"]["rel"] >= RATE_REL_MIN or rate["theta"]["cosine"] <= RATE_COS_MAX
        rows.append(
            {
                "centers_seed": seed,
                "spiking": spiking,
                "rate": rate,
                "spiking_ok": spiking_ok,
                "rate_ok": rate_ok,
            }
        )
        print(
            f"中心 {seed}: 脉冲 θ rel {spiking['theta']['rel']:.4f} cos {spiking['theta']['cosine']:.5f}；"
            f"速率 θ rel {rate['theta']['rel']:.3f} cos {rate['theta']['cosine']:.3f}",
            flush=True,
        )

    spiking_hits = sum(1 for row in rows if row["spiking_ok"])
    rate_hits = sum(1 for row in rows if row["rate_ok"])
    summary = {
        "kind": "diagnostic: encoder sensitivity to the angle dimension",
        "axes": [{"name": name, "index": index, "half": half} for name, index, half in AXES],
        "points": POINTS,
        "spiking": {"half_widths": list(SPIKING["half_widths"]), "sigma": SPIKING["sigma"]},
        "rate": {"half_widths": list(RATE["half_widths"]), "sigma": RATE["sigma"]},
        "criteria": {
            "spiking_rel_max": SPIKING_REL_MAX,
            "spiking_cos_min": SPIKING_COS_MIN,
            "rate_rel_min": RATE_REL_MIN,
            "rate_cos_max": RATE_COS_MAX,
            "centers_required": CENTERS_MIN,
        },
        "spiking_hits": spiking_hits,
        "rate_hits": rate_hits,
        "encoder_blind_to_angle": spiking_hits >= CENTERS_MIN and rate_hits >= CENTERS_MIN,
        "median_spiking_theta_rel": statistics.median(
            row["spiking"]["theta"]["rel"] for row in rows
        ),
        "median_spiking_theta_cosine": statistics.median(
            row["spiking"]["theta"]["cosine"] for row in rows
        ),
        "median_rate_theta_rel": statistics.median(row["rate"]["theta"]["rel"] for row in rows),
        "median_rate_theta_cosine": statistics.median(
            row["rate"]["theta"]["cosine"] for row in rows
        ),
        "rows": rows,
        "repro": collect(
            "W3 脉冲化：D17 编码器对极角的分辨力",
            seeds=",".join(str(seed) for seed in seeds),
            notes=[
                "每维 21 点、覆盖物理范围；其余维度为 0",
                "脉冲线 CARTPOLE_HALF_WIDTHS + σ=1 对照速率线半宽 1.5 + σ=0.5",
            ],
        ).render_block(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "blind": summary["encoder_blind_to_angle"],
                "spiking_hits": spiking_hits,
                "rate_hits": rate_hits,
                "spiking_theta_rel_median": round(summary["median_spiking_theta_rel"], 4),
                "rate_theta_rel_median": round(summary["median_rate_theta_rel"], 4),
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

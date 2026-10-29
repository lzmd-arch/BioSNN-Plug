"""D10：读出层的差方向去哪了（判据见 sweep_results/rstdp_spiking_direction_prereg.md）。

D8 量到末期贪心动作恒定、计数差 Δ 的极差只有 0–0.4。本轮把「为什么」拆成两个可测的量：

* **A 相对差驱动** `r(s) = (in_0 − in_1) / ((in_0 + in_1)/2)`，`in_j(s) = Σ_i W[i,j]·p_i(s)`；
  它只看权重几何与状态编码，不含 LIF 动力学；
* **C 差分方向放大**：把 `w_j` 相对列均值放大 `m` 倍后重新 L1 归一化，看计数差能否重新
  随状态变——若能，方向可表达；若不能，输出层对输入不敏感。

B（输入整体缩放下的传递曲线）与 D（随机同范数方向）只作交叉验证，不进判据。
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from research.common.provenance import collect
from research.components.cartpole_encoding import encode_state
from research.rstdp_spiking.cartpole import train

OUT = Path(__file__).resolve().parents[2] / "sweep_results" / "rstdp_spiking_direction"

ANGLES = [round(-0.2 + 0.02 * index, 2) for index in range(21)]
AXES = (
    ("theta", 2, [-0.2, -0.1, 0.0, 0.1, 0.2]),
    ("thetadot", 3, [-1.0, -0.5, 0.0, 0.5, 1.0]),
    ("x", 0, [-1.0, -0.5, 0.0, 0.5, 1.0]),
    ("xdot", 1, [-1.0, -0.5, 0.0, 0.5, 1.0]),
)
GAINS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
AMPLIFICATIONS = (1, 2, 4, 8, 16)
REPEATS = 20

#: 预注册阈值（改这里必须改预注册文件）。
A_SPREAD_MIN = 0.05
B_RANGE_MIN = 1.0
C_RANGE_MIN = 0.5


def _state_vector(index: int, value: float) -> torch.Tensor:
    state = torch.zeros(4)
    state[index] = value
    return state


def _grid_states() -> list[torch.Tensor]:
    return [_state_vector(2, angle) for angle in ANGLES]


def _relative_drive(features: torch.Tensor, weight: torch.Tensor) -> dict:
    """`r(s)` 及其分子分母（同一套权重）。

    分母取两列驱动的**平均绝对值**：突触有正有负，两列驱动的均值可以是负的，
    用带符号均值做分母会让 `r` 在过零点附近发散或恒为 0（冒烟时踩到过这一条）。
    """
    per_feature = (features.unsqueeze(0) @ weight).squeeze(0)
    drive_0 = float(per_feature[0])
    drive_1 = float(per_feature[1])
    scale = (abs(drive_0) + abs(drive_1)) / 2
    difference = drive_0 - drive_1
    ratio = difference / scale if scale > 1e-12 else 0.0
    return {"r": ratio, "mean_drive": scale, "difference": difference}


def _seed_for(seed: int, index: int, value: float) -> int:
    """与 D8 的 `_scan` 同式：同一个网格点在同一随机流下比较。"""
    return seed * 100_000 + index * 100 + int(value * 100)


def _delta_frozen(actor, features: torch.Tensor, *, seed: int) -> tuple[float, float]:
    """逐窗口重采（不额外跑一遍 ``counts``），与 D8 的窗口取样方式一致。"""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    deltas = []
    for _ in range(REPEATS):
        _, post = actor.window_spikes(features, generator=generator)
        counts = post.sum(dim=0).squeeze(0)
        deltas.append(float(counts[0] - counts[1]))
    mean = sum(deltas) / REPEATS
    sd = (sum((value - mean) ** 2 for value in deltas) / REPEATS) ** 0.5
    return mean, sd


@torch.no_grad()
def _measure_drive(actor, centers, sigma: float) -> dict:
    weight = actor.layer.weight.detach()
    columns = [weight[:, 0], weight[:, 1]]
    delta_w = (columns[0] - columns[1]).clone()
    column_mean = (columns[0] + columns[1]) / 2

    grid = []
    for angle, state in zip(ANGLES, _grid_states(), strict=True):
        features = encode_state(state, centers, sigma)
        record = _relative_drive(features, weight)
        record["angle"] = angle
        grid.append(record)
    axis_spreads = {}
    for axis, index, values in AXES:
        ratios = [
            _relative_drive(encode_state(_state_vector(index, value), centers, sigma), weight)["r"]
            for value in values
        ]
        axis_spreads[axis] = max(ratios) - min(ratios)
    return {
        "r_grid": [record["r"] for record in grid],
        "mean_drive_grid": [record["mean_drive"] for record in grid],
        "difference_grid": [record["difference"] for record in grid],
        "spread": max(record["r"] for record in grid) - min(record["r"] for record in grid),
        "axis_spread": axis_spreads,
        "delta_w_l2": float(delta_w.norm()),
        "column_l2": [float(columns[0].norm()), float(columns[1].norm())],
        "delta_w_relative": float(delta_w.norm() / (column_mean.norm() + 1e-12)),
    }


@torch.no_grad()
def _measure_transfer(actor, centers, sigma: float, *, seed: int) -> dict:
    """B：中性状态、整体缩放输入，看每列计数是否随输入变。"""
    neutral = torch.zeros(4)
    features = encode_state(neutral, centers, sigma)
    rows = []
    for gain in GAINS:
        scaled = (features * gain).clamp(max=1.0)
        generator = torch.Generator(device="cpu").manual_seed(
            seed * 100_000 + 900 + int(gain * 100)
        )
        counts = actor.counts(scaled, generator=generator, repeats=REPEATS)
        rows.append([float(counts[0]), float(counts[1])])
    # 传递曲线不是单调的：权重有正有负，输入放大后可能把膜电位压到阈值以下（冒烟时
    # 见过 0.25× 有 2.15 计数、≥0.75× 完全沉默）。所以只报动态范围，不假设单调。
    ranges = [
        max(row[column] for row in rows) - min(row[column] for row in rows) for column in (0, 1)
    ]
    return {
        "gains": list(GAINS),
        "counts": rows,
        "dynamic_range": min(ranges),
    }


@torch.no_grad()
def _measure_amplification(actor, centers, sigma: float, *, seed: int) -> dict:
    """C：放大两列相对均值的差方向，每列重新 L1 归一化后量网格 Δ 的极差。"""
    original = actor.layer.weight.detach().clone()
    column_l1 = original.abs().sum(dim=0)
    column_mean = original.mean(dim=1, keepdim=True)
    results = {}
    try:
        for factor in AMPLIFICATIONS:
            amplified = column_mean + factor * (original - column_mean)
            norms = amplified.abs().sum(dim=0, keepdim=True).clamp(min=1e-8)
            amplified = amplified * (column_l1.unsqueeze(0) / norms)
            actor.layer.weight.data.copy_(amplified)
            means = [
                _delta_frozen(
                    actor,
                    encode_state(state, centers, sigma),
                    seed=_seed_for(seed, 2, round(float(state[2]), 2)),
                )[0]
                for state in _grid_states()
            ]
            results[str(factor)] = {
                "range": max(means) - min(means),
                "mean": statistics.fmean(means),
                "delta_w_relative": float(
                    (amplified[:, 0] - amplified[:, 1]).norm()
                    / (amplified.mean(dim=1).norm() + 1e-12)
                ),
            }
    finally:
        actor.layer.weight.data.copy_(original)
    return results


@torch.no_grad()
def _measure_random_direction(actor, centers, sigma: float, *, seed: int) -> dict:
    """D：同范数的随机方向，看 r 的极差是否与训练出来的差方向同量级。"""
    weight = actor.layer.weight.detach()
    column_mean = weight.mean(dim=1, keepdim=True)
    delta_w = weight[:, 0] - weight[:, 1]
    generator = torch.Generator(device="cpu").manual_seed(seed * 7919 + 13)
    spreads = []
    for _ in range(5):
        direction = torch.randn(delta_w.shape, generator=generator)
        direction = direction * (delta_w.norm() / (direction.norm() + 1e-12))
        candidate = torch.cat(
            [column_mean + direction.unsqueeze(1) / 2, column_mean - direction.unsqueeze(1) / 2],
            dim=1,
        )
        ratios = [
            _relative_drive(encode_state(state, centers, sigma), candidate)["r"]
            for state in _grid_states()
        ]
        spreads.append(max(ratios) - min(ratios))
    return {"spreads": spreads, "median": statistics.median(spreads)}


def _verdict(drive: dict, amplification: dict, transfer: dict) -> dict:
    direction = drive["spread"] >= A_SPREAD_MIN
    expressible = amplification["16"]["range"] >= C_RANGE_MIN
    saturated = transfer["dynamic_range"] < B_RANGE_MIN
    if not direction:
        reason = "no_differential_direction" if not expressible else "other_unattributed"
    elif not expressible:
        reason = "lif_saturated"
    else:
        reason = "amplitude_too_small"
    return {
        "direction": direction,
        "expressible": expressible,
        "saturated": saturated,
        "reason": reason,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D10 读出差方向诊断")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    seeds = _parse_seeds(args.seeds)

    rows = []
    for seed in seeds:
        result = train(
            seed=seed,
            episodes=args.episodes,
            eval_every=25,
            eval_episodes=10,
            n_features=64,
            sigma=1.0,
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
            keep_actor=True,
            actor_after="final",
        )
        drive = _measure_drive(result.actor, result.centers, 1.0)
        transfer = _measure_transfer(result.actor, result.centers, 1.0, seed=seed)
        amplification = _measure_amplification(result.actor, result.centers, 1.0, seed=seed)
        random_control = _measure_random_direction(result.actor, result.centers, 1.0, seed=seed)
        verdict = _verdict(drive, amplification, transfer)
        rows.append(
            {
                "seed": seed,
                "drive": drive,
                "transfer": transfer,
                "amplification": amplification,
                "random_direction": random_control,
                "verdict": verdict,
                "best_steps": result.best_steps,
                "last_steps": result.eval_scores[-1]["mean_steps"] if result.eval_scores else None,
            }
        )
        print(
            f"seed {seed}: {verdict['reason']} "
            f"(A {drive['spread']:.3f}, C16 {amplification['16']['range']:.2f}, "
            f"B {transfer['dynamic_range']:.2f})",
            flush=True,
        )

    reasons = [row["verdict"]["reason"] for row in rows]
    counts = {reason: reasons.count(reason) for reason in set(reasons)}
    majority = [reason for reason, count in counts.items() if count >= 3]
    summary = {
        "kind": "diagnostic: where the readout's differential direction went",
        "arm": "D6-B (count-aligned, spiking_temperature 0.1->0.02)",
        "angles": ANGLES,
        "thresholds": {
            "A_spread_min": A_SPREAD_MIN,
            "B_range_min": B_RANGE_MIN,
            "C_range_min": C_RANGE_MIN,
        },
        "seeds": rows,
        "reason_counts": counts,
        "majority_reason": majority[0] if len(majority) == 1 else None,
        "repro": collect(
            "W3 脉冲化：D10 读出差方向",
            seeds=",".join(str(seed) for seed in seeds),
            notes=[
                "臂：D6-B（logit_scale 0.1→0.02，count-aligned）",
                "A 相对差驱动 / B 输入缩放传递曲线 / C 差方向放大（逐列 L1 归一化）/ D 随机方向对照",
            ],
            device=torch.device("cpu"),
        ).render_block(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps({"counts": counts, "majority": summary["majority_reason"]}, ensure_ascii=False)
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

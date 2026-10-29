"""D16：把差方向对「状态均值」正交化（判据见 rstdp_spiking_orthogonalise_prereg.md）。

诊断，不训练新策略：训练一次（D6-B），把最终的 `Δw = w_0 − w_1` 沿网格均值编码模式 `p̄`
的分量投影掉，看差分响应是否**跨零**、贪心动作是否开始随状态变（极角 slope）。
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

OUT = Path(__file__).resolve().parents[2] / "sweep_results" / "rstdp_spiking_orthogonalise"

ANGLES = [round(-0.2 + 0.02 * index, 2) for index in range(21)]
REPEATS = 20
SIGMA = 1.0

#: 判据（改这里必须改预注册文件）。
SLOPE_MIN = 0.20
SPREAD_RETENTION = 0.5
SEEDS_MIN = 4


def _grid_states() -> list[torch.Tensor]:
    states = []
    for angle in ANGLES:
        state = torch.zeros(4)
        state[2] = angle
        states.append(state)
    return states


@torch.no_grad()
def _orthogonalise(actor, centers) -> torch.Tensor:
    """把 `Δw` 沿网格均值模式 `p̄` 的分量投影掉，再重建两列并逐列归一化。"""
    weight = actor.layer.weight.detach()
    left = weight[:, 0]
    right = weight[:, 1]
    common = (left + right) / 2
    delta = left - right
    features = [encode_state(state, centers, SIGMA) for state in _grid_states()]
    mean_pattern = torch.stack(features).mean(dim=0)
    projection = float(delta @ mean_pattern) / float(mean_pattern @ mean_pattern)
    delta_prime = delta - projection * mean_pattern
    rebuilt = torch.stack([common + delta_prime / 2, common - delta_prime / 2], dim=1)
    column_l1 = weight.abs().sum(dim=0)
    norms = rebuilt.abs().sum(dim=0).clamp(min=1e-8)
    rebuilt = rebuilt * (column_l1 / norms).unsqueeze(0)
    actor.layer.weight.data.copy_(rebuilt)
    return mean_pattern


def _drive_signs(weight: torch.Tensor, centers) -> list[float]:
    drives = []
    for state in _grid_states():
        features = encode_state(state, centers, SIGMA)
        projected = features @ weight
        drives.append(float(projected[0] - projected[1]))
    return drives


@torch.no_grad()
def _counts_and_slope(actor, centers, *, seed: int) -> tuple[list[float], list[float]]:
    deltas = []
    lefts = []
    for index, state in enumerate(_grid_states()):
        features = encode_state(state, centers, SIGMA)
        values = []
        left_votes = 0
        for repeat in range(REPEATS):
            generator = torch.Generator(device="cpu").manual_seed(
                seed * 100_000 + 2 * 100 + round(ANGLES[index] * 100) + repeat
            )
            _, post = actor.window_spikes(features, generator=generator)
            counts = post.sum(dim=0).squeeze(0)
            values.append(float(counts[0] - counts[1]))
            left_votes += int(counts.argmax().item() == 0)
        deltas.append(statistics.fmean(values))
        lefts.append(left_votes / REPEATS)
    return deltas, lefts


def _crosses_zero(values: list[float]) -> bool:
    return min(values) < 0.0 < max(values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D16 差方向对状态均值正交化")
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
            sigma=SIGMA,
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
        actor = result.actor
        centers = result.centers

        before_drive = _drive_signs(actor.layer.weight.detach(), centers)
        before_counts, before_left = _counts_and_slope(actor, centers, seed=seed)
        before_spread = max(before_counts) - min(before_counts)
        before_slope = before_left[-1] - before_left[0]

        _orthogonalise(actor, centers)

        after_drive = _drive_signs(actor.layer.weight.detach(), centers)
        after_counts, after_left = _counts_and_slope(actor, centers, seed=seed)
        after_spread = max(after_counts) - min(after_counts)
        after_slope = after_left[-1] - after_left[0]

        rows.append(
            {
                "seed": seed,
                "before": {
                    "drive_crosses_zero": _crosses_zero(before_drive),
                    "drive": before_drive,
                    "spread": before_spread,
                    "slope": before_slope,
                },
                "after": {
                    "drive_crosses_zero": _crosses_zero(after_drive),
                    "drive": after_drive,
                    "spread": after_spread,
                    "slope": after_slope,
                },
                "retention": after_spread / before_spread if before_spread > 1e-9 else None,
            }
        )
        print(
            f"seed {seed}: 跨零 {_crosses_zero(before_drive)}→"
            f"{_crosses_zero(after_drive)}，Δ 极差 {before_spread:.2f}→{after_spread:.2f}，"
            f"slope {before_slope:+.2f}→{after_slope:+.2f}",
            flush=True,
        )

    crossing = sum(1 for row in rows if row["after"]["drive_crosses_zero"])
    slopes = sum(1 for row in rows if abs(row["after"]["slope"]) >= SLOPE_MIN)
    retention = sum(
        1 for row in rows if row["retention"] is not None and row["retention"] >= SPREAD_RETENTION
    )
    summary = {
        "kind": "diagnostic: orthogonalise the differential direction against the mean state pattern",
        "angles": ANGLES,
        "repeats": REPEATS,
        "criteria": {
            "slope_min": SLOPE_MIN,
            "spread_retention": SPREAD_RETENTION,
            "seeds_required": SEEDS_MIN,
        },
        "seeds_crossing_after": crossing,
        "seeds_slope_after": slopes,
        "seeds_retention_ok": retention,
        "orthogonalisation_helps": (
            crossing >= SEEDS_MIN and slopes >= SEEDS_MIN and retention >= SEEDS_MIN
        ),
        "seeds": rows,
        "repro": collect(
            "W3 脉冲化：D16 差方向正交化",
            seeds=",".join(str(seed) for seed in seeds),
            notes=[
                "臂 D6-B（logit_scale 0.1→0.02、count-aligned）；只改最终权重，不再训练",
                "干预：Δw ← Δw − (Δw·p̄/‖p̄‖²)·p̄，p̄ 为 21 点网格的编码均值",
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
                "crossing": crossing,
                "slope": slopes,
                "retention": retention,
                "helps": summary["orthogonalisation_helps"],
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

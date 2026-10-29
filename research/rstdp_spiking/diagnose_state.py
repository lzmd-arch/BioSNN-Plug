"""D4：策略的状态依赖性诊断（判据见 sweep_results/rstdp_spiking_state_dependence_prereg.md）。

**诊断，不是验收。** 训练一条 count-aligned 的脉冲 Actor（与历史臂同配置），冻结后用
**最终权重**在极角网格上测 ``P(动作=左)``，与同种子的未训练网络对照，看动作有没有随状态
变——这是「信用分配至少产生了状态依赖策略」的最低证据。

不训练新策略、不换规则、不调超参。
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
from research.rstdp_spiking.actor import SpikingActor
from research.rstdp_spiking.cartpole import train

OUT = Path(__file__).resolve().parents[2] / "sweep_results" / "rstdp_spiking_state_dependence"

#: 极角网格：两端用于算斜率，中间点只记录。
ANGLES = [round(-0.2 + 0.02 * index, 2) for index in range(21)]


def _build_net(n_features: int, seed: int) -> tuple[torch.Generator, torch.Tensor, SpikingActor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    centers = population_centers(
        n_features,
        half_widths=CARTPOLE_HALF_WIDTHS,
        generator=generator,
        device=torch.device("cpu"),
    )
    actor = SpikingActor(
        n_features,
        2,
        steps=25,
        learning_rate=1e-3,
        trace_rule="count-aligned",
        generator=generator,
        device=torch.device("cpu"),
    )
    return generator, centers, actor


def _policy_curve(
    actor: SpikingActor, centers: torch.Tensor, *, seed: int, repeats: int
) -> list[float]:
    """每个极角上 ``P(动作=左)`` 的估计（Bernoulli 编码 × 重复采样）。"""
    curve = []
    for index, angle in enumerate(ANGLES):
        state = torch.tensor([0.0, 0.0, angle, 0.0])
        features = encode_state(state, centers, sigma=1.0)
        generator = torch.Generator(device="cpu").manual_seed(seed * 10_000 + index)
        left = 0
        for _ in range(repeats):
            if actor.step(features, generator=generator, learn_trace=False) == 0:
                left += 1
        curve.append(left / repeats)
    return curve


def _slope(curve: list[float]) -> float:
    return curve[-1] - curve[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D4 策略状态依赖性诊断")
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--repeats", type=int, default=20)
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
            keep_actor=True,
            actor_after="final",
        )
        trained_curve = _policy_curve(result.actor, result.centers, seed=seed, repeats=args.repeats)

        _, centers, untrained = _build_net(64, seed)
        untrained_curve = _policy_curve(untrained, centers, seed=seed, repeats=args.repeats)
        rows.append(
            {
                "seed": seed,
                "trained_slope": _slope(trained_curve),
                "untrained_slope": _slope(untrained_curve),
                "trained_curve": trained_curve,
                "untrained_curve": untrained_curve,
                "best_steps": result.best_steps,
                "last_steps": (
                    result.eval_scores[-1]["mean_steps"] if result.eval_scores else None
                ),
            }
        )
        print(
            f"seed {seed}: trained slope {rows[-1]['trained_slope']:+.3f}, "
            f"untrained {rows[-1]['untrained_slope']:+.3f}",
            flush=True,
        )

    trained = [row["trained_slope"] for row in rows]
    untrained = [row["untrained_slope"] for row in rows]
    gate = {
        "sign": sum(1 for value in trained if value > 0) >= 8,
        "magnitude": statistics.median(trained) >= 0.10,
        "control": statistics.median(untrained) <= 0.05,
    }
    gate["state_dependence_detected"] = all(gate.values())
    summary = {
        "kind": "diagnostic: does the trained spiking actor depend on the pole angle",
        "angles": ANGLES,
        "seeds": rows,
        "median_trained_slope": statistics.median(trained),
        "median_untrained_slope": statistics.median(untrained),
        "gate": gate,
        "repro": collect(
            "W3 脉冲化：D4 状态依赖性诊断",
            seeds=",".join(str(seed) for seed in seeds),
            notes=[
                "训练配置与 count-aligned 臂相同；诊断用**最终**权重",
                "P(左) 由 20 次采样估计；随机流按 (seed, 角度) 冻结",
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
                "median_trained": summary["median_trained_slope"],
                "median_untrained": summary["median_untrained_slope"],
                "gate": gate,
            },
            ensure_ascii=False,
            indent=2,
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

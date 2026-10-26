"""「规则改变后重学快 3×」的执行入口（预注册见 ``sweep_results/relearn_3x_prereg.md``）。

**这是验收实验，不是调参。** 判据、种子、阈值与有效性门都在预注册里写死；
本模块只负责按顺序跑三组：

1. Phase A：在 R0（重力 9.8）上训练 10 个种子，必须达到掌握门；
2. Phase B warm：保留 Phase A 的完整 agent（Actor/Critic/基线）在 R1（重力 14.0）上重学；
3. Phase B scratch：同种子随机初始化，在 R1 上从头训练。

速度 = best-so-far 贪心评测首次 ≥ 阈值所需的回合数；比值 = scratch / warm。
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from research.common.provenance import collect
from research.rstdp.cartpole import _greedy_score, make_cartpole, run_trial

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "relearn_3x"

#: 预注册写死的常数。
GRAVITY_NEW = 14.0
THRESHOLD = 150.0
EVAL_EVERY = 10
EVAL_EPISODES = 5
MASTERY_MIN_MEDIAN = 200.0
MASTERY_MIN_SEED = 150.0
MIN_VALID_FRACTION = 0.8
MIN_SCRATCH_MEDIAN = 100.0
FINAL_TOLERANCE = 10.0
RATIO_TARGET = 3.0


def _parse_seeds(text: str) -> list[int]:
    if "-" in text:
        first, last = text.split("-", 1)
        return list(range(int(first), int(last) + 1))
    return [int(item) for item in text.split(",") if item.strip()]


def _first_reaching(curve: list[float], threshold: float, every: int) -> int | None:
    """best-so-far 首次 ≥ 阈值时的回合数；曲线是每 ``every`` 回合一个点。"""
    best = -float("inf")
    for index, value in enumerate(curve):
        best = max(best, value)
        if best >= threshold:
            return (index + 1) * every
    return None


def _initial_score(context, seed: int, gravity: float) -> float:
    """warm 臂在**训练前**的新规则上评测；独立环境，不推进训练环境的 RNG。"""
    env = make_cartpole(gravity)
    env.reset(seed=seed * 1000 + 7)
    try:
        return _greedy_score(
            env,
            context.agent,
            context.centers,
            context.encoding_sigma,
            torch.Generator().manual_seed(seed * 1000 + 7),
            EVAL_EPISODES,
        )
    finally:
        env.close()


def run(
    seeds: list[int], *, episodes: int, out: Path, gravity: float = GRAVITY_NEW
) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        # Phase A：旧规则，完整掌握门。
        result_a, context = run_trial(
            seed,
            episodes=episodes,
            greedy_eval_every=50,
            greedy_eval_episodes=3,
            device=torch.device("cpu"),
            return_context=True,
        )
        # Phase B warm：保留完整 agent 状态，在 R1 上重学。
        warm_initial = _initial_score(context, seed, gravity)
        result_warm = run_trial(
            seed,
            episodes=episodes,
            greedy_eval_every=EVAL_EVERY,
            greedy_eval_episodes=EVAL_EPISODES,
            device=torch.device("cpu"),
            gravity=GRAVITY_NEW,
            initial_context=context,
        )
        # Phase B scratch：同种子、随机初始化、同一套调度。
        result_scratch = run_trial(
            seed,
            episodes=episodes,
            greedy_eval_every=EVAL_EVERY,
            greedy_eval_episodes=EVAL_EPISODES,
            device=torch.device("cpu"),
            gravity=GRAVITY_NEW,
        )
        warm_time = _first_reaching(result_warm.curve, THRESHOLD, EVAL_EVERY)
        scratch_time = _first_reaching(result_scratch.curve, THRESHOLD, EVAL_EVERY)
        rows.append(
            {
                "seed": seed,
                "phase_a_final": result_a.mean_steps,
                "phase_a_peak": result_a.peak_mean_steps,
                "warm_initial": warm_initial,
                "warm_time": warm_time,
                "scratch_time": scratch_time,
                "warm_curve": result_warm.curve,
                "scratch_curve": result_scratch.curve,
                "warm_final": result_warm.mean_steps,
                "scratch_final": result_scratch.mean_steps,
                "warm_last_two": statistics.mean(result_warm.curve[-2:]),
                "scratch_last_two": statistics.mean(result_scratch.curve[-2:]),
                "ratio": (scratch_time / warm_time) if warm_time and scratch_time else None,
            }
        )
        print(
            f"seed {seed}: A {result_a.mean_steps:.1f} | warm {warm_initial:.1f}→"
            f"{warm_time} eps / {result_warm.mean_steps:.1f} | scratch {scratch_time} eps / "
            f"{result_scratch.mean_steps:.1f}",
            flush=True,
        )

    phase_a = [row["phase_a_final"] for row in rows]
    valid = [
        row
        for row in rows
        if row["phase_a_final"] >= MASTERY_MIN_SEED
        and row["warm_time"] is not None
        and row["scratch_time"] is not None
    ]
    ratios = [row["ratio"] for row in rows if row["ratio"] is not None]
    nonzero = [row for row in rows if row["warm_initial"] >= THRESHOLD]
    gate = {
        "phase_a_median": statistics.median(phase_a),
        "phase_a_mastery": statistics.median(phase_a) >= MASTERY_MIN_MEDIAN,
        "valid_pairs": len(valid),
        "nonzero_shot": len(nonzero),
        "scratch_reaches": sum(1 for row in rows if row["scratch_time"] is not None),
        "scratch_median_time": (
            statistics.median([row["scratch_time"] for row in rows if row["scratch_time"]])
            if any(row["scratch_time"] for row in rows)
            else None
        ),
        "ratio_median": statistics.median(ratios) if ratios else None,
        "warm_faster": sum(1 for row in valid if row["ratio"] and row["ratio"] > 1.0),
        "final_parity": all(
            row["warm_last_two"] >= row["scratch_last_two"] - FINAL_TOLERANCE for row in valid
        ),
    }
    passed = (
        gate["phase_a_mastery"]
        and len(valid) >= MIN_VALID_FRACTION * len(rows)
        and len(nonzero) <= 2
        and gate["scratch_reaches"] >= MIN_VALID_FRACTION * len(rows)
        and (gate["scratch_median_time"] or 0) >= MIN_SCRATCH_MEDIAN
        and gate["ratio_median"] is not None
        and gate["ratio_median"] >= RATIO_TARGET
        and gate["warm_faster"] >= 0.8 * len(rows)
        and gate["final_parity"]
    )
    summary = {
        "experiment": "规则改变后重学快 3×（角标：CartPole 重力 9.8 → 14.0）",
        "config": {
            "gravity_new": gravity,
            "threshold": THRESHOLD,
            "eval_every": EVAL_EVERY,
            "eval_episodes": EVAL_EPISODES,
            "episodes": episodes,
            "seeds": seeds,
        },
        "per_seed": rows,
        "gate": gate,
        "passed": passed,
        "repro": collect(
            "规则改变后重学快 3×",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "Phase A：R0 重力 9.8；Phase B：R1 重力 14.0",
                "warm 保留完整 agent 状态；scratch 同种子随机初始化",
                "速度 = best-so-far 贪心评测首次 ≥150 步的回合数",
            ],
            device=torch.device("cpu"),
        ).render_block(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    print(f"passed = {passed}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="规则改变后重学快 3×")
    parser.add_argument("--seeds", default="65-74")
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--gravity", type=float, default=GRAVITY_NEW)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    run(_parse_seeds(args.seeds), episodes=args.episodes, out=args.out, gravity=args.gravity)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

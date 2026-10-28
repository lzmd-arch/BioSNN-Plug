"""重学 3× v4：保留探索状态的实用重学（预注册见 sweep_results/relearn_3x_v4_prereg.md）。"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from research.common.provenance import collect
from research.rstdp.cartpole import run_trial
from research.rstdp.relearn import EVAL_EPISODES, EVAL_EVERY, _first_reaching, _initial_score

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "relearn_3x_v4"

GRAVITY_NEW = 20.0
THRESHOLD = 200.0
MASTERY_MEDIAN = 200.0
MASTERY_SEED = 150.0
RATIO_TARGET = 3.0
FINAL_TOLERANCE = 10.0
MIN_VALID_FRACTION = 0.8


def _parse_seeds(text: str) -> list[int]:
    if "-" in text:
        first, last = text.split("-", 1)
        return list(range(int(first), int(last) + 1))
    return [int(item) for item in text.split(",") if item.strip()]


def run(seeds: list[int], *, episodes: int, out: Path) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        result_a, context = run_trial(
            seed,
            episodes=episodes,
            greedy_eval_every=50,
            greedy_eval_episodes=3,
            device=torch.device("cpu"),
            return_context=True,
        )
        warm_initial = _initial_score(context, seed, GRAVITY_NEW, None)
        result_warm = run_trial(
            seed,
            episodes=episodes,
            greedy_eval_every=EVAL_EVERY,
            greedy_eval_episodes=EVAL_EPISODES,
            device=torch.device("cpu"),
            gravity=GRAVITY_NEW,
            initial_context=context,
            exploration_start=0.02,
            exploration_end=0.02,
            logit_scale_start=40.0,
            logit_scale_end=40.0,
            actor_lr_final_fraction=1.0,
        )
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
                "warm_initial": warm_initial,
                "warm_time": warm_time,
                "scratch_time": scratch_time,
                "warm_final": result_warm.mean_steps,
                "scratch_final": result_scratch.mean_steps,
                "warm_last_two": statistics.mean(result_warm.curve[-2:]),
                "scratch_last_two": statistics.mean(result_scratch.curve[-2:]),
                "ratio": (scratch_time / warm_time) if warm_time and scratch_time else None,
            }
        )
        print(
            f"seed {seed}: A {result_a.mean_steps:.1f} | warm {warm_initial:.1f}→"
            f"{warm_time} / {result_warm.mean_steps:.1f} | scratch {scratch_time} / "
            f"{result_scratch.mean_steps:.1f}",
            flush=True,
        )

    phase_a = [row["phase_a_final"] for row in rows]
    valid = [
        row
        for row in rows
        if row["phase_a_final"] >= MASTERY_SEED
        and row["warm_time"] is not None
        and row["scratch_time"] is not None
    ]
    ratios = [row["ratio"] for row in valid if row["ratio"] is not None]
    nonzero = [row for row in rows if row["warm_initial"] >= THRESHOLD]
    scratch_times = [row["scratch_time"] for row in rows if row["scratch_time"] is not None]
    gate = {
        "phase_a_median": statistics.median(phase_a),
        "phase_a_mastery": statistics.median(phase_a) >= MASTERY_MEDIAN,
        "valid_pairs": len(valid),
        "nonzero_shot": len(nonzero),
        "scratch_reaches": len(scratch_times),
        "scratch_median_time": statistics.median(scratch_times) if scratch_times else None,
        "ratio_median": statistics.median(ratios) if ratios else None,
        "warm_faster": sum(1 for row in valid if row["ratio"] and row["ratio"] > 1.0),
        "final_parity": all(
            row["warm_last_two"] >= row["scratch_last_two"] - FINAL_TOLERANCE for row in valid
        ),
    }
    gate["relearn_3x_v4_ready"] = bool(
        gate["phase_a_mastery"]
        and len(valid) >= MIN_VALID_FRACTION * len(rows)
        and len(nonzero) <= 2
        and gate["scratch_reaches"] >= MIN_VALID_FRACTION * len(rows)
        and (gate["scratch_median_time"] or 0) >= 100
        and gate["ratio_median"] is not None
        and gate["ratio_median"] >= RATIO_TARGET
        and gate["warm_faster"] >= MIN_VALID_FRACTION * len(rows)
        and gate["final_parity"]
    )
    summary = {
        "experiment": "重学 3× v4：保留探索状态的实用重学",
        "config": {
            "gravity_new": GRAVITY_NEW,
            "threshold": THRESHOLD,
            "episodes": episodes,
            "seeds": seeds,
        },
        "per_seed": rows,
        "gate": gate,
        "repro": collect(
            "重学 3× v4",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "Phase A R0 9.8；Phase B warm 保留探索状态（0.02/40），R1 20.0；scratch 默认调度",
                "速度 = best-so-far 首次 ≥200 的回合数",
                "原 v1-v3 与 3× 未达结论不变",
            ],
            device=torch.device("cpu"),
        ).render_block(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="重学 3× v4：保留探索状态")
    parser.add_argument("--seeds", default="103-112")
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    run(_parse_seeds(args.seeds), episodes=args.episodes, out=args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

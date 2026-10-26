"""多时间尺度工作记忆 v0 的执行入口（预注册见 sweep_results/working_memory_v0_prereg.md）。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.cognitive.working_memory import (
    N_DIMS,
    average_r2,
    encode_episode,
    fit_ridge,
    predict,
)
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "working_memory_v0"

MULTI_MIN = 0.50
FAST_A_MAX = 0.30
SLOW_B_MAX = 0.45


def _episodes(episodes: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = []
    targets = []
    for _ in range(episodes):
        cue_a = rng.normal(size=N_DIMS)
        cue_b = rng.normal(size=N_DIMS)
        states.append(encode_episode(cue_a, cue_b, rng=rng))
        targets.append(np.concatenate([cue_a, cue_b]))
    return np.stack(states), np.stack(targets)


def run(seeds: list[int], *, train_episodes: int, test_episodes: int, out: Path) -> dict:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        train_states, train_targets = _episodes(train_episodes, seed=seed)
        test_states, test_targets = _episodes(test_episodes, seed=seed + 10_000)
        arms = {
            "multi": slice(0, train_states.shape[1]),
            "fast": slice(0, N_DIMS),
            "slow": slice(train_states.shape[1] - N_DIMS, train_states.shape[1]),
        }
        row: dict[str, Any] = {"seed": seed}
        for arm, columns in arms.items():
            coefficients = fit_ridge(train_states[:, columns], train_targets)
            predicted = predict(test_states[:, columns], coefficients)
            row[f"{arm}_r2_a"] = average_r2(predicted[:, :N_DIMS], test_targets[:, :N_DIMS])
            row[f"{arm}_r2_b"] = average_r2(predicted[:, N_DIMS:], test_targets[:, N_DIMS:])
        rows.append(row)
        print(
            f"seed {seed}: multi A/B {row['multi_r2_a']:.3f}/{row['multi_r2_b']:.3f} | "
            f"fast {row['fast_r2_a']:.3f}/{row['fast_r2_b']:.3f} | "
            f"slow {row['slow_r2_a']:.3f}/{row['slow_r2_b']:.3f}",
            flush=True,
        )

    mean = lambda key: float(np.mean([row[key] for row in rows]))  # noqa: E731
    gate = {
        "multi_r2_a": mean("multi_r2_a"),
        "multi_r2_b": mean("multi_r2_b"),
        "fast_r2_a": mean("fast_r2_a"),
        "slow_r2_b": mean("slow_r2_b"),
        "multi_ok": mean("multi_r2_a") >= MULTI_MIN and mean("multi_r2_b") >= MULTI_MIN,
        "fast_a_ok": mean("fast_r2_a") < FAST_A_MAX,
        "slow_b_ok": mean("slow_r2_b") < SLOW_B_MAX,
    }
    gate["w0_ready"] = gate["multi_ok"] and gate["fast_a_ok"] and gate["slow_b_ok"]
    summary = {
        "experiment": "多时间尺度工作记忆 v0",
        "config": {
            "train_episodes": train_episodes,
            "test_episodes": test_episodes,
            "seeds": seeds,
        },
        "per_seed": rows,
        "gate": gate,
        "repro": collect(
            "多时间尺度工作记忆 v0",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "线索 A t=0–4、B t=20–24、测试 t=50；τ={2,5,10,20,50,100,200}",
                "线性读出 ridge 闭式；单尺度对照用同一读出",
            ],
            device=None,
        ).render_block(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="多时间尺度工作记忆 v0")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--train-episodes", type=int, default=2000)
    parser.add_argument("--test-episodes", type=int, default=500)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    run(
        parse_seeds(args.seeds),
        train_episodes=50 if args.smoke else args.train_episodes,
        test_episodes=20 if args.smoke else args.test_episodes,
        out=args.out,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

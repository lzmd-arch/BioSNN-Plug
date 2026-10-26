"""多时间尺度工作记忆 v1：三线索 + 单尺度不能全做（预注册见 sweep_results/working_memory_v1_prereg.md）。"""

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
    encode_timeline,
    fit_ridge,
    predict,
)
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "working_memory_v1"

CUE_WINDOWS = ((0, 4), (15, 19), (30, 34))
TEST_STEP = 60
MULTI_MIN = 0.50
FAST_A_MAX = 0.30


def _episodes(episodes: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = []
    targets = []
    for _ in range(episodes):
        cues = [rng.normal(size=N_DIMS) for _ in CUE_WINDOWS]
        states.append(
            encode_timeline(
                [(start, end, cue) for (start, end), cue in zip(CUE_WINDOWS, cues, strict=True)],
                rng=rng,
                test_step=TEST_STEP,
            )
        )
        targets.append(np.concatenate(cues))
    return np.stack(states), np.stack(targets)


def _r2(predicted: np.ndarray, targets: np.ndarray) -> list[float]:
    return [
        average_r2(
            predicted[:, index * N_DIMS : (index + 1) * N_DIMS],
            targets[:, index * N_DIMS : (index + 1) * N_DIMS],
        )
        for index in range(len(CUE_WINDOWS))
    ]


def run(seeds: list[int], *, train_episodes: int, test_episodes: int, out: Path) -> dict:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        train_states, train_targets = _episodes(train_episodes, seed=seed)
        test_states, test_targets = _episodes(test_episodes, seed=seed + 10_000)
        width = train_states.shape[1]
        arms = {
            "multi": slice(0, width),
            "fast": slice(0, N_DIMS),
            "mid": slice(3 * N_DIMS, 4 * N_DIMS),
            "slow": slice(width - N_DIMS, width),
        }
        row: dict[str, Any] = {"seed": seed}
        for arm, columns in arms.items():
            coefficients = fit_ridge(train_states[:, columns], train_targets)
            predicted = predict(test_states[:, columns], coefficients)
            scores = _r2(predicted, test_targets)
            for index, value in enumerate(scores):
                row[f"{arm}_r2_{'abc'[index]}"] = value
        rows.append(row)
        print(
            f"seed {seed}: multi {row['multi_r2_a']:.3f}/{row['multi_r2_b']:.3f}/{row['multi_r2_c']:.3f} | "
            f"fast {row['fast_r2_a']:.3f}/{row['fast_r2_b']:.3f}/{row['fast_r2_c']:.3f} | "
            f"mid {row['mid_r2_a']:.3f}/{row['mid_r2_b']:.3f}/{row['mid_r2_c']:.3f} | "
            f"slow {row['slow_r2_a']:.3f}/{row['slow_r2_b']:.3f}/{row['slow_r2_c']:.3f}",
            flush=True,
        )

    mean = lambda key: float(np.mean([row[key] for row in rows]))  # noqa: E731
    gate = {
        "multi_r2": [mean(f"multi_r2_{c}") for c in "abc"],
        "fast_r2": [mean(f"fast_r2_{c}") for c in "abc"],
        "mid_r2": [mean(f"mid_r2_{c}") for c in "abc"],
        "slow_r2": [mean(f"slow_r2_{c}") for c in "abc"],
    }
    gate["multi_ok"] = all(value >= MULTI_MIN for value in gate["multi_r2"])
    gate["fast_a_ok"] = gate["fast_r2"][0] < FAST_A_MAX
    gate["singles_fail"] = {
        arm: any(value < MULTI_MIN for value in gate[f"{arm}_r2"])
        for arm in ("fast", "mid", "slow")
    }
    gate["w1_ready"] = gate["multi_ok"] and gate["fast_a_ok"] and all(gate["singles_fail"].values())
    summary = {
        "experiment": "多时间尺度工作记忆 v1：三线索",
        "config": {
            "train_episodes": train_episodes,
            "test_episodes": test_episodes,
            "seeds": seeds,
            "cue_windows": CUE_WINDOWS,
            "test_step": TEST_STEP,
        },
        "per_seed": rows,
        "gate": gate,
        "repro": collect(
            "多时间尺度工作记忆 v1",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "三线索 A/B/C 分别在 t=0–4/15–19/30–34，测试 t=60",
                "τ=2..200；单尺度 fast/mid/slow 各只取一块 4 维",
                "ridge 闭式读出；判据要求每个单尺度至少漏掉一个线索",
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
    parser = argparse.ArgumentParser(description="多时间尺度工作记忆 v1")
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

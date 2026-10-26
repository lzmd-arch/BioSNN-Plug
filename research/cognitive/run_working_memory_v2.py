"""多时间尺度工作记忆 v2：局部 delta 读出（预注册见 sweep_results/working_memory_v2_prereg.md）。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.cognitive.run_working_memory_v1 import (
    CUE_WINDOWS,
    N_DIMS,
    TEST_STEP,
    _episodes,
    _r2,
)
from research.cognitive.working_memory import LinearDeltaReadout, fit_ridge, predict
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "working_memory_v2"

MULTI_MIN = 0.50
FAST_A_MAX = 0.30
RIDGE_GAP_MAX = 0.10


def run(
    seeds: list[int],
    *,
    train_episodes: int,
    test_episodes: int,
    out: Path,
    learning_rate: float = 0.01,
    epochs: int = 20,
    batch: int = 128,
) -> dict[str, Any]:
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
        rng = np.random.default_rng(seed)
        for arm, columns in arms.items():
            readout = LinearDeltaReadout(
                columns.stop - columns.start,
                train_targets.shape[1],
                learning_rate=learning_rate,
                seed=seed,
            )
            readout.fit(
                train_states[:, columns],
                train_targets,
                epochs=epochs,
                batch=batch,
                rng=rng,
            )
            predicted = readout.predict(test_states[:, columns])
            for index, value in enumerate(_r2(predicted, test_targets)):
                row[f"{arm}_r2_{'abc'[index]}"] = value
        coefficients = fit_ridge(train_states, train_targets)
        ridge_scores = _r2(predict(test_states, coefficients), test_targets)
        for index, value in enumerate(ridge_scores):
            row[f"ridge_r2_{'abc'[index]}"] = value
        rows.append(row)
        print(
            f"seed {seed}: delta {row['multi_r2_a']:.3f}/{row['multi_r2_b']:.3f}/"
            f"{row['multi_r2_c']:.3f} | ridge {row['ridge_r2_a']:.3f}/"
            f"{row['ridge_r2_b']:.3f}/{row['ridge_r2_c']:.3f}",
            flush=True,
        )

    mean = lambda key: float(np.mean([row[key] for row in rows]))  # noqa: E731
    gate = {
        "delta_multi_r2": [mean(f"multi_r2_{c}") for c in "abc"],
        "ridge_multi_r2": [mean(f"ridge_r2_{c}") for c in "abc"],
        "fast_r2": [mean(f"fast_r2_{c}") for c in "abc"],
        "mid_r2": [mean(f"mid_r2_{c}") for c in "abc"],
        "slow_r2": [mean(f"slow_r2_{c}") for c in "abc"],
    }
    gate["multi_ok"] = all(value >= MULTI_MIN for value in gate["delta_multi_r2"])
    gate["fast_a_ok"] = gate["fast_r2"][0] < FAST_A_MAX
    gate["singles_fail"] = {
        arm: any(value < MULTI_MIN for value in gate[f"{arm}_r2"])
        for arm in ("fast", "mid", "slow")
    }
    delta_mean = float(np.mean(gate["delta_multi_r2"]))
    ridge_mean = float(np.mean(gate["ridge_multi_r2"]))
    gate["delta_mean"] = delta_mean
    gate["ridge_mean"] = ridge_mean
    gate["no_degradation"] = delta_mean >= ridge_mean - RIDGE_GAP_MAX
    gate["w2_ready"] = (
        gate["multi_ok"]
        and gate["fast_a_ok"]
        and all(gate["singles_fail"].values())
        and gate["no_degradation"]
    )
    summary = {
        "experiment": "多时间尺度工作记忆 v2：局部 delta 读出",
        "config": {
            "train_episodes": train_episodes,
            "test_episodes": test_episodes,
            "seeds": seeds,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch": batch,
            "cue_windows": CUE_WINDOWS,
            "test_step": TEST_STEP,
        },
        "per_seed": rows,
        "gate": gate,
        "repro": collect(
            "多时间尺度工作记忆 v2",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "状态/任务/线索与 v1 相同；读出换成局部 delta（η=0.01, 20 epoch, B=128）",
                "ridge 只作不退化对照，不混入 delta 列",
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
    parser = argparse.ArgumentParser(description="多时间尺度工作记忆 v2")
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

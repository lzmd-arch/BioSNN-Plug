"""多时间尺度工作记忆 v4：RLS 先验对齐 ridge（预注册见 sweep_results/working_memory_v4_prereg.md）。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.cognitive.run_working_memory_v1 import N_DIMS, _episodes, _r2
from research.cognitive.working_memory import LinearRLSReadout, fit_ridge, predict
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "working_memory_v4"

MULTI_MIN = 0.50
FAST_A_MAX = 0.30
RIDGE_GAP_MAX = 0.10
EPOCHS = 20
DELTA = 1000.0


def run(
    seeds: list[int],
    *,
    train_episodes: int,
    test_episodes: int,
    out: Path,
    epochs: int = EPOCHS,
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
            readout = LinearRLSReadout(
                columns.stop - columns.start, train_targets.shape[1], delta=DELTA
            )
            readout.fit(train_states[:, columns], train_targets, epochs=epochs, rng=rng)
            predicted = readout.predict(test_states[:, columns])
            for index, value in enumerate(_r2(predicted, test_targets)):
                row[f"{arm}_r2_{'abc'[index]}"] = value
        coefficients = fit_ridge(train_states, train_targets)
        for index, value in enumerate(_r2(predict(test_states, coefficients), test_targets)):
            row[f"ridge_r2_{'abc'[index]}"] = value
        rows.append(row)
        print(
            f"seed {seed}: rls {row['multi_r2_a']:.3f}/{row['multi_r2_b']:.3f}/"
            f"{row['multi_r2_c']:.3f} | ridge {row['ridge_r2_a']:.3f}/"
            f"{row['ridge_r2_b']:.3f}/{row['ridge_r2_c']:.3f}",
            flush=True,
        )

    mean = lambda key: float(np.mean([row[key] for row in rows]))  # noqa: E731
    multi = [mean(f"multi_r2_{c}") for c in "abc"]
    fast = [mean(f"fast_r2_{c}") for c in "abc"]
    mid = [mean(f"mid_r2_{c}") for c in "abc"]
    slow = [mean(f"slow_r2_{c}") for c in "abc"]
    ridge = [mean(f"ridge_r2_{c}") for c in "abc"]
    multi_mean = float(np.mean(multi))
    ridge_mean = float(np.mean(ridge))
    gate = {
        "multi_r2": multi,
        "ridge_r2": ridge,
        "fast_r2": fast,
        "mid_r2": mid,
        "slow_r2": slow,
        "multi_mean": multi_mean,
        "ridge_mean": ridge_mean,
        "multi_ok": all(value >= MULTI_MIN for value in multi),
        "fast_a_ok": fast[0] < FAST_A_MAX,
        "singles_fail": {
            "fast": any(value < MULTI_MIN for value in fast),
            "mid": any(value < MULTI_MIN for value in mid),
            "slow": any(value < MULTI_MIN for value in slow),
        },
        "no_degradation": multi_mean >= ridge_mean - RIDGE_GAP_MAX,
    }
    gate["w4_ready"] = (
        gate["multi_ok"]
        and gate["fast_a_ok"]
        and all(gate["singles_fail"].values())
        and gate["no_degradation"]
    )
    summary = {
        "experiment": "多时间尺度工作记忆 v4：RLS 先验对齐",
        "config": {
            "train_episodes": train_episodes,
            "test_episodes": test_episodes,
            "seeds": seeds,
            "epochs": epochs,
            "rls_forgetting": 1.0,
            "rls_delta": DELTA,
        },
        "per_seed": rows,
        "gate": gate,
        "repro": collect(
            "多时间尺度工作记忆 v4",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "RLS δ=1000（=1/ridge λ=1e-3），逐样本 20 epoch；状态/任务/判据与 v3 相同",
                "ridge 只作不退化对照",
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
    parser = argparse.ArgumentParser(description="多时间尺度工作记忆 v4")
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

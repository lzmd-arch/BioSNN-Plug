"""多时间尺度工作记忆 v8：脉冲迹读出（预注册见 sweep_results/working_memory_v8_prereg.md）。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.cognitive.spiking_working_memory import (
    N_DIMS,
    TAUS,
    SpikingPopulationTimescale,
)
from research.cognitive.working_memory import LinearRLSReadout, average_r2, fit_ridge, predict
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "working_memory_v8"

MULTI_MIN = 0.50
FAST_A_MAX = 0.30
V4_MEAN = 0.9990
RIDGE_GAP_MAX = 0.10
EPOCHS = 20
POPULATION = 32
GAIN_RANGE = (0.5, 4.0)
MEMBRANE_WINDOW = 10
READOUT = "trace"


def _episodes(
    episodes: int, *, seed: int, beta: float
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    network = SpikingPopulationTimescale(
        beta=beta,
        seed=seed,
        population=POPULATION,
        gain_range=GAIN_RANGE,
        membrane_window=MEMBRANE_WINDOW,
        readout=READOUT,
    )
    states = []
    targets = []
    rates = []
    membrane_max = 0.0
    for _ in range(episodes):
        cue_a = rng.normal(size=N_DIMS)
        cue_b = rng.normal(size=N_DIMS)
        generator = torch.Generator().manual_seed(int(rng.integers(0, 2**31 - 1)))
        states.append(network.run(cue_a, cue_b, generator=generator))
        targets.append(np.concatenate([cue_a, cue_b]))
        rates.append(network.last_spike_rates)
        membrane_max = max(membrane_max, network.last_membrane_max)
    matrix = np.stack(states)
    diagnostics = {
        "min": float(matrix.min()),
        "max": float(matrix.max()),
        "mean": float(matrix.mean()),
        "std": float(matrix.std()),
        "near_threshold_frac": float(np.mean(matrix >= 0.95)),
        "spike_rates": np.mean(np.stack(rates), axis=0).tolist(),
        "spike_rate_mean": float(np.mean(np.stack(rates))),
        "membrane_max": float(membrane_max),
    }
    return matrix, np.stack(targets), diagnostics


def _r2(predicted: np.ndarray, targets: np.ndarray) -> list[float]:
    return [
        average_r2(
            predicted[:, i * N_DIMS : (i + 1) * N_DIMS], targets[:, i * N_DIMS : (i + 1) * N_DIMS]
        )
        for i in range(2)
    ]


def _run_seed(
    seed: int, *, train_episodes: int, test_episodes: int, epochs: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch.set_num_threads(1)
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for arm, beta in (("lif", 0.0), ("alif", 0.1)):
        train_states, train_targets, train_diag = _episodes(train_episodes, seed=seed, beta=beta)
        test_states, test_targets, test_diag = _episodes(
            test_episodes, seed=seed + 10_000, beta=beta
        )
        diagnostics[f"{seed}-{arm}"] = {"train": train_diag, "test": test_diag}
        arms = {
            "multi": slice(0, train_states.shape[1]),
            "fast": slice(0, N_DIMS),
            "mid": slice(3 * N_DIMS, 4 * N_DIMS),
            "slow": slice(6 * N_DIMS, 7 * N_DIMS),
        }
        row: dict[str, Any] = {"seed": seed, "arm": arm}
        rng = np.random.default_rng(seed)
        for name, columns in arms.items():
            readout = LinearRLSReadout(
                columns.stop - columns.start, train_targets.shape[1], delta=1000.0
            )
            readout.fit(
                train_states[:, columns],
                train_targets,
                epochs=epochs,
                rng=rng,
            )
            scores = _r2(readout.predict(test_states[:, columns]), test_targets)
            for index, value in enumerate(scores):
                row[f"{name}_r2_{'ab'[index]}"] = value
        coefficients = fit_ridge(train_states, train_targets)
        ridge_scores = _r2(predict(test_states, coefficients), test_targets)
        for index, value in enumerate(ridge_scores):
            row[f"ridge_r2_{'ab'[index]}"] = value
        rows.append(row)
        print(
            f"seed {seed} {arm}: multi {row['multi_r2_a']:.3f}/{row['multi_r2_b']:.3f} "
            f"| fast {row['fast_r2_a']:.3f}/{row['fast_r2_b']:.3f} | "
            f"ridge {row['ridge_r2_a']:.3f}/{row['ridge_r2_b']:.3f} "
            f"| spikes {train_diag['spike_rate_mean']:.4f}",
            flush=True,
        )
    return rows, diagnostics


def _run_seed_job(job: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return _run_seed(
        int(job["seed"]),
        train_episodes=int(job["train_episodes"]),
        test_episodes=int(job["test_episodes"]),
        epochs=int(job["epochs"]),
    )


def run(
    seeds: list[int],
    *,
    train_episodes: int,
    test_episodes: int,
    out: Path,
    epochs: int = EPOCHS,
    workers: int = 1,
) -> dict[str, Any]:
    started = time.perf_counter()
    jobs = [
        {
            "seed": seed,
            "train_episodes": train_episodes,
            "test_episodes": test_episodes,
            "epochs": epochs,
        }
        for seed in seeds
    ]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_run_seed_job, jobs))
    else:
        results = [_run_seed_job(job) for job in jobs]
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for seed_rows, seed_diagnostics in results:
        rows.extend(seed_rows)
        diagnostics.update(seed_diagnostics)

    mean = lambda key, arm: float(np.mean([row[key] for row in rows if row["arm"] == arm]))  # noqa: E731
    gate: dict[str, Any] = {}
    for arm in ("lif", "alif"):
        multi = [mean(f"multi_r2_{c}", arm) for c in "ab"]
        fast = [mean(f"fast_r2_{c}", arm) for c in "ab"]
        mid = [mean(f"mid_r2_{c}", arm) for c in "ab"]
        slow = [mean(f"slow_r2_{c}", arm) for c in "ab"]
        ridge = [mean(f"ridge_r2_{c}", arm) for c in "ab"]
        multi_mean = float(np.mean(multi))
        ridge_mean = float(np.mean(ridge))
        gate[arm] = {
            "multi_r2": multi,
            "fast_r2": fast,
            "mid_r2": mid,
            "slow_r2": slow,
            "ridge_r2": ridge,
            "multi_mean": multi_mean,
            "ridge_mean": ridge_mean,
            "multi_ok": all(value >= MULTI_MIN for value in multi),
            "fast_a_ok": fast[0] < FAST_A_MAX,
            "singles_fail": {
                "fast": any(value < MULTI_MIN for value in fast),
                "mid": any(value < MULTI_MIN for value in mid),
                "slow": any(value < MULTI_MIN for value in slow),
            },
            "no_spiking_degradation": multi_mean >= V4_MEAN - RIDGE_GAP_MAX,
        }
        gate[arm]["ready"] = (
            gate[arm]["multi_ok"]
            and gate[arm]["fast_a_ok"]
            and all(gate[arm]["singles_fail"].values())
            and gate[arm]["no_spiking_degradation"]
        )
        spike_rates = [
            diagnostics[key]["test"]["spike_rate_mean"]
            for key in diagnostics
            if key.endswith(f"-{arm}")
        ]
        gate[arm]["spike_rate_mean"] = float(np.mean(spike_rates)) if spike_rates else 0.0
        gate[arm]["spiking_ok"] = gate[arm]["spike_rate_mean"] > 0.0
    gate["v7_ready"] = (gate["lif"]["ready"] or gate["alif"]["ready"]) and all(
        gate[arm]["spiking_ok"] for arm in ("lif", "alif")
    )
    summary = {
        "experiment": "多时间尺度工作记忆 v8：脉冲迹读出",
        "config": {
            "train_episodes": train_episodes,
            "test_episodes": test_episodes,
            "seeds": seeds,
            "epochs": epochs,
            "taus": TAUS,
            "beta_alif": 0.1,
            "population": POPULATION,
            "gain_range": GAIN_RANGE,
            "membrane_window": MEMBRANE_WINDOW,
            "readout": READOUT,
        },
        "per_seed": rows,
        "diagnostics": diagnostics,
        "gate": gate,
        "repro": collect(
            "多时间尺度工作记忆 v8",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "每个 τ、每个维度 32 个独立传入通道驱动 ALIF 单元；无循环权重",
                "输入增益对数间隔 0.5-4.0；读出为最后 10 步群体膜电位均值 + 局部 RLS",
                "任务/两臂/判据与 v5/v6 相同；population 是传入群体，不是循环连接",
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
    parser = argparse.ArgumentParser(description="多时间尺度工作记忆 v8")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--train-episodes", type=int, default=2000)
    parser.add_argument("--test-episodes", type=int, default=500)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    run(
        parse_seeds(args.seeds),
        train_episodes=50 if args.smoke else args.train_episodes,
        test_episodes=20 if args.smoke else args.test_episodes,
        out=args.out,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

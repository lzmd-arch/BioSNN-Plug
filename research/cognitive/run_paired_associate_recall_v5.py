"""情景记忆 v5：局部 delta 规则（预注册见 sweep_results/episodic_paired_associate_v5_prereg.md）。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.cognitive.episodic_memory import (
    DeltaEpisodicMemory,
    HebbianEpisodicMemory,
    PseudoInverseMemory,
    RawNearestNeighbor,
    SparseEncoder,
)
from research.cognitive.run_paired_associate_recall import (
    DIM,
    N_DIMS,
    SPARSITY,
    TEST_EPISODES,
    TRAIN_EPISODES,
    _mse,
    _raw_episodes,
    _state_control,
    _top1,
)
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "episodic_paired_associate_v5"

CAPACITIES = (4, 8, 16)
PRIMARY_K = 8
PRIMARY_NOISE = 0.1
NOISE_CHECK = (0.1, 0.3)
DELTA_EPOCHS = 50
DELTA_LR = 1.0

K8_TOP1_MIN = 0.80
K4_TOP1_MIN = 0.90
STATE_CONTROL_MAX = 0.25


def _memory_scores(
    episodes: list[tuple[np.ndarray, np.ndarray, int]],
    *,
    sigma: float,
    seed: int,
) -> dict[str, float]:
    encoder = SparseEncoder(n_in=N_DIMS, dim=DIM, k=SPARSITY, seed=0)
    rng = np.random.default_rng(seed)
    totals = dict.fromkeys(
        (
            "delta_top1",
            "hebbian_top1",
            "pseudo_top1",
            "rawnn_top1",
            "delta_mse",
            "hebbian_mse",
            "pseudo_mse",
            "rawnn_mse",
        ),
        0.0,
    )
    for keys, values, query in episodes:
        codes = encoder.encode(keys)
        query_key = keys[query] + rng.normal(scale=sigma, size=N_DIMS)
        query_code = encoder.encode(query_key[None, :])
        truth = values[query]

        delta = DeltaEpisodicMemory(epochs=DELTA_EPOCHS, learning_rate=DELTA_LR)
        delta.store(codes, values)
        delta_recall = delta.recall(query_code)[0]

        hebbian = HebbianEpisodicMemory()
        hebbian.store(codes, values)
        hebbian_recall = hebbian.recall(query_code)[0]

        pseudo = PseudoInverseMemory()
        pseudo.store(codes, values)
        pseudo_recall = pseudo.recall(query_code)[0]

        nearest = RawNearestNeighbor()
        nearest.store(codes, values)
        nearest_recall = nearest.recall(query_code)[0]

        for name, recalled in (
            ("delta", delta_recall),
            ("hebbian", hebbian_recall),
            ("pseudo", pseudo_recall),
            ("rawnn", nearest_recall),
        ):
            totals[f"{name}_top1"] += float(_top1(recalled, values) == query)
            totals[f"{name}_mse"] += _mse(recalled, truth)
    count = len(episodes)
    return {key: value / count for key, value in totals.items()}


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def run(
    seeds: list[int],
    *,
    out: Path,
    test_episodes: int = TEST_EPISODES,
    train_episodes: int = TRAIN_EPISODES,
) -> dict[str, Any]:
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    for k in CAPACITIES:
        sigmas = NOISE_CHECK if k == PRIMARY_K else (PRIMARY_NOISE,)
        for sigma in sigmas:
            rows = []
            for seed in seeds:
                test = _raw_episodes(test_episodes, k, seed=seed + 10_000)
                scores = _memory_scores(
                    test, sigma=sigma, seed=seed + 300_000 + k * 100 + int(sigma * 1000)
                )
                rows.append({"seed": seed, **scores})
            mean = {
                key: _mean(rows, key)
                for key in (
                    "delta_top1",
                    "hebbian_top1",
                    "pseudo_top1",
                    "rawnn_top1",
                    "delta_mse",
                    "hebbian_mse",
                    "pseudo_mse",
                    "rawnn_mse",
                )
            }
            results.append(
                {
                    "k": k,
                    "sigma": sigma,
                    "per_seed": rows,
                    "mean": mean,
                }
            )
            print(
                f"K{k} sigma {sigma}: delta {mean['delta_top1']:.3f} "
                f"| hebb {mean['hebbian_top1']:.3f} | pseudo {mean['pseudo_top1']:.3f} "
                f"| rawnn {mean['rawnn_top1']:.3f} | delta MSE {mean['delta_mse']:.3f}",
                flush=True,
            )

    primary = next(
        row for row in results if row["k"] == PRIMARY_K and row["sigma"] == PRIMARY_NOISE
    )
    k4 = next(row for row in results if row["k"] == 4 and row["sigma"] == PRIMARY_NOISE)
    state_rows = []
    for seed in seeds:
        train = _raw_episodes(train_episodes, PRIMARY_K, seed=seed)
        test = _raw_episodes(test_episodes, PRIMARY_K, seed=seed + 10_000)
        state_rows.append(
            {
                "seed": seed,
                **_state_control(train, test, seed=seed + 500_000),
            }
        )
    state = {
        "top1": float(np.mean([row["top1"] for row in state_rows])),
        "mse": float(np.mean([row["mse"] for row in state_rows])),
        "per_seed": state_rows,
    }
    gate = {
        "primary_k8_delta": primary["mean"]["delta_top1"],
        "k4_delta": k4["mean"]["delta_top1"],
        "state_control_top1": state["top1"],
        "hebbian_k8": primary["mean"]["hebbian_top1"],
        "k8_ok": primary["mean"]["delta_top1"] >= K8_TOP1_MIN,
        "k4_ok": k4["mean"]["delta_top1"] >= K4_TOP1_MIN,
        "state_control_ok": state["top1"] <= STATE_CONTROL_MAX,
        "beats_hebbian": primary["mean"]["delta_top1"] > primary["mean"]["hebbian_top1"],
    }
    gate["v5_ready"] = all(
        gate[key] for key in ("k8_ok", "k4_ok", "state_control_ok", "beats_hebbian")
    )
    summary = {
        "experiment": "情景记忆 v5：局部 delta 规则",
        "config": {
            "seeds": seeds,
            "test_episodes": test_episodes,
            "train_episodes": train_episodes,
            "capacities": CAPACITIES,
            "delta_epochs": DELTA_EPOCHS,
            "delta_lr": DELTA_LR,
            "dim": DIM,
            "sparsity": SPARSITY,
        },
        "conditions": results,
        "state_control": state,
        "gate": gate,
        "repro": collect(
            "情景记忆 v5：局部 delta 规则",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "DPAR 与 v4 相同；新增局部 delta 规则 W←W+η(v−Wk)kᵀ/‖k‖²，50 epoch",
                "最终种子 5-9；0-4 只用于预注册前的机制试点，不混报",
                "伪逆/字面 NN/Hebbian 同流程对照；状态控制同 v4",
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
    parser = argparse.ArgumentParser(description="情景记忆 v5：局部 delta 规则")
    parser.add_argument("--seeds", default="5-9")
    parser.add_argument("--test-episodes", type=int, default=TEST_EPISODES)
    parser.add_argument("--train-episodes", type=int, default=TRAIN_EPISODES)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    run(
        parse_seeds(args.seeds),
        out=args.out,
        test_episodes=20 if args.smoke else args.test_episodes,
        train_episodes=50 if args.smoke else args.train_episodes,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

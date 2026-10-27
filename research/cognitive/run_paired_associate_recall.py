"""情景记忆 v4：延迟配对联想回忆（DPAR）。

预注册见 ``sweep_results/episodic_paired_associate_prereg.md``。任务先把 K 对
“键→值”写进时间线，延迟后只给查询键，回忆它当时配对的值；最终状态的线性读出
无法按查询键选中对应值，因此这个任务必须依赖联想记忆。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.cognitive.episodic_memory import (
    HebbianEpisodicMemory,
    PseudoInverseMemory,
    RawNearestNeighbor,
    SparseEncoder,
)
from research.cognitive.working_memory import (
    LinearRLSReadout,
    encode_timeline,
)
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "episodic_paired_associate"

N_DIMS = 4
DIM = 512
SPARSITY = 32
CAPACITIES = (2, 4, 8, 16, 32)
NOISES = (0.0, 0.1, 0.3)
PRIMARY_K = 8
PRIMARY_NOISE = 0.1
TEST_EPISODES = 500
TRAIN_EPISODES = 2000
EPOCHS = 20

BEST_TOP1_MIN = 0.80
K4_TOP1_MIN = 0.90
STATE_CONTROL_MAX = 0.25


def _raw_episodes(n: int, k: int, *, seed: int) -> list[tuple[np.ndarray, np.ndarray, int]]:
    rng = np.random.default_rng(seed)
    episodes = []
    for _ in range(n):
        keys = rng.normal(size=(k, N_DIMS))
        values = rng.normal(size=(k, N_DIMS))
        query = int(rng.integers(0, k))
        episodes.append((keys, values, query))
    return episodes


def _cues(keys: np.ndarray, values: np.ndarray, query: int) -> list[tuple[int, int, np.ndarray]]:
    cues: list[tuple[int, int, np.ndarray]] = []
    for index in range(keys.shape[0]):
        cues.append((10 * index, 10 * index + 4, keys[index]))
        cues.append((10 * index + 5, 10 * index + 9, values[index]))
    cues.append((10 * keys.shape[0], 10 * keys.shape[0] + 4, keys[query]))
    return cues


def _top1(recalled: np.ndarray, values: np.ndarray) -> int:
    distances = ((values - recalled[None, :]) ** 2).sum(axis=1)
    return int(np.argmin(distances))


def _mse(recalled: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((recalled - truth) ** 2))


def _memory_scores(
    episodes: list[tuple[np.ndarray, np.ndarray, int]],
    *,
    sigma: float,
    seed: int,
) -> dict[str, float]:
    encoder = SparseEncoder(n_in=N_DIMS, dim=DIM, k=SPARSITY, seed=0)
    rng = np.random.default_rng(seed)
    totals = {
        "hebbian_top1": 0.0,
        "pseudo_top1": 0.0,
        "rawnn_top1": 0.0,
        "hebbian_mse": 0.0,
        "pseudo_mse": 0.0,
        "rawnn_mse": 0.0,
    }
    for keys, values, query in episodes:
        codes = encoder.encode(keys)
        query_key = keys[query] + rng.normal(scale=sigma, size=N_DIMS)
        query_code = encoder.encode(query_key[None, :])
        truth = values[query]

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
            ("hebbian", hebbian_recall),
            ("pseudo", pseudo_recall),
            ("rawnn", nearest_recall),
        ):
            totals[f"{name}_top1"] += float(_top1(recalled, values) == query)
            totals[f"{name}_mse"] += _mse(recalled, truth)
    count = len(episodes)
    return {key: value / count for key, value in totals.items()}


def _state_control(
    train: list[tuple[np.ndarray, np.ndarray, int]],
    test: list[tuple[np.ndarray, np.ndarray, int]],
    *,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    k = train[0][0].shape[0]
    test_step = 10 * k + 5
    train_states = []
    train_targets = []
    for keys, values, query in train:
        train_states.append(
            encode_timeline(_cues(keys, values, query), rng=rng, test_step=test_step, noise=0.0)
        )
        train_targets.append(values[query])
    test_states = []
    test_truth = []
    for keys, values, query in test:
        test_states.append(
            encode_timeline(_cues(keys, values, query), rng=rng, test_step=test_step, noise=0.0)
        )
        test_truth.append((values, query))
    train_states = np.stack(train_states)
    train_targets = np.stack(train_targets)
    readout = LinearRLSReadout(train_states.shape[1], train_targets.shape[1], delta=1000.0)
    readout.fit(train_states, train_targets, epochs=EPOCHS, rng=np.random.default_rng(seed))
    predicted = readout.predict(np.stack(test_states))
    top1 = 0.0
    mse = 0.0
    for prediction, (values, query) in zip(predicted, test_truth, strict=True):
        top1 += float(_top1(prediction, values) == query)
        mse += _mse(prediction, values[query])
    return {"top1": top1 / len(test), "mse": mse / len(test)}


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
    raw_test: dict[int, list[list[tuple[np.ndarray, np.ndarray, int]]]] = {}
    raw_train: dict[int, list[list[tuple[np.ndarray, np.ndarray, int]]]] = {}
    for seed in seeds:
        raw_test[seed] = {}
        raw_train[seed] = {}
        for k in CAPACITIES:
            raw_test[seed][k] = _raw_episodes(test_episodes, k, seed=seed + 10_000)
            raw_train[seed][k] = _raw_episodes(train_episodes, k, seed=seed)

    conditions: list[dict[str, Any]] = []
    for k in CAPACITIES:
        for sigma in NOISES if k == PRIMARY_K else (PRIMARY_NOISE,):
            rows = []
            for seed in seeds:
                scores = _memory_scores(
                    raw_test[seed][k],
                    sigma=sigma,
                    seed=seed + 300_000 + k * 100 + int(sigma * 1000),
                )
                rows.append({"seed": seed, **scores})
            condition = {
                "kind": "capacity" if sigma == PRIMARY_NOISE else "noise",
                "k": k,
                "sigma": sigma,
                "per_seed": rows,
                "mean": {
                    key: _mean(rows, key)
                    for key in (
                        "hebbian_top1",
                        "pseudo_top1",
                        "rawnn_top1",
                        "hebbian_mse",
                        "pseudo_mse",
                        "rawnn_mse",
                    )
                },
            }
            if k == PRIMARY_K and sigma == PRIMARY_NOISE:
                state_rows = []
                for seed in seeds:
                    state_rows.append(
                        {
                            "seed": seed,
                            **_state_control(
                                raw_train[seed][k],
                                raw_test[seed][k],
                                seed=seed + 500_000,
                            ),
                        }
                    )
                condition["state_control"] = {
                    "per_seed": state_rows,
                    "top1": _mean(state_rows, "top1"),
                    "mse": _mean(state_rows, "mse"),
                }
            conditions.append(condition)
            main = condition["mean"]
            print(
                f"K={k} sigma={sigma}: H {main['hebbian_top1']:.3f} "
                f"P {main['pseudo_top1']:.3f} NN {main['rawnn_top1']:.3f} "
                f"| state {condition.get('state_control', {}).get('top1', float('nan')):.3f}",
                flush=True,
            )

    def find(k: int, sigma: float) -> dict[str, Any]:
        return next(c for c in conditions if c["k"] == k and c["sigma"] == sigma)

    primary = find(PRIMARY_K, PRIMARY_NOISE)
    k4 = find(4, PRIMARY_NOISE)
    main = primary["mean"]
    best_arm = "hebbian" if main["hebbian_top1"] >= main["pseudo_top1"] else "pseudo"
    best_top1 = main[f"{best_arm}_top1"]
    k4_best = max(k4["mean"]["hebbian_top1"], k4["mean"]["pseudo_top1"])
    state_top1 = primary["state_control"]["top1"]
    gate = {
        "best_arm": best_arm,
        "best_top1": best_top1,
        "k4_best_top1": k4_best,
        "state_control_top1": state_top1,
        "hebbian_top1": main["hebbian_top1"],
        "pseudo_top1": main["pseudo_top1"],
        "rawnn_top1": main["rawnn_top1"],
        "best_ok": best_top1 >= BEST_TOP1_MIN,
        "k4_ok": k4_best >= K4_TOP1_MIN,
        "state_control_ok": state_top1 <= STATE_CONTROL_MAX,
    }
    gate["paired_associate_ready"] = gate["best_ok"] and gate["k4_ok"] and gate["state_control_ok"]
    capacity_curve = {
        str(condition["k"]): {
            "hebbian_top1": condition["mean"]["hebbian_top1"],
            "pseudo_top1": condition["mean"]["pseudo_top1"],
            "rawnn_top1": condition["mean"]["rawnn_top1"],
            "best_top1": max(condition["mean"]["hebbian_top1"], condition["mean"]["pseudo_top1"]),
        }
        for condition in conditions
        if condition["kind"] == "capacity"
    }
    noise_curve = {
        str(condition["sigma"]): {
            "hebbian_top1": condition["mean"]["hebbian_top1"],
            "pseudo_top1": condition["mean"]["pseudo_top1"],
            "rawnn_top1": condition["mean"]["rawnn_top1"],
            "best_top1": max(condition["mean"]["hebbian_top1"], condition["mean"]["pseudo_top1"]),
        }
        for condition in conditions
        if condition["k"] == PRIMARY_K
    }
    summary = {
        "experiment": "情景记忆 v4：延迟配对联想回忆（DPAR）",
        "config": {
            "seeds": seeds,
            "capacities": CAPACITIES,
            "noises": NOISES,
            "primary_k": PRIMARY_K,
            "primary_noise": PRIMARY_NOISE,
            "train_episodes": train_episodes,
            "test_episodes": test_episodes,
            "dim": DIM,
            "sparsity": SPARSITY,
            "epochs": EPOCHS,
        },
        "conditions": conditions,
        "capacity_curve": capacity_curve,
        "noise_curve": noise_curve,
        "gate": gate,
        "repro": collect(
            "情景记忆 v4：延迟配对联想回忆（DPAR）",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "K 对键值按时间线写入；查询键在延迟后给出，回忆配对值",
                "Hebbian / Pseudo-inverse / RawNN 每幕独立建库；状态控制为最终状态 RLS 读出",
                "主档 K=8、σ=0.1；容量曲线 K=2/4/8/16/32；噪声曲线 0/0.1/0.3",
            ],
            device=None,
        ).render_block(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    print(json.dumps(capacity_curve, ensure_ascii=False, indent=2))
    print(json.dumps(noise_curve, ensure_ascii=False, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="情景记忆 v4：延迟配对联想回忆")
    parser.add_argument("--seeds", default="0-4")
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

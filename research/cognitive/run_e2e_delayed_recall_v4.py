"""阶段四端到端 v4：DPAR + 工作记忆共存。

预注册见 ``sweep_results/stage4_e2e_v4_prereg.md``。同一幕里既有 A/B/Q 工作记忆，
也有 K=4 对延迟配对联想；分别评估，不把两种目标混成一个 R²。
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
    average_r2,
    encode_timeline,
    fit_ridge,
    predict,
)
from research.cognitive.world_model import auroc
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "stage4_e2e_v4"

N_DIMS = 4
PAIRS = 4
TEST_STEP = 85
MEMBERS = 5
EPOCHS = 20
COVERAGE = 0.80
QUERY_NOISE = 0.1
CUE_NOISE = 0.3
DIM = 512
SPARSITY = 32

R2_MIN = 0.50
DPAR_MIN = 0.80
STATE_CONTROL_MAX = 0.35
OOD_AUROC_MIN = 0.75
OOD_ABSTAIN_MIN = 0.80
SELECTIVE_MSE_MAX = 0.10


def _episode(
    rng: np.random.Generator, *, ood: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    cue_a = rng.normal(size=N_DIMS)
    cue_b = rng.normal(size=N_DIMS)
    if ood:
        cue_a = cue_a * 3.0
        cue_b = cue_b * 3.0
    else:
        cue_a = cue_a + rng.normal(scale=CUE_NOISE, size=N_DIMS)
        cue_b = cue_b + rng.normal(scale=CUE_NOISE, size=N_DIMS)
    choose_b = bool(rng.integers(0, 2))
    query = np.array([0.0, 1.0, 0.0, 0.0] if choose_b else [1.0, 0.0, 0.0, 0.0])
    keys = rng.normal(size=(PAIRS, N_DIMS))
    values = rng.normal(size=(PAIRS, N_DIMS))
    pair = int(rng.integers(0, PAIRS))
    cues: list[tuple[int, int, np.ndarray]] = [
        (0, 4, cue_a),
        (15, 19, cue_b),
        (30, 34, query),
    ]
    for index in range(PAIRS):
        cues.append((40 + 10 * index, 44 + 10 * index, keys[index]))
        cues.append((45 + 10 * index, 49 + 10 * index, values[index]))
    cues.append((80, 84, keys[pair]))
    previous = encode_timeline(cues, rng=rng, test_step=TEST_STEP - 1)
    state = encode_timeline(cues, rng=rng, test_step=TEST_STEP)
    wm_target = np.concatenate([cue_a, cue_b, query])
    return state, previous, wm_target, keys, values, pair


def _dataset(
    episodes: int, *, seed: int, ood: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states, previous, wm_targets, keys, values, pairs = [], [], [], [], [], []
    for _ in range(episodes):
        state, prev, target, key, value, pair = _episode(rng, ood=ood)
        states.append(state)
        previous.append(prev)
        wm_targets.append(target)
        keys.append(key)
        values.append(value)
        pairs.append(pair)
    return (
        np.stack(states),
        np.stack(previous),
        np.stack(wm_targets),
        np.stack(keys),
        np.stack(values),
        np.asarray(pairs),
    )


def _top1(recalled: np.ndarray, values: np.ndarray) -> int:
    return int(np.argmin(((values - recalled[None, :]) ** 2).sum(axis=1)))


def _dpar_scores(
    keys: np.ndarray,
    values: np.ndarray,
    pairs: np.ndarray,
    *,
    seed: int,
) -> dict[str, float]:
    encoder = SparseEncoder(n_in=N_DIMS, dim=DIM, k=SPARSITY, seed=0)
    rng = np.random.default_rng(seed)
    totals = {"pseudo": 0.0, "rawnn": 0.0, "hebbian": 0.0}
    for index in range(keys.shape[0]):
        codes = encoder.encode(keys[index])
        query = keys[index, pairs[index]] + rng.normal(scale=QUERY_NOISE, size=N_DIMS)
        query_code = encoder.encode(query[None, :])

        pseudo = PseudoInverseMemory()
        pseudo.store(codes, values[index])
        nearest = RawNearestNeighbor()
        nearest.store(codes, values[index])
        hebbian = HebbianEpisodicMemory()
        hebbian.store(codes, values[index])

        for name, recalled in (
            ("pseudo", pseudo.recall(query_code)[0]),
            ("rawnn", nearest.recall(query_code)[0]),
            ("hebbian", hebbian.recall(query_code)[0]),
        ):
            totals[name] += float(_top1(recalled, values[index]) == pairs[index])
    return {name: value / keys.shape[0] for name, value in totals.items()}


def _wm_features(
    split: tuple[np.ndarray, ...],
    *,
    readout: LinearRLSReadout,
    members: list[LinearRLSReadout],
    world: np.ndarray,
) -> dict[str, np.ndarray]:
    states, previous, targets = split[0], split[1], split[2]
    predicted = readout.predict(states)
    member_predictions = np.stack([member.predict(states) for member in members])
    return {
        "errors": _answer_error(predicted, targets),
        "disagreement": member_predictions.std(axis=0).mean(axis=1),
        "surprise": ((predict(previous, world) - states) ** 2).mean(axis=1),
    }


def _answer_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    choose_b = target[:, 8] < 0.5
    clues = np.where(choose_b[:, None], predicted[:, 4:8], predicted[:, 0:4])
    truth = np.where(choose_b[:, None], target[:, 4:8], target[:, 0:4])
    return ((clues - truth) ** 2).mean(axis=1)


def run(seeds: list[int], *, out: Path, test_episodes: int = 500) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        train = _dataset(2000, seed=seed)
        train_states, train_previous, train_targets = train[0], train[1], train[2]
        calibration = _dataset(test_episodes, seed=seed + 30_000)
        test = _dataset(test_episodes, seed=seed + 10_000)
        ood = _dataset(test_episodes, seed=seed + 20_000, ood=True)

        readout = LinearRLSReadout(train_states.shape[1], train_targets.shape[1], delta=1000.0)
        readout.fit(train_states, train_targets, epochs=EPOCHS, rng=np.random.default_rng(seed))
        members = []
        for member_index in range(MEMBERS):
            rng = np.random.default_rng(seed + 100 + member_index)
            index = rng.choice(train_states.shape[0], size=train_states.shape[0], replace=True)
            member = LinearRLSReadout(train_states.shape[1], train_targets.shape[1], delta=1000.0)
            member.fit(train_states[index], train_targets[index], epochs=EPOCHS, rng=rng)
            members.append(member)
        world = fit_ridge(train_previous, train_states)

        calibration_features = _wm_features(
            calibration, readout=readout, members=members, world=world
        )
        test_features = _wm_features(test, readout=readout, members=members, world=world)
        ood_features = _wm_features(ood, readout=readout, members=members, world=world)

        predicted = readout.predict(test[0])
        ood_predicted = readout.predict(ood[0])
        r2 = [
            average_r2(
                predicted[:, i * N_DIMS : (i + 1) * N_DIMS],
                test[2][:, i * N_DIMS : (i + 1) * N_DIMS],
            )
            for i in range(3)
        ]
        ood_errors = _answer_error(ood_predicted, ood[2])

        def zscore(values, reference):
            return (values - reference.mean()) / max(reference.std(), 1e-8)

        confidence = -zscore(
            test_features["disagreement"], calibration_features["disagreement"]
        ) - zscore(test_features["surprise"], calibration_features["surprise"])
        ood_confidence = -zscore(
            ood_features["disagreement"], calibration_features["disagreement"]
        ) - zscore(ood_features["surprise"], calibration_features["surprise"])
        threshold = float(
            np.quantile(
                -zscore(calibration_features["disagreement"], calibration_features["disagreement"])
                - zscore(calibration_features["surprise"], calibration_features["surprise"]),
                1.0 - COVERAGE,
            )
        )
        answered = confidence >= threshold
        ood_answered = ood_confidence >= threshold
        selective_mse = (
            float(np.mean(test_features["errors"][answered])) if np.any(answered) else float("nan")
        )
        ood_abstain = float(np.mean(~ood_answered))
        auc = auroc(
            np.concatenate([confidence, ood_confidence]),
            np.concatenate([np.ones_like(confidence), np.zeros_like(ood_confidence)]),
        )

        dpar = _dpar_scores(test[3], test[4], test[5], seed=seed + 400_000)
        dpar_best = max(dpar["pseudo"], dpar["rawnn"])

        state_readout = LinearRLSReadout(train_states.shape[1], N_DIMS, delta=1000.0)
        train_values = np.stack(
            [values[pair] for values, pair in zip(train[4], train[5], strict=True)]
        )
        state_readout.fit(
            train_states, train_values, epochs=EPOCHS, rng=np.random.default_rng(seed + 500_000)
        )
        state_predictions = state_readout.predict(test[0])
        state_top1 = float(
            np.mean(
                [
                    _top1(prediction, values) == pair
                    for prediction, values, pair in zip(
                        state_predictions, test[4], test[5], strict=True
                    )
                ]
            )
        )

        rows.append(
            {
                "seed": seed,
                "r2_a": r2[0],
                "r2_b": r2[1],
                "r2_q": r2[2],
                "r2_mean": float(np.mean(r2)),
                "selective_mse": selective_mse,
                "ood_auroc": auc,
                "ood_abstain": ood_abstain,
                "ood_answer_mse": float(np.mean(ood_errors)),
                "dpar_pseudo": dpar["pseudo"],
                "dpar_rawnn": dpar["rawnn"],
                "dpar_hebbian": dpar["hebbian"],
                "dpar_best": dpar_best,
                "state_control_top1": state_top1,
            }
        )
        print(
            f"seed {seed}: R2 {np.round(r2, 3).tolist()} | sel MSE {selective_mse:.3f} | "
            f"OOD {auc:.3f}/{ood_abstain:.3f} | DPAR P/N/H "
            f"{dpar['pseudo']:.3f}/{dpar['rawnn']:.3f}/{dpar['hebbian']:.3f} | "
            f"state {state_top1:.3f}",
            flush=True,
        )

    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in rows]))

    gate = {
        "r2_mean": mean("r2_mean"),
        "selective_mse": mean("selective_mse"),
        "ood_auroc": mean("ood_auroc"),
        "ood_abstain": mean("ood_abstain"),
        "dpar_best": mean("dpar_best"),
        "state_control_top1": mean("state_control_top1"),
        "r2_ok": mean("r2_mean") >= R2_MIN,
        "selective_ok": mean("selective_mse") <= SELECTIVE_MSE_MAX,
        "ood_auroc_ok": mean("ood_auroc") >= OOD_AUROC_MIN,
        "ood_abstain_ok": mean("ood_abstain") >= OOD_ABSTAIN_MIN,
        "dpar_ok": mean("dpar_best") >= DPAR_MIN,
        "state_control_ok": mean("state_control_top1") <= STATE_CONTROL_MAX,
    }
    gate["e2e_v4_ready"] = all(
        gate[key]
        for key in (
            "r2_ok",
            "selective_ok",
            "ood_auroc_ok",
            "ood_abstain_ok",
            "dpar_ok",
            "state_control_ok",
        )
    )
    summary = {
        "experiment": "阶段四端到端 v4：DPAR + 工作记忆共存",
        "config": {
            "seeds": seeds,
            "pairs": PAIRS,
            "test_step": TEST_STEP,
            "cue_noise": CUE_NOISE,
            "query_noise": QUERY_NOISE,
            "test_episodes": test_episodes,
            "coverage": COVERAGE,
            "dim": DIM,
            "sparsity": SPARSITY,
        },
        "per_seed": rows,
        "gate": gate,
        "repro": collect(
            "阶段四端到端 v4",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "A/B/Q + K=4 DPAR 键值 + 查询键；test_step=85",
                "WM R² 与 DPAR top-1 分开；状态控制为最终状态 RLS 读出",
                "OOD 为 A/B ×3；阈值只在 calibration（seed+30000）上取",
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
    parser = argparse.ArgumentParser(description="阶段四端到端 v4")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--test-episodes", type=int, default=500)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    run(
        parse_seeds(args.seeds),
        out=args.out,
        test_episodes=20 if args.smoke else args.test_episodes,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

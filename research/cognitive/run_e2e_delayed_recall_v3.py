"""阶段四端到端 v3：容量/噪声/延迟压力测试（预注册见 sweep_results/stage4_e2e_v3_prereg.md）。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.cognitive.episodic_memory import RawNearestNeighbor
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
OUT = REPO_ROOT / "sweep_results" / "stage4_e2e_v3"

N_DIMS = 4
CAPACITIES = (2, 4, 8, 16)
PRIMARY_EVENTS = 8
PRIMARY_NOISE = 0.3
PRIMARY_TEST_STEP = 100
MEMBERS = 5
EPOCHS = 20
COVERAGE = 0.80


def event_codes(n_events: int) -> np.ndarray:
    """K≤16 的 4-bit ±1 码；每个码位在类间都变化，避免 R² 除零。"""
    if n_events == 2:
        return np.array([[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]])
    if n_events == 4:
        return np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0, 1.0],
                [-1.0, 1.0, 1.0, -1.0],
                [-1.0, -1.0, -1.0, -1.0],
            ]
        )
    if n_events == 8:
        codes = []
        for code in range(8):
            bits = [1.0 if (code >> bit) & 1 else -1.0 for bit in range(3)]
            parity = 1.0 if sum(bits) > 0 else -1.0
            codes.append([*bits, parity])
        return np.array(codes)
    if n_events == 16:
        return np.array(
            [[1.0 if (code >> bit) & 1 else -1.0 for bit in range(4)] for code in range(16)]
        )
    raise ValueError("n_events 只支持 2/4/8/16。")


R2_MIN = 0.50
SELECTIVE_MSE_MAX = 0.10
OOD_AUROC_MIN = 0.75
OOD_ABSTAIN_MIN = 0.80
EPISODIC_ACC_MIN = 0.80


def _episode(
    rng: np.random.Generator,
    *,
    ood: bool,
    n_events: int,
    noise: float,
    test_step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    cue_a = rng.normal(size=N_DIMS)
    cue_b = rng.normal(size=N_DIMS)
    if ood:
        cue_a = cue_a * 3.0
        cue_b = cue_b * 3.0
    else:
        cue_a = cue_a + rng.normal(scale=noise, size=N_DIMS)
        cue_b = cue_b + rng.normal(scale=noise, size=N_DIMS)
    choose_b = bool(rng.integers(0, 2))
    query = np.array([0.0, 1.0, 0.0, 0.0] if choose_b else [1.0, 0.0, 0.0, 0.0])
    event_class = int(rng.integers(0, n_events))
    event = event_codes(n_events)[event_class].copy()
    cues = [
        (0, 4, cue_a),
        (15, 19, cue_b),
        (30, 34, query),
        (45, 49, event),
    ]
    state_before = encode_timeline(cues, rng=rng, test_step=test_step - 1)
    state = encode_timeline(cues, rng=rng, test_step=test_step)
    target = np.concatenate([cue_a, cue_b, query, event])
    return state, state_before, target, event_class


def _dataset(
    episodes: int,
    *,
    seed: int,
    ood: bool,
    n_events: int,
    noise: float,
    test_step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states, previous, targets, classes = [], [], [], []
    for _ in range(episodes):
        state, state_before, target, event_class = _episode(
            rng,
            ood=ood,
            n_events=n_events,
            noise=noise,
            test_step=test_step,
        )
        states.append(state)
        previous.append(state_before)
        targets.append(target)
        classes.append(event_class)
    return (
        np.stack(states),
        np.stack(previous),
        np.stack(targets),
        np.asarray(classes),
    )


def _r2(predicted: np.ndarray, target: np.ndarray) -> list[float]:
    return [
        average_r2(
            predicted[:, i * N_DIMS : (i + 1) * N_DIMS], target[:, i * N_DIMS : (i + 1) * N_DIMS]
        )
        for i in range(4)
    ]


def _answer_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    choose_b = target[:, 8] < 0.5
    clues = np.where(choose_b[:, None], predicted[:, 4:8], predicted[:, 0:4])
    truth = np.where(choose_b[:, None], target[:, 4:8], target[:, 0:4])
    return ((clues - truth) ** 2).mean(axis=1)


def _event_accuracy(retrieved: np.ndarray, truth: np.ndarray, n_events: int) -> float:
    codes = event_codes(n_events)
    predicted = np.argmin(((retrieved[:, None, :] - codes[None, :, :]) ** 2).sum(axis=2), axis=1)
    return float(np.mean(predicted == truth))


def _condition(
    seeds: list[int],
    *,
    n_events: int,
    noise: float,
    test_step: int,
    train_episodes: int,
    test_episodes: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        train_states, train_previous, train_targets, _ = _dataset(
            train_episodes,
            seed=seed,
            ood=False,
            n_events=n_events,
            noise=noise,
            test_step=test_step,
        )
        test_states, test_previous, test_targets, test_classes = _dataset(
            test_episodes,
            seed=seed + 10_000,
            ood=False,
            n_events=n_events,
            noise=noise,
            test_step=test_step,
        )
        ood_states, ood_previous, ood_targets, _ = _dataset(
            test_episodes,
            seed=seed + 20_000,
            ood=True,
            n_events=n_events,
            noise=noise,
            test_step=test_step,
        )

        readout = LinearRLSReadout(train_states.shape[1], train_targets.shape[1], delta=1000.0)
        readout.fit(train_states, train_targets, epochs=EPOCHS, rng=np.random.default_rng(seed))
        predicted = readout.predict(test_states)
        ood_predicted = readout.predict(ood_states)
        r2 = _r2(predicted, test_targets)
        errors = _answer_error(predicted, test_targets)
        ood_errors = _answer_error(ood_predicted, ood_targets)

        members = []
        for member_index in range(MEMBERS):
            rng = np.random.default_rng(seed + 100 + member_index)
            index = rng.choice(train_states.shape[0], size=train_states.shape[0], replace=True)
            member = LinearRLSReadout(train_states.shape[1], train_targets.shape[1], delta=1000.0)
            member.fit(train_states[index], train_targets[index], epochs=EPOCHS, rng=rng)
            members.append(member)
        test_predictions = np.stack([member.predict(test_states) for member in members])
        ood_predictions = np.stack([member.predict(ood_states) for member in members])
        disagreement = test_predictions.std(axis=0).mean(axis=1)
        ood_disagreement = ood_predictions.std(axis=0).mean(axis=1)

        world = fit_ridge(train_previous, train_states)
        surprise = ((predict(test_previous, world) - test_states) ** 2).mean(axis=1)
        ood_surprise = ((predict(ood_previous, world) - ood_states) ** 2).mean(axis=1)

        def zscore(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
            return (values - reference.mean()) / max(reference.std(), 1e-8)

        confidence = -zscore(disagreement, disagreement) - zscore(surprise, surprise)
        ood_confidence = -zscore(ood_disagreement, disagreement) - zscore(ood_surprise, surprise)
        threshold = float(np.quantile(confidence, 1.0 - COVERAGE))
        answered = confidence >= threshold
        ood_answered = ood_confidence >= threshold
        selective_mse = float(np.mean(errors[answered])) if np.any(answered) else float("nan")
        ood_abstain = float(np.mean(~ood_answered))
        auc = auroc(
            np.concatenate([confidence, ood_confidence]),
            np.concatenate([np.ones_like(confidence), np.zeros_like(ood_confidence)]),
        )

        nearest = RawNearestNeighbor()
        nearest.store(train_states, train_targets[:, 12:16])
        retrieved = nearest.recall(test_states)
        episodic_accuracy = _event_accuracy(retrieved, test_classes, n_events)
        rows.append(
            {
                "seed": seed,
                "r2_a": r2[0],
                "r2_b": r2[1],
                "r2_q": r2[2],
                "r2_e": r2[3],
                "r2_mean": float(np.mean(r2)),
                "selective_mse": selective_mse,
                "ood_auroc": auc,
                "ood_abstain": ood_abstain,
                "ood_answer_mse": float(np.mean(ood_errors)),
                "episodic_accuracy": episodic_accuracy,
            }
        )
        print(
            f"K={n_events} noise={noise} t={test_step} seed {seed}: "
            f"R2 {np.round(r2, 3).tolist()} | sel MSE {selective_mse:.3f} | "
            f"OOD AUROC {auc:.3f} abstain {ood_abstain:.3f} | E acc {episodic_accuracy:.3f}",
            flush=True,
        )
    return {
        "n_events": n_events,
        "noise": noise,
        "test_step": test_step,
        "per_seed": rows,
        "mean": {
            key: float(np.mean([row[key] for row in rows]))
            for key in (
                "r2_mean",
                "selective_mse",
                "ood_auroc",
                "ood_abstain",
                "episodic_accuracy",
            )
        },
    }


def run(
    seeds: list[int],
    *,
    out: Path,
    train_episodes: int = 2000,
    test_episodes: int = 500,
) -> dict[str, Any]:
    started = time.perf_counter()
    conditions = [
        ("capacity", capacity, PRIMARY_NOISE, PRIMARY_TEST_STEP) for capacity in CAPACITIES
    ]
    conditions += [("noise", PRIMARY_EVENTS, noise, PRIMARY_TEST_STEP) for noise in (0.0, 0.1)]
    conditions += [("delay", PRIMARY_EVENTS, PRIMARY_NOISE, 60)]
    results: list[dict[str, Any]] = []
    for axis, n_events, noise, test_step in conditions:
        print(f"=== {axis}: K={n_events} noise={noise} t={test_step} ===", flush=True)
        result = _condition(
            seeds,
            n_events=n_events,
            noise=noise,
            test_step=test_step,
            train_episodes=train_episodes,
            test_episodes=test_episodes,
        )
        result["axis"] = axis
        results.append(result)

    primary = next(
        result
        for result in results
        if result["axis"] == "capacity"
        and result["n_events"] == PRIMARY_EVENTS
        and result["noise"] == PRIMARY_NOISE
        and result["test_step"] == PRIMARY_TEST_STEP
    )
    mean = primary["mean"]
    gate = {
        **{key: mean[key] for key in mean},
        "r2_ok": mean["r2_mean"] >= R2_MIN,
        "selective_ok": mean["selective_mse"] <= SELECTIVE_MSE_MAX,
        "ood_auroc_ok": mean["ood_auroc"] >= OOD_AUROC_MIN,
        "ood_abstain_ok": mean["ood_abstain"] >= OOD_ABSTAIN_MIN,
        "episodic_ok": mean["episodic_accuracy"] >= EPISODIC_ACC_MIN,
    }
    gate["e2e_v3_ready"] = all(
        gate[key]
        for key in ("r2_ok", "selective_ok", "ood_auroc_ok", "ood_abstain_ok", "episodic_ok")
    )
    capacity_curve = {
        str(result["n_events"]): {
            "episodic_accuracy": result["mean"]["episodic_accuracy"],
            "r2_mean": result["mean"]["r2_mean"],
        }
        for result in results
        if result["axis"] == "capacity"
    }
    summary = {
        "experiment": "阶段四端到端 v3：容量/噪声/延迟压力测试",
        "config": {
            "seeds": seeds,
            "coverage": COVERAGE,
            "members": MEMBERS,
            "epochs": EPOCHS,
            "ood_scale": 3.0,
            "train_episodes": train_episodes,
            "test_episodes": test_episodes,
            "primary": {
                "n_events": PRIMARY_EVENTS,
                "noise": PRIMARY_NOISE,
                "test_step": PRIMARY_TEST_STEP,
            },
        },
        "conditions": results,
        "capacity_curve": capacity_curve,
        "gate": gate,
        "repro": collect(
            "阶段四端到端 v3",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "E 用 4-bit ±1 码，K≤16；A/B 噪声 σ；测试延迟 t",
                "主档 K=8、σ=0.3、t=100；容量曲线 K=2/4/8/16；其余为对照",
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
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段四端到端 v3 压力测试")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--train-episodes", type=int, default=2000)
    parser.add_argument("--test-episodes", type=int, default=500)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    run(
        parse_seeds(args.seeds),
        out=args.out,
        train_episodes=50 if args.smoke else args.train_episodes,
        test_episodes=20 if args.smoke else args.test_episodes,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

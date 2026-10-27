"""阶段四端到端 v6：单 bank vs 路由双 bank 的共存。

预注册见 ``sweep_results/stage4_e2e_v6_prereg.md``。两臂都跑：
单 bank 把 A/B/Q + K=4 对键值写进同一个 28 维状态；双 bank 把 WM 与 DPAR 的输入
路由到两个 28 维 bank 再拼接。判据一致，不做事后挑臂。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.cognitive.run_e2e_delayed_recall_v4 import _dpar_scores, _top1
from research.cognitive.working_memory import (
    LinearRLSReadout,
    encode_timeline,
    fit_ridge,
    predict,
)
from research.cognitive.world_model import auroc
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "stage4_e2e_v6"

N_DIMS = 4
PAIRS = 4
FINAL_STEP = 85
MEMBERS = 5
EPOCHS = 20
COVERAGE = 0.80
CUE_NOISE = 0.3

R2_MIN = 0.90
SELECTIVE_MSE_MAX = 0.10
DPAR_MIN = 0.80
STATE_CONTROL_MAX = 0.35
OOD_AUROC_MIN = 0.75
OOD_ABSTAIN_MIN = 0.80


def _episode(
    rng: np.random.Generator, *, ood: bool
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
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

    wm_cues = [
        (0, 4, cue_a),
        (15, 19, cue_b),
        (30, 34, query),
    ]
    dpar_cues: list[tuple[int, int, np.ndarray]] = []
    for index in range(PAIRS):
        dpar_cues.append((40 + 10 * index, 44 + 10 * index, keys[index]))
        dpar_cues.append((45 + 10 * index, 49 + 10 * index, values[index]))
    dpar_cues.append((80, 84, keys[pair]))

    single_state = encode_timeline(wm_cues + dpar_cues, rng=rng, test_step=FINAL_STEP)
    single_previous = encode_timeline(wm_cues + dpar_cues, rng=rng, test_step=FINAL_STEP - 1)
    wm_state = encode_timeline(wm_cues, rng=rng, test_step=FINAL_STEP)
    wm_previous = encode_timeline(wm_cues, rng=rng, test_step=FINAL_STEP - 1)
    dpar_state = encode_timeline(dpar_cues, rng=rng, test_step=FINAL_STEP)
    dpar_previous = encode_timeline(dpar_cues, rng=rng, test_step=FINAL_STEP - 1)
    target = np.concatenate([cue_a, cue_b, query])
    return (
        single_state,
        single_previous,
        np.concatenate([wm_state, dpar_state]),
        np.concatenate([wm_previous, dpar_previous]),
        target,
        keys,
        values,
        pair,
    )


def _dataset(
    episodes: int, *, seed: int, ood: bool = False
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(seed)
    single_states, single_previous, dual_states, dual_previous = [], [], [], []
    targets, keys, values, pairs = [], [], [], []
    for _ in range(episodes):
        single, single_prev, dual, dual_prev, target, key, value, pair = _episode(rng, ood=ood)
        single_states.append(single)
        single_previous.append(single_prev)
        dual_states.append(dual)
        dual_previous.append(dual_prev)
        targets.append(target)
        keys.append(key)
        values.append(value)
        pairs.append(pair)
    return (
        np.stack(single_states),
        np.stack(single_previous),
        np.stack(dual_states),
        np.stack(dual_previous),
        np.stack(targets),
        np.stack(keys),
        np.stack(values),
        np.asarray(pairs),
    )


def _features(
    states: np.ndarray,
    previous: np.ndarray,
    targets: np.ndarray,
    *,
    readout: LinearRLSReadout,
    members: list[LinearRLSReadout],
    world: np.ndarray,
) -> dict[str, np.ndarray]:
    predicted = readout.predict(states)
    member_predictions = np.stack([member.predict(states) for member in members])
    choose_b = targets[:, 8] < 0.5
    clues = np.where(choose_b[:, None], predicted[:, 4:8], predicted[:, 0:4])
    truth = np.where(choose_b[:, None], targets[:, 4:8], targets[:, 0:4])
    return {
        "errors": ((clues - truth) ** 2).mean(axis=1),
        "disagreement": member_predictions.std(axis=0).mean(axis=1),
        "surprise": ((predict(previous, world) - states) ** 2).mean(axis=1),
    }


def _evaluate_arm(
    *,
    train_states: np.ndarray,
    train_previous: np.ndarray,
    train_targets: np.ndarray,
    calibration,
    test,
    ood,
    train_state_targets: np.ndarray,
    test_values: np.ndarray,
    test_pairs: np.ndarray,
    seed: int,
) -> dict[str, float]:
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

    calibration_features = _features(
        calibration[0],
        calibration[1],
        calibration[2],
        readout=readout,
        members=members,
        world=world,
    )
    test_features = _features(
        test[0], test[1], test[2], readout=readout, members=members, world=world
    )
    ood_features = _features(ood[0], ood[1], ood[2], readout=readout, members=members, world=world)

    predicted = readout.predict(test[0])

    def safe_r2(predicted_values: np.ndarray, target_values: np.ndarray) -> float:
        residual = ((predicted_values - target_values) ** 2).sum(axis=0)
        total = ((target_values - target_values.mean(axis=0)) ** 2).sum(axis=0)
        varying = total > 1e-9
        return float(np.mean(1.0 - residual[varying] / total[varying]))

    r2 = [
        safe_r2(
            predicted[:, i * N_DIMS : (i + 1) * N_DIMS],
            test[2][:, i * N_DIMS : (i + 1) * N_DIMS],
        )
        for i in range(3)
    ]

    def zscore(values, reference):
        return (values - reference.mean()) / max(reference.std(), 1e-8)

    calibration_confidence = -zscore(
        calibration_features["disagreement"], calibration_features["disagreement"]
    ) - zscore(calibration_features["surprise"], calibration_features["surprise"])
    test_confidence = -zscore(
        test_features["disagreement"], calibration_features["disagreement"]
    ) - zscore(test_features["surprise"], calibration_features["surprise"])
    ood_confidence = -zscore(
        ood_features["disagreement"], calibration_features["disagreement"]
    ) - zscore(ood_features["surprise"], calibration_features["surprise"])
    threshold = float(np.quantile(calibration_confidence, 1.0 - COVERAGE))
    answered = test_confidence >= threshold
    ood_answered = ood_confidence >= threshold
    selective_mse = (
        float(np.mean(test_features["errors"][answered])) if np.any(answered) else float("nan")
    )
    auc = auroc(
        np.concatenate([test_confidence, ood_confidence]),
        np.concatenate([np.ones_like(test_confidence), np.zeros_like(ood_confidence)]),
    )

    state_readout = LinearRLSReadout(train_states.shape[1], N_DIMS, delta=1000.0)
    state_readout.fit(
        train_states,
        train_state_targets,
        epochs=EPOCHS,
        rng=np.random.default_rng(seed + 500_000),
    )
    state_predictions = state_readout.predict(test[0])
    state_top1 = float(
        np.mean(
            [
                _top1(prediction, values) == pair
                for prediction, values, pair in zip(
                    state_predictions, test_values, test_pairs, strict=True
                )
            ]
        )
    )

    return {
        "r2_a": r2[0],
        "r2_b": r2[1],
        "r2_q": r2[2],
        "r2_mean": float(np.mean(r2)),
        "selective_mse": selective_mse,
        "ood_auroc": auc,
        "ood_abstain": float(np.mean(~ood_answered)),
        "state_control_top1": state_top1,
    }


def run(seeds: list[int], *, out: Path, test_episodes: int = 500) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        train = _dataset(2000, seed=seed)
        calibration = _dataset(test_episodes, seed=seed + 30_000)
        test = _dataset(test_episodes, seed=seed + 10_000)
        ood = _dataset(test_episodes, seed=seed + 20_000, ood=True)
        dpar = _dpar_scores(test[5], test[6], test[7], seed=seed + 400_000)
        dpar_best = max(dpar["pseudo"], dpar["rawnn"])

        train_state_targets = np.stack(
            [values[pair] for values, pair in zip(train[6], train[7], strict=True)]
        )
        row: dict[str, Any] = {"seed": seed, **{f"dpar_{k}": v for k, v in dpar.items()}}
        for arm, train_states, train_previous, cal, te, oo in (
            (
                "single",
                train[0],
                train[1],
                (calibration[0], calibration[1], calibration[4]),
                (test[0], test[1], test[4]),
                (ood[0], ood[1], ood[4]),
            ),
            (
                "dual",
                train[2],
                train[3],
                (calibration[2], calibration[3], calibration[4]),
                (test[2], test[3], test[4]),
                (ood[2], ood[3], ood[4]),
            ),
        ):
            metrics = _evaluate_arm(
                train_states=train_states,
                train_previous=train_previous,
                train_targets=train[4],
                calibration=cal,
                test=te,
                ood=oo,
                train_state_targets=train_state_targets,
                test_values=test[6],
                test_pairs=test[7],
                seed=seed,
            )
            for key, value in metrics.items():
                row[f"{arm}_{key}"] = value
            row[f"{arm}_dpar_best"] = dpar_best
        rows.append(row)
        print(
            f"seed {seed}: "
            + " | ".join(
                f"{arm} R2 {row[f'{arm}_r2_mean']:.3f} MSE {row[f'{arm}_selective_mse']:.3f} "
                f"OOD {row[f'{arm}_ood_auroc']:.3f}/{row[f'{arm}_ood_abstain']:.3f} "
                f"state {row[f'{arm}_state_control_top1']:.3f}"
                for arm in ("single", "dual")
            )
            + f" | DPAR {dpar_best:.3f}",
            flush=True,
        )

    def mean(arm: str, key: str) -> float:
        return float(np.mean([row[f"{arm}_{key}"] for row in rows]))

    gates = {}
    for arm in ("single", "dual"):
        gate = {
            key: mean(arm, key)
            for key in (
                "r2_mean",
                "selective_mse",
                "ood_auroc",
                "ood_abstain",
                "state_control_top1",
                "dpar_best",
            )
        }
        gate.update(
            {
                "r2_ok": mean(arm, "r2_mean") >= R2_MIN,
                "selective_ok": mean(arm, "selective_mse") <= SELECTIVE_MSE_MAX,
                "dpar_ok": mean(arm, "dpar_best") >= DPAR_MIN,
                "state_control_ok": mean(arm, "state_control_top1") <= STATE_CONTROL_MAX,
                "ood_auroc_ok": mean(arm, "ood_auroc") >= OOD_AUROC_MIN,
                "ood_abstain_ok": mean(arm, "ood_abstain") >= OOD_ABSTAIN_MIN,
            }
        )
        gate["ready"] = all(
            gate[key]
            for key in (
                "r2_ok",
                "selective_ok",
                "dpar_ok",
                "state_control_ok",
                "ood_auroc_ok",
                "ood_abstain_ok",
            )
        )
        gates[arm] = gate
    gates["e2e_v6_single_ready"] = gates["single"]["ready"]
    gates["e2e_v6_dual_ready"] = gates["dual"]["ready"]
    summary = {
        "experiment": "阶段四端到端 v6：单 bank vs 路由双 bank",
        "config": {
            "seeds": seeds,
            "pairs": PAIRS,
            "final_step": FINAL_STEP,
            "test_episodes": test_episodes,
            "coverage": COVERAGE,
        },
        "per_seed": rows,
        "gates": gates,
        "repro": collect(
            "阶段四端到端 v6",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "single：A/B/Q + 键值对进入同一 28 维状态",
                "dual：A/B/Q 与键值对/查询分别进入两个 28 维 bank 后拼接",
                "OOD 为 A/B ×3；阈值只在 calibration（seed+30000）上取",
            ],
            device=None,
        ).render_block(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(gates, ensure_ascii=False, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段四端到端 v6")
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

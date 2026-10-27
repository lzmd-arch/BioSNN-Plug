"""阶段四端到端 v5：两段式 WM → DPAR。

预注册见 ``sweep_results/stage4_e2e_v5_prereg.md``。WM 在 t=35 读，DPAR 在 t=85 读；
两个探针不共享最终状态，避免 v4 的共存干扰，但保留 v4 的共享状态结果。
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
    average_r2,
    encode_timeline,
    fit_ridge,
    predict,
)
from research.cognitive.world_model import auroc
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "stage4_e2e_v5"

N_DIMS = 4
PAIRS = 4
WM_STEP = 35
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
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
    cues = list(wm_cues)
    for index in range(PAIRS):
        cues.append((40 + 10 * index, 44 + 10 * index, keys[index]))
        cues.append((45 + 10 * index, 49 + 10 * index, values[index]))
    cues.append((80, 84, keys[pair]))

    previous = encode_timeline(wm_cues, rng=rng, test_step=WM_STEP - 1)
    wm_state = encode_timeline(wm_cues, rng=rng, test_step=WM_STEP)
    final_state = encode_timeline(cues, rng=rng, test_step=FINAL_STEP)
    wm_target = np.concatenate([cue_a, cue_b, query])
    return wm_state, previous, wm_target, keys, values, pair, final_state


def _dataset(
    episodes: int, *, seed: int, ood: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    wm_states, previous, targets, keys, values, pairs, final_states = [], [], [], [], [], [], []
    for _ in range(episodes):
        wm, prev, target, key, value, pair, final = _episode(rng, ood=ood)
        wm_states.append(wm)
        previous.append(prev)
        targets.append(target)
        keys.append(key)
        values.append(value)
        pairs.append(pair)
        final_states.append(final)
    return (
        np.stack(wm_states),
        np.stack(previous),
        np.stack(targets),
        np.stack(keys),
        np.stack(values),
        np.asarray(pairs),
        np.stack(final_states),
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


def run(seeds: list[int], *, out: Path, test_episodes: int = 500) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        train = _dataset(2000, seed=seed)
        calibration = _dataset(test_episodes, seed=seed + 30_000)
        test = _dataset(test_episodes, seed=seed + 10_000)
        ood = _dataset(test_episodes, seed=seed + 20_000, ood=True)

        train_states, train_previous, train_targets = train[0], train[1], train[2]
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
        ood_features = _features(
            ood[0], ood[1], ood[2], readout=readout, members=members, world=world
        )

        predicted = readout.predict(test[0])
        r2 = [
            average_r2(
                predicted[:, i * N_DIMS : (i + 1) * N_DIMS],
                test[2][:, i * N_DIMS : (i + 1) * N_DIMS],
            )
            for i in range(3)
        ]
        ood_errors = _features(
            ood[0], ood[1], ood[2], readout=readout, members=members, world=world
        )["errors"]

        def zscore(values, reference):
            return (values - reference.mean()) / max(reference.std(), 1e-8)

        test_confidence = -zscore(
            test_features["disagreement"], calibration_features["disagreement"]
        ) - zscore(test_features["surprise"], calibration_features["surprise"])
        ood_confidence = -zscore(
            ood_features["disagreement"], calibration_features["disagreement"]
        ) - zscore(ood_features["surprise"], calibration_features["surprise"])
        calibration_confidence = -zscore(
            calibration_features["disagreement"], calibration_features["disagreement"]
        ) - zscore(calibration_features["surprise"], calibration_features["surprise"])
        threshold = float(np.quantile(calibration_confidence, 1.0 - COVERAGE))
        answered = test_confidence >= threshold
        ood_answered = ood_confidence >= threshold
        selective_mse = (
            float(np.mean(test_features["errors"][answered])) if np.any(answered) else float("nan")
        )
        ood_abstain = float(np.mean(~ood_answered))
        auc = auroc(
            np.concatenate([test_confidence, ood_confidence]),
            np.concatenate([np.ones_like(test_confidence), np.zeros_like(ood_confidence)]),
        )

        dpar = _dpar_scores(test[3], test[4], test[5], seed=seed + 400_000)
        dpar_best = max(dpar["pseudo"], dpar["rawnn"])

        state_readout = LinearRLSReadout(train[6].shape[1], N_DIMS, delta=1000.0)
        train_values = np.stack(
            [values[pair] for values, pair in zip(train[4], train[5], strict=True)]
        )
        state_readout.fit(
            train[6], train_values, epochs=EPOCHS, rng=np.random.default_rng(seed + 500_000)
        )
        state_predictions = state_readout.predict(test[6])
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
    gate["e2e_v5_ready"] = all(
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
        "experiment": "阶段四端到端 v5：两段式 WM → DPAR",
        "config": {
            "seeds": seeds,
            "pairs": PAIRS,
            "wm_step": WM_STEP,
            "final_step": FINAL_STEP,
            "cue_noise": CUE_NOISE,
            "test_episodes": test_episodes,
            "coverage": COVERAGE,
        },
        "per_seed": rows,
        "gate": gate,
        "repro": collect(
            "阶段四端到端 v5",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "WM probe at t=35（A/B/Q）；DPAR 在 t=85（K=4 键值 + 查询）",
                "两段不共享最终状态；v4 的共享状态干扰结果原样保留",
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
    parser = argparse.ArgumentParser(description="阶段四端到端 v5")
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

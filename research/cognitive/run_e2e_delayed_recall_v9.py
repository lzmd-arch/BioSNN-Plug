"""阶段四端到端 v9：v8 的尺度/K/噪声稳健性扫描。

预注册见 ``sweep_results/stage4_e2e_v9_prereg.md``。结构固定为 v8 的双 bank +
bank W 尺度门；一次只动一条轴（s / K / σ），主档 K=4、σ=0.3、s=3。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.cognitive.run_e2e_delayed_recall_v4 import _dpar_scores, _top1
from research.cognitive.run_e2e_delayed_recall_v8 import _feature_set
from research.cognitive.run_metacognitive_gate_v4 import _evaluate, _gaussian_gate
from research.cognitive.working_memory import (
    LinearRLSReadout,
    encode_timeline,
    fit_ridge,
)
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "stage4_e2e_v9"

N_DIMS = 4
BANK = 28
MEMBERS = 5
EPOCHS = 20
COVERAGE = 0.80

R2_MIN = 0.90
SELECTIVE_MSE_MAX = 0.10
DPAR_MIN = 0.80
OOD_AUROC_MIN = 0.75
OOD_ABSTAIN_MIN = 0.80

CONDITIONS = (
    ("main", 4, 0.3, 3.0),
    ("scale-1.5", 4, 0.3, 1.5),
    ("scale-2", 4, 0.3, 2.0),
    ("scale-5", 4, 0.3, 5.0),
    ("K-2", 2, 0.3, 3.0),
    ("K-8", 8, 0.3, 3.0),
    ("sigma-0.1", 4, 0.1, 3.0),
    ("sigma-0.5", 4, 0.5, 3.0),
)


def _episode(
    rng: np.random.Generator,
    *,
    pairs: int,
    cue_noise: float,
    scale: float,
    ood: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    cue_a = rng.normal(size=N_DIMS)
    cue_b = rng.normal(size=N_DIMS)
    if ood:
        cue_a = cue_a * scale
        cue_b = cue_b * scale
    else:
        cue_a = cue_a + rng.normal(scale=cue_noise, size=N_DIMS)
        cue_b = cue_b + rng.normal(scale=cue_noise, size=N_DIMS)
    choose_b = bool(rng.integers(0, 2))
    query = np.array([0.0, 1.0, 0.0, 0.0] if choose_b else [1.0, 0.0, 0.0, 0.0])
    keys = rng.normal(size=(pairs, N_DIMS))
    values = rng.normal(size=(pairs, N_DIMS))
    pair = int(rng.integers(0, pairs))

    wm_cues = [(0, 4, cue_a), (15, 19, cue_b), (30, 34, query)]
    dpar_cues: list[tuple[int, int, np.ndarray]] = []
    for index in range(pairs):
        dpar_cues.append((40 + 10 * index, 44 + 10 * index, keys[index]))
        dpar_cues.append((45 + 10 * index, 49 + 10 * index, values[index]))
    dpar_cues.append((40 + 10 * pairs, 44 + 10 * pairs, keys[pair]))
    final_step = 40 + 10 * pairs + 5

    single_state = encode_timeline(wm_cues + dpar_cues, rng=rng, test_step=final_step)
    single_previous = encode_timeline(wm_cues + dpar_cues, rng=rng, test_step=final_step - 1)
    wm_state = encode_timeline(wm_cues, rng=rng, test_step=final_step)
    wm_previous = encode_timeline(wm_cues, rng=rng, test_step=final_step - 1)
    dpar_state = encode_timeline(dpar_cues, rng=rng, test_step=final_step)
    dpar_previous = encode_timeline(dpar_cues, rng=rng, test_step=final_step - 1)
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
    episodes: int,
    *,
    seed: int,
    pairs: int,
    cue_noise: float,
    scale: float,
    ood: bool = False,
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
    targets, keys, values, pairs_out = [], [], [], []
    for _ in range(episodes):
        single, single_prev, dual, dual_prev, target, key, value, pair = _episode(
            rng, pairs=pairs, cue_noise=cue_noise, scale=scale, ood=ood
        )
        single_states.append(single)
        single_previous.append(single_prev)
        dual_states.append(dual)
        dual_previous.append(dual_prev)
        targets.append(target)
        keys.append(key)
        values.append(value)
        pairs_out.append(pair)
    return (
        np.stack(single_states),
        np.stack(single_previous),
        np.stack(dual_states),
        np.stack(dual_previous),
        np.stack(targets),
        np.stack(keys),
        np.stack(values),
        np.asarray(pairs_out),
    )


def _safe_r2(predicted: np.ndarray, target: np.ndarray) -> float:
    residual = ((predicted - target) ** 2).sum(axis=0)
    total = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    varying = total > 1e-9
    return float(np.mean(1.0 - residual[varying] / total[varying]))


def _evaluate_condition(
    train,
    calibration,
    test,
    ood,
    *,
    seed: int,
) -> dict[str, float]:
    train_wm = train[2][:, :BANK]
    train_wm_previous = train[3][:, :BANK]
    calibration_wm = calibration[2][:, :BANK]
    calibration_wm_previous = calibration[3][:, :BANK]
    test_wm = test[2][:, :BANK]
    test_wm_previous = test[3][:, :BANK]
    ood_wm = ood[2][:, :BANK]
    ood_wm_previous = ood[3][:, :BANK]

    readout = LinearRLSReadout(BANK, train[4].shape[1], delta=1000.0)
    readout.fit(train_wm, train[4], epochs=EPOCHS, rng=np.random.default_rng(seed))
    members = []
    for member_index in range(MEMBERS):
        rng = np.random.default_rng(seed + 100 + member_index)
        index = rng.choice(train_wm.shape[0], size=train_wm.shape[0], replace=True)
        member = LinearRLSReadout(BANK, train[4].shape[1], delta=1000.0)
        member.fit(train_wm[index], train[4][index], epochs=EPOCHS, rng=rng)
        members.append(member)
    world = fit_ridge(train_wm_previous, train_wm)

    calibration_features = _feature_set(
        calibration_wm,
        calibration_wm_previous,
        calibration[4],
        readout=readout,
        members=members,
        world=world,
    )
    test_features = _feature_set(
        test_wm, test_wm_previous, test[4], readout=readout, members=members, world=world
    )
    ood_features = _feature_set(
        ood_wm, ood_wm_previous, ood[4], readout=readout, members=members, world=world
    )
    confidence = _gaussian_gate(
        calibration_features,
        test_features,
        ood_features,
        ("disagreement", "surprise", "state_norm"),
    )
    metrics = _evaluate(calibration_features, test_features, ood_features, confidence=confidence)

    predicted = readout.predict(test_wm)
    r2 = [
        _safe_r2(
            predicted[:, i * N_DIMS : (i + 1) * N_DIMS],
            test[4][:, i * N_DIMS : (i + 1) * N_DIMS],
        )
        for i in range(3)
    ]

    dpar = _dpar_scores(test[5], test[6], test[7], seed=seed + 400_000)
    dpar_best = max(dpar["pseudo"], dpar["rawnn"])
    train_values = np.stack([values[pair] for values, pair in zip(train[6], train[7], strict=True)])
    state_readout = LinearRLSReadout(train[2].shape[1], N_DIMS, delta=1000.0)
    state_readout.fit(
        train[2], train_values, epochs=EPOCHS, rng=np.random.default_rng(seed + 500_000)
    )
    state_predictions = state_readout.predict(test[2])
    state_top1 = float(
        np.mean(
            [
                _top1(prediction, values) == pair
                for prediction, values, pair in zip(
                    state_predictions, test[6], test[7], strict=True
                )
            ]
        )
    )
    return {
        "r2_mean": float(np.mean(r2)),
        "selective_mse": metrics["selective_mse"],
        "test_coverage": metrics["test_coverage"],
        "ood_auroc": metrics["ood_auroc"],
        "ood_abstain": metrics["ood_abstain"],
        "dpar_best": dpar_best,
        "dpar_pseudo": dpar["pseudo"],
        "dpar_rawnn": dpar["rawnn"],
        "dpar_hebbian": dpar["hebbian"],
        "state_control_top1": state_top1,
    }


def _checks(row: dict[str, float], pairs: int) -> dict[str, bool]:
    return {
        "r2_ok": row["r2_mean"] >= R2_MIN,
        "selective_ok": row["selective_mse"] <= SELECTIVE_MSE_MAX,
        "dpar_ok": row["dpar_best"] >= DPAR_MIN,
        "state_control_ok": row["state_control_top1"] <= 1.0 / pairs + 0.10,
        "ood_auroc_ok": row["ood_auroc"] >= OOD_AUROC_MIN,
        "ood_abstain_ok": row["ood_abstain"] >= OOD_ABSTAIN_MIN,
    }


def run(seeds: list[int], *, out: Path, test_episodes: int = 500) -> dict[str, Any]:
    started = time.perf_counter()
    condition_results: dict[str, Any] = {}
    for name, pairs, cue_noise, scale in CONDITIONS:
        rows = []
        for seed in seeds:
            train = _dataset(2000, seed=seed, pairs=pairs, cue_noise=cue_noise, scale=scale)
            calibration = _dataset(
                test_episodes, seed=seed + 30_000, pairs=pairs, cue_noise=cue_noise, scale=scale
            )
            test = _dataset(
                test_episodes, seed=seed + 10_000, pairs=pairs, cue_noise=cue_noise, scale=scale
            )
            ood = _dataset(
                test_episodes,
                seed=seed + 20_000,
                pairs=pairs,
                cue_noise=cue_noise,
                scale=scale,
                ood=True,
            )
            row = {
                "seed": seed,
                **_evaluate_condition(train, calibration, test, ood, seed=seed),
            }
            rows.append(row)
            print(
                f"{name} seed {seed}: R2 {row['r2_mean']:.3f} "
                f"MSE {row['selective_mse']:.3f} OOD {row['ood_abstain']:.3f}/"
                f"{row['ood_auroc']:.3f} DPAR {row['dpar_best']:.3f} "
                f"state {row['state_control_top1']:.3f}",
                flush=True,
            )
        mean = {
            key: float(np.mean([row[key] for row in rows]))
            for key in (
                "r2_mean",
                "selective_mse",
                "test_coverage",
                "ood_auroc",
                "ood_abstain",
                "dpar_best",
                "state_control_top1",
            )
        }
        checks = _checks(mean, pairs)
        condition_results[name] = {
            "pairs": pairs,
            "cue_noise": cue_noise,
            "ood_scale": scale,
            "per_seed": rows,
            "mean": mean,
            "checks": checks,
            "ready": all(checks.values()),
        }

    required = [name for name in condition_results if name != "scale-1.5"]
    main_ready = condition_results["main"]["ready"]
    robust = all(condition_results[name]["ready"] for name in required)
    summary = {
        "experiment": "阶段四端到端 v9：v8 的稳健性扫描",
        "config": {
            "seeds": seeds,
            "test_episodes": test_episodes,
            "coverage": COVERAGE,
            "conditions": {
                name: {"pairs": p, "cue_noise": s, "ood_scale": sc} for name, p, s, sc in CONDITIONS
            },
        },
        "conditions": condition_results,
        "main_ready": main_ready,
        "robust": robust,
        "v9_ready": bool(main_ready and robust),
        "repro": collect(
            "阶段四端到端 v9",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "v8 双 bank + bank W gauss3 尺度门；一次只动 s/K/σ 一条轴",
                "稳健判据排除 scale-1.5；它只报告",
                "OOD A/B 乘 s；in-dist A/B 加 σ；阈值只在 calibration 上取",
            ],
            device=None,
        ).render_block(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                name: {
                    "mean": result["mean"],
                    "ready": result["ready"],
                }
                for name, result in condition_results.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(json.dumps({"main_ready": main_ready, "robust": robust, "v9_ready": summary["v9_ready"]}))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段四端到端 v9")
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

"""阶段四端到端 v15：只补 v14 的 4 个 s=2 失败格。

预注册见 ``sweep_results/stage4_e2e_v15_prereg.md``。每格两臂：
K4-s2-n0.5 用 baseline/longtau + shrunk；K8-s2 用 snapshot85/longtau + diag。
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
from research.cognitive.run_e2e_delayed_recall_v10 import (
    _diag_mahalanobis,
    _shrunk_mahalanobis,
)
from research.cognitive.run_e2e_delayed_recall_v11 import _safe_r2
from research.cognitive.run_metacognitive_gate_v4 import _evaluate
from research.cognitive.working_memory import (
    TAUS,
    LinearRLSReadout,
    encode_timeline,
    fit_ridge,
)
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "stage4_e2e_v15"

N_DIMS = 4
MEMBERS = 5
EPOCHS = 20
COVERAGE = 0.80
COVERAGE_LOW = 0.78
COVERAGE_HIGH = 0.82

R2_MIN = 0.90
SELECTIVE_MSE_MAX = 0.10
DPAR_MIN = 0.80
OOD_AUROC_MIN = 0.75
OOD_ABSTAIN_MIN = 0.80

# name, pairs, scale, noise, arms, primary_arm
CONDITIONS = (
    ("K4-s2-n0.5", 4, 2.0, 0.5, ("baseline", "longtau"), "longtau"),
    ("K8-s2-n0.1", 8, 2.0, 0.1, ("snapshot85", "longtau"), "snapshot85"),
    ("K8-s2-n0.3", 8, 2.0, 0.3, ("snapshot85", "longtau"), "snapshot85"),
    ("K8-s2-n0.5", 8, 2.0, 0.5, ("snapshot85", "longtau"), "snapshot85"),
)


def _episode(
    rng: np.random.Generator,
    *,
    pairs: int,
    scale: float,
    cue_noise: float,
    ood: bool,
    arm: str,
) -> tuple[np.ndarray, ...]:
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

    if arm == "snapshot85":
        wm_step = 85
        taus = TAUS
    elif arm == "longtau":
        wm_step = final_step
        taus = (*TAUS, 400)
    else:
        wm_step = final_step
        taus = TAUS
    wm_state = encode_timeline(wm_cues, rng=rng, test_step=wm_step, taus=taus)
    wm_previous = encode_timeline(wm_cues, rng=rng, test_step=wm_step - 1, taus=taus)
    dpar_state = encode_timeline(dpar_cues, rng=rng, test_step=final_step)
    dpar_previous = encode_timeline(dpar_cues, rng=rng, test_step=final_step - 1)
    target = np.concatenate([cue_a, cue_b, query])
    return wm_state, wm_previous, dpar_state, dpar_previous, target, keys, values, pair


def _dataset(
    episodes: int,
    *,
    seed: int,
    pairs: int,
    scale: float,
    cue_noise: float,
    arm: str,
    ood: bool = False,
) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    fields = [[] for _ in range(7)]
    pair_values = []
    for _ in range(episodes):
        *values, pair = _episode(
            rng,
            pairs=pairs,
            scale=scale,
            cue_noise=cue_noise,
            ood=ood,
            arm=arm,
        )
        for target, value in zip(fields, values, strict=True):
            target.append(value)
        pair_values.append(pair)
    return (*tuple(np.stack(field) for field in fields), np.asarray(pair_values))


def _checks(metrics: dict[str, float], pairs: int) -> dict[str, bool]:
    return {
        "r2_ok": metrics["r2_mean"] >= R2_MIN,
        "selective_ok": metrics["selective_mse"] <= SELECTIVE_MSE_MAX,
        "dpar_ok": metrics["dpar_best"] >= DPAR_MIN,
        "state_control_ok": metrics["state_control_top1"] <= 1.0 / pairs + 0.10,
        "ood_auroc_ok": metrics["ood_auroc"] >= OOD_AUROC_MIN,
        "ood_abstain_ok": metrics["ood_abstain"] >= OOD_ABSTAIN_MIN,
        "coverage_ok": COVERAGE_LOW <= metrics["test_coverage"] <= COVERAGE_HIGH,
    }


def run(
    seeds: list[int],
    *,
    out: Path,
    test_episodes: int = 500,
    train_episodes: int = 2000,
) -> dict[str, Any]:
    started = time.perf_counter()
    results: dict[str, Any] = {}
    for name, pairs, scale, cue_noise, arms, primary in CONDITIONS:
        cell: dict[str, Any] = {}
        for arm in arms:
            rows = []
            for seed in seeds:
                train = _dataset(
                    train_episodes,
                    seed=seed,
                    pairs=pairs,
                    scale=scale,
                    cue_noise=cue_noise,
                    arm=arm,
                )
                calibration = _dataset(
                    test_episodes,
                    seed=seed + 30_000,
                    pairs=pairs,
                    scale=scale,
                    cue_noise=cue_noise,
                    arm=arm,
                )
                test = _dataset(
                    test_episodes,
                    seed=seed + 10_000,
                    pairs=pairs,
                    scale=scale,
                    cue_noise=cue_noise,
                    arm=arm,
                )
                ood = _dataset(
                    test_episodes,
                    seed=seed + 20_000,
                    pairs=pairs,
                    scale=scale,
                    cue_noise=cue_noise,
                    arm=arm,
                    ood=True,
                )
                readout = LinearRLSReadout(train[0].shape[1], train[4].shape[1], delta=1000.0)
                readout.fit(train[0], train[4], epochs=EPOCHS, rng=np.random.default_rng(seed))
                members = []
                for member_index in range(MEMBERS):
                    rng = np.random.default_rng(seed + 100 + member_index)
                    index = rng.choice(train[0].shape[0], size=train[0].shape[0], replace=True)
                    member = LinearRLSReadout(train[0].shape[1], train[4].shape[1], delta=1000.0)
                    member.fit(train[0][index], train[4][index], epochs=EPOCHS, rng=rng)
                    members.append(member)
                world = fit_ridge(train[1], train[0])

                calibration_features = _feature_set(
                    calibration[0],
                    calibration[1],
                    calibration[4],
                    readout=readout,
                    members=members,
                    world=world,
                )
                test_features = _feature_set(
                    test[0], test[1], test[4], readout=readout, members=members, world=world
                )
                ood_features = _feature_set(
                    ood[0], ood[1], ood[4], readout=readout, members=members, world=world
                )
                if pairs == 4:
                    confidence = _shrunk_mahalanobis(calibration[0], test[0], ood[0])
                else:
                    confidence = _diag_mahalanobis(calibration[0], test[0], ood[0])
                metrics = _evaluate(
                    calibration_features,
                    test_features,
                    ood_features,
                    confidence=confidence,
                )
                predicted = readout.predict(test[0])
                r2 = [
                    _safe_r2(
                        predicted[:, i * N_DIMS : (i + 1) * N_DIMS],
                        test[4][:, i * N_DIMS : (i + 1) * N_DIMS],
                    )
                    for i in range(3)
                ]
                dpar = _dpar_scores(test[5], test[6], test[7], seed=seed + 400_000)
                train_values = np.stack(
                    [values[pair] for values, pair in zip(train[6], train[7], strict=True)]
                )
                train_control = np.concatenate([train[0], train[2]], axis=1)
                test_control = np.concatenate([test[0], test[2]], axis=1)
                state_readout = LinearRLSReadout(train_control.shape[1], N_DIMS, delta=1000.0)
                state_readout.fit(
                    train_control,
                    train_values,
                    epochs=EPOCHS,
                    rng=np.random.default_rng(seed + 500_000),
                )
                state_predictions = state_readout.predict(test_control)
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
                rows.append(
                    {
                        "seed": seed,
                        "r2_mean": float(np.mean(r2)),
                        "selective_mse": metrics["selective_mse"],
                        "test_coverage": metrics["test_coverage"],
                        "ood_auroc": metrics["ood_auroc"],
                        "ood_abstain": metrics["ood_abstain"],
                        "dpar_best": max(dpar["pseudo"], dpar["rawnn"]),
                        "state_control_top1": state_top1,
                    }
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
            cell[arm] = {
                "mean": mean,
                "checks": checks,
                "ready": all(checks.values()),
            }
            print(
                f"{name} {arm}: R² {mean['r2_mean']:.3f} MSE {mean['selective_mse']:.3f} "
                f"cov {mean['test_coverage']:.3f} abstain {mean['ood_abstain']:.3f} "
                f"AUROC {mean['ood_auroc']:.3f} DPAR {mean['dpar_best']:.3f} "
                f"state {mean['state_control_top1']:.3f} ready={cell[arm]['ready']}",
                flush=True,
            )
        cell["primary"] = primary
        cell["primary_ready"] = cell[primary]["ready"]
        results[name] = cell

    summary = {
        "experiment": "阶段四端到端 v15：补 v14 的 s=2 四格",
        "config": {
            "seeds": seeds,
            "test_episodes": test_episodes,
            "conditions": [name for name, *_ in CONDITIONS],
        },
        "conditions": results,
        "v15_ready": all(cell["primary_ready"] for cell in results.values()),
        "repro": collect(
            "阶段四端到端 v15",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "只跑 v14 的 4 个 s=2 失败格；K4-σ0.5 主臂 longtau，K8-s2 主臂 snapshot85",
                "门：K4 shrunk / K8 diag；覆盖率 0.78-0.82",
            ],
            device=None,
        ).render_block(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({name: cell["primary_ready"] for name, cell in results.items()}))
    print(json.dumps({"v15_ready": summary["v15_ready"]}))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段四端到端 v15")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--test-episodes", type=int, default=500)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    run(
        parse_seeds(args.seeds),
        out=args.out,
        test_episodes=20 if args.smoke else args.test_episodes,
        train_episodes=50 if args.smoke else 2000,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

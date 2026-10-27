"""阶段四端到端 v11：K=8 的容量/探针时间边界。

预注册见 ``sweep_results/stage4_e2e_v11_prereg.md``。三臂：baseline（bank W t=125）、
snapshot85（bank W t=85）、longtau（t=125 + tau=400）；bank D 与 DPAR 不变。
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
from research.cognitive.run_e2e_delayed_recall_v10 import _shrunk_mahalanobis
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
OUT = REPO_ROOT / "sweep_results" / "stage4_e2e_v11"

N_DIMS = 4
PAIRS = 8
FINAL_STEP = 125
SNAPSHOT_STEP = 85
MEMBERS = 5
EPOCHS = 20
COVERAGE = 0.80

R2_MIN = 0.90
SELECTIVE_MSE_MAX = 0.10
DPAR_MIN = 0.80
STATE_CONTROL_MAX = 1.0 / PAIRS + 0.10
OOD_AUROC_MIN = 0.75
OOD_ABSTAIN_MIN = 0.80
COVERAGE_LOW = 0.78
COVERAGE_HIGH = 0.82


def _episode(rng: np.random.Generator, *, ood: bool) -> tuple:
    cue_a = rng.normal(size=N_DIMS)
    cue_b = rng.normal(size=N_DIMS)
    if ood:
        cue_a = cue_a * 3.0
        cue_b = cue_b * 3.0
    else:
        cue_a = cue_a + rng.normal(scale=0.3, size=N_DIMS)
        cue_b = cue_b + rng.normal(scale=0.3, size=N_DIMS)
    choose_b = bool(rng.integers(0, 2))
    query = np.array([0.0, 1.0, 0.0, 0.0] if choose_b else [1.0, 0.0, 0.0, 0.0])
    keys = rng.normal(size=(PAIRS, N_DIMS))
    values = rng.normal(size=(PAIRS, N_DIMS))
    pair = int(rng.integers(0, PAIRS))

    wm_cues = [(0, 4, cue_a), (15, 19, cue_b), (30, 34, query)]
    dpar_cues: list[tuple[int, int, np.ndarray]] = []
    for index in range(PAIRS):
        dpar_cues.append((40 + 10 * index, 44 + 10 * index, keys[index]))
        dpar_cues.append((45 + 10 * index, 49 + 10 * index, values[index]))
    dpar_cues.append((40 + 10 * PAIRS, 44 + 10 * PAIRS, keys[pair]))

    wm_final = encode_timeline(wm_cues, rng=rng, test_step=FINAL_STEP)
    wm_final_previous = encode_timeline(wm_cues, rng=rng, test_step=FINAL_STEP - 1)
    wm_snapshot = encode_timeline(wm_cues, rng=rng, test_step=SNAPSHOT_STEP)
    wm_snapshot_previous = encode_timeline(wm_cues, rng=rng, test_step=SNAPSHOT_STEP - 1)
    wm_long = encode_timeline(wm_cues, rng=rng, test_step=FINAL_STEP, taus=(*TAUS, 400))
    wm_long_previous = encode_timeline(
        wm_cues, rng=rng, test_step=FINAL_STEP - 1, taus=(*TAUS, 400)
    )
    dpar_state = encode_timeline(dpar_cues, rng=rng, test_step=FINAL_STEP)
    dpar_previous = encode_timeline(dpar_cues, rng=rng, test_step=FINAL_STEP - 1)
    target = np.concatenate([cue_a, cue_b, query])
    return (
        wm_final,
        wm_final_previous,
        wm_snapshot,
        wm_snapshot_previous,
        wm_long,
        wm_long_previous,
        dpar_state,
        dpar_previous,
        target,
        keys,
        values,
        pair,
    )


def _dataset(episodes: int, *, seed: int, ood: bool = False) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    fields = [[] for _ in range(11)]
    pairs = []
    for _ in range(episodes):
        *values, pair = _episode(rng, ood=ood)
        for target, value in zip(fields, values, strict=True):
            target.append(value)
        pairs.append(pair)
    return (*tuple(np.stack(field) for field in fields), np.asarray(pairs))


def _safe_r2(predicted: np.ndarray, target: np.ndarray) -> float:
    residual = ((predicted - target) ** 2).sum(axis=0)
    total = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    varying = total > 1e-9
    return float(np.mean(1.0 - residual[varying] / total[varying]))


def _arm_metrics(
    *,
    train_wm: np.ndarray,
    train_wm_previous: np.ndarray,
    train_targets: np.ndarray,
    calibration,
    test,
    ood,
    dpar: dict[str, float],
    seed: int,
) -> dict[str, float]:
    readout = LinearRLSReadout(train_wm.shape[1], train_targets.shape[1], delta=1000.0)
    readout.fit(train_wm, train_targets, epochs=EPOCHS, rng=np.random.default_rng(seed))
    members = []
    for member_index in range(MEMBERS):
        rng = np.random.default_rng(seed + 100 + member_index)
        index = rng.choice(train_wm.shape[0], size=train_wm.shape[0], replace=True)
        member = LinearRLSReadout(train_wm.shape[1], train_targets.shape[1], delta=1000.0)
        member.fit(train_wm[index], train_targets[index], epochs=EPOCHS, rng=rng)
        members.append(member)
    world = fit_ridge(train_wm_previous, train_wm)

    calibration_features = _feature_set(
        calibration[0],
        calibration[1],
        calibration[2],
        readout=readout,
        members=members,
        world=world,
    )
    test_features = _feature_set(
        test[0], test[1], test[2], readout=readout, members=members, world=world
    )
    ood_features = _feature_set(
        ood[0], ood[1], ood[2], readout=readout, members=members, world=world
    )
    confidence = _shrunk_mahalanobis(calibration[0], test[0], ood[0])
    metrics = _evaluate(calibration_features, test_features, ood_features, confidence=confidence)

    predicted = readout.predict(test[0])
    r2 = [
        _safe_r2(
            predicted[:, i * N_DIMS : (i + 1) * N_DIMS],
            test[2][:, i * N_DIMS : (i + 1) * N_DIMS],
        )
        for i in range(3)
    ]
    return {
        "r2_mean": float(np.mean(r2)),
        "selective_mse": metrics["selective_mse"],
        "test_coverage": metrics["test_coverage"],
        "ood_auroc": metrics["ood_auroc"],
        "ood_abstain": metrics["ood_abstain"],
        "dpar_best": max(dpar["pseudo"], dpar["rawnn"]),
    }


def run(seeds: list[int], *, out: Path, test_episodes: int = 500) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        train = _dataset(2000, seed=seed)
        calibration = _dataset(test_episodes, seed=seed + 30_000)
        test = _dataset(test_episodes, seed=seed + 10_000)
        ood = _dataset(test_episodes, seed=seed + 20_000, ood=True)
        dpar = _dpar_scores(test[9], test[10], test[11], seed=seed + 400_000)
        train_values = np.stack(
            [values[pair] for values, pair in zip(train[10], train[11], strict=True)]
        )

        arms = {
            "baseline": (0, 1, 6),
            "snapshot85": (2, 3, 6),
            "longtau": (4, 5, 6),
        }
        row: dict[str, Any] = {"seed": seed}
        for arm, (wm_idx, prev_idx, dpar_idx) in arms.items():
            metrics = _arm_metrics(
                train_wm=train[wm_idx],
                train_wm_previous=train[prev_idx],
                train_targets=train[8],
                calibration=(calibration[wm_idx], calibration[prev_idx], calibration[8]),
                test=(test[wm_idx], test[prev_idx], test[8]),
                ood=(ood[wm_idx], ood[prev_idx], ood[8]),
                dpar=dpar,
                seed=seed,
            )
            train_control = np.concatenate([train[wm_idx], train[dpar_idx]], axis=1)
            test_control = np.concatenate([test[wm_idx], test[dpar_idx]], axis=1)
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
                            state_predictions, test[10], test[11], strict=True
                        )
                    ]
                )
            )
            for key, value in metrics.items():
                row[f"{arm}_{key}"] = value
            row[f"{arm}_state_control_top1"] = state_top1
        rows.append(row)
        print(
            f"seed {seed}: "
            + " | ".join(
                f"{arm} R2 {row[f'{arm}_r2_mean']:.3f} cov {row[f'{arm}_test_coverage']:.3f} "
                f"abstain {row[f'{arm}_ood_abstain']:.3f}"
                for arm in arms
            )
            + f" | DPAR {max(dpar['pseudo'], dpar['rawnn']):.3f}",
            flush=True,
        )

    def mean(arm: str, key: str) -> float:
        return float(np.mean([row[f"{arm}_{key}"] for row in rows]))

    results = {}
    for arm in ("baseline", "snapshot85", "longtau"):
        metrics = {
            key: mean(arm, key)
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
        checks = {
            "r2_ok": metrics["r2_mean"] >= R2_MIN,
            "selective_ok": metrics["selective_mse"] <= SELECTIVE_MSE_MAX,
            "dpar_ok": metrics["dpar_best"] >= DPAR_MIN,
            "state_control_ok": metrics["state_control_top1"] <= STATE_CONTROL_MAX,
            "ood_auroc_ok": metrics["ood_auroc"] >= OOD_AUROC_MIN,
            "ood_abstain_ok": metrics["ood_abstain"] >= OOD_ABSTAIN_MIN,
            "coverage_ok": COVERAGE_LOW <= metrics["test_coverage"] <= COVERAGE_HIGH,
        }
        results[arm] = {
            "mean": metrics,
            "checks": checks,
            "ready": all(checks.values()),
        }
    summary = {
        "experiment": "阶段四端到端 v11：K=8 的容量/探针时间边界",
        "config": {
            "seeds": seeds,
            "pairs": PAIRS,
            "final_step": FINAL_STEP,
            "snapshot_step": SNAPSHOT_STEP,
            "test_episodes": test_episodes,
            "coverage": COVERAGE,
        },
        "per_seed": rows,
        "arms": results,
        "v11_ready": bool(results["snapshot85"]["ready"] or results["longtau"]["ready"]),
        "repro": collect(
            "阶段四端到端 v11",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "K=8；三臂 baseline/snapshot85/longtau；bank D 与 DPAR 不变",
                "主门 shrunk Mahalanobis（lambda=0.1）；覆盖率判据 0.78-0.82",
                "s=3、sigma=0.3；阈值只在 calibration 上取",
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
                arm: {"mean": results[arm]["mean"], "ready": results[arm]["ready"]}
                for arm in results
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(json.dumps({"v11_ready": summary["v11_ready"]}))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段四端到端 v11")
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

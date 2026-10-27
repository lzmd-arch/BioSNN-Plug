"""阶段四端到端 v12：K=8 用对角协方差门。

预注册见 ``sweep_results/stage4_e2e_v12_prereg.md``。复用 v11 的三臂；每臂同时报
diag（主门）与 shrunk（对照）。
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
from research.cognitive.run_e2e_delayed_recall_v11 import _dataset, _safe_r2
from research.cognitive.run_metacognitive_gate_v4 import _evaluate
from research.cognitive.working_memory import LinearRLSReadout, fit_ridge
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "stage4_e2e_v12"

N_DIMS = 4
PAIRS = 8
MEMBERS = 5
EPOCHS = 20
COVERAGE = 0.80
COVERAGE_LOW = 0.78
COVERAGE_HIGH = 0.82

R2_MIN = 0.90
SELECTIVE_MSE_MAX = 0.10
DPAR_MIN = 0.80
STATE_CONTROL_MAX = 1.0 / PAIRS + 0.10
OOD_AUROC_MIN = 0.75
OOD_ABSTAIN_MIN = 0.80

ARMS = ("baseline", "snapshot85", "longtau")


def _checks(metrics: dict[str, float]) -> dict[str, bool]:
    return {
        "r2_ok": metrics["r2_mean"] >= R2_MIN,
        "selective_ok": metrics["selective_mse"] <= SELECTIVE_MSE_MAX,
        "dpar_ok": metrics["dpar_best"] >= DPAR_MIN,
        "state_control_ok": metrics["state_control_top1"] <= STATE_CONTROL_MAX,
        "ood_auroc_ok": metrics["ood_auroc"] >= OOD_AUROC_MIN,
        "ood_abstain_ok": metrics["ood_abstain"] >= OOD_ABSTAIN_MIN,
        "coverage_ok": COVERAGE_LOW <= metrics["test_coverage"] <= COVERAGE_HIGH,
    }


def run(seeds: list[int], *, out: Path, test_episodes: int = 500) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    arm_indices = {"baseline": (0, 1), "snapshot85": (2, 3), "longtau": (4, 5)}
    for seed in seeds:
        train = _dataset(2000, seed=seed)
        calibration = _dataset(test_episodes, seed=seed + 30_000)
        test = _dataset(test_episodes, seed=seed + 10_000)
        ood = _dataset(test_episodes, seed=seed + 20_000, ood=True)
        dpar = _dpar_scores(test[9], test[10], test[11], seed=seed + 400_000)
        train_values = np.stack(
            [values[pair] for values, pair in zip(train[10], train[11], strict=True)]
        )
        row: dict[str, Any] = {"seed": seed}
        for arm in ARMS:
            wm_idx, prev_idx = arm_indices[arm]
            train_wm = train[wm_idx]
            train_prev = train[prev_idx]
            readout = LinearRLSReadout(train_wm.shape[1], train[8].shape[1], delta=1000.0)
            readout.fit(train_wm, train[8], epochs=EPOCHS, rng=np.random.default_rng(seed))
            members = []
            for member_index in range(MEMBERS):
                rng = np.random.default_rng(seed + 100 + member_index)
                index = rng.choice(train_wm.shape[0], size=train_wm.shape[0], replace=True)
                member = LinearRLSReadout(train_wm.shape[1], train[8].shape[1], delta=1000.0)
                member.fit(train_wm[index], train[8][index], epochs=EPOCHS, rng=rng)
                members.append(member)
            world = fit_ridge(train_prev, train_wm)

            calibration_features = _feature_set(
                calibration[wm_idx],
                calibration[prev_idx],
                calibration[8],
                readout=readout,
                members=members,
                world=world,
            )
            test_features = _feature_set(
                test[wm_idx], test[prev_idx], test[8], readout=readout, members=members, world=world
            )
            ood_features = _feature_set(
                ood[wm_idx], ood[prev_idx], ood[8], readout=readout, members=members, world=world
            )
            confidences = {
                "diag": _diag_mahalanobis(calibration[wm_idx], test[wm_idx], ood[wm_idx]),
                "shrunk": _shrunk_mahalanobis(calibration[wm_idx], test[wm_idx], ood[wm_idx]),
            }
            predicted = readout.predict(test[wm_idx])
            r2 = [
                _safe_r2(
                    predicted[:, i * N_DIMS : (i + 1) * N_DIMS],
                    test[8][:, i * N_DIMS : (i + 1) * N_DIMS],
                )
                for i in range(3)
            ]
            for gate, confidence in confidences.items():
                metrics = _evaluate(
                    calibration_features, test_features, ood_features, confidence=confidence
                )
                row[f"{arm}_{gate}_r2_mean"] = float(np.mean(r2))
                row[f"{arm}_{gate}_selective_mse"] = metrics["selective_mse"]
                row[f"{arm}_{gate}_test_coverage"] = metrics["test_coverage"]
                row[f"{arm}_{gate}_ood_auroc"] = metrics["ood_auroc"]
                row[f"{arm}_{gate}_ood_abstain"] = metrics["ood_abstain"]
                row[f"{arm}_{gate}_dpar_best"] = max(dpar["pseudo"], dpar["rawnn"])

            train_control = np.concatenate([train_wm, train[6]], axis=1)
            test_control = np.concatenate([test[wm_idx], test[6]], axis=1)
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
            row[f"{arm}_state_control_top1"] = state_top1
        rows.append(row)
        print(
            f"seed {seed}: "
            + " | ".join(
                f"{arm} diag {row[f'{arm}_diag_r2_mean']:.3f}/"
                f"{row[f'{arm}_diag_test_coverage']:.3f}/"
                f"{row[f'{arm}_diag_ood_abstain']:.3f}"
                for arm in ARMS
            ),
            flush=True,
        )

    def mean(arm: str, gate: str, key: str) -> float:
        return float(np.mean([row[f"{arm}_{gate}_{key}"] for row in rows]))

    results: dict[str, Any] = {}
    for arm in ARMS:
        arm_result: dict[str, Any] = {}
        for gate in ("diag", "shrunk"):
            metrics = {
                key: mean(arm, gate, key)
                for key in (
                    "r2_mean",
                    "selective_mse",
                    "test_coverage",
                    "ood_auroc",
                    "ood_abstain",
                    "dpar_best",
                )
            }
            metrics["state_control_top1"] = float(
                np.mean([row[f"{arm}_state_control_top1"] for row in rows])
            )
            checks = _checks(metrics)
            arm_result[gate] = {
                "mean": metrics,
                "checks": checks,
                "ready": all(checks.values()),
            }
        results[arm] = arm_result
    summary = {
        "experiment": "阶段四端到端 v12：K=8 对角协方差门",
        "config": {
            "seeds": seeds,
            "pairs": PAIRS,
            "test_episodes": test_episodes,
            "coverage": COVERAGE,
            "primary_gate": "diag",
        },
        "per_seed": rows,
        "arms": results,
        "v12_ready": bool(
            results["snapshot85"]["diag"]["ready"] or results["longtau"]["diag"]["ready"]
        ),
        "repro": collect(
            "阶段四端到端 v12",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "K=8 三臂 baseline/snapshot85/longtau；diag 主门，shrunk 对照",
                "s=3、sigma=0.3；阈值只在 calibration 上取；覆盖率 0.78-0.82",
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
                arm: {
                    gate: {
                        "mean": results[arm][gate]["mean"],
                        "ready": results[arm][gate]["ready"],
                    }
                    for gate in ("diag", "shrunk")
                }
                for arm in ARMS
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(json.dumps({"v12_ready": summary["v12_ready"]}))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段四端到端 v12")
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

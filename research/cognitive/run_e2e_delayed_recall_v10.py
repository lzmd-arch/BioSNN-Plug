"""阶段四端到端 v10：收缩全状态协方差门，冲 s=2。

预注册见 ``sweep_results/stage4_e2e_v10_prereg.md``。结构同 v8/v9；主门换成 bank W
28 维状态的收缩 Mahalanobis（lambda=0.1），另报 gauss3/diag/norm。
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
from research.cognitive.run_e2e_delayed_recall_v9 import _dataset
from research.cognitive.run_metacognitive_gate_v4 import _evaluate, _gaussian_gate
from research.cognitive.working_memory import LinearRLSReadout, fit_ridge
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "stage4_e2e_v10"

N_DIMS = 4
BANK = 28
MEMBERS = 5
EPOCHS = 20
COVERAGE = 0.80
SHRINKAGE = 0.1

R2_MIN = 0.90
SELECTIVE_MSE_MAX = 0.10
DPAR_MIN = 0.80
OOD_AUROC_MIN = 0.75
OOD_ABSTAIN_MIN = 0.80

CONDITIONS = (
    ("s2", 4, 0.3, 2.0),
    ("s3", 4, 0.3, 3.0),
    ("s5", 4, 0.3, 5.0),
    ("K8", 8, 0.3, 3.0),
)


def _safe_r2(predicted: np.ndarray, target: np.ndarray) -> float:
    residual = ((predicted - target) ** 2).sum(axis=0)
    total = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    varying = total > 1e-9
    return float(np.mean(1.0 - residual[varying] / total[varying]))


def _shrunk_mahalanobis(
    calibration_states: np.ndarray,
    test_states: np.ndarray,
    ood_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = calibration_states.mean(axis=0)
    scale = np.maximum(calibration_states.std(axis=0), 1e-8)
    reference = (calibration_states - mean) / scale
    covariance = np.atleast_2d(np.cov(reference, rowvar=False))
    dimension = covariance.shape[0]
    covariance = (1.0 - SHRINKAGE) * covariance + SHRINKAGE * (
        np.trace(covariance) / dimension
    ) * np.eye(dimension)
    inverse = np.linalg.pinv(covariance)

    def confidence(states: np.ndarray) -> np.ndarray:
        delta = (states - mean) / scale
        return -np.einsum("ni,ij,nj->n", delta, inverse, delta)

    return confidence(calibration_states), confidence(test_states), confidence(ood_states)


def _diag_mahalanobis(
    calibration_states: np.ndarray,
    test_states: np.ndarray,
    ood_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = calibration_states.mean(axis=0)
    variance = np.maximum(calibration_states.var(axis=0), 1e-8)

    def confidence(states: np.ndarray) -> np.ndarray:
        return -(((states - mean) ** 2) / variance).sum(axis=1)

    return confidence(calibration_states), confidence(test_states), confidence(ood_states)


def _checks(row: dict[str, float], pairs: int) -> dict[str, bool]:
    return {
        "r2_ok": row["r2_mean"] >= R2_MIN,
        "selective_ok": row["selective_mse"] <= SELECTIVE_MSE_MAX,
        "dpar_ok": row["dpar_best"] >= DPAR_MIN,
        "state_control_ok": row["state_control_top1"] <= 1.0 / pairs + 0.10,
        "ood_auroc_ok": row["ood_auroc"] >= OOD_AUROC_MIN,
        "ood_abstain_ok": row["ood_abstain"] >= OOD_ABSTAIN_MIN,
        "coverage_ok": 0.78 <= row["shrunk_coverage"] <= 0.82,
    }


def _condition(
    name: str,
    pairs: int,
    cue_noise: float,
    scale: float,
    seeds: list[int],
    test_episodes: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
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

        confidences = {
            "shrunk": _shrunk_mahalanobis(calibration_wm, test_wm, ood_wm),
            "diag": _diag_mahalanobis(calibration_wm, test_wm, ood_wm),
            "gauss3": _gaussian_gate(
                calibration_features,
                test_features,
                ood_features,
                ("disagreement", "surprise", "state_norm"),
            ),
            "norm": _gaussian_gate(
                calibration_features, test_features, ood_features, ("state_norm",)
            ),
        }
        metrics = {}
        for gate, confidence in confidences.items():
            metrics[gate] = _evaluate(
                calibration_features, test_features, ood_features, confidence=confidence
            )

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
        train_values = np.stack(
            [values[pair] for values, pair in zip(train[6], train[7], strict=True)]
        )
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
        row = {
            "seed": seed,
            "r2_mean": float(np.mean(r2)),
            "dpar_best": dpar_best,
            "dpar_pseudo": dpar["pseudo"],
            "dpar_rawnn": dpar["rawnn"],
            "dpar_hebbian": dpar["hebbian"],
            "state_control_top1": state_top1,
        }
        for gate, values in metrics.items():
            row[f"{gate}_coverage"] = values["test_coverage"]
            row[f"{gate}_selective_mse"] = values["selective_mse"]
            row[f"{gate}_auroc"] = values["ood_auroc"]
            row[f"{gate}_abstain"] = values["ood_abstain"]
        row["selective_mse"] = row["shrunk_selective_mse"]
        row["ood_auroc"] = row["shrunk_auroc"]
        row["ood_abstain"] = row["shrunk_abstain"]
        rows.append(row)
        print(
            f"{name} seed {seed}: R2 {row['r2_mean']:.3f} MSE {row['selective_mse']:.3f} "
            f"shrunk {row['shrunk_abstain']:.3f}/{row['shrunk_auroc']:.3f} "
            f"gauss3 {row['gauss3_abstain']:.3f}/{row['gauss3_auroc']:.3f} "
            f"DPAR {dpar_best:.3f} state {state_top1:.3f}",
            flush=True,
        )

    mean = {}
    for key in (
        "r2_mean",
        "selective_mse",
        "dpar_best",
        "state_control_top1",
        "shrunk_coverage",
        "shrunk_auroc",
        "shrunk_abstain",
        "diag_coverage",
        "diag_auroc",
        "diag_abstain",
        "gauss3_coverage",
        "gauss3_auroc",
        "gauss3_abstain",
        "norm_coverage",
        "norm_auroc",
        "norm_abstain",
    ):
        mean[key] = float(np.mean([row[key] for row in rows]))
    mean["ood_auroc"] = mean["shrunk_auroc"]
    mean["ood_abstain"] = mean["shrunk_abstain"]
    checks = _checks(mean, pairs)
    return {
        "pairs": pairs,
        "cue_noise": cue_noise,
        "ood_scale": scale,
        "per_seed": rows,
        "mean": mean,
        "checks": checks,
        "ready": all(checks.values()),
    }


def run(seeds: list[int], *, out: Path, test_episodes: int = 500) -> dict[str, Any]:
    started = time.perf_counter()
    results = {
        name: _condition(name, pairs, cue_noise, scale, seeds, test_episodes)
        for name, pairs, cue_noise, scale in CONDITIONS
    }
    s2_ready = results["s2"]["ready"]
    scale_ready = results["s3"]["ready"] and results["s5"]["ready"]
    summary = {
        "experiment": "阶段四端到端 v10：收缩全状态协方差门",
        "config": {
            "seeds": seeds,
            "test_episodes": test_episodes,
            "coverage": COVERAGE,
            "shrinkage": SHRINKAGE,
            "conditions": {
                name: {"pairs": pairs, "cue_noise": cue_noise, "ood_scale": scale}
                for name, pairs, cue_noise, scale in CONDITIONS
            },
        },
        "conditions": results,
        "s2_ready": s2_ready,
        "scale_ready": scale_ready,
        "v10_ready": bool(s2_ready and scale_ready),
        "repro": collect(
            "阶段四端到端 v10",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "主门 shrunk-mahalanobis：bank W 28 维，标准化 + lambda=0.1 收缩",
                "条件 s2/s3/s5/K8；主档 s2；K8 只报容量",
                "OOD A/B 乘 s；in-dist 加 σ；阈值只在 calibration 上取",
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
                name: {"mean": result["mean"], "ready": result["ready"]}
                for name, result in results.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        json.dumps(
            {"s2_ready": s2_ready, "scale_ready": scale_ready, "v10_ready": summary["v10_ready"]}
        )
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段四端到端 v10")
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

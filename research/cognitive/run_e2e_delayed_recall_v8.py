"""阶段四端到端 v8：bank W 的尺度特征。

预注册见 ``sweep_results/stage4_e2e_v8_prereg.md``。任务/结构同 v7；只改 OOD 置信度：
主门 gauss3 在 bank W 上加入 state_norm，另报 zsum/gauss2/norm-only/全状态 Mahalanobis。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.cognitive.run_e2e_delayed_recall_v4 import _dpar_scores, _top1
from research.cognitive.run_e2e_delayed_recall_v6 import _dataset, _features
from research.cognitive.run_metacognitive_gate_v4 import (
    _evaluate,
    _gaussian_gate,
    _zsum_gate,
)
from research.cognitive.working_memory import LinearRLSReadout, fit_ridge
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "stage4_e2e_v8"

N_DIMS = 4
BANK = 28
MEMBERS = 5
EPOCHS = 20
COVERAGE = 0.80

R2_MIN = 0.90
SELECTIVE_MSE_MAX = 0.10
DPAR_MIN = 0.80
STATE_CONTROL_MAX = 0.35
OOD_AUROC_MIN = 0.75
OOD_ABSTAIN_MIN = 0.80


def _safe_r2(predicted: np.ndarray, target: np.ndarray) -> float:
    residual = ((predicted - target) ** 2).sum(axis=0)
    total = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    varying = total > 1e-9
    return float(np.mean(1.0 - residual[varying] / total[varying]))


def _full_mahalanobis(
    calibration_states: np.ndarray,
    test_states: np.ndarray,
    ood_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = calibration_states.mean(axis=0)
    centered = calibration_states - mean
    covariance = np.atleast_2d(np.cov(centered, rowvar=False))
    covariance = covariance + 1e-6 * np.eye(centered.shape[1])
    inverse = np.linalg.pinv(covariance)

    def confidence(states: np.ndarray) -> np.ndarray:
        delta = states - mean
        return -np.einsum("ni,ij,nj->n", delta, inverse, delta)

    return confidence(calibration_states), confidence(test_states), confidence(ood_states)


def _feature_set(
    states: np.ndarray,
    previous: np.ndarray,
    targets: np.ndarray,
    *,
    readout: LinearRLSReadout,
    members: list[LinearRLSReadout],
    world: np.ndarray,
) -> dict[str, np.ndarray]:
    features = _features(states, previous, targets, readout=readout, members=members, world=world)
    features["state_norm"] = np.linalg.norm(states, axis=1)
    return features


def run(seeds: list[int], *, out: Path, test_episodes: int = 500) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    gates = ("zsum", "gauss2", "gauss3", "norm", "mahalanobis")
    for seed in seeds:
        train = _dataset(2000, seed=seed)
        calibration = _dataset(test_episodes, seed=seed + 30_000)
        test = _dataset(test_episodes, seed=seed + 10_000)
        ood = _dataset(test_episodes, seed=seed + 20_000, ood=True)

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

        predicted = readout.predict(test_wm)
        r2 = [
            _safe_r2(
                predicted[:, i * N_DIMS : (i + 1) * N_DIMS],
                test[4][:, i * N_DIMS : (i + 1) * N_DIMS],
            )
            for i in range(3)
        ]

        confidences = {
            "zsum": _zsum_gate(calibration_features, test_features, ood_features),
            "gauss2": _gaussian_gate(
                calibration_features, test_features, ood_features, ("disagreement", "surprise")
            ),
            "gauss3": _gaussian_gate(
                calibration_features,
                test_features,
                ood_features,
                ("disagreement", "surprise", "state_norm"),
            ),
            "norm": _gaussian_gate(
                calibration_features, test_features, ood_features, ("state_norm",)
            ),
            "mahalanobis": _full_mahalanobis(calibration_wm, test_wm, ood_wm),
        }

        row: dict[str, Any] = {"seed": seed}
        for gate in gates:
            metrics = _evaluate(
                calibration_features,
                test_features,
                ood_features,
                confidence=confidences[gate],
            )
            for key, value in metrics.items():
                row[f"{gate}_{key}"] = value

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
        row.update(
            {
                "r2_mean": float(np.mean(r2)),
                "selective_mse": row["gauss3_selective_mse"],
                "dpar_best": dpar_best,
                "dpar_pseudo": dpar["pseudo"],
                "dpar_rawnn": dpar["rawnn"],
                "dpar_hebbian": dpar["hebbian"],
                "state_control_top1": state_top1,
            }
        )
        rows.append(row)
        print(
            f"seed {seed}: R2 {row['r2_mean']:.3f} MSE {row['selective_mse']:.3f} "
            f"DPAR {dpar_best:.3f} state {state_top1:.3f} | "
            + " | ".join(
                f"{gate} {row[f'{gate}_ood_abstain']:.3f}/{row[f'{gate}_ood_auroc']:.3f}"
                for gate in gates
            ),
            flush=True,
        )

    def mean(gate: str, key: str) -> float:
        return float(np.mean([row[f"{gate}_{key}"] for row in rows]))

    gate_results = {
        gate: {
            key: mean(gate, key)
            for key in ("test_coverage", "selective_mse", "ood_abstain", "ood_auroc")
        }
        for gate in gates
    }
    primary = gate_results["gauss3"]
    gate = {
        "primary": "gauss3",
        **primary,
        "r2_mean": float(np.mean([row["r2_mean"] for row in rows])),
        "dpar_best": float(np.mean([row["dpar_best"] for row in rows])),
        "state_control_top1": float(np.mean([row["state_control_top1"] for row in rows])),
    }
    gate.update(
        {
            "r2_ok": gate["r2_mean"] >= R2_MIN,
            "selective_ok": gate["selective_mse"] <= SELECTIVE_MSE_MAX,
            "dpar_ok": gate["dpar_best"] >= DPAR_MIN,
            "state_control_ok": gate["state_control_top1"] <= STATE_CONTROL_MAX,
            "ood_auroc_ok": gate["ood_auroc"] >= OOD_AUROC_MIN,
            "ood_abstain_ok": gate["ood_abstain"] >= OOD_ABSTAIN_MIN,
        }
    )
    gate["e2e_v8_ready"] = all(
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
    summary = {
        "experiment": "阶段四端到端 v8：bank W 的尺度特征",
        "config": {
            "seeds": seeds,
            "pairs": 4,
            "test_episodes": test_episodes,
            "coverage": COVERAGE,
            "primary": "gauss3",
            "bank": BANK,
        },
        "per_seed": rows,
        "gates": gate_results,
        "gate": gate,
        "repro": collect(
            "阶段四端到端 v8",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "v7 结构；OOD 置信度加入 bank W 的 state_norm",
                "主门 gauss3；对照 zsum/gauss2/norm-only/全状态 Mahalanobis",
                "OOD 为 A/B ×3；阈值只在 calibration（seed+30000）上取",
            ],
            device=None,
        ).render_block(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(gate_results, ensure_ascii=False, indent=2))
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段四端到端 v8")
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

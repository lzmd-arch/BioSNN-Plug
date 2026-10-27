"""元认知 v4：e2e v3 上的本地风险控制（预注册见 sweep_results/metacognitive_gate_v4_prereg.md）。

主门是 calibration-only 的三维高斯密度（disagreement / surprise / state_norm）；所有阈值
只在独立校准集上取，test/OOD 不参与拟合。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.cognitive.run_e2e_delayed_recall_v3 import (
    EPOCHS,
    MEMBERS,
    PRIMARY_EVENTS,
    PRIMARY_NOISE,
    PRIMARY_TEST_STEP,
    _answer_error,
    _dataset,
)
from research.cognitive.working_memory import LinearRLSReadout, fit_ridge, predict
from research.cognitive.world_model import auroc
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "metacognitive_gate_v4"

COVERAGE = 0.80
TEST_COVERAGE_LOW = 0.78
TEST_COVERAGE_HIGH = 0.82
SELECTIVE_MSE_MAX = 0.10
OOD_ABSTAIN_MIN = 0.80
OOD_AUROC_MIN = 0.835


def _dataset_from_seed(episodes: int, *, seed: int, ood: bool = False):
    return _dataset(
        episodes,
        seed=seed,
        ood=ood,
        n_events=PRIMARY_EVENTS,
        noise=PRIMARY_NOISE,
        test_step=PRIMARY_TEST_STEP,
    )


def _zscore(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return (values - reference.mean()) / max(reference.std(), 1e-8)


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
    predictions = np.stack([member.predict(states) for member in members])
    return {
        "disagreement": predictions.std(axis=0).mean(axis=1),
        "surprise": ((predict(previous, world) - states) ** 2).mean(axis=1),
        "state_norm": np.linalg.norm(states, axis=1),
        "errors": _answer_error(predicted, targets),
    }


def _gaussian_gate(
    calibration: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    ood: dict[str, np.ndarray],
    columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = np.column_stack([calibration[name] for name in columns])
    mean = reference.mean(axis=0)
    scale = np.maximum(reference.std(axis=0), 1e-8)
    normalized = (reference - mean) / scale
    center = normalized.mean(axis=0)
    covariance = np.cov(normalized, rowvar=False)
    covariance = np.atleast_2d(covariance) + 1e-6 * np.eye(len(columns))
    inverse = np.linalg.pinv(covariance)

    def confidence(split: dict[str, np.ndarray]) -> np.ndarray:
        values = (np.column_stack([split[name] for name in columns]) - mean) / scale
        delta = values - center
        return -np.einsum("ni,ij,nj->n", delta, inverse, delta)

    return confidence(calibration), confidence(test), confidence(ood)


def _zsum_gate(
    calibration: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    ood: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def confidence(split: dict[str, np.ndarray]) -> np.ndarray:
        return -_zscore(split["disagreement"], calibration["disagreement"]) - _zscore(
            split["surprise"], calibration["surprise"]
        )

    return confidence(calibration), confidence(test), confidence(ood)


def _knn_gate(
    calibration: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    ood: dict[str, np.ndarray],
    columns: tuple[str, ...] = ("disagreement", "surprise", "state_norm"),
    neighbors: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = np.column_stack([calibration[name] for name in columns])
    mean = reference.mean(axis=0)
    scale = np.maximum(reference.std(axis=0), 1e-8)
    normalized = (reference - mean) / scale

    def confidence(split: dict[str, np.ndarray], *, exclude_self: bool = False) -> np.ndarray:
        values = (np.column_stack([split[name] for name in columns]) - mean) / scale
        distances = (
            (values**2).sum(axis=1)[:, None]
            + (normalized**2).sum(axis=1)[None, :]
            - 2.0 * values @ normalized.T
        )
        if exclude_self:
            # 校准点自己的距离为 0；不排除就会把自身算进 5 近邻，阈值系统性偏高。
            np.fill_diagonal(distances, np.inf)
        nearest = np.partition(distances, neighbors - 1, axis=1)[:, :neighbors]
        return -nearest.mean(axis=1)

    return (
        confidence(calibration, exclude_self=True),
        confidence(test),
        confidence(ood),
    )


def _evaluate(
    calibration: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    ood: dict[str, np.ndarray],
    *,
    confidence: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, float]:
    calibration_confidence, test_confidence, ood_confidence = confidence
    threshold = float(np.quantile(calibration_confidence, 1.0 - COVERAGE))
    answered = test_confidence >= threshold
    ood_answered = ood_confidence >= threshold
    selective_mse = float(np.mean(test["errors"][answered])) if np.any(answered) else float("nan")
    return {
        "threshold": threshold,
        "test_coverage": float(np.mean(answered)),
        "selective_mse": selective_mse,
        "ood_abstain": float(np.mean(~ood_answered)),
        "ood_auroc": auroc(
            np.concatenate([test_confidence, ood_confidence]),
            np.concatenate([np.ones_like(test_confidence), np.zeros_like(ood_confidence)]),
        ),
        "ood_answer_mse": float(np.mean(ood["errors"][ood_answered]))
        if np.any(ood_answered)
        else float("nan"),
        "ood_mse_all": float(np.mean(ood["errors"])),
    }


def run(seeds: list[int], *, out: Path, test_episodes: int = 500) -> dict[str, Any]:
    started = time.perf_counter()
    gates = ("zsum", "gauss2", "gauss3", "knn3")
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        train_states, train_previous, train_targets, _ = _dataset_from_seed(2000, seed=seed)
        calibration = _dataset_from_seed(test_episodes, seed=seed + 30_000)
        test = _dataset_from_seed(test_episodes, seed=seed + 10_000)
        ood = _dataset_from_seed(test_episodes, seed=seed + 20_000, ood=True)

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

        feature_sets = {
            "calibration": _features(
                calibration[0],
                calibration[1],
                calibration[2],
                readout=readout,
                members=members,
                world=world,
            ),
            "test": _features(
                test[0], test[1], test[2], readout=readout, members=members, world=world
            ),
            "ood": _features(ood[0], ood[1], ood[2], readout=readout, members=members, world=world),
        }

        confidences = {
            "zsum": _zsum_gate(
                feature_sets["calibration"], feature_sets["test"], feature_sets["ood"]
            ),
            "gauss2": _gaussian_gate(
                feature_sets["calibration"],
                feature_sets["test"],
                feature_sets["ood"],
                ("disagreement", "surprise"),
            ),
            "gauss3": _gaussian_gate(
                feature_sets["calibration"],
                feature_sets["test"],
                feature_sets["ood"],
                ("disagreement", "surprise", "state_norm"),
            ),
            "knn3": _knn_gate(
                feature_sets["calibration"], feature_sets["test"], feature_sets["ood"]
            ),
        }
        row: dict[str, Any] = {"seed": seed}
        for gate in gates:
            metrics = _evaluate(
                feature_sets["calibration"],
                feature_sets["test"],
                feature_sets["ood"],
                confidence=confidences[gate],
            )
            for key, value in metrics.items():
                row[f"{gate}_{key}"] = value
        rows.append(row)
        print(
            f"seed {seed}: "
            + " | ".join(
                f"{gate} abstain {row[f'{gate}_ood_abstain']:.3f} "
                f"AUROC {row[f'{gate}_ood_auroc']:.3f} "
                f"cov {row[f'{gate}_test_coverage']:.3f}"
                for gate in gates
            ),
            flush=True,
        )

    def mean(gate: str, key: str) -> float:
        return float(np.mean([row[f"{gate}_{key}"] for row in rows]))

    gate_results = {
        gate: {
            key: mean(gate, key)
            for key in (
                "test_coverage",
                "selective_mse",
                "ood_abstain",
                "ood_auroc",
                "ood_answer_mse",
            )
        }
        for gate in gates
    }
    primary = gate_results["gauss3"]
    gate = {
        "primary": "gauss3",
        **primary,
        "coverage_ok": TEST_COVERAGE_LOW <= primary["test_coverage"] <= TEST_COVERAGE_HIGH,
        "selective_ok": primary["selective_mse"] <= SELECTIVE_MSE_MAX,
        "abstain_ok": primary["ood_abstain"] >= OOD_ABSTAIN_MIN,
        "auroc_ok": primary["ood_auroc"] >= OOD_AUROC_MIN,
    }
    gate["metacognitive_v4_ready"] = all(
        gate[key] for key in ("coverage_ok", "selective_ok", "abstain_ok", "auroc_ok")
    )
    summary = {
        "experiment": "元认知 v4：e2e v3 上的本地风险控制",
        "config": {
            "seeds": seeds,
            "test_episodes": test_episodes,
            "coverage": COVERAGE,
            "k": PRIMARY_EVENTS,
            "noise": PRIMARY_NOISE,
            "test_step": PRIMARY_TEST_STEP,
            "primary": "gauss3",
        },
        "per_seed": rows,
        "gates": gate_results,
        "gate": gate,
        "repro": collect(
            "元认知 v4：e2e v3 上的本地风险控制",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "阈值只在 calibration（seed+30000）上取；test/OOD 不参与拟合",
                "主门 gauss3 = calibration 上的三维高斯密度（disagreement/surprise/state_norm）",
                "对照 zsum/gauss2/knn3 同阈值口径；OOD 为 A/B ×3 尺度偏移",
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
    parser = argparse.ArgumentParser(description="元认知 v4：本地风险控制")
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

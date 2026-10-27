"""元认知 v7：高阶矩门 vs 信息丢失 OOD。

预注册见 ``sweep_results/metacognitive_gate_v7_prereg.md``。不改 v6 的量化 OOD，
只把状态特征换成逐样本高阶矩（mean z^2 / mean |z| / mean z^4 / 尾比例），检验
“一、二阶矩相同但分布离散化”的偏移能否被检测。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.cognitive.run_metacognitive_gate_v4 import (
    _dataset_from_seed,
    _evaluate,
    _features,
)
from research.cognitive.run_metacognitive_gate_v5 import _gate_confidence
from research.cognitive.run_metacognitive_gate_v6 import _quantized_dataset
from research.cognitive.working_memory import LinearRLSReadout, fit_ridge
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "metacognitive_gate_v7"

COVERAGE = 0.80
TEST_COVERAGE_LOW = 0.78
TEST_COVERAGE_HIGH = 0.82
SELECTIVE_MSE_MAX = 0.10
OOD_ABSTAIN_MIN = 0.80
OOD_AUROC_MIN = 0.75
VALIDITY_RATIO = 2.0
MEMBERS = 5
EPOCHS = 20


def _moment_features(states: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = (np.asarray(states, dtype=np.float64) - mean) / scale
    return np.column_stack(
        [
            (z**2).mean(axis=1),
            np.abs(z).mean(axis=1),
            (z**4).mean(axis=1),
            (np.abs(z) > 1.5).mean(axis=1),
        ]
    )


def _moment_gate(
    calibration_states: np.ndarray,
    test_states: np.ndarray,
    ood_states: np.ndarray,
    *,
    columns: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = calibration_states.mean(axis=0)
    scale = np.maximum(calibration_states.std(axis=0), 1e-8)
    reference = _moment_features(calibration_states, mean, scale)[:, columns]
    center = reference.mean(axis=0)
    spread = np.maximum(reference.std(axis=0), 1e-8)
    normalized = (reference - center) / spread
    covariance = np.atleast_2d(np.cov(normalized, rowvar=False))
    covariance = covariance + 1e-6 * np.eye(len(columns))
    inverse = np.linalg.pinv(covariance)

    def confidence(states: np.ndarray) -> np.ndarray:
        values = (_moment_features(states, mean, scale)[:, columns] - center) / spread
        delta = values - normalized.mean(axis=0)
        return -np.einsum("ni,ij,nj->n", delta, inverse, delta)

    return confidence(calibration_states), confidence(test_states), confidence(ood_states)


def run(seeds: list[int], *, out: Path, test_episodes: int = 500) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    gates = ("zsum", "gauss2", "gauss3", "knn3", "moment2", "moment4")
    for seed in seeds:
        train_states, train_previous, train_targets, _ = _dataset_from_seed(2000, seed=seed)
        calibration = _dataset_from_seed(test_episodes, seed=seed + 30_000)
        test = _dataset_from_seed(test_episodes, seed=seed + 10_000)
        scale_ood = _dataset_from_seed(test_episodes, seed=seed + 20_000, ood=True)
        quantized_ood = _quantized_dataset(test_episodes, seed=seed + 20_000)

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
            name: _features(
                split[0], split[1], split[2], readout=readout, members=members, world=world
            )
            for name, split in (
                ("calibration", calibration),
                ("test", test),
                ("scale", scale_ood),
                ("quantized", quantized_ood),
            )
        }
        moment_confidences = {
            "moment2": _moment_gate(calibration[0], test[0], quantized_ood[0], columns=(0, 1)),
            "moment4": _moment_gate(
                calibration[0], test[0], quantized_ood[0], columns=(0, 1, 2, 3)
            ),
        }
        moment_scale = {
            "moment2": _moment_gate(calibration[0], test[0], scale_ood[0], columns=(0, 1)),
            "moment4": _moment_gate(calibration[0], test[0], scale_ood[0], columns=(0, 1, 2, 3)),
        }

        row: dict[str, Any] = {"seed": seed}
        for gate in gates:
            split_confidences = {
                "scale": None,
                "quantized": None,
            }
            if gate in moment_confidences:
                split_confidences["quantized"] = moment_confidences[gate]
                split_confidences["scale"] = moment_scale[gate]
            else:
                split_confidences["quantized"] = _gate_confidence(
                    gate,
                    feature_sets["calibration"],
                    feature_sets["test"],
                    feature_sets["quantized"],
                )
                split_confidences["scale"] = _gate_confidence(
                    gate,
                    feature_sets["calibration"],
                    feature_sets["test"],
                    feature_sets["scale"],
                )
            for split in ("scale", "quantized"):
                metrics = _evaluate(
                    feature_sets["calibration"],
                    feature_sets["test"],
                    feature_sets[split],
                    confidence=split_confidences[split],
                )
                for key, value in metrics.items():
                    row[f"{gate}_{split}_{key}"] = value
        rows.append(row)
        print(
            f"seed {seed}: quant MSE {row['gauss3_quantized_ood_mse_all']:.3f} "
            f"(in-dist {row['gauss3_quantized_test_mse_all']:.3f}) | "
            + " | ".join(
                f"{gate} q {row[f'{gate}_quantized_ood_abstain']:.3f}/"
                f"{row[f'{gate}_quantized_ood_auroc']:.3f} s "
                f"{row[f'{gate}_scale_ood_abstain']:.3f}"
                for gate in gates
            ),
            flush=True,
        )

    def mean(gate: str, split: str, key: str) -> float:
        return float(np.mean([row[f"{gate}_{split}_{key}"] for row in rows]))

    gate_results = {
        gate: {
            split: {
                key: mean(gate, split, key)
                for key in (
                    "test_coverage",
                    "selective_mse",
                    "test_mse_all",
                    "ood_abstain",
                    "ood_auroc",
                    "ood_mse_all",
                )
            }
            for split in ("scale", "quantized")
        }
        for gate in gates
    }
    primary = gate_results["moment4"]
    validity = primary["quantized"]["ood_mse_all"] >= (
        VALIDITY_RATIO * primary["quantized"]["test_mse_all"]
    )
    gate = {
        "primary": "moment4",
        "validity": validity,
        "validity_ratio": primary["quantized"]["ood_mse_all"]
        / max(primary["quantized"]["test_mse_all"], 1e-12),
        "scale": primary["scale"],
        "quantized": primary["quantized"],
        "coverage_ok": TEST_COVERAGE_LOW
        <= primary["quantized"]["test_coverage"]
        <= TEST_COVERAGE_HIGH,
        "selective_ok": primary["quantized"]["selective_mse"] <= SELECTIVE_MSE_MAX,
        "quantized_abstain_ok": primary["quantized"]["ood_abstain"] >= OOD_ABSTAIN_MIN,
        "quantized_auroc_ok": primary["quantized"]["ood_auroc"] >= OOD_AUROC_MIN,
        "scale_abstain_ok": primary["scale"]["ood_abstain"] >= OOD_ABSTAIN_MIN,
        "scale_auroc_ok": primary["scale"]["ood_auroc"] >= OOD_AUROC_MIN,
    }
    gate["moment_gate_ready"] = all(
        gate[key]
        for key in (
            "validity",
            "coverage_ok",
            "selective_ok",
            "quantized_abstain_ok",
            "quantized_auroc_ok",
            "scale_abstain_ok",
            "scale_auroc_ok",
        )
    )
    summary = {
        "experiment": "元认知 v7：高阶矩门 vs 信息丢失 OOD",
        "config": {
            "seeds": seeds,
            "test_episodes": test_episodes,
            "coverage": COVERAGE,
            "validity_ratio": VALIDITY_RATIO,
            "primary": "moment4",
            "moments": ["mean_z2", "mean_abs_z", "mean_z4", "tail_abs_z_gt_1.5"],
        },
        "per_seed": rows,
        "gates": gate_results,
        "gate": gate,
        "repro": collect(
            "元认知 v7：高阶矩门 vs 信息丢失 OOD",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "moment4 = calibration 标准化后的 mean z^2 / mean |z| / mean z^4 / 尾比例",
                "quantized：A/B 输入取 sign、目标保留原始幅度；scale 为 sanity",
                "有效性：quantized 全量 MSE ≥2× in-dist 全量 MSE",
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
    parser = argparse.ArgumentParser(description="元认知 v7：高阶矩门")
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

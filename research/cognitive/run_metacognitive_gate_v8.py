"""元认知 v8：重建一致性门（预注册见 sweep_results/metacognitive_gate_v8_prereg.md）。"""

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
    _knn_gate,
)
from research.cognitive.run_metacognitive_gate_v6 import _quantized_dataset
from research.cognitive.working_memory import LinearRLSReadout, fit_ridge
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "metacognitive_gate_v8"

COVERAGE = 0.80
TEST_COVERAGE_LOW = 0.78
TEST_COVERAGE_HIGH = 0.82
SELECTIVE_MSE_MAX = 0.10
OOD_ABSTAIN_MIN = 0.80
OOD_AUROC_MIN = 0.75
SCALE_ABSTAIN_MIN = 0.80
VALIDITY_RATIO = 2.0
MEMBERS = 5
EPOCHS = 20
COLUMNS = ("errors", "state_norm")
NEIGHBORS = 5


def run(seeds: list[int], *, out: Path, test_episodes: int = 500) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        train = _dataset_from_seed(2000, seed=seed)
        calibration = _dataset_from_seed(test_episodes, seed=seed + 30_000)
        test = _dataset_from_seed(test_episodes, seed=seed + 10_000)
        scale_ood = _dataset_from_seed(test_episodes, seed=seed + 20_000, ood=True)
        quantized_ood = _quantized_dataset(test_episodes, seed=seed + 20_000)

        readout = LinearRLSReadout(train[0].shape[1], train[2].shape[1], delta=1000.0)
        readout.fit(train[0], train[2], epochs=EPOCHS, rng=np.random.default_rng(seed))
        members = []
        for member_index in range(MEMBERS):
            rng = np.random.default_rng(seed + 100 + member_index)
            index = rng.choice(train[0].shape[0], size=train[0].shape[0], replace=True)
            member = LinearRLSReadout(train[0].shape[1], train[2].shape[1], delta=1000.0)
            member.fit(train[0][index], train[2][index], epochs=EPOCHS, rng=rng)
            members.append(member)
        world = fit_ridge(train[1], train[0])
        features = {
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
        confidence = _knn_gate(
            features["calibration"],
            features["test"],
            features["quantized"],
            COLUMNS,
            NEIGHBORS,
        )
        metrics = _evaluate(
            features["calibration"], features["test"], features["quantized"], confidence=confidence
        )
        scale_confidence = _knn_gate(
            features["calibration"],
            features["test"],
            features["scale"],
            COLUMNS,
            NEIGHBORS,
        )
        scale_metrics = _evaluate(
            features["calibration"],
            features["test"],
            features["scale"],
            confidence=scale_confidence,
        )
        rows.append(
            {
                "seed": seed,
                "test_coverage": metrics["test_coverage"],
                "selective_mse": metrics["selective_mse"],
                "test_mse_all": metrics["test_mse_all"],
                "quantized_abstain": metrics["ood_abstain"],
                "quantized_auroc": metrics["ood_auroc"],
                "quantized_mse_all": metrics["ood_mse_all"],
                "scale_abstain": scale_metrics["ood_abstain"],
                "scale_auroc": scale_metrics["ood_auroc"],
            }
        )
        print(
            f"seed {seed}: quant MSE {metrics['ood_mse_all']:.3f} "
            f"(in-dist {metrics['test_mse_all']:.3f}, ratio "
            f"{metrics['ood_mse_all'] / max(metrics['test_mse_all'], 1e-12):.1f}x) | "
            f"coverage {metrics['test_coverage']:.3f} selective {metrics['selective_mse']:.3f} "
            f"| quant abstain {metrics['ood_abstain']:.3f} AUROC {metrics['ood_auroc']:.3f} "
            f"| scale abstain {scale_metrics['ood_abstain']:.3f}",
            flush=True,
        )

    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in rows]))

    values = {
        key: mean(key)
        for key in (
            "test_coverage",
            "selective_mse",
            "test_mse_all",
            "quantized_abstain",
            "quantized_auroc",
            "quantized_mse_all",
            "scale_abstain",
            "scale_auroc",
        )
    }
    gate = {
        "columns": list(COLUMNS),
        "neighbors": NEIGHBORS,
        "metrics": values,
        "validity_ratio": values["quantized_mse_all"] / max(values["test_mse_all"], 1e-12),
        "validity": values["quantized_mse_all"] >= VALIDITY_RATIO * values["test_mse_all"],
        "coverage_ok": TEST_COVERAGE_LOW <= values["test_coverage"] <= TEST_COVERAGE_HIGH,
        "selective_ok": values["selective_mse"] <= SELECTIVE_MSE_MAX,
        "abstain_ok": values["quantized_abstain"] >= OOD_ABSTAIN_MIN,
        "auroc_ok": values["quantized_auroc"] >= OOD_AUROC_MIN,
        "scale_abstain_ok": values["scale_abstain"] >= SCALE_ABSTAIN_MIN,
    }
    gate["v8_ready"] = all(
        gate[key]
        for key in (
            "validity",
            "coverage_ok",
            "selective_ok",
            "abstain_ok",
            "auroc_ok",
            "scale_abstain_ok",
        )
    )
    summary = {
        "experiment": "元认知 v8：重建一致性门",
        "config": {
            "seeds": seeds,
            "test_episodes": test_episodes,
            "coverage": COVERAGE,
            "columns": list(COLUMNS),
            "neighbors": NEIGHBORS,
            "validity_ratio": VALIDITY_RATIO,
        },
        "per_seed": rows,
        "gate": gate,
        "repro": collect(
            "元认知 v8：重建一致性门",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "quantized：A/B 输入取 sign，目标保留原始幅度；Q/E 不变",
                "门：kNN(5) on [errors, state_norm]；阈值只在 calibration（seed+30000）上取",
                "最终种子 5-9；0-4 只用于预注册前的门族试点",
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
    parser = argparse.ArgumentParser(description="元认知 v8：重建一致性门")
    parser.add_argument("--seeds", default="5-9")
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

"""元认知 v10：逐维最大重建误差门（预注册见 sweep_results/metacognitive_gate_v10_prereg.md）。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.cognitive.run_e2e_delayed_recall_v3 import PRIMARY_TEST_STEP
from research.cognitive.run_metacognitive_gate_v4 import (
    _dataset_from_seed,
    _evaluate,
    _features,
    _knn_gate,
)
from research.cognitive.working_memory import LinearRLSReadout, encode_timeline, fit_ridge
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "metacognitive_gate_v10"

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
FAMILIES = ("time_shift", "time_reverse", "dim_dropout", "correlated")


def _ood_dataset(episodes: int, *, seed: int, family: str) -> tuple[np.ndarray, ...]:
    base = _dataset_from_seed(episodes, seed=seed)
    targets = base[2]
    rng = np.random.default_rng(seed)
    states, previous, kept = [], [], []
    for target in targets:
        cue_a = target[:4].copy()
        cue_b = target[4:8].copy()
        query = target[8:12].copy()
        event = target[12:16].copy()
        if family == "time_shift":
            cues = [(5, 9, cue_a), (20, 24, cue_b), (35, 39, query), (50, 54, event)]
        elif family == "time_reverse":
            cues = [(0, 4, cue_b), (15, 19, cue_a), (30, 34, query), (45, 49, event)]
        elif family == "dim_dropout":
            cue_a[0] = 0.0
            cue_b[0] = 0.0
            cues = [(0, 4, cue_a), (15, 19, cue_b), (30, 34, query), (45, 49, event)]
        elif family == "correlated":
            cues = [(0, 4, cue_a), (15, 19, cue_a), (30, 34, query), (45, 49, event)]
        else:
            raise ValueError(f"未知 OOD 家族：{family!r}")
        previous.append(encode_timeline(cues, rng=rng, test_step=PRIMARY_TEST_STEP - 1))
        states.append(encode_timeline(cues, rng=rng, test_step=PRIMARY_TEST_STEP))
        kept.append(target)
    return (
        np.stack(states),
        np.stack(previous),
        np.stack(kept),
        base[3],
    )


def run(seeds: list[int], *, out: Path, test_episodes: int = 500) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        train = _dataset_from_seed(2000, seed=seed)
        calibration = _dataset_from_seed(test_episodes, seed=seed + 30_000)
        test = _dataset_from_seed(test_episodes, seed=seed + 10_000)
        scale_ood = _dataset_from_seed(test_episodes, seed=seed + 20_000, ood=True)
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

        def features(split, readout=readout, members=members, world=world):
            values = _features(
                split[0], split[1], split[2], readout=readout, members=members, world=world
            )
            residual = (readout.predict(split[0]) - split[2]) ** 2
            values["max_errors"] = residual.max(axis=1)
            return values

        cal_features = features(calibration)
        test_features = features(test)
        row: dict[str, Any] = {"seed": seed}
        for family in FAMILIES:
            ood = _ood_dataset(test_episodes, seed=seed + 20_000, family=family)
            ood_features = features(ood)
            confidence = _knn_gate(
                cal_features, test_features, ood_features, ("max_errors", "state_norm")
            )
            metrics = _evaluate(cal_features, test_features, ood_features, confidence=confidence)
            for key, value in metrics.items():
                row[f"{family}_{key}"] = value
            print(
                f"seed {seed} {family}: ratio "
                f"{metrics['ood_mse_all'] / max(metrics['test_mse_all'], 1e-12):.1f}x "
                f"cov {metrics['test_coverage']:.3f} abstain {metrics['ood_abstain']:.3f} "
                f"AUROC {metrics['ood_auroc']:.3f}",
                flush=True,
            )
        scale_features = features(scale_ood)
        scale_confidence = _knn_gate(
            cal_features, test_features, scale_features, ("max_errors", "state_norm")
        )
        scale_metrics = _evaluate(
            cal_features, test_features, scale_features, confidence=scale_confidence
        )
        row["scale_abstain"] = scale_metrics["ood_abstain"]
        row["scale_auroc"] = scale_metrics["ood_auroc"]
        rows.append(row)

    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in rows]))

    families: dict[str, Any] = {}
    for family in FAMILIES:
        metrics = {
            key: mean(f"{family}_{key}")
            for key in (
                "test_coverage",
                "selective_mse",
                "test_mse_all",
                "ood_mse_all",
                "ood_abstain",
                "ood_auroc",
            )
        }
        valid = metrics["ood_mse_all"] >= VALIDITY_RATIO * metrics["test_mse_all"]
        metrics["validity_ratio"] = metrics["ood_mse_all"] / max(metrics["test_mse_all"], 1e-12)
        metrics["valid"] = valid
        metrics["passes"] = bool(
            valid
            and TEST_COVERAGE_LOW <= metrics["test_coverage"] <= TEST_COVERAGE_HIGH
            and metrics["selective_mse"] <= SELECTIVE_MSE_MAX
            and metrics["ood_abstain"] >= OOD_ABSTAIN_MIN
            and metrics["ood_auroc"] >= OOD_AUROC_MIN
        )
        families[family] = metrics
    scale_abstain = mean("scale_abstain")
    valid_count = sum(1 for metrics in families.values() if metrics["valid"])
    gate = {
        "families": families,
        "valid_count": valid_count,
        "scale_abstain": scale_abstain,
        "scale_abstain_ok": scale_abstain >= SCALE_ABSTAIN_MIN,
        "all_valid_pass": all(
            metrics["passes"] for metrics in families.values() if metrics["valid"]
        ),
    }
    gate["v9_ready"] = bool(
        valid_count >= 3 and gate["all_valid_pass"] and gate["scale_abstain_ok"]
    )
    summary = {
        "experiment": "元认知 v10：逐维最大重建误差门",
        "config": {
            "seeds": seeds,
            "test_episodes": test_episodes,
            "coverage": COVERAGE,
            "columns": ["errors", "state_norm"],
            "families": list(FAMILIES),
            "validity_ratio": VALIDITY_RATIO,
        },
        "per_seed": rows,
        "gate": gate,
        "repro": collect(
            "元认知 v10：逐维最大重建误差门",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "OOD 目标保留原始 A/B/Q/E；只改输入时间窗/维度/相关性",
                "门为 kNN(5) on [max_errors, state_norm]；阈值只在 calibration 上取",
                "最终种子 10-14；先过有效性门（OOD MSE >=2x in-dist）再谈弃权",
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
    parser = argparse.ArgumentParser(description="元认知 v10：逐维最大重建误差门")
    parser.add_argument("--seeds", default="15-19")
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

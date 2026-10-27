"""元认知 v6：量化 OOD 与有效性门。

预注册见 ``sweep_results/metacognitive_gate_v6_prereg.md``。OOD 对 A/B 做逐元素符号
量化但保留原始目标，先检查 OOD 全量误差是否真的升高，再解释弃权/AUROC。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.cognitive.run_e2e_delayed_recall_v3 import (
    N_DIMS,
    PRIMARY_EVENTS,
    PRIMARY_TEST_STEP,
    event_codes,
)
from research.cognitive.run_metacognitive_gate_v4 import (
    _dataset_from_seed,
    _evaluate,
    _features,
)
from research.cognitive.run_metacognitive_gate_v5 import _gate_confidence
from research.cognitive.working_memory import LinearRLSReadout, encode_timeline, fit_ridge
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "metacognitive_gate_v6"

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


def _quantized_dataset(
    episodes: int, *, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    codes = event_codes(PRIMARY_EVENTS)
    states, previous, targets, classes = [], [], [], []
    for _ in range(episodes):
        raw_a = rng.normal(size=N_DIMS)
        raw_b = rng.normal(size=N_DIMS)
        cue_a = np.sign(raw_a)
        cue_b = np.sign(raw_b)
        choose_b = bool(rng.integers(0, 2))
        query = np.array([0.0, 1.0, 0.0, 0.0] if choose_b else [1.0, 0.0, 0.0, 0.0])
        event_class = int(rng.integers(0, PRIMARY_EVENTS))
        event = codes[event_class].copy()
        cues = [
            (0, 4, cue_a),
            (15, 19, cue_b),
            (30, 34, query),
            (45, 49, event),
        ]
        previous.append(encode_timeline(cues, rng=rng, test_step=PRIMARY_TEST_STEP - 1))
        states.append(encode_timeline(cues, rng=rng, test_step=PRIMARY_TEST_STEP))
        # 目标是原始幅度；量化只发生在输入。
        targets.append(np.concatenate([raw_a, raw_b, query, event]))
        classes.append(event_class)
    return (
        np.stack(states),
        np.stack(previous),
        np.stack(targets),
        np.asarray(classes),
    )


def run(seeds: list[int], *, out: Path, test_episodes: int = 500) -> dict[str, Any]:
    started = time.perf_counter()
    gates = ("zsum", "gauss2", "gauss3", "knn3")
    rows: list[dict[str, Any]] = []
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
        row: dict[str, Any] = {"seed": seed}
        for gate in gates:
            for split in ("scale", "quantized"):
                confidence = _gate_confidence(
                    gate,
                    feature_sets["calibration"],
                    feature_sets["test"],
                    feature_sets[split],
                )
                metrics = _evaluate(
                    feature_sets["calibration"],
                    feature_sets["test"],
                    feature_sets[split],
                    confidence=confidence,
                )
                for key, value in metrics.items():
                    row[f"{gate}_{split}_{key}"] = value
        rows.append(row)
        print(
            f"seed {seed}: quant MSE {row['gauss3_quantized_ood_mse_all']:.3f} "
            f"(in-dist {row['gauss3_quantized_test_mse_all']:.3f}) | "
            + " | ".join(
                f"{gate} q-abstain {row[f'{gate}_quantized_ood_abstain']:.3f} "
                f"AUROC {row[f'{gate}_quantized_ood_auroc']:.3f} "
                f"| scale {row[f'{gate}_scale_ood_abstain']:.3f}"
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
    primary = gate_results["gauss3"]
    validity = primary["quantized"]["ood_mse_all"] >= (
        VALIDITY_RATIO * primary["quantized"]["test_mse_all"]
    )
    gate = {
        "primary": "gauss3",
        "validity": validity,
        "validity_ratio": primary["quantized"]["ood_mse_all"]
        / max(primary["quantized"]["test_mse_all"], 1e-12),
        "scale": primary["scale"],
        "quantized": primary["quantized"],
        "coverage_ok": TEST_COVERAGE_LOW
        <= primary["quantized"]["test_coverage"]
        <= TEST_COVERAGE_HIGH,
        "selective_ok": primary["quantized"]["selective_mse"] <= SELECTIVE_MSE_MAX,
        "abstain_ok": primary["quantized"]["ood_abstain"] >= OOD_ABSTAIN_MIN,
        "auroc_ok": primary["quantized"]["ood_auroc"] >= OOD_AUROC_MIN,
        "scale_abstain_ok": primary["scale"]["ood_abstain"] >= SCALE_ABSTAIN_MIN,
    }
    gate["quantization_ood_ready"] = bool(
        validity
        and gate["coverage_ok"]
        and gate["selective_ok"]
        and gate["abstain_ok"]
        and gate["auroc_ok"]
        and gate["scale_abstain_ok"]
    )
    summary = {
        "experiment": "元认知 v6：量化 OOD 与有效性门",
        "config": {
            "seeds": seeds,
            "test_episodes": test_episodes,
            "coverage": COVERAGE,
            "validity_ratio": VALIDITY_RATIO,
            "primary": "gauss3",
        },
        "per_seed": rows,
        "gates": gate_results,
        "gate": gate,
        "repro": collect(
            "元认知 v6：量化 OOD 与有效性门",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "quantized：A/B 输入取 sign，目标保留原始幅度；Q/E 不变",
                "validity：quantized 全量 MSE ≥2× in-dist 全量 MSE，否则本轮无效",
                "阈值只在 calibration（seed+30000）上取；scale 为 v4 sanity",
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
    parser = argparse.ArgumentParser(description="元认知 v6：量化 OOD")
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

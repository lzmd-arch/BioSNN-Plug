"""元认知 v5：尺度无关 OOD（正交旋转）检验。

预注册见 ``sweep_results/metacognitive_gate_v5_prereg.md``。与 v4 相同的门和校准口径，
只把 OOD 换成保范的正交旋转，检验 v4 的增益是否只来自 ``state_norm``。
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
    _gaussian_gate,
    _knn_gate,
    _zsum_gate,
)
from research.cognitive.working_memory import LinearRLSReadout, encode_timeline, fit_ridge
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "metacognitive_gate_v5"

COVERAGE = 0.80
TEST_COVERAGE_LOW = 0.78
TEST_COVERAGE_HIGH = 0.82
SELECTIVE_MSE_MAX = 0.10
ROTATION_ABSTAIN_MIN = 0.80
ROTATION_AUROC_MIN = 0.75
SCALE_ABSTAIN_MIN = 0.80
MEMBERS = 5
EPOCHS = 20


def _rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix, _ = np.linalg.qr(rng.normal(size=(N_DIMS, N_DIMS)))
    return matrix


def _rotation_dataset(
    episodes: int, *, seed: int, rotation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    codes = event_codes(PRIMARY_EVENTS)
    states, previous, targets, classes = [], [], [], []
    for _ in range(episodes):
        cue_a = rng.normal(size=N_DIMS) @ rotation.T
        cue_b = rng.normal(size=N_DIMS) @ rotation.T
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
        targets.append(np.concatenate([cue_a, cue_b, query, event]))
        classes.append(event_class)
    return (
        np.stack(states),
        np.stack(previous),
        np.stack(targets),
        np.asarray(classes),
    )


def _gate_confidence(
    gate: str, calibration, test, ood
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if gate == "zsum":
        return _zsum_gate(calibration, test, ood)
    if gate == "gauss2":
        return _gaussian_gate(calibration, test, ood, ("disagreement", "surprise"))
    if gate == "gauss3":
        return _gaussian_gate(calibration, test, ood, ("disagreement", "surprise", "state_norm"))
    return _knn_gate(calibration, test, ood)


def run(seeds: list[int], *, out: Path, test_episodes: int = 500) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    gates = ("zsum", "gauss2", "gauss3", "knn3")
    for seed in seeds:
        train_states, train_previous, train_targets, _ = _dataset_from_seed(2000, seed=seed)
        calibration = _dataset_from_seed(test_episodes, seed=seed + 30_000)
        test = _dataset_from_seed(test_episodes, seed=seed + 10_000)
        scale_ood = _dataset_from_seed(test_episodes, seed=seed + 20_000, ood=True)
        rotation = _rotation(seed + 700_000)
        rotation_ood = _rotation_dataset(test_episodes, seed=seed + 20_000, rotation=rotation)

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
                ("rotation", rotation_ood),
            )
        }
        row: dict[str, Any] = {"seed": seed}
        for gate in gates:
            for split in ("scale", "rotation"):
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
            f"seed {seed}: "
            + " | ".join(
                f"{gate} rot abstain {row[f'{gate}_rotation_ood_abstain']:.3f} "
                f"AUROC {row[f'{gate}_rotation_ood_auroc']:.3f} "
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
                for key in ("test_coverage", "selective_mse", "ood_abstain", "ood_auroc")
            }
            for split in ("scale", "rotation")
        }
        for gate in gates
    }
    primary = gate_results["gauss3"]
    gate = {
        "primary": "gauss3",
        "scale": primary["scale"],
        "rotation": primary["rotation"],
        "coverage_ok": TEST_COVERAGE_LOW
        <= primary["rotation"]["test_coverage"]
        <= TEST_COVERAGE_HIGH,
        "selective_ok": primary["rotation"]["selective_mse"] <= SELECTIVE_MSE_MAX,
        "rotation_abstain_ok": primary["rotation"]["ood_abstain"] >= ROTATION_ABSTAIN_MIN,
        "rotation_auroc_ok": primary["rotation"]["ood_auroc"] >= ROTATION_AUROC_MIN,
        "scale_abstain_ok": primary["scale"]["ood_abstain"] >= SCALE_ABSTAIN_MIN,
    }
    gate["scale_independent_ready"] = all(
        gate[key]
        for key in (
            "coverage_ok",
            "selective_ok",
            "rotation_abstain_ok",
            "rotation_auroc_ok",
            "scale_abstain_ok",
        )
    )
    summary = {
        "experiment": "元认知 v5：尺度无关 OOD（正交旋转）检验",
        "config": {
            "seeds": seeds,
            "test_episodes": test_episodes,
            "coverage": COVERAGE,
            "primary": "gauss3",
        },
        "per_seed": rows,
        "gates": gate_results,
        "gate": gate,
        "repro": collect(
            "元认知 v5：尺度无关 OOD（正交旋转）检验",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "rotation：每种子随机正交 Q 作用于 A/B，保持线索 L2 范数",
                "scale：A/B ×3，作 v4 复现 sanity",
                "阈值只在 calibration（seed+30000）上取；test/OOD 不参与拟合",
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
    parser = argparse.ArgumentParser(description="元认知 v5：正交旋转 OOD")
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

"""元认知门控 v3：ensemble 不确定性组合（预注册见 sweep_results/metacognitive_gate_v3_prereg.md）。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.cognitive.metacognitive_gate import (
    LogisticCalibrator,
    coverage_threshold,
    ece_confidence,
    margin_scores,
    selective_accuracy_at,
    softmax,
    train_readout,
)
from research.cognitive.world_model import auroc
from research.common.provenance import collect
from research.common.seeding import parse_seeds
from research.continual import image_features, load_cifar100

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "metacognitive_gate_v3"

COVERAGE = 0.80
MEMBERS = 5
BOOTSTRAP = 0.8
GAIN_MIN = 0.05
AUROC_MIN = 0.75
ECE_MAX = 0.10


def _confidence(
    features: np.ndarray,
    correct: np.ndarray,
    *,
    calibration: np.ndarray,
    evaluation: np.ndarray,
) -> tuple[np.ndarray, float, float, float, float]:
    calibrator = LogisticCalibrator()
    calibrator.fit(features[calibration], correct[calibration])
    calibration_confidence = calibrator.predict_proba(features[calibration])
    evaluation_confidence = calibrator.predict_proba(features[evaluation])
    evaluation_correct = correct[evaluation]
    threshold = coverage_threshold(calibration_confidence, COVERAGE)
    selective, coverage = selective_accuracy_at(
        evaluation_confidence, evaluation_correct, threshold=threshold
    )
    return (
        evaluation_confidence,
        selective,
        coverage,
        auroc(evaluation_confidence, evaluation_correct.astype(int)),
        ece_confidence(evaluation_confidence, evaluation_correct),
    )


def run(
    seeds: list[int],
    *,
    data_root: Path,
    out: Path,
    epochs: int = 20,
    batch: int = 128,
    members: int = MEMBERS,
    train_limit: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    train, test = load_cifar100(data_root)
    cache = Path(data_root) / "_features"
    train_features = image_features(
        train.x, backbone="resnet18", cache=cache / "resnet18_train.npz", device="cpu"
    )
    test_features = image_features(
        test.x, backbone="resnet18", cache=cache / "resnet18_test.npz", device="cpu"
    )
    if train_limit is not None:
        train_features = train_features[:train_limit]
        train_labels = train.y[:train_limit]
    else:
        train_labels = train.y

    member_logits = []
    for member in range(members):
        rng = np.random.default_rng(1000 + member)
        index = rng.choice(
            train_features.shape[0],
            size=int(BOOTSTRAP * train_features.shape[0]),
            replace=True,
        )
        readout = train_readout(
            train_features[index],
            train_labels[index],
            epochs=epochs,
            batch=batch,
            seed=1000 + member,
        )
        member_logits.append(readout.predict(test_features))
    stacked = np.stack(member_logits)  # (K, n, 100)
    probabilities = np.stack([softmax(logits, 1.0) for logits in member_logits])
    max_probabilities = probabilities.max(axis=2)  # (K, n)
    mean_max = max_probabilities.mean(axis=0)
    std_max = max_probabilities.std(axis=0)
    mean_logits = stacked.mean(axis=0)
    margin = margin_scores(mean_logits)
    labels = test.y

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        order = rng.permutation(labels.shape[0])
        half = labels.shape[0] // 2
        calibration, evaluation = order[:half], order[half:]
        correct = np.argmax(mean_logits, axis=1) == labels
        base = float(np.mean(correct[evaluation]))
        features = np.column_stack([mean_max, std_max, margin])
        combo = _confidence(features, correct, calibration=calibration, evaluation=evaluation)
        m_only = _confidence(
            mean_max[:, None], correct, calibration=calibration, evaluation=evaluation
        )
        rows.append(
            {
                "seed": seed,
                "base_accuracy": base,
                "combo_selective_80": combo[1],
                "combo_coverage": combo[2],
                "combo_gain": combo[1] - base if not np.isnan(combo[1]) else None,
                "combo_auroc": combo[3],
                "combo_ece": combo[4],
                "m_selective_80": m_only[1],
                "m_gain": m_only[1] - base if not np.isnan(m_only[1]) else None,
                "m_auroc": m_only[3],
                "m_ece": m_only[4],
            }
        )
        print(
            f"seed {seed}: combo {rows[-1]['combo_selective_80']:.3f} "
            f"AUROC {rows[-1]['combo_auroc']:.3f} ECE {rows[-1]['combo_ece']:.3f} | "
            f"M-only AUROC {rows[-1]['m_auroc']:.3f} ECE {rows[-1]['m_ece']:.3f}",
            flush=True,
        )

    mean = lambda key: float(np.mean([row[key] for row in rows]))  # noqa: E731
    gate = {
        "base_accuracy": mean("base_accuracy"),
        "combo": {
            "selective_accuracy_80": mean("combo_selective_80"),
            "gain": mean("combo_gain"),
            "auroc": mean("combo_auroc"),
            "ece": mean("combo_ece"),
            "risk_coverage_auc": None,
        },
        "m_only": {
            "selective_accuracy_80": mean("m_selective_80"),
            "gain": mean("m_gain"),
            "auroc": mean("m_auroc"),
            "ece": mean("m_ece"),
        },
    }
    gate["combo"]["ready"] = (
        gate["combo"]["gain"] >= GAIN_MIN
        and gate["combo"]["auroc"] >= AUROC_MIN
        and gate["combo"]["ece"] <= ECE_MAX
    )
    gate["v3_ready"] = gate["combo"]["ready"]
    summary = {
        "experiment": "元认知门控 v3：ensemble 不确定性组合",
        "config": {
            "epochs": epochs,
            "batch": batch,
            "train_limit": train_limit,
            "coverage": COVERAGE,
            "members": members,
            "bootstrap": BOOTSTRAP,
            "seeds": seeds,
        },
        "per_seed": rows,
        "gate": gate,
        "repro": collect(
            "元认知门控 v3",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "K=5 bootstrap 读出；置信度特征 M/S/G，标定头只在校准半集拟合",
                "M-only Platt 对照同表",
            ],
            device=torch.device("cpu"),
        ).render_block(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="元认知门控 v3")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "cifar100")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--members", type=int, default=MEMBERS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    run(
        parse_seeds(args.seeds),
        data_root=args.data_root,
        out=args.out,
        epochs=2 if args.smoke else args.epochs,
        batch=args.batch,
        members=2 if args.smoke else args.members,
        train_limit=5000 if args.smoke else None,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

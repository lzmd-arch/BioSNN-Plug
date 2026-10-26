"""元认知门控 v1：宽温度网格与 margin 置信度（预注册见 sweep_results/metacognitive_gate_v1_prereg.md）。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.cognitive.metacognitive_gate import (
    coverage_threshold,
    ece,
    ece_confidence,
    fit_sigmoid_temperature,
    fit_temperature,
    margin_scores,
    risk_coverage_auc,
    selective_accuracy_at,
    sigmoid,
    softmax,
    train_readout,
)
from research.cognitive.world_model import auroc
from research.common.provenance import collect
from research.common.seeding import parse_seeds
from research.continual import image_features, load_cifar100

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "metacognitive_gate_v1"

COVERAGE = 0.80
GAIN_MIN = 0.05
AUROC_MIN = 0.75
ECE_MAX = 0.10


def _arm_ready(row: dict[str, Any]) -> bool:
    return bool(
        row["gain"] is not None
        and row["gain"] >= GAIN_MIN
        and row["auroc"] >= AUROC_MIN
        and row["ece"] <= ECE_MAX
    )


def run(
    seeds: list[int],
    *,
    data_root: Path,
    out: Path,
    epochs: int = 20,
    batch: int = 128,
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
        labels_train = train.y[:train_limit]
    else:
        labels_train = train.y
    readout = train_readout(train_features, labels_train, epochs=epochs, batch=batch, seed=seeds[0])
    logits = readout.predict(test_features)
    labels = test.y

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        order = rng.permutation(labels.shape[0])
        half = labels.shape[0] // 2
        calibration, evaluation = order[:half], order[half:]
        correct_cal = np.argmax(logits[calibration], axis=1) == labels[calibration]
        correct_eval = np.argmax(logits[evaluation], axis=1) == labels[evaluation]
        base = float(np.mean(correct_eval))

        # A：宽温度网格
        temperature = fit_temperature(logits[calibration], labels[calibration], low=0.01, high=5.0)
        confidence_cal = softmax(logits[calibration], temperature).max(axis=1)
        probabilities_eval = softmax(logits[evaluation], temperature)
        confidence_eval = probabilities_eval.max(axis=1)
        threshold = coverage_threshold(confidence_cal, COVERAGE)
        selective, actual = selective_accuracy_at(
            confidence_eval, correct_eval, threshold=threshold
        )
        wide = {
            "arm": "temperature_wide",
            "temperature": temperature,
            "selective_accuracy_80": selective,
            "actual_coverage": actual,
            "gain": selective - base if not np.isnan(selective) else None,
            "auroc": auroc(confidence_eval, correct_eval.astype(int)),
            "ece": ece(probabilities_eval, correct_eval),
            "risk_coverage_auc": risk_coverage_auc(confidence_eval, correct_eval),
            "threshold": threshold,
        }

        # B：margin 置信度 + sigmoid 标定
        margin_cal = margin_scores(logits[calibration])
        margin_eval = margin_scores(logits[evaluation])
        margin_temperature = fit_sigmoid_temperature(margin_cal, correct_cal)
        probability_correct = sigmoid(margin_eval / margin_temperature)
        margin_threshold = coverage_threshold(margin_cal, COVERAGE)
        margin_selective, margin_actual = selective_accuracy_at(
            margin_eval, correct_eval, threshold=margin_threshold
        )
        margin = {
            "arm": "margin",
            "temperature": margin_temperature,
            "selective_accuracy_80": margin_selective,
            "actual_coverage": margin_actual,
            "gain": margin_selective - base if not np.isnan(margin_selective) else None,
            "auroc": auroc(margin_eval, correct_eval.astype(int)),
            "ece": ece_confidence(probability_correct, correct_eval),
            "risk_coverage_auc": risk_coverage_auc(margin_eval, correct_eval),
            "threshold": margin_threshold,
        }
        for row in (wide, margin):
            row["seed"] = seed
            row["base_accuracy"] = base
            row["ready"] = _arm_ready(row)
            rows.append(row)
        print(
            f"seed {seed}: base {base:.3f} | T-wide {wide['selective_accuracy_80']:.3f} "
            f"AUROC {wide['auroc']:.3f} ECE {wide['ece']:.3f} | margin "
            f"{margin['selective_accuracy_80']:.3f} AUROC {margin['auroc']:.3f} "
            f"ECE {margin['ece']:.3f}",
            flush=True,
        )

    def summarize(arm: str) -> dict[str, Any]:
        subset = [row for row in rows if row["arm"] == arm]
        mean = lambda key: float(np.mean([row[key] for row in subset]))  # noqa: E731
        return {
            "selective_accuracy_80": mean("selective_accuracy_80"),
            "gain": mean("gain"),
            "auroc": mean("auroc"),
            "ece": mean("ece"),
            "risk_coverage_auc": mean("risk_coverage_auc"),
            "ready": all(row["ready"] for row in subset),
        }

    arms = {"temperature_wide": summarize("temperature_wide"), "margin": summarize("margin")}
    gate = {
        "base_accuracy": float(np.mean([row["base_accuracy"] for row in rows[::2]])),
        "arms": arms,
        "v1_ready": any(arm["ready"] for arm in arms.values()),
    }
    summary = {
        "experiment": "元认知门控 v1：宽温度网格 + margin 置信度",
        "config": {
            "epochs": epochs,
            "batch": batch,
            "train_limit": train_limit,
            "coverage": COVERAGE,
            "seeds": seeds,
        },
        "per_seed": rows,
        "gate": gate,
        "repro": collect(
            "元认知门控 v1",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "两臂：T-wide（T∈[0.01,5]）与 margin（sigmoid 标定，T∈[0.01,10]）",
                "阈值只在校准集上取 80% 覆盖；评估集不参与选择",
                "读出/切分/CIFAR-100 与 v0 相同",
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
    parser = argparse.ArgumentParser(description="元认知门控 v1")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "cifar100")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    run(
        parse_seeds(args.seeds),
        data_root=args.data_root,
        out=args.out,
        epochs=2 if args.smoke else args.epochs,
        batch=args.batch,
        train_limit=5000 if args.smoke else None,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

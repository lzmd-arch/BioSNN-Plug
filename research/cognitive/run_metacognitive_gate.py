"""元认知门控 v0：CIFAR-100 选择性预测（预注册见 sweep_results/metacognitive_gate_v0_prereg.md）。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.cognitive.metacognitive_gate import (
    ece,
    fit_temperature,
    risk_coverage_auc,
    selective_accuracy,
    softmax,
    train_readout,
)
from research.cognitive.world_model import auroc
from research.common.provenance import collect
from research.common.seeding import parse_seeds
from research.continual import image_features, load_cifar100

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "metacognitive_gate_v0"

SELECTIVE_GAIN_MIN = 0.05
ERROR_AUROC_MIN = 0.75
ECE_MAX = 0.10
COVERAGE = 0.80


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
        train_labels = train.y[:train_limit]
    else:
        train_labels = train.y
    readout = train_readout(train_features, train_labels, epochs=epochs, batch=batch, seed=seeds[0])
    logits = readout.predict(test_features)
    labels = test.y

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        order = rng.permutation(labels.shape[0])
        half = labels.shape[0] // 2
        calibration, evaluation = order[:half], order[half:]
        temperature = fit_temperature(logits[calibration], labels[calibration])
        probs_pre = softmax(logits[evaluation], 1.0)
        probs_post = softmax(logits[evaluation], temperature)
        correct = np.argmax(logits[evaluation], axis=1) == labels[evaluation]
        confidence = probs_post.max(axis=1)
        selective, actual_coverage, threshold = selective_accuracy(
            confidence, correct, coverage=COVERAGE
        )
        base = float(np.mean(correct))
        rows.append(
            {
                "seed": seed,
                "temperature": temperature,
                "base_accuracy": base,
                "selective_accuracy_80": selective,
                "actual_coverage": actual_coverage,
                "gain": selective - base if not np.isnan(selective) else None,
                "error_auroc": auroc(confidence, correct.astype(int)),
                "ece_pre": ece(probs_pre, correct),
                "ece_post": ece(probs_post, correct),
                "risk_coverage_auc": risk_coverage_auc(confidence, correct),
                "threshold": threshold,
            }
        )
        print(
            f"seed {seed}: base {base:.3f} selective80 {selective:.3f} "
            f"(+{rows[-1]['gain']:.3f}) | AUROC {rows[-1]['error_auroc']:.3f} | "
            f"ECE {rows[-1]['ece_pre']:.3f}->{rows[-1]['ece_post']:.3f} T={temperature:.2f}",
            flush=True,
        )

    mean = lambda key: float(np.mean([row[key] for row in rows]))  # noqa: E731
    gate = {
        "base_accuracy": mean("base_accuracy"),
        "selective_accuracy_80": mean("selective_accuracy_80"),
        "gain": mean("gain"),
        "gain_ok": mean("gain") >= SELECTIVE_GAIN_MIN,
        "error_auroc": mean("error_auroc"),
        "error_auroc_ok": mean("error_auroc") >= ERROR_AUROC_MIN,
        "ece_pre": mean("ece_pre"),
        "ece_post": mean("ece_post"),
        "ece_ok": mean("ece_post") <= ECE_MAX,
        "risk_coverage_auc": mean("risk_coverage_auc"),
    }
    gate["v0_ready"] = gate["gain_ok"] and gate["error_auroc_ok"] and gate["ece_ok"]
    summary = {
        "experiment": "元认知门控 v0：选择性预测",
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
            "元认知门控 v0",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "冻结 ResNet18 特征 + 100 类 delta 读出",
                "温度与阈值只在校准半集上选；评估半集不参与选择",
                "选择性准确率在 80% 覆盖率下报告",
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
    parser = argparse.ArgumentParser(description="元认知门控 v0")
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

"""元认知门控 v2：K 个 bootstrap 读出的 ensemble（预注册见 sweep_results/metacognitive_gate_v2_prereg.md）。"""

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
    fit_temperature,
    risk_coverage_auc,
    selective_accuracy_at,
    softmax,
    train_readout,
)
from research.cognitive.world_model import auroc
from research.common.provenance import collect
from research.common.seeding import parse_seeds
from research.continual import image_features, load_cifar100

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "metacognitive_gate_v2"

COVERAGE = 0.80
MEMBERS = 5
BOOTSTRAP = 0.8
GAIN_MIN = 0.05
AUROC_MIN = 0.75
ECE_MAX = 0.10


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
    member_aurocs: list[float] = []
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
        logits = readout.predict(test_features)
        member_logits.append(logits)
        member_aurocs.append(
            auroc(
                np.max(softmax(logits, 1.0), axis=1),
                (np.argmax(logits, axis=1) == test.y).astype(int),
            )
        )
    ensemble_logits = np.mean(member_logits, axis=0)
    labels = test.y

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        order = rng.permutation(labels.shape[0])
        half = labels.shape[0] // 2
        calibration, evaluation = order[:half], order[half:]
        temperature = fit_temperature(
            ensemble_logits[calibration], labels[calibration], low=0.01, high=5.0
        )
        probabilities_cal = softmax(ensemble_logits[calibration], temperature)
        probabilities_eval = softmax(ensemble_logits[evaluation], temperature)
        correct = np.argmax(ensemble_logits[evaluation], axis=1) == labels[evaluation]
        confidence = probabilities_eval.max(axis=1)
        threshold = coverage_threshold(probabilities_cal.max(axis=1), COVERAGE)
        selective, actual = selective_accuracy_at(confidence, correct, threshold=threshold)
        base = float(np.mean(correct))
        rows.append(
            {
                "seed": seed,
                "temperature": temperature,
                "base_accuracy": base,
                "selective_accuracy_80": selective,
                "actual_coverage": actual,
                "gain": selective - base if not np.isnan(selective) else None,
                "auroc": auroc(confidence, correct.astype(int)),
                "ece": ece(probabilities_eval, correct),
                "risk_coverage_auc": risk_coverage_auc(confidence, correct),
                "threshold": threshold,
            }
        )
        print(
            f"seed {seed}: base {base:.3f} selective80 {selective:.3f} "
            f"(+{rows[-1]['gain']:.3f}) | AUROC {rows[-1]['auroc']:.3f} | "
            f"ECE {rows[-1]['ece']:.3f} | T={temperature:.2f}",
            flush=True,
        )

    mean = lambda key: float(np.mean([row[key] for row in rows]))  # noqa: E731
    gate = {
        "base_accuracy": mean("base_accuracy"),
        "selective_accuracy_80": mean("selective_accuracy_80"),
        "gain": mean("gain"),
        "gain_ok": mean("gain") >= GAIN_MIN,
        "auroc": mean("auroc"),
        "auroc_ok": mean("auroc") >= AUROC_MIN,
        "ece": mean("ece"),
        "ece_ok": mean("ece") <= ECE_MAX,
        "members": members,
        "member_auroc_mean": float(np.mean(member_aurocs)),
        "member_auroc_range": [float(min(member_aurocs)), float(max(member_aurocs))],
    }
    gate["v2_ready"] = gate["gain_ok"] and gate["auroc_ok"] and gate["ece_ok"]
    summary = {
        "experiment": "元认知门控 v2：ensemble",
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
            "元认知门控 v2",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                f"K={members} bootstrap({BOOTSTRAP}) 读出的均值 logits；温度只在校准集选",
                "阈值只在校准集取 80% 覆盖；成员 AUROC 单列",
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
    parser = argparse.ArgumentParser(description="元认知门控 v2")
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

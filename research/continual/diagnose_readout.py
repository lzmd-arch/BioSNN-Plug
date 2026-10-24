"""CL 读出诊断：分清"没学会"与"学会了但被旧类压住"。

**诊断，不是验收。** 计划与判读规则见
``sweep_results/continual_readout_diagnosis_plan.md``；正式判据（ACC 65%+ / AF<8%）
一个字不改。

跑法::

    uv run python -m research.continual.diagnose_readout --device cuda
    uv run python -m research.continual.diagnose_readout --smoke

产物：``sweep_results/continual_readout_diagnosis/summary.json`` + 自证块。
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.common.provenance import collect
from research.continual.data import load_cifar100, task_classes
from research.continual.readout import DeltaReadout, evaluate
from research.continual.run_continual import features_of, train_hidden
from research.ib_hebbian import IBHebbianPerceptron

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "continual_readout_diagnosis"

#: 正式运行的配置（只读复现，不改）。
HIDDEN_EPOCHS = 1
READOUT_EPOCHS = 20
READOUT_LR = 0.01
BATCH = 128

__all__ = ["diagnose", "main", "sweep_task_two"]


def _to_tensor(split: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(split.x).float().reshape(split.x.shape[0], -1) / 255.0
    y = torch.from_numpy(split.y).long()
    return x.to(device), y.to(device)


def _logit_stats(
    readout: DeltaReadout,
    features: np.ndarray,
    current: np.ndarray,
    old: np.ndarray,
) -> tuple[float, float]:
    """当前类与旧类在本任务测试样本上的最大 logit 均值（旧类为空时给 nan）。"""
    logits = readout.predict(features)
    current_max = float(np.max(logits[:, np.asarray(current, dtype=int)], axis=1).mean())
    if np.asarray(old).size == 0:
        return current_max, float("nan")
    old_max = float(np.max(logits[:, np.asarray(old, dtype=int)], axis=1).mean())
    return current_max, old_max


def sweep_task_two(
    base: DeltaReadout,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    current: np.ndarray,
) -> list[dict[str, Any]]:
    """固定 task-2 特征，只换读出的 eta 与轮数。"""
    rows: list[dict[str, Any]] = []
    for learning_rate in (0.01, 0.1, 0.5):
        for epochs in (20, 100):
            readout = copy.deepcopy(base)
            readout.learning_rate = float(learning_rate)
            diverged = False
            try:
                for _ in range(epochs):
                    readout.step(train_features, train_labels)
            except RuntimeError:
                diverged = True
            if diverged:
                rows.append(
                    {
                        "learning_rate": learning_rate,
                        "epochs": epochs,
                        "diverged": True,
                    }
                )
                continue
            rows.append(
                {
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                    "diverged": False,
                    "oracle_acc": evaluate(readout, test_features, test_labels, current),
                    "train_acc": evaluate(readout, train_features, train_labels, current),
                }
            )
    return rows


def diagnose(
    *,
    seed: int,
    n_tasks: int,
    limit_per_class: int | None,
    device: torch.device,
    data_root: Path,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train, test = load_cifar100(data_root)
    tasks = task_classes(n_tasks, 100)
    model = IBHebbianPerceptron(n_in=32 * 32 * 3, width=1024, n_layers=3, n_classes=100).to(device)
    readout = DeltaReadout(1024, 100, learning_rate=READOUT_LR, seed=seed)
    test_x, test_y = _to_tensor(test, device)
    test_labels_all = test_y.cpu().numpy()

    seen: list[int] = []
    rows: list[dict[str, Any]] = []
    sweep: list[dict[str, Any]] | None = None
    readout_after_task_one: DeltaReadout | None = None

    for index, classes in enumerate(tasks):
        current = np.asarray(sorted(int(c) for c in classes), dtype=int)
        seen.extend(int(c) for c in current)
        subset = train.subset(classes, limit=limit_per_class)
        x, y = _to_tensor(subset, device)

        train_hidden(model, x, y, epochs=HIDDEN_EPOCHS, batch=BATCH)
        features = features_of(model, x)
        labels = y.cpu().numpy()
        for _ in range(READOUT_EPOCHS):
            readout.step(features, labels)

        seen_classes = np.asarray(sorted(seen), dtype=int)
        current_mask = np.isin(test_labels_all, current)
        current_features = features_of(model, test_x[current_mask])
        current_labels = test_labels_all[current_mask]
        seen_mask = np.isin(test_labels_all, seen_classes)
        seen_features = features_of(model, test_x[seen_mask])
        seen_labels = test_labels_all[seen_mask]
        old = np.asarray([c for c in seen_classes if c not in set(current.tolist())], dtype=int)

        current_logit, old_logit = _logit_stats(readout, current_features, current, old)
        row = {
            "task": index + 1,
            "classes": current.tolist(),
            "seen_acc": evaluate(readout, seen_features, seen_labels, seen_classes),
            "oracle_acc": evaluate(readout, current_features, current_labels, current),
            "train_acc": evaluate(readout, features, labels, current),
            "current_logit": current_logit,
            "old_logit": old_logit,
            "feature_norm": float(np.linalg.norm(features, axis=1).mean()),
            "chance_seen": 1.0 / float(seen_classes.size),
            "chance_oracle": 1.0 / float(current.size),
        }
        rows.append(row)

        if index == 0:
            readout_after_task_one = copy.deepcopy(readout)
        elif index == 1 and readout_after_task_one is not None:
            # task-2 的特征已经算好（隐藏层只在上面训过一次），扫参只换读出。
            sweep = sweep_task_two(
                readout_after_task_one,
                features,
                labels,
                current_features,
                current_labels,
                current,
            )

    return {"seed": seed, "tasks": rows, "task_two_sweep": sweep or []}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CL 读出诊断")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "cifar100")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    if args.smoke:
        args.tasks = 2
        args.limit_per_class = 20
        args.out = REPO_ROOT / "data" / "_smoke_readout_diagnosis"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    result = diagnose(
        seed=args.seed,
        n_tasks=args.tasks,
        limit_per_class=args.limit_per_class,
        device=device,
        data_root=args.data_root,
    )
    elapsed = time.perf_counter() - started
    result["config"] = {
        "tasks": args.tasks,
        "limit_per_class": args.limit_per_class,
        "hidden_epochs": HIDDEN_EPOCHS,
        "readout_epochs": READOUT_EPOCHS,
        "readout_lr": READOUT_LR,
        "batch": BATCH,
        "device": str(device),
    }
    result["repro"] = collect(
        "CL 读出诊断：delta 读出为什么只学会任务 1",
        seeds=str(args.seed),
        elapsed_s=elapsed,
        notes=[
            "诊断，不是验收；判读规则见 continual_readout_diagnosis_plan.md。",
            "oracle_acc = 只在本任务 10 类里 argmax；train_acc 同口径但看训练集。",
        ],
        device=device,
    ).render_block()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(
        f"{'任务':>4}{'seen':>8}{'oracle':>8}{'train':>8}"
        f"{'当前logit':>11}{'旧类logit':>11}{'特征范数':>10}"
    )
    for row in result["tasks"]:
        print(
            f"{row['task']:>4}{row['seen_acc']:>8.3f}{row['oracle_acc']:>8.3f}"
            f"{row['train_acc']:>8.3f}{row['current_logit']:>11.2f}"
            f"{row['old_logit']:>11.2f}{row['feature_norm']:>10.1f}"
        )
    if result["task_two_sweep"]:
        print("\ntask-2 扫参（固定特征，只换读出）：")
        for row in result["task_two_sweep"]:
            if row["diverged"]:
                print(f"  eta={row['learning_rate']:<5} epochs={row['epochs']:<4} 发散")
            else:
                print(
                    f"  eta={row['learning_rate']:<5} epochs={row['epochs']:<4}"
                    f" oracle={row['oracle_acc']:.3f} train={row['train_acc']:.3f}"
                )
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

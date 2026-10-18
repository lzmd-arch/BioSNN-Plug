"""CIFAR-100 类增量基准（10 步 × 10 类）。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

预注册：[`sweep_results/continual_cifar100_prereg.md`](../../sweep_results/continual_cifar100_prereg.md)。

用法::

    uv run python -m research.continual.run_continual --smoke
    uv run python -m research.continual.run_continual --seeds 0,1,2,3,4

## 协议（与预注册一致）

* 10 个任务 × 10 类，类别按 id 升序分组；
* 特征：W1 感知层（`IBHebbianPerceptron`，3 层 × 1024），**隐藏层在每个任务上继续用局部规则更新**；
* 读出：100 维 delta 规则，只喂当前任务的类，**无回放、无正则**——测的就是朴素增量；
* 每学完一个任务，在**已见过的所有类**上评测（不提供任务 id）。

## 对照

``--joint`` 跑联合训练（一次性在全部类上训练），作为"没有遗忘压力"的上界。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from research.common.provenance import collect
from research.continual.data import CifarSplit, load_cifar100, task_classes
from research.continual.metrics import summarise
from research.continual.readout import DeltaReadout, evaluate
from research.ib_hebbian.model import IBHebbianPerceptron

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "continual_cifar100"

__all__ = ["evaluate_seen", "features_of", "main", "run_incremental", "train_hidden"]


def _to_tensor(split: CifarSplit, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """``(n, 32, 32, 3)`` uint8 → ``(n, 3072)`` 浮点，缩放到 [0, 1]。"""
    x = torch.from_numpy(split.x).float().reshape(split.x.shape[0], -1) / 255.0
    y = torch.from_numpy(split.y).long()
    return x.to(device), y.to(device)


def train_hidden(model, x: torch.Tensor, y: torch.Tensor, *, epochs: int, batch: int) -> None:
    """让 W1 的隐藏层在给定数据上继续用局部规则更新（读出层不参与）。"""
    n = x.shape[0]
    for _ in range(epochs):
        order = torch.randperm(n)
        for start in range(0, n, batch):
            idx = order[start : start + batch]
            model.hidden_activations(x[idx], y[idx], update=True)


@torch.no_grad()
def features_of(model, x: torch.Tensor, *, batch: int = 512) -> np.ndarray:
    parts = [model.features(x[i : i + batch]).cpu().numpy() for i in range(0, x.shape[0], batch)]
    return np.concatenate(parts, axis=0).astype(np.float64)


def evaluate_seen(
    model, readout: DeltaReadout, test_x: torch.Tensor, test_y: torch.Tensor, seen: np.ndarray
) -> float:
    """在已见类上评测：先筛出属于 ``seen`` 的测试样本，再算准确率。"""
    mask = np.isin(test_y.cpu().numpy(), seen)
    if not mask.any():
        return float("nan")
    feats = features_of(model, test_x[mask])
    return evaluate(readout, feats, test_y.cpu().numpy()[mask], seen)


def run_incremental(
    *,
    seed: int,
    n_tasks: int,
    limit_per_class: int | None,
    hidden_epochs: int,
    readout_epochs: int,
    batch: int,
    device: torch.device,
    data_root: Path,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train, test = load_cifar100(data_root)
    tasks = task_classes(n_tasks, 100)
    model = IBHebbianPerceptron(n_in=32 * 32 * 3, width=1024, n_layers=3, n_classes=100).to(device)
    readout = DeltaReadout(1024, 100, learning_rate=0.01)

    test_x, test_y = _to_tensor(test, device)
    matrix = np.full((n_tasks, n_tasks), np.nan)
    seen: list[int] = []
    for t, classes in enumerate(tasks):
        seen.extend(int(c) for c in classes)
        subset = train.subset(classes, limit=limit_per_class)
        x, y = _to_tensor(subset, device)
        train_hidden(model, x, y, epochs=hidden_epochs, batch=batch)

        feats = features_of(model, x)
        labels = y.cpu().numpy()
        for _ in range(readout_epochs):
            readout.step(feats, labels)

        matrix[t, : t + 1] = [
            evaluate_seen(
                model, readout, test_x, test_y, np.asarray(sorted(seen[: (i + 1) * len(classes)]))
            )
            for i in range(t + 1)
        ]
        print(f"  task {t + 1}/{n_tasks}: 本任务 {matrix[t, t]:.3f}")

    scores = summarise(matrix)
    return {
        "seed": seed,
        "mode": "incremental",
        "n_tasks": n_tasks,
        "limit_per_class": limit_per_class,
        "accuracy_matrix": matrix.tolist(),
        "acc": scores.acc,
        "af": scores.af,
        "per_task_final": list(scores.per_task_final),
    }


def run_joint(
    *,
    seed: int,
    limit_per_class: int | None,
    hidden_epochs: int,
    readout_epochs: int,
    batch: int,
    device: torch.device,
    data_root: Path,
) -> dict:
    """联合训练上界：一次性在全部 100 类上训练。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    train, test = load_cifar100(data_root)
    if limit_per_class is not None:
        train = train.subset(np.arange(100), limit=limit_per_class)

    model = IBHebbianPerceptron(n_in=32 * 32 * 3, width=1024, n_layers=3, n_classes=100).to(device)
    x, y = _to_tensor(train, device)
    train_hidden(model, x, y, epochs=hidden_epochs, batch=batch)

    readout = DeltaReadout(1024, 100, learning_rate=0.01)
    feats = features_of(model, x)
    labels = y.cpu().numpy()
    for _ in range(readout_epochs):
        readout.step(feats, labels)

    test_x, test_y = _to_tensor(test, device)
    acc = evaluate_seen(model, readout, test_x, test_y, np.arange(100))
    return {"seed": seed, "mode": "joint", "acc": acc, "af": float("nan")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CIFAR-100 类增量基准")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--hidden-epochs", type=int, default=1)
    parser.add_argument("--readout-epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "cifar100")
    parser.add_argument("--joint", action="store_true", help="跑联合训练上界")
    parser.add_argument("--smoke", action="store_true", help="2 任务、每类 20 张")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    if args.smoke:
        args.tasks = 2
        args.limit_per_class = 20
        args.readout_epochs = 5
        args.out = Path(__file__).resolve().parents[2] / "data" / "_smoke_continual"

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    device = torch.device(args.device)
    started = time.perf_counter()
    results = []
    for seed in seeds:
        if args.joint:
            results.append(
                run_joint(
                    seed=seed,
                    limit_per_class=args.limit_per_class,
                    hidden_epochs=args.hidden_epochs,
                    readout_epochs=args.readout_epochs,
                    batch=args.batch,
                    device=device,
                    data_root=args.data_root,
                )
            )
        else:
            results.append(
                run_incremental(
                    seed=seed,
                    n_tasks=args.tasks,
                    limit_per_class=args.limit_per_class,
                    hidden_epochs=args.hidden_epochs,
                    readout_epochs=args.readout_epochs,
                    batch=args.batch,
                    device=device,
                    data_root=args.data_root,
                )
            )

    summary = {
        "experiment": "CIFAR-100 class-incremental (W1 features + delta readout)",
        "mode": "joint" if args.joint else "incremental",
        "config": {
            "n_tasks": args.tasks,
            "limit_per_class": args.limit_per_class,
            "hidden_epochs": args.hidden_epochs,
            "readout_epochs": args.readout_epochs,
            "batch": args.batch,
        },
        "per_seed": results,
        "mean_acc": float(np.mean([r["acc"] for r in results])),
        "mean_af": float(np.nanmean([r["af"] for r in results])),
        "repro": collect(
            "W5 持续学习：CIFAR-100 类增量",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=["隐藏层每任务继续局部更新；读出为 delta 规则，无回放无正则"],
            device=device,
        ).render_block(),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"\nACC = {summary['mean_acc']:.3f}   AF = {summary['mean_af']:.3f}")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

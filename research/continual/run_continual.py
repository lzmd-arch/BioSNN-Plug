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
from research.continual.frontends import image_features
from research.continual.metrics import summarise
from research.continual.ncm import ClassPrototypes
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


def _prototype_accuracy(model, prototypes, test_x, test_y, classes: np.ndarray) -> float:
    """原型读出在给定类别子集上的准确率。**只在已见类里**取最近原型。"""
    labels = test_y.cpu().numpy()
    mask = np.isin(labels, classes)
    if not mask.any():
        return float("nan")
    features = features_of(model, test_x[mask])
    predicted = prototypes.predict(features)
    return float(np.mean(predicted == labels[mask]))


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
    readout_kind: str = "delta",
    whiten: bool = False,
    whiten_mode: str = "diag",
    freeze_features: bool = False,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train, test = load_cifar100(data_root)
    tasks = task_classes(n_tasks, 100)
    model = IBHebbianPerceptron(n_in=32 * 32 * 3, width=1024, n_layers=3, n_classes=100).to(device)
    readout = DeltaReadout(1024, 100, learning_rate=0.01)
    prototypes = (
        ClassPrototypes(100, whiten=whiten, whiten_mode=whiten_mode)
        if readout_kind == "ncm"
        else None
    )

    test_x, test_y = _to_tensor(test, device)
    matrix = np.full((n_tasks, n_tasks), np.nan)
    seen: list[int] = []
    for t, classes in enumerate(tasks):
        seen.extend(int(c) for c in classes)
        subset = train.subset(classes, limit=limit_per_class)
        x, y = _to_tensor(subset, device)
        # 冻结特征：只在第一个任务上训练隐藏层，之后不再改动——去掉"特征漂移"这一遗忘来源。
        if not (freeze_features and t > 0):
            train_hidden(model, x, y, epochs=hidden_epochs, batch=batch)

        feats = features_of(model, x)
        labels = y.cpu().numpy()
        if prototypes is not None:
            prototypes.add(feats, labels)
        else:
            for _ in range(readout_epochs):
                readout.step(feats, labels)

        if prototypes is not None:
            matrix[t, : t + 1] = [
                _prototype_accuracy(
                    model,
                    prototypes,
                    test_x,
                    test_y,
                    np.asarray(sorted(seen[: (i + 1) * len(classes)])),
                )
                for i in range(t + 1)
            ]
        else:
            matrix[t, : t + 1] = [
                evaluate_seen(
                    model,
                    readout,
                    test_x,
                    test_y,
                    np.asarray(sorted(seen[: (i + 1) * len(classes)])),
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


def run_with_frozen_frontend(
    *,
    seed: int,
    n_tasks: int,
    limit_per_class: int | None,
    whiten: bool,
    data_root: Path,
    device: torch.device,
    backbone: str = "resnet18",
    whiten_mode: str = "diag",
    consolidation: str = "none",
    consolidation_rounds: int = 3,
    consolidation_lambda: float = 0.02,
) -> dict:
    """冻结 ResNet18 前端 + 类原型：**没有特征漂移，也没有逐任务覆盖**。

    与 W1 那条路线的差别只有特征来源；读出一条是同一份 `ClassPrototypes`。
    规格里说明过：这条测的是「任何强表示 + 局部原型记忆」，不是本仓的 Hebbian 表示。
    """
    np.random.seed(seed)
    train, test = load_cifar100(data_root)
    cache_dir = Path(data_root) / "_features"
    print(f"提取 {backbone} 特征（训练集）…")
    train_features = image_features(
        train.x, backbone=backbone, cache=cache_dir / f"{backbone}_train.npz", device=str(device)
    )
    print(f"提取 {backbone} 特征（测试集）…")
    test_features = image_features(
        test.x, backbone=backbone, cache=cache_dir / f"{backbone}_test.npz", device=str(device)
    )

    tasks = task_classes(n_tasks, 100)
    prototypes = ClassPrototypes(100, whiten=whiten, whiten_mode=whiten_mode)
    matrix = np.full((n_tasks, n_tasks), np.nan)
    seen: list[int] = []
    for t, classes in enumerate(tasks):
        seen.extend(int(c) for c in classes)
        mask = np.isin(train.y, classes)
        features = train_features[mask]
        if limit_per_class is not None:
            keep = []
            for label in classes:
                keep.append(np.flatnonzero(train.y[mask] == label)[:limit_per_class])
            subset = np.concatenate(keep) if keep else np.array([], dtype=int)
            features = features[subset]
            labels = train.y[mask][subset]
        else:
            labels = train.y[mask]
        prototypes.add(features, labels)

        # 巩固（预注册见 sweep_results/consolidation_prereg.md）：
        # 只操作已存统计，不重放原始样本。C1 朝全体已见类均值收缩；
        # C2 同量算力但目标换成一个坐标被打乱的均值（排除"多算了几遍"）。
        if consolidation != "none" and prototypes.seen_classes.size:
            shrink_rng = np.random.default_rng(seed + 7919)
            for _ in range(consolidation_rounds):
                mu_all = prototypes.prototypes().mean(axis=0)
                if consolidation == "shuffled":
                    mu_all = mu_all[shrink_rng.permutation(mu_all.shape[0])]
                prototypes.shrink_toward(mu_all, consolidation_lambda)

        for i in range(t + 1):
            subset_classes = np.asarray(sorted(seen[: (i + 1) * len(classes)]))
            pick = np.isin(test.y, subset_classes)
            predicted = prototypes.predict(test_features[pick])
            matrix[t, i] = float(np.mean(predicted == test.y[pick]))
        print(f"  task {t + 1}/{n_tasks}: 本任务 {matrix[t, t]:.3f}")

    scores = summarise(matrix)
    return {
        "seed": seed,
        "mode": "incremental-resnet18",
        "consolidation": consolidation,
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
    parser.add_argument("--features", choices=("w1", "resnet18"), default="w1")
    parser.add_argument(
        "--backbone",
        default="resnet18",
        help="--features resnet18 时用哪个冻结骨干（resnet18 / resnet50 / vit_b_16）",
    )
    parser.add_argument("--readout", choices=("delta", "ncm"), default="delta")
    parser.add_argument("--whiten", action="store_true", help="NCM 读出做白化")
    parser.add_argument(
        "--whiten-mode",
        choices=("diag", "shared"),
        default="diag",
        help="diag=每维方差（O(d) 内存）；shared=完整协方差（高维会 OOM）",
    )
    parser.add_argument(
        "--freeze-features",
        action="store_true",
        help="隐藏层只在第一个任务上训练，之后冻结（去掉特征漂移）",
    )
    parser.add_argument(
        "--consolidation",
        choices=("none", "stats", "shuffled"),
        default="none",
        help="巩固臂：none=无（C0）、stats=朝全体均值收缩（C1）、shuffled=目标打乱（C2）",
    )
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
        if args.features == "resnet18":
            results.append(
                run_with_frozen_frontend(
                    seed=seed,
                    n_tasks=args.tasks,
                    limit_per_class=args.limit_per_class,
                    whiten=args.whiten,
                    whiten_mode=args.whiten_mode,
                    consolidation=args.consolidation,
                    data_root=args.data_root,
                    device=device,
                    backbone=args.backbone,
                )
            )
            continue
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
                    readout_kind=args.readout,
                    whiten=args.whiten,
                    whiten_mode=args.whiten_mode,
                    freeze_features=args.freeze_features,
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

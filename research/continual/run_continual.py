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
* 归一化是显式的独立臂（`--normalize`，默认关），统计量只在任务 1 上拟合并冻结，
  见 [`normalization.py`](normalization.py) 与预注册 v3；
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
from research.continual.normalization import build_transform
from research.continual.readout import DeltaReadout, evaluate
from research.ib_hebbian import IBHebbianPerceptron

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "continual_cifar100"

__all__ = [
    "evaluate_seen",
    "evaluate_task",
    "features_of",
    "main",
    "run_incremental",
    "run_joint",
    "run_with_frozen_frontend",
    "run_with_frozen_frontend_delta",
    "train_hidden",
]


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


def evaluate_task(
    model,
    readout: DeltaReadout,
    test_x,
    test_y,
    task_classes: np.ndarray,
    seen_classes: np.ndarray,
) -> float:
    """预注册的 CIL 口径：样本只取本任务，**argmax 在全部已见类里**。

    与 :func:`evaluate_seen` 的区别在样本：``evaluate_seen`` 把样本也过滤到给定类集，
    那是"在这个类集上评测"；本函数固定只看任务 i 的样本，但分类器面对的是任务 t 时
    的全部已见类——**不提供任务 id**，这才是类增量矩阵 ``a[t][i]`` 的定义。
    """
    labels = test_y.cpu().numpy()
    mask = np.isin(labels, np.asarray(task_classes, dtype=int))
    if not mask.any():
        return float("nan")
    features = features_of(model, test_x[mask])
    return evaluate(readout, features, labels[mask], np.asarray(seen_classes, dtype=int))


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

        # **a[t][i] = 任务 i 的样本、在"任务 t 时的全部已见类"上 argmax。**
        # 两种错误写法都出现过：填累计类集的联合准确率（原 formal/），或把 argmax
        # 限制到任务 i 自己（oracle，formal_corrected/）。预注册的口径是不提供
        # 任务 id，所以样本按任务切、分类器面对全部已见类。
        seen_now = np.asarray(sorted(seen), dtype=int)
        task_sets = [np.asarray(sorted(int(c) for c in task)) for task in tasks[: t + 1]]
        if prototypes is not None:
            # 原型读出本来就在"全部已见原型"里取最近，样本按任务切即可。
            matrix[t, : t + 1] = [
                _prototype_accuracy(model, prototypes, test_x, test_y, task_set)
                for task_set in task_sets
            ]
        else:
            matrix[t, : t + 1] = [
                evaluate_task(model, readout, test_x, test_y, task_set, seen_now)
                for task_set in task_sets
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

        # a[t][i] = 任务 i 的样本、在全部已见原型里取最近（CIL，不提供任务 id）。
        for i, task in enumerate(tasks[: t + 1]):
            pick = np.isin(test.y, task)
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


GATE_MIN_WEIGHT_CHANGE = 0.05
"""预注册 v2 的跑门：每个任务读出权重的相对变化不得低于此值。"""


def _active_rows_for_task(
    isolation: str,
    classes: np.ndarray,
    rng: np.random.Generator,
    n_classes: int = 100,
) -> np.ndarray | None:
    """按隔离臂给出本任务允许更新的输出行；``none`` 返回 ``None``（全行更新）。"""
    if isolation == "none":
        return None
    if isolation == "current-rows":
        return np.asarray(classes, dtype=int)
    if isolation == "random-rows":
        return np.sort(rng.choice(n_classes, size=len(classes), replace=False)).astype(int)
    raise ValueError(f"未知的 isolation：{isolation!r}；可选 none / current-rows / random-rows。")


def _sample_replay(
    features: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    per_class: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """从当前任务特征里每类抽 ``per_class`` 条进回放缓冲。"""
    picks: list[int] = []
    for label in classes:
        where = np.flatnonzero(labels == label)
        if where.size == 0:
            continue
        take = min(int(per_class), int(where.size))
        picks.extend(int(i) for i in rng.choice(where, size=take, replace=False))
    if not picks:
        return features[:0], labels[:0]
    index = np.asarray(picks, dtype=np.int64)
    return features[index], labels[index]


def _train_delta_readout(
    readout: DeltaReadout,
    features: np.ndarray,
    labels: np.ndarray,
    replay_features: np.ndarray,
    replay_labels: np.ndarray,
    *,
    epochs: int,
    batch: int,
    rho: float,
    rng: np.random.Generator,
    active_classes: np.ndarray | None = None,
) -> None:
    """训练 delta 读出；``batch <= 0`` 走 v1 的全批量，否则按 ρ 混入回放。"""
    if rho > 0 and batch <= 0:
        raise ValueError("replay 需要 --readout-batch > 0（预注册 v2 的 minibatch 混合）。")
    if batch <= 0:
        for _ in range(epochs):
            readout.step(features, labels, active_classes=active_classes)
        return
    n_replay = round(rho * batch) if replay_features.shape[0] else 0
    n_current = batch - n_replay
    if n_current <= 0:
        raise ValueError(f"readout batch={batch} 太小，扣掉回放 {n_replay} 后没有当前样本。")
    for _ in range(epochs):
        order = rng.permutation(features.shape[0])
        for start in range(0, features.shape[0], n_current):
            index = order[start : start + n_current]
            x = features[index]
            y = labels[index]
            if n_replay:
                replay_index = rng.integers(0, replay_features.shape[0], size=n_replay)
                x = np.concatenate([x, replay_features[replay_index]], axis=0)
                y = np.concatenate([y, replay_labels[replay_index]], axis=0)
            readout.step(x, y, active_classes=active_classes)


def run_with_frozen_frontend_delta(
    *,
    seed: int,
    n_tasks: int,
    limit_per_class: int | None,
    data_root: Path,
    device: torch.device,
    backbone: str = "resnet18",
    readout_epochs: int = 20,
    learning_rate: float = 0.01,
    readout_batch: int = 0,
    replay_rho: float = 0.0,
    replay_per_class: int = 20,
    normalize: str = "none",
    normalize_scale: str = "native",
    isolation: str = "none",
) -> dict:
    """Route R 的 delta 读出：冻结特征 + 线性读出 + 可选 latent replay / 归一化。

    与 :func:`run_with_frozen_frontend` 的区别只在读出：NCM 不遗忘，delta 会——
    回放臂要对照的就是后者。评估口径是 CIL（全部已见类 argmax、不提供任务 id）。

    ``normalize`` 是预注册 v3 的独立机制：``task1-standardize`` 在任务 1 的训练特征上
    拟合一次并**冻结**，同一组统计量用于全部任务的训练与测试（见 normalization.py）。
    """
    np.random.seed(seed)
    train, test = load_cifar100(data_root)
    cache_dir = Path(data_root) / "_features"
    train_features = image_features(
        train.x, backbone=backbone, cache=cache_dir / f"{backbone}_train.npz", device=str(device)
    )
    test_features = image_features(
        test.x, backbone=backbone, cache=cache_dir / f"{backbone}_test.npz", device=str(device)
    )

    tasks = task_classes(n_tasks, 100)
    readout = DeltaReadout(
        int(train_features.shape[1]), 100, learning_rate=learning_rate, seed=seed
    )
    matrix = np.full((n_tasks, n_tasks), np.nan)
    seen: list[int] = []
    replay_features = np.empty((0, int(train_features.shape[1])))
    replay_labels = np.empty((0,), dtype=int)
    rng = np.random.default_rng(seed)
    ratios: list[float] = []
    transform = build_transform(normalize, scale=normalize_scale)
    isolation_rng = np.random.default_rng(seed + 7919)
    active_rows: list[list[int]] = []

    for t, classes in enumerate(tasks):
        seen.extend(int(c) for c in classes)
        mask = np.isin(train.y, classes)
        features = train_features[mask]
        labels = train.y[mask]
        if limit_per_class is not None:
            keep = []
            for label in classes:
                keep.append(np.flatnonzero(labels == label)[:limit_per_class])
            subset = np.concatenate(keep) if keep else np.array([], dtype=int)
            features = features[subset]
            labels = labels[subset]

        if t == 0:
            # 统计量只在这一批（任务 1 的训练特征）上算一次；之后冻结。
            transform.fit(features)
            test_features = transform.transform(test_features)
        features = transform.transform(features)
        active = _active_rows_for_task(isolation, classes, isolation_rng)
        active_rows.append([] if active is None else [int(c) for c in active])

        before = readout.weights.copy()
        _train_delta_readout(
            readout,
            features,
            labels,
            replay_features,
            replay_labels,
            epochs=readout_epochs,
            batch=readout_batch,
            rho=replay_rho,
            rng=rng,
            active_classes=active,
        )
        denominator = float(np.linalg.norm(before))
        if t == 0 or denominator < 1e-12:
            # 第一个任务从零初始化出发，相对变化没有分母；不参与跑门。
            ratios.append(float("nan"))
        else:
            ratios.append(float(np.linalg.norm(readout.weights - before) / denominator))

        if replay_rho > 0:
            new_features, new_labels = _sample_replay(
                features, labels, classes, replay_per_class, rng
            )
            replay_features = np.concatenate([replay_features, new_features], axis=0)
            replay_labels = np.concatenate([replay_labels, new_labels], axis=0)

        seen_now = np.asarray(sorted(seen), dtype=int)
        for i, task in enumerate(tasks[: t + 1]):
            pick = np.isin(test.y, task)
            matrix[t, i] = evaluate(readout, test_features[pick], test.y[pick], seen_now)
        print(f"  task {t + 1}/{n_tasks}: 本任务 {matrix[t, t]:.3f}")

    scores = summarise(matrix)
    gate_failed = any(np.isfinite(ratio) and ratio < GATE_MIN_WEIGHT_CHANGE for ratio in ratios)
    return {
        "seed": seed,
        "mode": "incremental-resnet18-delta",
        "n_tasks": n_tasks,
        "limit_per_class": limit_per_class,
        "replay_rho": replay_rho,
        "replay_per_class": replay_per_class,
        "readout_batch": readout_batch,
        "normalize": normalize,
        "normalize_scale": normalize_scale,
        "normalizer": transform.digest(),
        "isolation": isolation,
        "active_rows": active_rows,
        "accuracy_matrix": matrix.tolist(),
        "acc": scores.acc,
        "af": scores.af,
        "per_task_final": list(scores.per_task_final),
        "weight_change_ratios": ratios,
        "gate_failed": gate_failed,
        "valid": not gate_failed,
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
    parser.add_argument("--replay-rho", type=float, default=0.0, help="latent replay 比例；0=关闭")
    parser.add_argument("--replay-per-class", type=int, default=20)
    parser.add_argument(
        "--readout-batch",
        type=int,
        default=0,
        help="0=全批量（v1 协议）；>0=minibatch 大小（回放臂用 128）",
    )
    parser.add_argument(
        "--isolation",
        choices=("none", "current-rows", "random-rows"),
        default="none",
        help="输出行参数隔离：none=全行更新；current-rows=只更新当前任务类；"
        "random-rows=只更新同样数量的随机行（控制臂）",
    )
    parser.add_argument(
        "--normalize",
        choices=("none", "task1-standardize"),
        default="none",
        help="归一化臂：none=关（默认）；task1-standardize=任务 1 统计量冻结的逐维标准化",
    )
    parser.add_argument(
        "--normalize-scale",
        choices=("native", "match-task1"),
        default="native",
        help="native=标准化本身；match-task1=再对齐任务 1 的平均行范数（控制臂）",
    )
    parser.add_argument("--joint", action="store_true", help="跑联合训练上界")
    parser.add_argument("--smoke", action="store_true", help="2 任务、每类 20 张")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if args.normalize == "none" and args.normalize_scale != "native":
        parser.error("--normalize none 时 --normalize-scale 必须为 native。")
    if args.readout == "ncm" and args.normalize != "none":
        parser.error("归一化臂只接 delta 读出；NCM 的白化是另一条路径。")
    if args.readout == "ncm" and args.isolation != "none":
        parser.error("参数隔离臂只接 delta 读出；NCM 没有可冻结的输出行。")

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
            if args.readout == "ncm":
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
            else:
                results.append(
                    run_with_frozen_frontend_delta(
                        seed=seed,
                        n_tasks=args.tasks,
                        limit_per_class=args.limit_per_class,
                        data_root=args.data_root,
                        device=device,
                        backbone=args.backbone,
                        readout_epochs=args.readout_epochs,
                        readout_batch=args.readout_batch,
                        replay_rho=args.replay_rho,
                        replay_per_class=args.replay_per_class,
                        normalize=args.normalize,
                        normalize_scale=args.normalize_scale,
                        isolation=args.isolation,
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

    valid = [row for row in results if row.get("valid", True)]
    summary = {
        "experiment": "CIFAR-100 class-incremental",
        "mode": "joint" if args.joint else "incremental",
        "config": {
            "features": args.features,
            "backbone": args.backbone,
            "readout": args.readout,
            "n_tasks": args.tasks,
            "limit_per_class": args.limit_per_class,
            "hidden_epochs": args.hidden_epochs,
            "readout_epochs": args.readout_epochs,
            "batch": args.batch,
            "readout_batch": args.readout_batch,
            "replay_rho": args.replay_rho,
            "replay_per_class": args.replay_per_class,
            "normalize": args.normalize,
            "normalize_scale": args.normalize_scale,
            "isolation": args.isolation,
        },
        "per_seed": results,
        "void_seeds": [row["seed"] for row in results if not row.get("valid", True)],
        "mean_acc": float(np.mean([row["acc"] for row in valid])) if valid else float("nan"),
        "mean_af": float(np.nanmean([row["af"] for row in valid])) if valid else float("nan"),
        "repro": collect(
            "W5 持续学习：CIFAR-100 类增量",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                f"特征：{args.features}/{args.backbone}；读出：{args.readout}；"
                f"归一化：{args.normalize}（{args.normalize_scale}）；"
                f"隔离：{args.isolation}；"
                f"回放 ρ：{args.replay_rho}；batching：{args.readout_batch or '全批量'}"
            ],
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

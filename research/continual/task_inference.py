"""CL 任务推断保底臂：per-task 头 + 最近任务原型路由（预注册 v8）。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

预注册：[`sweep_results/continual_v2_task_inference_prereg.md`](../../sweep_results/continual_v2_task_inference_prereg.md)。

## 这个模块回答什么

输出行轴与特征轴都已封。本臂不改变"纯 CIL"的结论，只测量：**任务边界信息是否必须
以某种形式进入推理**。训练时任务标签可用（增量学习的常规设定）；**推理时只看输入**：

```text
每任务一个 10 类 delta 头 h_t（旧头冻结）
任务原型 μ_t = 本任务各类均值的均值（只用训练特征）
推理：t̂ = argmin_k ||x − μ_k||；ŷ = argmax_c h_t̂(x)
```

oracle 臂用真实任务下标选头，只作上界；routed 臂用最近原型。所有组件都是局部的
（delta、均值、最近邻），没有反向传播。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from research.continual.data import load_cifar100, task_classes
from research.continual.frontends import image_features
from research.continual.metrics import summarise
from research.continual.readout import DeltaReadout

__all__ = ["run_with_task_inference"]

#: 头的训练准确率下限。从零初始化的头训到收敛后，末 epoch 的相对变化会饱和在
#: lr 的有效尺度（实测 0.047），不能用来判"更新有没有进权重"；用训练准确率更直接。
GATE_MIN_TRAIN_ACCURACY = 0.5


def _task_stats(
    features: np.ndarray, labels: np.ndarray, classes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """本任务的均值与**类内**逐维方差；只用训练特征。"""
    means = [features[labels == label].mean(axis=0) for label in classes]
    variances = [features[labels == label].var(axis=0) for label in classes]
    return np.mean(means, axis=0), np.mean(variances, axis=0)


def _train_head(
    head: DeltaReadout,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int,
    batch: int,
    rng: np.random.Generator,
) -> None:
    """训练一个头；跑门在训练后用训练准确率与权重范数判。"""
    for _ in range(epochs):
        order = rng.permutation(features.shape[0])
        for start in range(0, features.shape[0], batch):
            index = order[start : start + batch]
            head.step(features[index], labels[index])


def _route(
    features: np.ndarray,
    prototypes: list[np.ndarray],
    variances: list[np.ndarray] | None = None,
) -> np.ndarray:
    """最近任务原型（欧氏）或对角 Mahalanobis 最近任务。"""
    matrix = np.stack(prototypes)
    if variances is None:
        distances = (
            (features**2).sum(axis=1)[:, None]
            + (matrix**2).sum(axis=1)[None, :]
            - 2.0 * features @ matrix.T
        )
    else:
        scale = np.stack(variances).clip(min=1e-6)
        # (n, tasks, dim) 的临时张量太大；逐任务算。
        squared = (features[:, None, :] - matrix[None, :, :]) ** 2 / scale[None, :, :]
        distances = squared.sum(axis=2)
    return np.argmin(distances, axis=1)


def _predict_task(
    features: np.ndarray,
    task: np.ndarray,
    head: DeltaReadout,
) -> np.ndarray:
    """在任务的局部头里 argmax，再映射回全局类 id。"""
    local = np.argmax(head.predict(features), axis=1)
    return task[local]


def run_with_task_inference(
    *,
    seed: int,
    n_tasks: int,
    limit_per_class: int | None,
    data_root: Path,
    device: torch.device,
    backbone: str = "resnet18",
    readout_epochs: int = 20,
    learning_rate: float = 0.01,
    readout_batch: int = 128,
    routing: str = "routed",
) -> dict:
    """跑一个种子的 task-inference 臂；``routing`` 为 ``routed`` 或 ``oracle``。"""
    if routing not in ("routed", "mahalanobis", "oracle"):
        raise ValueError(f"routing 只能是 'routed' / 'mahalanobis' / 'oracle'，收到 {routing!r}。")
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
    matrix = np.full((n_tasks, n_tasks), np.nan)
    routing_matrix = np.full((n_tasks, n_tasks), np.nan)
    heads: list[DeltaReadout] = []
    prototypes: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    head_train_accuracy: list[float] = []
    head_weight_norm: list[float] = []
    rng = np.random.default_rng(seed)

    for t, classes in enumerate(tasks):
        mask = np.isin(train.y, classes)
        features = train_features[mask]
        labels = train.y[mask]
        if limit_per_class is not None:
            keep = [np.flatnonzero(labels == label)[:limit_per_class] for label in classes]
            subset = np.concatenate(keep) if keep else np.array([], dtype=int)
            features = features[subset]
            labels = labels[subset]

        local = np.searchsorted(classes, labels).astype(int)
        head = DeltaReadout(
            int(features.shape[1]), len(classes), learning_rate=learning_rate, seed=seed + t
        )
        _train_head(
            head,
            features,
            local,
            epochs=readout_epochs,
            batch=readout_batch,
            rng=rng,
        )
        heads.append(head)
        mean, variance = _task_stats(features, labels, classes)
        prototypes.append(mean)
        variances.append(variance)
        head_train_accuracy.append(
            float(np.mean(np.argmax(head.predict(features), axis=1) == local))
        )
        head_weight_norm.append(float(np.linalg.norm(head.weights)))

        for i, task in enumerate(tasks[: t + 1]):
            pick = np.isin(test.y, task)
            x = test_features[pick]
            y = test.y[pick]
            if routing == "oracle":
                inferred = np.full(x.shape[0], i, dtype=int)
            elif routing == "mahalanobis":
                inferred = _route(x, prototypes[: t + 1], variances[: t + 1])
            else:
                inferred = _route(x, prototypes[: t + 1])
            predicted = np.empty(x.shape[0], dtype=int)
            for k in np.unique(inferred):
                rows = inferred == k
                predicted[rows] = _predict_task(x[rows], tasks[int(k)], heads[int(k)])
            matrix[t, i] = float(np.mean(predicted == y))
            routing_matrix[t, i] = float(np.mean(inferred == i))

        print(f"  task {t + 1}/{n_tasks}: 本任务 {matrix[t, t]:.3f}")

    # 最终一步的任务混淆矩阵（真实任务 × 推断任务）
    confusion = np.zeros((n_tasks, n_tasks), dtype=int)
    for i, task in enumerate(tasks):
        pick = np.isin(test.y, task)
        x = test_features[pick]
        if routing == "oracle":
            inferred = np.full(x.shape[0], i, dtype=int)
        elif routing == "mahalanobis":
            inferred = _route(x, prototypes, variances)
        else:
            inferred = _route(x, prototypes)
        for k in range(n_tasks):
            confusion[i, k] = int(np.sum(inferred == k))

    scores = summarise(matrix)
    gate_failed = any(
        norm <= 0.0 or accuracy < GATE_MIN_TRAIN_ACCURACY
        for norm, accuracy in zip(head_weight_norm, head_train_accuracy, strict=True)
    )
    return {
        "seed": seed,
        "mode": "task-inference",
        "routing": routing,
        "n_tasks": n_tasks,
        "limit_per_class": limit_per_class,
        "accuracy_matrix": matrix.tolist(),
        "acc": scores.acc,
        "af": scores.af,
        "per_task_final": list(scores.per_task_final),
        "routing_accuracy_matrix": routing_matrix.tolist(),
        "routing_accuracy": float(np.nanmean(np.diag(routing_matrix))),
        "confusion_matrix": confusion.tolist(),
        "head_train_accuracy": head_train_accuracy,
        "head_weight_norm": head_weight_norm,
        "gate_failed": gate_failed,
        "valid": not gate_failed,
    }

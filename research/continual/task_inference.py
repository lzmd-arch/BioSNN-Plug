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

#: 头在最后一个 epoch 的相对权重变化下限（与全局跑门同一条纪律）。
GATE_MIN_WEIGHT_CHANGE = 0.05


def _task_prototype(features: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """本任务各类均值的再平均；只用训练特征。"""
    means = [features[labels == label].mean(axis=0) for label in classes]
    return np.mean(means, axis=0)


def _train_head(
    head: DeltaReadout,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int,
    batch: int,
    rng: np.random.Generator,
) -> float:
    """训练一个头，返回最后一个 epoch 的相对权重变化（跑门用）。"""
    before_last = head.weights.copy()
    for epoch in range(epochs):
        if epoch == epochs - 1:
            before_last = head.weights.copy()
        order = rng.permutation(features.shape[0])
        for start in range(0, features.shape[0], batch):
            index = order[start : start + batch]
            head.step(features[index], labels[index])
    denominator = float(np.linalg.norm(before_last))
    if denominator < 1e-12:
        return float("inf") if np.linalg.norm(head.weights) > 0 else 0.0
    return float(np.linalg.norm(head.weights - before_last) / denominator)


def _route(
    features: np.ndarray,
    prototypes: list[np.ndarray],
) -> np.ndarray:
    """最近任务原型；用范数展开避免 ``n × tasks × dim`` 的临时张量。"""
    matrix = np.stack(prototypes)
    distances = (
        (features**2).sum(axis=1)[:, None]
        + (matrix**2).sum(axis=1)[None, :]
        - 2.0 * features @ matrix.T
    )
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
    if routing not in ("routed", "oracle"):
        raise ValueError(f"routing 只能是 'routed' 或 'oracle'，收到 {routing!r}。")
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
    gate_ratios: list[float] = []
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
        gate_ratios.append(
            _train_head(
                head,
                features,
                local,
                epochs=readout_epochs,
                batch=readout_batch,
                rng=rng,
            )
        )
        heads.append(head)
        prototypes.append(_task_prototype(features, labels, classes))

        for i, task in enumerate(tasks[: t + 1]):
            pick = np.isin(test.y, task)
            x = test_features[pick]
            y = test.y[pick]
            if routing == "oracle":
                inferred = np.full(x.shape[0], i, dtype=int)
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
        else:
            inferred = _route(x, prototypes)
        for k in range(n_tasks):
            confusion[i, k] = int(np.sum(inferred == k))

    scores = summarise(matrix)
    gate_failed = any(
        not np.isfinite(ratio) or ratio < GATE_MIN_WEIGHT_CHANGE for ratio in gate_ratios
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
        "head_gate_ratios": gate_ratios,
        "gate_failed": gate_failed,
        "valid": not gate_failed,
    }

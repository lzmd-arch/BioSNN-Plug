"""特征可分性：衡量"共享权重把类别分得有多开"。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

规格：[`sweep_results/arbiter_synergy_task_spec.md`](../../sweep_results/arbiter_synergy_task_spec.md)。

## 为什么不用端到端准确率

实测（2026-10-18）：MNIST 上任何 256 维特征 + 任意读出都有 30–50%，共享权重改动 71% 也不变。
**读出容量把表示质量盖住了**，于是三条臂的数字几乎一样，测不出规则的差别。

这里改为直接量特征本身：

| 指标 | 含义 | 为什么合适 |
| :--- | :--- | :--- |
| **Fisher 比** `Sb/Sw` | 类间散度 / 类内散度 | 无量纲、不需要训练、不受读出容量影响 |
| **线性探针准确率** | 在特征上做闭式岭回归 | 用**全部训练标签**：测的是"表示里有多少信息"，不是"少标签下能学多少" |

⚠️ **探针用全部标签是有意的**：训练时规则只看到有标签子集，但**测量**表示质量时应当把
标签全部用上——否则测到的是"这个规则在少标签下的样本效率"，而不是"它把表示塑造成了什么样"。
两件事都要能分开说，所以探针标签集与训练标签集**不是同一个**。
"""

from __future__ import annotations

import numpy as np

__all__ = ["fisher_ratio", "linear_probe_accuracy"]

#: 闭式探针的 λ。固定不调——探针不是自变量。
PROBE_LAMBDA = 1.0


def fisher_ratio(features: np.ndarray, labels: np.ndarray) -> float:
    """类间散度 / 类内散度。越大越好；``nan`` 表示类内散度为 0（完全可分）。

    Args:
        features: ``(n, d)``。
        labels: ``(n,)`` 的整数类别。
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError(f"特征 {x.shape} 与标签 {y.shape} 不一致。")
    if x.shape[0] == 0:
        raise ValueError("特征为空。")

    overall = x.mean(axis=0, keepdims=True)
    between = 0.0
    within = 0.0
    for label in np.unique(y):
        group = x[y == label]
        centroid = group.mean(axis=0, keepdims=True)
        between += group.shape[0] * float(((centroid - overall) ** 2).sum())
        within += float(((group - centroid) ** 2).sum())
    if within <= 0:
        return float("nan")
    return between / within


def linear_probe_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    n_classes: int = 10,
) -> float:
    """在**训练特征上**做闭式岭回归，返回**训练集自身**的准确率。

    ⚠️ 这是**可分性探针**，不是泛化指标：读出的就是"特征把训练样本分得开不开"。
    泛化要靠端到端任务去测，那条路已经证明会被读出容量盖住，所以这里换成这个。

    直接用全部训练标签（见模块文档）。
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError(f"特征 {x.shape} 与标签 {y.shape} 不一致。")
    if x.shape[0] == 0:
        return float("nan")

    design = np.hstack([x, np.ones((x.shape[0], 1))])
    target = np.zeros((x.shape[0], n_classes), dtype=np.float64)
    target[np.arange(x.shape[0]), y] = 1.0
    gram = design.T @ design + PROBE_LAMBDA * np.eye(design.shape[1])
    weights = np.linalg.solve(gram, design.T @ target)
    predicted = np.argmax(design @ weights, axis=1)
    return float(np.mean(predicted == y))

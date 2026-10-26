"""CL 特征级掩码：任务级的 Fisher top-k 与随机控制（预注册 v7）。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

预注册：[`sweep_results/continual_v2_feature_isolation_prereg.md`](../../sweep_results/continual_v2_feature_isolation_prereg.md)。

## 掩码落在哪一层

Route R 没有"输入→隐藏投影"：冻结 ResNet18 的 512 维 avgpool 特征直接进线性 delta
读出。所以掩码作用在**输入特征**上，等价于把每个类那行的权重按掩码屏蔽：

```text
logit_c = x · (m_c ⊙ w_c) + b_c
Δw_c   = η · (target_c − logit_c) · (m_c ⊙ x)
```

实现上不必逐类切片：`predict` 用 `x @ (W ⊙ M).T`，`step` 的权重增量乘 `M` 即可。

## 掩码怎么学（局部）

每个任务在**自己的训练特征与标签**上算逐维 Fisher 分数：

```text
score_j = 类间方差_j / (类内方差_j + eps)
m_t = top-k 个 score_j 置 1
```

没有反向传播、没有跨任务数据；任务结束时冻结。随机控制臂用同样大小的随机掩码。
"""

from __future__ import annotations

import numpy as np

__all__ = ["fisher_task_mask", "jaccard", "random_task_mask"]

DEFAULT_EPS = 1e-6


def _check(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError(f"特征必须是非空二维数组，收到 {x.shape}。")
    if y.shape[0] != x.shape[0]:
        raise ValueError(f"标签数 {y.shape[0]} 与样本数 {x.shape[0]} 不一致。")
    return x, y


def fisher_task_mask(
    features: np.ndarray,
    labels: np.ndarray,
    k: int,
    *,
    eps: float = DEFAULT_EPS,
) -> np.ndarray:
    """本任务训练数据上的 Fisher top-k 二值掩码，``(n_in,)``。

    类间方差按类均值（等权）算，类内方差按各类内部方差（等权）平均；分数只用于排序，
    不做阈值。``k`` 必须落在 ``[1, n_in]``。
    """
    x, y = _check(features, labels)
    n_in = x.shape[1]
    if not 1 <= k <= n_in:
        raise ValueError(f"k 必须在 [1, {n_in}]，收到 {k}。")
    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError(f"Fisher 分数至少需要 2 个类，收到 {classes.size} 个。")
    means = np.stack([x[y == label].mean(axis=0) for label in classes])
    within = np.mean(
        [((x[y == label] - means[index]) ** 2).mean(axis=0) for index, label in enumerate(classes)],
        axis=0,
    )
    between = means.var(axis=0)
    score = between / (within + eps)
    order = np.argsort(-score, kind="stable")[:k]
    mask = np.zeros(n_in, dtype=np.float64)
    mask[order] = 1.0
    return mask


def random_task_mask(n_in: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """同尺寸的随机二值掩码（控制臂）；``rng`` 由调用方固定。"""
    if not 1 <= k <= n_in:
        raise ValueError(f"k 必须在 [1, {n_in}]，收到 {k}。")
    mask = np.zeros(n_in, dtype=np.float64)
    mask[rng.choice(n_in, size=k, replace=False)] = 1.0
    return mask


def jaccard(first: np.ndarray, second: np.ndarray) -> float:
    """两个二值掩码的 Jaccard 重叠率；空掩码报 0 而不是 NaN。"""
    a = np.asarray(first) > 0
    b = np.asarray(second) > 0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0

"""类原型读出：冻结特征上的最近类均值（NCM），可选协方差白化。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

协议：[`sweep_results/continual_cifar100_prereg.md`](../../sweep_results/continual_cifar100_prereg.md)。

## 为什么换掉逐任务的 delta 读出

实测（200 张/类、1 epoch）：delta 读出在类增量下每个新任务都停在随机附近，
而**旧任务几乎不掉**（任务 1 一直是 0.37–0.39）。给读出加 5 倍预算只从 0.208 到 0.224——
瓶颈不在训练量。

类原型读出把"学习"换成**逐类累加一个均值**：

```text
μ_c ← (n_c·μ_c + Σ_{i∈task, y_i=c} x_i) / (n_c + |task∩c|)
ŷ(x) = argmin_c  d(x, μ_c)
```

它天然抗遗忘：**新任务只写它自己的类，不碰旧类的原型**。代价是它不使用任何
判别信息（不做类间优化），所以是"冻结特征 + 记忆"这一族的**下界**。

## 白化（可选）

类内分布各向异性时，欧氏距离会被方差大的方向主导。用**所有已见类的共享协方差**
做一次收缩白化（FeCAM 的简化版）：

```text
Σ ← shrink·I + (1−shrink)·Σ_all
d(x, μ) = (x−μ)ᵀ Σ⁻¹ (x−μ)
```

⚠️ **共享协方差只用已见类的特征**，不看类别标签之外的任何信息；累加协方差与累加均值
一样是逐类增量的。
"""

from __future__ import annotations

import numpy as np

__all__ = ["ClassPrototypes"]


def _symmetric_sqrt(matrix: np.ndarray) -> np.ndarray:
    """实对称半正定矩阵的平方根（经特征分解）。用于把白化写成"先变换再算欧氏距离"。"""
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, 0.0, None)
    return (vectors * np.sqrt(values)) @ vectors.T


class ClassPrototypes:
    """逐类累加原型；可选共享协方差白化。

    Args:
        n_classes: 总类别数（CIFAR-100 是 100）。
        whiten: 是否用共享协方差白化。
        shrinkage: 白化时往单位阵收缩的强度，``0`` = 不收缩、``1`` = 退化成欧氏。

    Note:
        ``predict`` 只在**已见类**上取最小距离——与类增量口径一致（模型不知道任务 id，
        但未学过的类本来就不该出现在候选里）。
    """

    def __init__(
        self,
        n_classes: int,
        *,
        whiten: bool = False,
        shrinkage: float = 0.1,
        whiten_mode: str = "diag",
    ) -> None:
        if n_classes <= 0:
            raise ValueError(f"n_classes 必须为正，收到 {n_classes}。")
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError(f"shrinkage 应在 [0, 1]，收到 {shrinkage}。")
        self.n_classes = int(n_classes)
        if whiten_mode not in ("diag", "shared"):
            raise ValueError(f"whiten_mode 只能是 'diag' 或 'shared'，收到 {whiten_mode!r}。")
        self.whiten = bool(whiten)
        self.shrinkage = float(shrinkage)
        self.whiten_mode = whiten_mode
        self.sums = np.zeros((n_classes, 0), dtype=np.float64)
        self.counts = np.zeros(n_classes, dtype=np.int64)
        self._second_moment: np.ndarray | None = None
        self._diag_second: np.ndarray | None = None
        self._n_seen = 0

    # ------------------------------------------------------------------ 累加
    def add(self, features: np.ndarray, labels: np.ndarray) -> None:
        """把一个任务的样本并入原型。**只更新这些标签对应的类**。"""
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=int)
        if x.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError(f"特征 {x.shape} 与标签 {y.shape} 不一致。")
        if self.sums.shape[1] == 0:
            self.sums = np.zeros((self.n_classes, x.shape[1]), dtype=np.float64)

        for label in np.unique(y):
            rows = x[y == label]
            self.sums[label] += rows.sum(axis=0)
            self.counts[label] += rows.shape[0]

        # 共享协方差：累加二阶矩（对已见样本，按行标准化前先减自身类均值会在后面做）
        if self.whiten:
            centred = x - np.stack([self._mean_of(y[i]) for i in range(x.shape[0])])
            if self.whiten_mode == "shared":
                self._second_moment = (
                    centred.T @ centred
                    if self._second_moment is None
                    else self._second_moment + centred.T @ centred
                )
            else:
                # 对角模式：只累加每维平方和，O(d) 内存。2048 维上完整协方差的 SVD 会 OOM（实测）。
                diag = (centred**2).sum(axis=0)
                self._diag_second = diag if self._diag_second is None else self._diag_second + diag
            self._n_seen += x.shape[0]

    def shrink_toward(self, target: np.ndarray, lam: float) -> None:
        """把每个**已见类**的原型朝 ``target`` 收缩一轮：``μ_c ← (1-λ)·μ_c + λ·target``。

        这是「巩固」在原型读出上的最小实现（ADR-0016 / `sweep_results/consolidation_prereg.md`）：
        **只操作已存统计**，不重放原始样本，因此不改变基准「无回放」的前提。
        """
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"lam 必须落在 [0, 1]，收到 {lam!r}。")
        seen = self.seen_classes
        if seen.size == 0:
            return
        target = np.asarray(target, dtype=np.float64)
        moved = (1.0 - lam) * self.prototypes() + lam * target
        self.sums[seen] = moved * self.counts[seen][:, None]

    def _mean_of(self, label: int) -> np.ndarray:
        if self.counts[label] == 0:
            return np.zeros(self.sums.shape[1])
        return self.sums[label] / self.counts[label]

    @property
    def seen_classes(self) -> np.ndarray:
        return np.flatnonzero(self.counts > 0)

    def prototypes(self) -> np.ndarray:
        seen = self.seen_classes
        return self.sums[seen] / self.counts[seen][:, None]

    # ------------------------------------------------------------------ 预测
    def predict(self, features: np.ndarray) -> np.ndarray:
        """返回**已见类**里距离最近的那个类的标签。"""
        seen = self.seen_classes
        if seen.size == 0:
            raise RuntimeError("还没有任何类被加入。")
        x = np.asarray(features, dtype=np.float64)
        mu = self.prototypes()

        if self.whiten and self.whiten_mode == "shared" and self._second_moment is not None:
            cov = self._second_moment / self._n_seen + 1e-6 * np.eye(self.sums.shape[1])
            cov = self.shrinkage * np.eye(cov.shape[0]) + (1 - self.shrinkage) * cov
            inverse = np.linalg.pinv(cov)
            # 白化等价于先把双方变换到 Σ^{-1/2} 空间，再用欧氏距离
            root = _symmetric_sqrt(inverse)
            x, mu = x @ root, mu @ root
        elif self.whiten and self.whiten_mode == "diag" and self._diag_second is not None:
            # 对角白化：每个维度除以自己的标准差（收缩到 1），O(d) 内存。
            variance = self._diag_second / max(self._n_seen, 1)
            variance = self.shrinkage + (1.0 - self.shrinkage) * variance
            scale = 1.0 / np.sqrt(np.maximum(variance, 1e-8))
            x, mu = x * scale, mu * scale

        # ⚠️ 用 (x−μ)² = ‖x‖² − 2x·μ + ‖μ‖² 展开，**不实例化 (n, C, d) 的三维数组**。
        # 之前直接相减，在 8000×90×1000 上要 5.36 GiB，实测 OOM。
        x_norm = (x**2).sum(axis=1)[:, None]
        mu_norm = (mu**2).sum(axis=1)[None, :]
        distance = x_norm - 2.0 * (x @ mu.T) + mu_norm

        return seen[np.argmin(distance, axis=1)]

    def accuracy(self, features: np.ndarray, labels: np.ndarray) -> float:
        predicted = self.predict(features)
        return float(np.mean(predicted == np.asarray(labels, dtype=int)))

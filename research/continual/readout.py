"""线性读出：Widrow–Hoff delta 规则（局部、有教师）。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

预注册：[`sweep_results/continual_cifar100_prereg.md`](../../sweep_results/continual_cifar100_prereg.md)。

## 规则

```text
ŷ = W·x + b
ΔW = η · (y* − ŷ) ⊗ x
Δb = η · (y* − ŷ)
```

每个输出单元只用到自己的误差与输入——没有反向传播穿过特征。类增量时**输出维度固定在 100**，
每个任务只喂它自己的类；旧类的输出不会被重训（这正是遗忘的来源）。

⚠️ **这不是"防护"**：本类没有任何抗遗忘机制（无回放、无正则、无参数隔离）。它测的就是
朴素增量下的遗忘——计划书的基准要的是这个数，不是把它修好。
"""

from __future__ import annotations

import numpy as np

__all__ = ["DeltaReadout", "evaluate"]


class DeltaReadout:
    """固定输出维度的线性读出。

    Args:
        n_in: 特征维度。
        n_classes: 输出维度（CIFAR-100 恒为 100）。
        learning_rate: 步长 η。**稳定性条件**：``η · λ_max(XᵀX)/n < 2``。特征未归一化时
            这个条件很容易被违反（实测 η=0.1 在 8 维、120 样本上直接发散），所以默认取
            ``0.01``；发散会被 :meth:`step` 显式检测并抛错，不会静默变成 NaN。
        seed: 权重初始化种子（零初始化时不影响结果，但保留以统一配置）。
    """

    def __init__(
        self, n_in: int, n_classes: int, *, learning_rate: float = 0.01, seed: int = 0
    ) -> None:
        if n_in <= 0 or n_classes <= 0:
            raise ValueError(f"n_in 与 n_classes 必须为正，收到 {n_in}、{n_classes}。")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate 必须为正，收到 {learning_rate}。")
        self.n_in = int(n_in)
        self.n_classes = int(n_classes)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.weights = np.zeros((n_classes, n_in), dtype=np.float64)
        self.bias = np.zeros(n_classes, dtype=np.float64)
        # 特征级掩码（预注册 v7）：``(n_classes, n_in)`` 的 0/1 矩阵，按类共享任务掩码。
        # ``None`` 表示不启用——默认路径逐位不变。
        self.masks: np.ndarray | None = None

    def predict(self, features: np.ndarray) -> np.ndarray:
        """``(n, n_in)`` → ``(n, n_classes)`` 的 logits；有掩码时按 ``W ⊙ M`` 算。"""
        x = np.asarray(features, dtype=np.float64)
        if self.masks is None:
            return x @ self.weights.T + self.bias
        return x @ (self.weights * self.masks).T + self.bias

    def set_masks_for_classes(self, classes: np.ndarray, mask: np.ndarray) -> None:
        """把 ``mask`` 写到这些类的行上；未涉及的类保持原来的掩码（旧掩码冻结）。"""
        values = np.asarray(mask, dtype=np.float64)
        if values.shape != (self.n_in,):
            raise ValueError(f"掩码必须是 ({self.n_in},)，收到 {values.shape}。")
        index = np.asarray(classes, dtype=int)
        if index.ndim != 1 or index.size == 0:
            raise ValueError("classes 必须是非空一维数组。")
        if index.min() < 0 or index.max() >= self.n_classes:
            raise ValueError(f"classes 必须在 [0, {self.n_classes}) 内，收到 {index}。")
        if self.masks is None:
            self.masks = np.zeros((self.n_classes, self.n_in), dtype=np.float64)
        self.masks[index] = values

    def step(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        active_classes: np.ndarray | None = None,
        inactive_scale: float | None = None,
    ) -> None:
        """一次批量 delta 更新（按样本数的均值，避免步长随批量大小漂移）。

        ``active_classes`` 给出时，激活行按全步长更新，其余行按 ``inactive_scale``
        缩放更新——``0`` 就是预注册 v4 的完全冻结，``1`` 等于全行更新。``None`` 走原路径
        （全行更新），逐位不变。
        """
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=int)
        if x.ndim != 2 or x.shape[1] != self.n_in:
            raise ValueError(f"特征必须是 (n, {self.n_in})，收到 {x.shape}。")
        if y.shape[0] != x.shape[0]:
            raise ValueError(f"标签数 {y.shape[0]} 与样本数 {x.shape[0]} 不一致。")

        # 默认语义：给了 active_classes 就**冻结**其余行（P1）；没给就全行同一步长。
        # 显式传 0/1 时按显式值走。这样 v4 的调用方不必改，α 又是一个独立的旋钮。
        if inactive_scale is None:
            inactive_scale = 0.0 if active_classes is not None else 1.0
        if not 0.0 <= inactive_scale <= 1.0:
            raise ValueError(f"inactive_scale 应在 [0, 1]，收到 {inactive_scale}。")
        if active_classes is None and inactive_scale != 1.0:
            raise ValueError("没有 active_classes 时 inactive_scale 必须为 1（全行同一步长）。")

        target = np.zeros((x.shape[0], self.n_classes), dtype=np.float64)
        target[np.arange(x.shape[0]), y] = 1.0
        with np.errstate(over="ignore", invalid="ignore"):
            error = target - self.predict(x)
            scale = self.learning_rate / x.shape[0]
            if active_classes is None:
                update = scale * (error.T @ x)
                if self.masks is not None:
                    update = update * self.masks
                self.weights += update
                self.bias += scale * error.sum(axis=0)
            else:
                active = np.asarray(active_classes, dtype=int)
                if active.ndim != 1 or active.size == 0:
                    raise ValueError("active_classes 必须是非空一维数组。")
                if active.min() < 0 or active.max() >= self.n_classes:
                    raise ValueError(
                        f"active_classes 必须在 [0, {self.n_classes}) 内，收到 {active}。"
                    )
                index = np.unique(active)
                weight_update = scale * (error.T @ x)
                bias_update = scale * error.sum(axis=0)
                if self.masks is not None:
                    # 特征掩码与行隔离可以叠加：先按掩码屏蔽输入侧，再按 α 缩放非激活行。
                    weight_update = weight_update * self.masks
                if inactive_scale == 0.0:
                    # 与 v4 的完全冻结**逐位一致**：只加激活行，不碰其余行。
                    self.weights[index] += weight_update[index]
                    self.bias[index] += bias_update[index]
                else:
                    mask = np.full(self.n_classes, inactive_scale, dtype=np.float64)
                    mask[index] = 1.0
                    self.weights += weight_update * mask[:, None]
                    self.bias += bias_update * mask
        if not (np.all(np.isfinite(self.weights)) and np.all(np.isfinite(self.bias))):
            raise RuntimeError(
                "delta 读出发散（权重出现 inf/NaN）。检查步长与特征尺度："
                "稳定性要求 η·λ_max(XᵀX)/n < 2。"
            )


def evaluate(
    readout: DeltaReadout, features: np.ndarray, labels: np.ndarray, classes: np.ndarray
) -> float:
    """在给定类别子集上算准确率。

    **只在 ``classes`` 里取最大值**——这与类增量评测的口径一致：模型不知道任务 id，
    但这里限制到"已见过的类"是因为还没学过的类本来就不该出现在评测里。
    """
    if features.shape[0] == 0:
        raise ValueError("评测集为空。")
    logits = readout.predict(features)
    logits = logits[:, np.asarray(classes, dtype=int)]
    predicted = np.asarray(classes, dtype=int)[np.argmax(logits, axis=1)]
    return float(np.mean(predicted == np.asarray(labels, dtype=int)))

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

    def predict(self, features: np.ndarray) -> np.ndarray:
        """``(n, n_in)`` → ``(n, n_classes)`` 的 logits。"""
        return np.asarray(features, dtype=np.float64) @ self.weights.T + self.bias

    def step(self, features: np.ndarray, labels: np.ndarray) -> None:
        """一次批量 delta 更新（按样本数的均值，避免步长随批量大小漂移）。"""
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=int)
        if x.ndim != 2 or x.shape[1] != self.n_in:
            raise ValueError(f"特征必须是 (n, {self.n_in})，收到 {x.shape}。")
        if y.shape[0] != x.shape[0]:
            raise ValueError(f"标签数 {y.shape[0]} 与样本数 {x.shape[0]} 不一致。")

        target = np.zeros((x.shape[0], self.n_classes), dtype=np.float64)
        target[np.arange(x.shape[0]), y] = 1.0
        with np.errstate(over="ignore", invalid="ignore"):
            error = target - self.predict(x)
            scale = self.learning_rate / x.shape[0]
            self.weights += scale * (error.T @ x)
            self.bias += scale * error.sum(axis=0)
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

"""跨模态联想：一层外积 Hebbian 映射，把一路的表示翻到另一路的空间里。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

## 为什么必须有这一层

`biosnn-bus` 的两个通道各自经过**独立种子**的随机投影，所以「只有文本」与「只有图像」
的总线输出虽然在同一个 `bus_dim` 维空间里，却是两个互不相干的线性像。直接算余弦，
期望为 0——预注册里把这条写在最前面，免得跑出随机水平被误读成编码坏了。

这一层就是那条「纯局部的联想规则」：**外积 Hebb 学习，没有梯度、没有反向传播**。

```text
W = η · Σ_i  v_i ⊗ t_i          # t_i 文本侧、v_i 图像侧
```

推理时 `文本 → W·t / (1 + λ)`，再与图像侧表示比余弦。反方向用**另一个** W′，
不假设 W 与 W′ 互为转置——局部规则没有理由保证这件事。

## 两个超参怎么定

`η` 与 `λ` **只在训练集上选**（留一法），测试集不参与任何选择。留一法是**精确**的，
不是近似：外积和是线性的，`W_{-i} = W − η·v_i⊗t_i` 一步就能减出来，不必重训。

⚠️ **`η` 单独没有意义**：余弦对全局缩放不敏感，`η` 只有相对 `1/(1+λ)` 才有作用。
所以网格给的是「η × λ」的组合，报的是组合，不是 η 一个数。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import recall_at_k

__all__ = ["HebbianAssociator", "select_hyperparameters"]

#: 选参网格。**先写死**，不在看到测试集之后扩大（那就是过拟合到测试集）。
ETA_GRID = (0.001, 0.01, 0.1, 1.0)
DECAY_GRID = (0.0, 0.1, 1.0)


@dataclass
class HebbianAssociator:
    """外积 Hebb 联想：``source → target``。"""

    eta: float = 0.01
    decay: float = 0.0
    weight: np.ndarray | None = None

    def fit(self, source: np.ndarray, target: np.ndarray) -> HebbianAssociator:
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("source 与 target 都必须是二维 (n, d)。")
        if source.shape[0] != target.shape[0]:
            raise ValueError(f"样本数必须相同，收到 {source.shape[0]} 与 {target.shape[0]}。")
        self.weight = self.eta * (target.T @ source)
        return self

    def transform(self, source: np.ndarray) -> np.ndarray:
        if self.weight is None:
            raise RuntimeError("先 fit() 再 transform()。")
        source = np.asarray(source, dtype=np.float64)
        return (source @ self.weight.T) / (1.0 + self.decay)

    def __repr__(self) -> str:
        shape = None if self.weight is None else self.weight.shape
        return f"HebbianAssociator(eta={self.eta}, decay={self.decay}, weight={shape})"


def _loo_transform(
    source: np.ndarray, target: np.ndarray, index: int, eta: float, decay: float
) -> np.ndarray:
    """第 ``index`` 条样本在**去掉自己**的 W 下的变换结果。"""
    full = eta * (target.T @ source)  # (dt, ds)
    own = eta * np.outer(target[index], source[index])
    reduced = full - own
    return (source[index] @ reduced.T) / (1.0 + decay)


def select_hyperparameters(
    source: np.ndarray,
    target: np.ndarray,
    *,
    eta_grid: tuple[float, ...] = ETA_GRID,
    decay_grid: tuple[float, ...] = DECAY_GRID,
) -> tuple[float, float, dict[tuple[float, float], float]]:
    """在训练集上做留一法选 ``(eta, decay)``，返回最优组合与全网格分数。

    ⚠️ **留一法在这里有它的上限**：训练样本同时是"记忆对象"和"泛化证据"。
    如果最优解是记住训练集（`eta` 很大、`decay` 很小），留一法会看出来——被留出的那一条
    不在和里——但它看不出"这个映射换一个数据集还成不成立"。所以测试集那一栏是必须的，
    不能拿训练集的留一分数替代。
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n = source.shape[0]
    scores: dict[tuple[float, float], float] = {}

    for eta in eta_grid:
        for decay in decay_grid:
            hits = 0
            for i in range(n):
                predicted = _loo_transform(source, target, i, eta, decay)
                similarity = (target @ predicted) / (
                    np.linalg.norm(target, axis=1) * np.linalg.norm(predicted) + 1e-12
                )
                order = np.argsort(-similarity, kind="stable")
                rank = int(np.where(order == i)[0][0]) + 1
                hits += rank <= 1
            scores[(eta, decay)] = hits / n

    best = max(scores, key=lambda key: (scores[key], -key[0]))
    return best[0], best[1], scores


def evaluate_direction(
    associator: HebbianAssociator,
    queries: np.ndarray,
    gallery: np.ndarray,
    *,
    direction: str,
    ks: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    """把查询翻到候选空间，算 R@K。"""
    predicted = associator.transform(queries)
    return {f"r{k}": recall_at_k(predicted, gallery, k=k, direction=direction).recall for k in ks}

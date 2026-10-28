"""阶段四神经发生的最小实现：可增长随机特征层 + 冻结旧类读出。

边界：这是**玩具规模**的控制器验证，不是多模态真实插件实验；特征是固定随机投影，
增长只加特征神经元，不做反向传播。旧类读出冻结是控制变量，不是抗遗忘算法。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["GrowableFeatureReadout", "GrowthController", "GrowthDecision"]


@dataclass(frozen=True)
class GrowthDecision:
    grow: bool
    current: int
    next_size: int
    reason: str


class GrowthController:
    def __init__(
        self, *, n0: int, growth: int = 16, max_ratio: float = 2.0, min_accuracy: float = 0.90
    ) -> None:
        if n0 <= 0 or growth <= 0 or max_ratio < 1.0:
            raise ValueError("n0/growth 必须为正，max_ratio >= 1。")
        self.n0 = int(n0)
        self.growth = int(growth)
        self.max_size = int(n0 * max_ratio)
        self.min_accuracy = float(min_accuracy)

    def decide(self, accuracy: float, current: int) -> GrowthDecision:
        if accuracy >= self.min_accuracy:
            return GrowthDecision(False, current, current, "accuracy_ok")
        if current >= self.max_size:
            return GrowthDecision(False, current, current, "max_size")
        return GrowthDecision(
            True, current, min(current + self.growth, self.max_size), "low_accuracy"
        )


class GrowableFeatureReadout:
    """ReLU 随机特征 + 线性读出；增长时补零旧权重，旧类行冻结。"""

    def __init__(
        self,
        n_in: int,
        n_classes: int,
        *,
        n0: int = 64,
        growth: int = 16,
        max_ratio: float = 2.0,
        min_accuracy: float = 0.90,
        learning_rate: float = 0.001,
        epochs: int = 200,
        seed: int = 0,
    ) -> None:
        self.n_in = int(n_in)
        self.n_classes = int(n_classes)
        self.rng = np.random.default_rng(seed)
        self.n_features = int(n0)
        self.projection = self._projection(n0)
        self.weights = np.zeros((self.n_classes, self.n_features))
        self.bias = np.zeros(self.n_classes)
        self.controller = GrowthController(
            n0=n0, growth=growth, max_ratio=max_ratio, min_accuracy=min_accuracy
        )
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.decisions: list[GrowthDecision] = []

    def _projection(self, n: int) -> np.ndarray:
        return self.rng.normal(size=(self.n_in, n)) / np.sqrt(self.n_in)

    def _features(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(np.asarray(x, dtype=np.float64) @ self.projection, 0.0)

    def grow(self) -> None:
        added = self.controller.growth
        self.projection = np.hstack([self.projection, self._projection(added)])
        self.weights = np.hstack([self.weights, np.zeros((self.n_classes, added))])
        self.n_features += added

    def fit_task(self, x: np.ndarray, y: np.ndarray, *, classes: np.ndarray) -> float:
        """只训练 ``classes`` 对应的读出 rows；旧 rows 冻结。返回训练后本任务准确率。"""
        f = self._features(x)
        target = np.zeros((len(y), len(classes)))
        target[np.arange(len(y)), np.searchsorted(classes, y)] = 1.0
        rows = np.asarray(classes, dtype=int)
        for _ in range(self.epochs):
            predicted = f @ self.weights[rows].T + self.bias[rows]
            error = target - predicted
            self.weights[rows] += self.learning_rate * error.T @ f / len(y)
            self.bias[rows] += self.learning_rate * error.mean(axis=0)
        predicted = np.argmax(f @ self.weights[rows].T + self.bias[rows], axis=1)
        return float(np.mean(classes[predicted] == y))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self._features(x) @ self.weights.T + self.bias, axis=1)

    def maybe_grow(self, accuracy: float) -> GrowthDecision:
        decision = self.controller.decide(accuracy, self.n_features)
        self.decisions.append(decision)
        if decision.grow:
            self.grow()
        return decision

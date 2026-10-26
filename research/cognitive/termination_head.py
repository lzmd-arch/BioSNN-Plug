"""世界模型 P3 的终止头 v1：类平衡 L2 逻辑回归（测量头，不是局部学习规则）。

预注册见 ``sweep_results/termination_head_v1_prereg.md``。特征图与
``RidgeActionModel(quadratic=True)`` 相同；这里只把终止从联合回归里拆出来。
"""

from __future__ import annotations

import numpy as np

__all__ = ["TerminationHead", "build_features", "sigmoid"]


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=np.float64)))


def build_features(
    states: np.ndarray, actions: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """``[z, onehot(a), a·z, z², (a·z)²]``，与 ridge 的二次特征同图。"""
    s = np.asarray(states, dtype=np.float64)
    a = np.asarray(actions, dtype=np.int64)
    if s.ndim != 2 or a.shape[0] != s.shape[0]:
        raise ValueError("states 与 actions 形状不一致。")
    z = (s - mean) / np.maximum(scale, 1e-8)
    onehot = np.zeros((s.shape[0], 2), dtype=np.float64)
    onehot[np.arange(s.shape[0]), a] = 1.0
    interaction = np.hstack([onehot[:, [i]] * z for i in range(2)])
    return np.hstack([z, onehot, interaction, z**2, interaction**2])


class TerminationHead:
    """类平衡 L2 逻辑回归；IRLS/牛顿迭代，不反向传播。"""

    def __init__(self, *, l2: float = 1.0, iterations: int = 50) -> None:
        if l2 < 0:
            raise ValueError(f"l2 必须非负，收到 {l2}。")
        self.l2 = float(l2)
        self.iterations = int(iterations)
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.weights: np.ndarray | None = None

    def fit(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        terminated: np.ndarray,
    ) -> None:
        states = np.asarray(states, dtype=np.float64)
        terminated = np.asarray(terminated, dtype=np.float64)
        self.mean = states.mean(axis=0)
        self.scale = np.maximum(states.std(axis=0), 1e-8)
        design = build_features(states, actions, self.mean, self.scale)
        # 特征再标准化（截距除外），让 IRLS 条件数正常。
        design_mean = design.mean(axis=0)
        design_scale = np.maximum(design.std(axis=0), 1e-8)
        design = (design - design_mean) / design_scale
        augmented = np.hstack([np.ones((design.shape[0], 1)), design])

        positives = float(np.sum(terminated == 1.0))
        negatives = float(terminated.size - positives)
        if positives == 0 or negatives == 0:
            raise ValueError("训练集必须同时包含终止与未终止样本。")
        positive_weight = terminated.size / (2.0 * positives)
        negative_weight = terminated.size / (2.0 * negatives)
        sample_weight = np.where(terminated == 1.0, positive_weight, negative_weight)

        weights = np.zeros(augmented.shape[1], dtype=np.float64)
        penalty = self.l2 * np.eye(augmented.shape[1])
        penalty[0, 0] = 0.0  # 不罚截距
        for _ in range(self.iterations):
            probabilities = sigmoid(augmented @ weights)
            gradient = (
                augmented.T @ (sample_weight * (probabilities - terminated)) + penalty @ weights
            )
            curvature = (
                augmented.T
                @ (augmented * (sample_weight * probabilities * (1.0 - probabilities))[:, None])
                + penalty
            )
            weights -= np.linalg.solve(curvature, gradient)
        self.weights = weights
        self._design_mean = design_mean
        self._design_scale = design_scale

    def predict_proba(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        if self.weights is None or self.mean is None or self.scale is None:
            raise RuntimeError("头为空；先 fit。")
        design = build_features(states, actions, self.mean, self.scale)
        design = (design - self._design_mean) / self._design_scale
        augmented = np.hstack([np.ones((design.shape[0], 1)), design])
        return sigmoid(augmented @ self.weights)

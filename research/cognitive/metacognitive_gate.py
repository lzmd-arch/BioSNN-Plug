"""元认知门控 v0：选择性预测、温度缩放、风险-覆盖率。

预注册见 ``sweep_results/metacognitive_gate_v0_prereg.md``。**不训练新网络**：在冻结
ResNet18 特征上训练一个 100 类 delta 读出，用 softmax 最大概率当置信度；温度与阈值
只在校准集上选，评估集不参与选择。
"""

from __future__ import annotations

import numpy as np

from research.continual import DeltaReadout

__all__ = [
    "LogisticCalibrator",
    "coverage_threshold",
    "ece",
    "ece_confidence",
    "fit_sigmoid_temperature",
    "fit_temperature",
    "margin_scores",
    "risk_coverage_auc",
    "selective_accuracy",
    "selective_accuracy_at",
    "sigmoid",
    "softmax",
    "train_readout",
]


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError(f"temperature 必须为正，收到 {temperature}。")
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    probs = softmax(logits, temperature)
    picks = probs[np.arange(labels.shape[0]), labels]
    return float(-np.mean(np.log(np.clip(picks, 1e-12, None))))


def fit_temperature(
    logits: np.ndarray, labels: np.ndarray, *, low: float = 0.5, high: float = 5.0, points: int = 40
) -> float:
    """在校准集上网格搜索一个标量温度，最小化 NLL。"""
    grid = np.geomspace(low, high, points)
    losses = [_nll(logits, labels, float(t)) for t in grid]
    return float(grid[int(np.argmin(losses))])


def ece(probabilities: np.ndarray, correct: np.ndarray, *, bins: int = 10) -> float:
    """期望校准误差：按最大概率分箱，|准确率 − 平均置信度| 的加权和。"""
    return ece_confidence(np.max(probabilities, axis=1), correct, bins=bins)


def ece_confidence(confidence: np.ndarray, correct: np.ndarray, *, bins: int = 10) -> float:
    """直接给置信度（如 margin 标定后的 p）算 ECE。"""
    confidence = np.asarray(confidence, dtype=np.float64)
    accuracy = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        left, right = edges[index], edges[index + 1]
        mask = (confidence > left) & (confidence <= right)
        if not np.any(mask):
            continue
        total += float(np.mean(mask)) * abs(
            float(np.mean(accuracy[mask])) - float(np.mean(confidence[mask]))
        )
    return total


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=np.float64)))


def margin_scores(logits: np.ndarray) -> np.ndarray:
    """``top1 − top2`` 的 logit margin；排序用。"""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"logits 必须是 (n, ≥2)，收到 {values.shape}。")
    top2 = np.partition(values, -2, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def _sigmoid_nll(margin: np.ndarray, correct: np.ndarray, temperature: float) -> float:
    p = sigmoid(np.asarray(margin) / temperature)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(correct * np.log(p) + (1.0 - correct) * np.log(1.0 - p)))


def fit_sigmoid_temperature(
    margin: np.ndarray,
    correct: np.ndarray,
    *,
    low: float = 0.01,
    high: float = 10.0,
    points: int = 40,
) -> float:
    """把 margin 校准成 P(正确)：``sigmoid(margin / T)``，校准集最小 NLL。"""
    grid = np.geomspace(low, high, points)
    truth = np.asarray(correct, dtype=np.float64)
    losses = [_sigmoid_nll(margin, truth, float(t)) for t in grid]
    return float(grid[int(np.argmin(losses))])


def selective_accuracy(
    confidence: np.ndarray, correct: np.ndarray, *, coverage: float
) -> tuple[float, float, float]:
    """按置信度取前 ``coverage`` 比例，返回 (选择性准确率, 实际覆盖率, 阈值)。"""
    if not 0.0 < coverage <= 1.0:
        raise ValueError(f"coverage 应在 (0, 1]，收到 {coverage}。")
    threshold = float(np.quantile(confidence, 1.0 - coverage))
    mask = confidence >= threshold
    if not np.any(mask):
        return float("nan"), 0.0, threshold
    return float(np.mean(correct[mask])), float(np.mean(mask)), threshold


def coverage_threshold(confidence: np.ndarray, coverage: float) -> float:
    """在**校准集**置信度上取阈值，使保留比例为 ``coverage``。"""
    if not 0.0 < coverage <= 1.0:
        raise ValueError(f"coverage 应在 (0, 1]，收到 {coverage}。")
    return float(np.quantile(np.asarray(confidence, dtype=np.float64), 1.0 - coverage))


def selective_accuracy_at(
    confidence: np.ndarray, correct: np.ndarray, *, threshold: float
) -> tuple[float, float]:
    """用给定阈值（来自校准集）在评估集上算选择性准确率与实际覆盖率。"""
    mask = np.asarray(confidence) >= threshold
    if not np.any(mask):
        return float("nan"), 0.0
    return float(np.mean(np.asarray(correct)[mask])), float(np.mean(mask))


def risk_coverage_auc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """按置信度从高到低排序后的准确率-覆盖率曲线下面积（越高越好）。"""
    order = np.argsort(-confidence)
    running = np.cumsum(np.asarray(correct, dtype=float)[order])
    steps = np.arange(1, order.size + 1)
    curve = running / steps
    # 覆盖率等分，梯形积分；避免依赖 numpy 版本的 trapz/trapezoid 命名差异。
    return float(np.mean((curve[:-1] + curve[1:]) / 2.0)) if curve.size > 1 else float(curve[0])


class LogisticCalibrator:
    """通用 L2 逻辑回归标定头（IRLS）；把特征映射成 P(正确)。"""

    def __init__(self, *, l2: float = 1.0, iterations: int = 50) -> None:
        if l2 < 0:
            raise ValueError("l2 必须非负。")
        self.l2 = float(l2)
        self.iterations = int(iterations)
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.weights: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        values = np.asarray(features, dtype=np.float64)
        truth = np.asarray(labels, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        if values.shape[0] != truth.shape[0]:
            raise ValueError("features 与 labels 行数不一致。")
        self.mean = values.mean(axis=0)
        self.scale = np.maximum(values.std(axis=0), 1e-8)
        design = (values - self.mean) / self.scale
        augmented = np.hstack([np.ones((design.shape[0], 1)), design])

        positives = float(np.sum(truth == 1.0))
        negatives = float(truth.size - positives)
        if positives == 0 or negatives == 0:
            raise ValueError("标定集必须同时包含正确与错误样本。")
        sample_weight = np.where(
            truth == 1.0,
            truth.size / (2.0 * positives),
            truth.size / (2.0 * negatives),
        )
        weights = np.zeros(augmented.shape[1], dtype=np.float64)
        penalty = self.l2 * np.eye(augmented.shape[1])
        penalty[0, 0] = 0.0
        for _ in range(self.iterations):
            probabilities = sigmoid(augmented @ weights)
            gradient = augmented.T @ (sample_weight * (probabilities - truth)) + penalty @ weights
            curvature = (
                augmented.T
                @ (augmented * (sample_weight * probabilities * (1.0 - probabilities))[:, None])
                + penalty
            )
            weights -= np.linalg.solve(curvature, gradient)
        self.weights = weights

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.weights is None or self.mean is None or self.scale is None:
            raise RuntimeError("标定头为空；先 fit。")
        values = np.asarray(features, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        design = (values - self.mean) / self.scale
        augmented = np.hstack([np.ones((design.shape[0], 1)), design])
        return sigmoid(augmented @ self.weights)


def train_readout(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int,
    batch: int,
    learning_rate: float = 0.01,
    seed: int = 0,
) -> DeltaReadout:
    readout = DeltaReadout(features.shape[1], 100, learning_rate=learning_rate, seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(features.shape[0])
        for start in range(0, features.shape[0], batch):
            index = order[start : start + batch]
            readout.step(features[index], labels[index])
    return readout

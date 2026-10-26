"""元认知门控 v0：选择性预测、温度缩放、风险-覆盖率。

预注册见 ``sweep_results/metacognitive_gate_v0_prereg.md``。**不训练新网络**：在冻结
ResNet18 特征上训练一个 100 类 delta 读出，用 softmax 最大概率当置信度；温度与阈值
只在校准集上选，评估集不参与选择。
"""

from __future__ import annotations

import numpy as np

from research.continual.readout import DeltaReadout

__all__ = [
    "ece",
    "fit_temperature",
    "risk_coverage_auc",
    "selective_accuracy",
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
    confidence = np.max(probabilities, axis=1)
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


def risk_coverage_auc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """按置信度从高到低排序后的准确率-覆盖率曲线下面积（越高越好）。"""
    order = np.argsort(-confidence)
    running = np.cumsum(np.asarray(correct, dtype=float)[order])
    steps = np.arange(1, order.size + 1)
    curve = running / steps
    # 覆盖率等分，梯形积分；避免依赖 numpy 版本的 trapz/trapezoid 命名差异。
    return float(np.mean((curve[:-1] + curve[1:]) / 2.0)) if curve.size > 1 else float(curve[0])


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

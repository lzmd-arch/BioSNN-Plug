"""多时间尺度工作记忆 v0：固定 τ 的漏积分器 + 线性读出。

预注册见 ``sweep_results/working_memory_v0_prereg.md``。两个线索 A（早）与 B（晚）在
延迟后同时回忆；快 τ 记 B、慢 τ 记 A，单 4 维状态在秩上无法同时重建两者。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "CUE_A",
    "CUE_B",
    "N_DIMS",
    "TAUS",
    "TEST_STEP",
    "LinearDeltaReadout",
    "average_r2",
    "encode_episode",
    "encode_timeline",
    "fit_ridge",
    "predict",
]

TAUS = (2, 5, 10, 20, 50, 100, 200)
N_DIMS = 4
CUE_A = (0, 4)
CUE_B = (20, 24)
TEST_STEP = 50
NOISE = 0.01


def encode_episode(
    cue_a: np.ndarray,
    cue_b: np.ndarray,
    *,
    rng: np.random.Generator,
    taus: tuple[int, ...] = TAUS,
    n_dims: int = N_DIMS,
    noise: float = NOISE,
    test_step: int = TEST_STEP,
) -> np.ndarray:
    """跑一条时间线，返回 ``(len(taus)·n_dims,)`` 的测试时刻状态。"""
    cue_a = np.asarray(cue_a, dtype=np.float64)
    cue_b = np.asarray(cue_b, dtype=np.float64)
    if cue_a.shape != (n_dims,) or cue_b.shape != (n_dims,):
        raise ValueError(f"线索必须是 ({n_dims},)。")
    alphas = np.exp(-1.0 / np.asarray(taus, dtype=np.float64))
    state = np.zeros((len(taus), n_dims), dtype=np.float64)
    for step in range(test_step):
        inputs = rng.normal(scale=noise, size=n_dims)
        if CUE_A[0] <= step <= CUE_A[1]:
            inputs = inputs + cue_a
        if CUE_B[0] <= step <= CUE_B[1]:
            inputs = inputs + cue_b
        state = alphas[:, None] * state + (1.0 - alphas)[:, None] * inputs[None, :]
    return state.reshape(-1)


def encode_timeline(
    cues: list[tuple[int, int, np.ndarray]],
    *,
    rng: np.random.Generator,
    taus: tuple[int, ...] = TAUS,
    n_dims: int = N_DIMS,
    noise: float = NOISE,
    test_step: int = TEST_STEP,
) -> np.ndarray:
    """通用时间线：``cues`` 是 ``(start, end, vector)`` 列表（含端点）。"""
    alphas = np.exp(-1.0 / np.asarray(taus, dtype=np.float64))
    vectors = [
        (int(start), int(end), np.asarray(vector, dtype=np.float64)) for start, end, vector in cues
    ]
    for _, _, vector in vectors:
        if vector.shape != (n_dims,):
            raise ValueError(f"线索必须是 ({n_dims},)，收到 {vector.shape}。")
    state = np.zeros((len(taus), n_dims), dtype=np.float64)
    for step in range(test_step):
        inputs = rng.normal(scale=noise, size=n_dims)
        for start, end, vector in vectors:
            if start <= step <= end:
                inputs = inputs + vector
        state = alphas[:, None] * state + (1.0 - alphas)[:, None] * inputs[None, :]
    return state.reshape(-1)


class LinearDeltaReadout:
    """局部 delta 读出：只用读出误差与突触前状态更新，无闭式求解。"""

    def __init__(
        self, n_in: int, n_out: int, *, learning_rate: float = 0.01, seed: int = 0
    ) -> None:
        if n_in <= 0 or n_out <= 0:
            raise ValueError("n_in / n_out 必须为正。")
        if learning_rate <= 0:
            raise ValueError("learning_rate 必须为正。")
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.weights = np.zeros((n_out, n_in), dtype=np.float64)
        self.bias = np.zeros(n_out, dtype=np.float64)

    def fit(
        self,
        states: np.ndarray,
        targets: np.ndarray,
        *,
        epochs: int,
        batch: int,
        rng: np.random.Generator,
    ) -> None:
        values = np.asarray(states, dtype=np.float64)
        truth = np.asarray(targets, dtype=np.float64)
        if values.shape[0] != truth.shape[0]:
            raise ValueError("states 与 targets 行数不一致。")
        for _ in range(epochs):
            order = rng.permutation(values.shape[0])
            for start in range(0, values.shape[0], batch):
                index = order[start : start + batch]
                x = values[index]
                error = truth[index] - (x @ self.weights.T + self.bias)
                scale = self.learning_rate / x.shape[0]
                self.weights += scale * (error.T @ x)
                self.bias += scale * error.sum(axis=0)
        if not (np.all(np.isfinite(self.weights)) and np.all(np.isfinite(self.bias))):
            raise RuntimeError("delta 读出 diverged（inf/NaN）。")

    def predict(self, states: np.ndarray) -> np.ndarray:
        return np.asarray(states, dtype=np.float64) @ self.weights.T + self.bias


def fit_ridge(states: np.ndarray, targets: np.ndarray, *, l2: float = 1e-3) -> np.ndarray:
    """闭式 ridge（带截距）；测量头，不是局部学习规则。"""
    design = np.asarray(states, dtype=np.float64)
    values = np.asarray(targets, dtype=np.float64)
    augmented = np.hstack([np.ones((design.shape[0], 1)), design])
    penalty = l2 * np.eye(augmented.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(augmented.T @ augmented + penalty, augmented.T @ values)


def predict(states: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    design = np.asarray(states, dtype=np.float64)
    augmented = np.hstack([np.ones((design.shape[0], 1)), design])
    return augmented @ coefficients


def average_r2(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    residual = np.sum((predicted - target) ** 2, axis=0)
    total = np.sum((target - target.mean(axis=0)) ** 2, axis=0)
    return float(np.mean(1.0 - residual / np.maximum(total, 1e-12)))

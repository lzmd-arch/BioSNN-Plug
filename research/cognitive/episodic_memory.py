"""情景记忆 v0：稀疏二值键 + Hebbian 外积的局部联想记忆。

预注册见 ``sweep_results/episodic_memory_v0_prereg.md``。**回答"曾经"**：存事件
``(状态, 动作, 终止, 时间戳)``，用带噪状态线索回忆。

本模块是 v0：键是类脉冲的稀疏 ±1 码，存储是纯外积；没有梯度、没有脉冲动力学、
没有巩固。字面最近邻只作基线，**不是**联想记忆的成绩。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "EpisodicMemoryV1",
    "HebbianEpisodicMemory",
    "PseudoInverseMemory",
    "RawNearestNeighbor",
    "SparseEncoder",
    "StateIndexedMemory",
    "build_action_term",
    "build_values",
]

#: CartPole 四维状态的已知界；只用于归一化，不看测试数据。
CARTPOLE_SCALE = np.array([2.4, 3.0, 0.21, 3.0], dtype=np.float64)


class SparseEncoder:
    """固定随机投影 + top-k 符号，得到 ``(n, dim)`` 的稀疏 ±1 键。"""

    def __init__(
        self,
        n_in: int = 4,
        *,
        dim: int = 512,
        k: int = 32,
        seed: int = 0,
        scale: np.ndarray | None = None,
        mode: str = "sparse",
    ) -> None:
        if mode not in ("sparse", "dense_sign", "dense"):
            raise ValueError(f"mode 只能是 sparse/dense_sign/dense，收到 {mode!r}。")
        if mode != "sparse":
            k = dim
        if not 1 <= k <= dim:
            raise ValueError(f"k 必须在 [1, {dim}]，收到 {k}。")
        self.n_in = int(n_in)
        self.dim = int(dim)
        self.k = int(k)
        self.mode = mode
        self.scale = np.asarray(scale if scale is not None else CARTPOLE_SCALE[:n_in])
        if self.scale.shape != (self.n_in,):
            raise ValueError(f"scale 必须是 ({self.n_in},)，收到 {self.scale.shape}。")
        rng = np.random.default_rng(seed)
        self.projection = rng.normal(size=(self.dim, self.n_in)) / np.sqrt(self.n_in)

    def encode(self, states: np.ndarray) -> np.ndarray:
        values = np.asarray(states, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.n_in:
            raise ValueError(f"状态必须是 (n, {self.n_in})，收到 {values.shape}。")
        projected = (values / self.scale) @ self.projection.T
        if self.mode == "dense":
            return projected
        if self.mode == "dense_sign":
            return np.where(projected >= 0.0, 1.0, -1.0)
        keys = np.zeros_like(projected)
        index = np.argpartition(-np.abs(projected), self.k - 1, axis=1)[:, : self.k]
        rows = np.arange(values.shape[0])[:, None]
        selected = projected[rows, index]
        # 只对被选中的 k 个位置取符号；0 归到 +1。**不能**把整张 keys 的 0 改成 1——
        # 那会把稀疏码变稠密，记忆容量与指标全变。
        keys[rows, index] = np.where(selected >= 0.0, 1.0, -1.0)
        return keys


def build_action_term(actions: np.ndarray, terminated: np.ndarray) -> np.ndarray:
    """``(n, 4)``：动作 one-hot(2) + 终止 one-hot(2)；时间戳单独存。"""
    actions = np.asarray(actions, dtype=int)
    terminated = np.asarray(terminated, dtype=int)
    if actions.shape != terminated.shape:
        raise ValueError("actions 与 terminated 形状必须一致。")
    values = np.zeros((actions.shape[0], 4), dtype=np.float64)
    values[np.arange(actions.shape[0]), actions] = 1.0
    values[np.arange(actions.shape[0]), 2 + terminated] = 1.0
    return values


def _sparsify(keys: np.ndarray, k: int) -> np.ndarray:
    if not 1 <= k <= keys.shape[1]:
        raise ValueError(f"k 必须在 [1, {keys.shape[1]}]，收到 {k}。")
    output = np.zeros_like(keys)
    index = np.argpartition(-np.abs(keys), k - 1, axis=1)[:, :k]
    rows = np.arange(keys.shape[0])[:, None]
    selected = keys[rows, index]
    output[rows, index] = np.where(selected >= 0.0, 1.0, -1.0)
    return output


def build_values(actions: np.ndarray, terminated: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """``(n, 5)``：动作 one-hot(2) + 终止 one-hot(2) + 归一化时间戳。"""
    actions = np.asarray(actions, dtype=int)
    terminated = np.asarray(terminated, dtype=int)
    times = np.asarray(timestamps, dtype=np.float64)
    if not (actions.shape == terminated.shape == times.shape):
        raise ValueError("actions / terminated / timestamps 形状必须一致。")
    values = np.zeros((actions.shape[0], 5), dtype=np.float64)
    values[np.arange(actions.shape[0]), actions] = 1.0
    values[np.arange(actions.shape[0]), 2 + terminated] = 1.0
    denominator = max(float(times.max()), 1e-12)
    values[:, 4] = times / denominator
    return values


class HebbianEpisodicMemory:
    """``M = Σ v_i k_iᵀ``；回忆 ``r = q M``。"""

    def __init__(self) -> None:
        self.matrix: np.ndarray | None = None
        self.count = 0

    def store(self, keys: np.ndarray, values: np.ndarray) -> None:
        keys = np.asarray(keys, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        if keys.ndim != 2 or values.ndim != 2 or keys.shape[0] != values.shape[0]:
            raise ValueError("keys (n,d) 与 values (n,v) 必须行数一致。")
        self.matrix = keys.T @ values
        self.count = int(keys.shape[0])

    def recall(self, keys: np.ndarray) -> np.ndarray:
        if self.matrix is None:
            raise RuntimeError("记忆为空；先 store。")
        return np.asarray(keys, dtype=np.float64) @ self.matrix


class RawNearestNeighbor:
    """字面基线：存干净状态，按欧氏距离找最近的事件，返回它的值。"""

    def __init__(self) -> None:
        self.states: np.ndarray | None = None
        self.values: np.ndarray | None = None

    def store(self, states: np.ndarray, values: np.ndarray) -> None:
        self.states = np.asarray(states, dtype=np.float64)
        self.values = np.asarray(values, dtype=np.float64)

    def recall(self, states: np.ndarray) -> np.ndarray:
        if self.states is None or self.values is None:
            raise RuntimeError("记忆为空；先 store。")
        query = np.asarray(states, dtype=np.float64)
        distances = (
            (query**2).sum(axis=1)[:, None]
            + (self.states**2).sum(axis=1)[None, :]
            - 2.0 * query @ self.states.T
        )
        return self.values[np.argmin(distances, axis=1)]


class EpisodicMemoryV1:
    """动作/终止 4 维值 + **独立**时间标量记忆；可选局部吸引子迭代。"""

    def __init__(self, *, sparsity: int = 32) -> None:
        self.sparsity = int(sparsity)
        self.keys: np.ndarray | None = None
        self.value_matrix: np.ndarray | None = None
        self.time_vector: np.ndarray | None = None

    def store(self, keys: np.ndarray, action_term: np.ndarray, times: np.ndarray) -> None:
        keys = np.asarray(keys, dtype=np.float64)
        action_term = np.asarray(action_term, dtype=np.float64)
        times = np.asarray(times, dtype=np.float64)
        if keys.shape[0] != action_term.shape[0] or keys.shape[0] != times.shape[0]:
            raise ValueError("keys / action_term / times 行数必须一致。")
        self.keys = keys
        self.value_matrix = keys.T @ action_term
        self.time_vector = keys.T @ times

    def recall(
        self, keys: np.ndarray, *, attractor_steps: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.keys is None or self.value_matrix is None or self.time_vector is None:
            raise RuntimeError("记忆为空；先 store。")
        query = np.asarray(keys, dtype=np.float64)
        for _ in range(attractor_steps):
            # 局部吸引子：q ← top-k(sign(Kᵀ(K q)))，只有矩阵-向量与 top-k。
            query = _sparsify(np.sign((query @ self.keys.T) @ self.keys), self.sparsity)
        return query @ self.value_matrix, query @ self.time_vector


class StateIndexedMemory:
    """非局部的状态索引读出：k 近邻 / softmax attention；时间戳取最近邻。"""

    def __init__(self, *, neighbors: int = 5, beta: float = 10.0) -> None:
        if neighbors < 1:
            raise ValueError("neighbors 必须 ≥1。")
        if beta <= 0:
            raise ValueError("beta 必须为正。")
        self.neighbors = int(neighbors)
        self.beta = float(beta)
        self.states: np.ndarray | None = None
        self.action_term: np.ndarray | None = None
        self.times: np.ndarray | None = None

    def store(self, states: np.ndarray, action_term: np.ndarray, times: np.ndarray) -> None:
        values = np.asarray(states, dtype=np.float64)
        self.mean = values.mean(axis=0)
        self.scale = np.maximum(values.std(axis=0), 1e-8)
        self.states = (values - self.mean) / self.scale
        self.action_term = np.asarray(action_term, dtype=np.float64)
        self.times = np.asarray(times, dtype=np.float64)

    def recall(self, states: np.ndarray, *, attention: bool) -> tuple[np.ndarray, np.ndarray]:
        if self.states is None or self.action_term is None or self.times is None:
            raise RuntimeError("记忆为空；先 store。")
        query = (np.asarray(states, dtype=np.float64) - self.mean) / self.scale
        distances = (
            (query**2).sum(axis=1)[:, None]
            + (self.states**2).sum(axis=1)[None, :]
            - 2.0 * query @ self.states.T
        )
        k = min(self.neighbors, self.states.shape[0])
        index = np.argpartition(distances, k - 1, axis=1)[:, :k]
        if attention:
            selected = np.take_along_axis(distances, index, axis=1)
            weights = np.exp(-(selected - selected.min(axis=1, keepdims=True)) / self.beta)
            weights /= weights.sum(axis=1, keepdims=True)
        else:
            weights = np.full((query.shape[0], k), 1.0 / k)
        values = np.einsum("nk,nkv->nv", weights, self.action_term[index])
        times = self.times[index[:, 0]]
        return values, times


class PseudoInverseMemory:
    """``V K⁺`` 的闭式读出（非局部，只作上限对照）。"""

    def __init__(self) -> None:
        self.coef: np.ndarray | None = None

    def store(self, keys: np.ndarray, values: np.ndarray) -> None:
        self.coef = np.linalg.lstsq(
            np.asarray(keys, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
            rcond=None,
        )[0]

    def recall(self, keys: np.ndarray) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("记忆为空；先 store。")
        return np.asarray(keys, dtype=np.float64) @ self.coef

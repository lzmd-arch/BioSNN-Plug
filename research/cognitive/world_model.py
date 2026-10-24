"""世界模型的打底零件：预测对象、四个基线与两项评估量。

**这一层是打底，不是阶段四。** 定位与判据见
``sweep_results/world_model_groundwork_prereg.md``：它不训练神经世界模型、不做多步
rollout、不做 PINN，只回答「一步动力学是否可测」。

## 为什么基线全是 numpy 闭式解

``copy_last`` / ``mean_delta`` / ``per_action_mean`` 是「不做预测」的三个强度档；
``ridge_action`` 是动作条件化的线性上界。它不是局部学习规则——这里只把它当
**测量标尺**，与 W1 的岭回归读出同一个姿态：闭式、无梯度、可复现。

## 奖励那一侧是空的

CartPole 的即时奖励恒为 1，预测它没有区分力；本模块的离散目标取 ``terminated``。
这是对 ADR-0015「状态与奖励分开报」的**收窄**，不是满足——换环境时要补。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "CopyLastModel",
    "MeanDeltaModel",
    "PerActionMeanModel",
    "Prediction",
    "RidgeActionModel",
    "WorldModel",
    "action_sensitivity",
    "auroc",
    "per_dim_nmse",
]


@dataclass(frozen=True)
class Prediction:
    """一个模型对一批 ``(状态, 动作)`` 的预测。

    Attributes:
        delta: ``(n, n_dims)`` 的状态增量 ``s' − s``。用增量而不是绝对值，是因为
            CartPole 各维尺度差一个数量级，绝对值误差会被位置项主导。
        termination: ``(n,)`` 的终止分数。**只需要能排序**——``ridge_action`` 输出
            的是回归值，不在 ``[0, 1]`` 里，AUROC 仍可用。
    """

    delta: np.ndarray
    termination: np.ndarray

    def __post_init__(self) -> None:
        delta = np.asarray(self.delta, dtype=np.float64)
        termination = np.asarray(self.termination, dtype=np.float64).reshape(-1)
        if delta.ndim != 2:
            raise ValueError(f"delta 必须是 (n, n_dims)，收到 {delta.shape}。")
        if termination.shape[0] != delta.shape[0]:
            raise ValueError(
                f"termination 与 delta 的条数不一致：{termination.shape[0]} vs {delta.shape[0]}。"
            )
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "termination", termination)


@runtime_checkable
class WorldModel(Protocol):
    """一步动力学模型的最小契约。

    ``fit`` 只吃训练集；``predict`` 只吃测试集允许知道的东西（状态与动作），
    **不许碰真实的下一个状态**——否则测出来的是拟合，不是预测。
    """

    def fit(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        terminated: np.ndarray,
    ) -> None: ...

    def predict(self, states: np.ndarray, actions: np.ndarray) -> Prediction: ...


def _checked(states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(states, dtype=np.float64)
    a = np.asarray(actions, dtype=np.int64).reshape(-1)
    if s.ndim != 2:
        raise ValueError(f"states 必须是 (n, n_dims)，收到 {s.shape}。")
    if a.shape[0] != s.shape[0]:
        raise ValueError(f"actions 与 states 条数不一致：{a.shape[0]} vs {s.shape[0]}。")
    if np.any((a < 0) | (a > 1)):
        raise ValueError("CartPole 的动作只能是 0 或 1。")
    return s, a


def _deltas(states: np.ndarray, next_states: np.ndarray) -> np.ndarray:
    nxt = np.asarray(next_states, dtype=np.float64)
    if nxt.shape != states.shape:
        raise ValueError(f"next_states 形状 {nxt.shape} 与 states {states.shape} 不一致。")
    return nxt - states


class CopyLastModel:
    """不预测：``Δs = 0``，终止分数取训练集基率。"""

    def __init__(self) -> None:
        self.termination_rate_ = 0.0

    def fit(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        terminated: np.ndarray,
    ) -> None:
        _checked(states, actions)
        self.termination_rate_ = float(np.mean(np.asarray(terminated, dtype=np.float64)))

    def predict(self, states: np.ndarray, actions: np.ndarray) -> Prediction:
        s, _ = _checked(states, actions)
        return Prediction(
            delta=np.zeros_like(s),
            termination=np.full(s.shape[0], self.termination_rate_),
        )


class MeanDeltaModel:
    """只看全局平均漂移：忽略状态与动作。"""

    def __init__(self) -> None:
        self.delta_ = np.zeros(1)
        self.termination_rate_ = 0.0

    def fit(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        terminated: np.ndarray,
    ) -> None:
        s, _ = _checked(states, actions)
        self.delta_ = _deltas(s, next_states).mean(axis=0)
        self.termination_rate_ = float(np.mean(np.asarray(terminated, dtype=np.float64)))

    def predict(self, states: np.ndarray, actions: np.ndarray) -> Prediction:
        s, _ = _checked(states, actions)
        return Prediction(
            delta=np.tile(self.delta_, (s.shape[0], 1)),
            termination=np.full(s.shape[0], self.termination_rate_),
        )


class PerActionMeanModel:
    """每个动作一个平均漂移：动作条件化的最弱形式（与状态无关）。"""

    def __init__(self) -> None:
        self.delta_ = np.zeros((2, 1))
        self.termination_rate_ = np.zeros(2)

    def fit(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        terminated: np.ndarray,
    ) -> None:
        s, a = _checked(states, actions)
        delta = _deltas(s, next_states)
        term = np.asarray(terminated, dtype=np.float64)
        self.delta_ = np.zeros((2, s.shape[1]))
        global_delta = delta.mean(axis=0)
        global_rate = float(term.mean())
        for action in (0, 1):
            mask = a == action
            self.delta_[action] = delta[mask].mean(axis=0) if mask.any() else global_delta
            self.termination_rate_[action] = float(term[mask].mean()) if mask.any() else global_rate

    def predict(self, states: np.ndarray, actions: np.ndarray) -> Prediction:
        _, a = _checked(states, actions)
        return Prediction(
            delta=self.delta_[a],
            termination=self.termination_rate_[a],
        )


class RidgeActionModel:
    """动作条件化的线性动力学：``Δs`` 与终止分数对 ``[1, z(s), a, a·z(s)]`` 的岭回归。

    ``z`` 用**训练集**的均值与标准差标准化；``a`` 是 one-hot。显式含 ``a·z(s)``
    交互项，才能测出「动作改变动力学」而不只是「两个动作的平均不同」。

    ``quadratic=True`` 时再加 ``z²`` 与 ``(a·z)²``：终止是双侧阈值，线性分数会
    把两侧互相抵消（原预注册的 `ridge_action` 在终止 AUROC 上只剩 0.53）。
    这一档是 `world_model_groundwork_prereg_amend1.md` 加的，不改原判据。


    **λ 同时惩罚截距**（实现更简单，且标准化后截距接近 0）。这不是局部学习规则，
    只是测量标尺——见模块文档。
    """

    def __init__(self, ridge: float = 1.0, *, quadratic: bool = False) -> None:
        if ridge <= 0:
            raise ValueError(f"ridge 必须为正，收到 {ridge!r}。")
        self.ridge = float(ridge)
        self.quadratic = bool(quadratic)
        self.mean_ = np.zeros(1)
        self.scale_ = np.ones(1)
        self.coef_ = np.zeros((1, 1))

    def _features(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        z = (states - self.mean_) / self.scale_
        onehot = np.zeros((states.shape[0], 2), dtype=np.float64)
        onehot[np.arange(states.shape[0]), actions] = 1.0
        # 每个动作一组 z：交互项是 (n, 8)，不是 onehot (n,2) 与 z (n,4) 的逐元素乘。
        interaction = np.hstack([onehot[:, [0]] * z, onehot[:, [1]] * z])
        parts = [np.ones((states.shape[0], 1)), z, onehot, interaction]
        if self.quadratic:
            # 终止是**双侧阈值**（|θ'| 超过界限），单个线性分数表达不了：两边的贡献
            # 会互相抵消。二次项让一个分数能同时抬高低侧与高侧——这是 amendment 1
            # 加的那一项，不是原预注册的线性头。
            parts.append(z**2)
            parts.append(interaction**2)
        return np.hstack(parts)

    def fit(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        terminated: np.ndarray,
    ) -> None:
        s, a = _checked(states, actions)
        delta = _deltas(s, next_states)
        self.mean_ = s.mean(axis=0)
        # 零方差维（理论上不会有，防御一下）不能除零；标准差下限是"不缩放"。
        self.scale_ = np.maximum(s.std(axis=0), 1e-8)
        design = self._features(s, a)
        targets = np.hstack([delta, np.asarray(terminated, dtype=np.float64).reshape(-1, 1)])
        gram = design.T @ design + self.ridge * np.eye(design.shape[1])
        self.coef_ = np.linalg.solve(gram, design.T @ targets)

    def predict(self, states: np.ndarray, actions: np.ndarray) -> Prediction:
        s, a = _checked(states, actions)
        out = self._features(s, a) @ self.coef_
        return Prediction(delta=out[:, :-1], termination=out[:, -1])


def per_dim_nmse(
    predicted_delta: np.ndarray, true_delta: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """逐维归一化 MSE：``MSE_d / scale_d``。

    ``scale`` 由调用方用**训练集**的 ``Var(s')`` 给出——测试集的方差不能用来当分母，
    那等于让分母看见答案。
    """
    pred = np.asarray(predicted_delta, dtype=np.float64)
    true = np.asarray(true_delta, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"预测与真值形状不一致：{pred.shape} vs {true.shape}。")
    mse = ((pred - true) ** 2).mean(axis=0)
    return mse / np.maximum(np.asarray(scale, dtype=np.float64), 1e-12)


def action_sensitivity(model: WorldModel, states: np.ndarray, scale: np.ndarray) -> float:
    """换一个动作，预测差多少（按 ``scale`` 逐维归一化后的均值）。

    动作无关的模型恒为 0。这是 ADR-0015「换动作必须换预测」的可测形式。
    """
    s = np.asarray(states, dtype=np.float64)
    zero = np.zeros(s.shape[0], dtype=np.int64)
    one = np.ones(s.shape[0], dtype=np.int64)
    delta = np.abs(model.predict(s, zero).delta - model.predict(s, one).delta)
    return float(np.mean(delta / np.maximum(np.asarray(scale, dtype=np.float64), 1e-12)))


def _rankdata(values: np.ndarray) -> np.ndarray:
    """平均秩（并列取平均），不依赖 scipy。"""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)
    index = 0
    while index < values.shape[0]:
        end = index
        while end + 1 < values.shape[0] and sorted_values[end + 1] == sorted_values[index]:
            end += 1
        ranks[order[index : end + 1]] = (index + end) / 2.0 + 1.0
        index = end + 1
    return ranks


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """终止事件的 AUROC（Mann–Whitney 形式，并列算 0.5）。

    单类（测试集里没有终止或全是终止）时返回 ``nan``——那是"这个测试集分辨不了"，
    不是 1.0 也不是 0.5。
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=bool).reshape(-1)
    if s.shape[0] != y.shape[0]:
        raise ValueError(f"scores 与 labels 条数不一致：{s.shape[0]} vs {y.shape[0]}。")
    n_pos = int(y.sum())
    n_neg = int(y.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

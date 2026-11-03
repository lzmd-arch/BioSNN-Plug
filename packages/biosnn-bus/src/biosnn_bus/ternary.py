"""有符号多值膜（TI-LIF）：把"发不发"扩展成"发多少、往哪个方向发"。

## 为什么需要它

:class:`~biosnn_bus.membrane.AdaptiveMembrane` 返回布尔脉冲：一个神经元一步只能
等于"发了"或"没发"。而**风格这类连续量是有方向的**（更冷淡 / 更热情），二值脉冲
丢掉了符号；把方向拆成两组神经元是常见做法，但它把"同一个量的正负"变成了"两个
互不相干的群体"，读出时还得再拼回去。

TI-LIF（Ternary-Integer LIF，见 SpikingMamba，arXiv:2510.04595 v2）把这一步的脉冲
改成有符号整数 ``{-D, …, -1, 0, +1, …, +D}``：符号保留方向，幅值保留强度。它仍然
是**整数、稀疏**的，推理侧与二值脉冲一样只做累加。

## 更新式（先写死，再实现）

::

    a_t   = rho * a_{t-1} + |z_{t-1}|                    # 适应变量看幅值
    v_t   = alpha * v_{t-1} + I_t - z_{t-1} * threshold  # 复位用基阈值
    eff   = threshold + beta * a_t
    s_t   = 0                                            # 默认不发
    if v_t >= eff:      s_t = +min(floor(v_t / eff), D)
    elif signed and v_t <= -eff:  s_t = -min(floor(-v_t / eff), D)
    if r_{t-1} > 0:     s_t = 0                          # 不应期
    r_t   = clip(r_{t-1} + n_refractory * |s_t| - 1, 0, n_refractory)

``signed=False, D=1`` 时它**逐位退化为** :class:`AdaptiveMembrane`：判定、复位、
适应、不应期四项完全一致（`tests/test_membrane_ternary.py` 用随机输入逐步比对）。
这条退化路径不是巧合——TI-LIF 是 ALIF 的超集，退化测试保证"超集"这件事真的成立，
而不是两套各写各的动力学。

## 它不做什么

* **不含权重与学习规则**：输入仍是电流，权重属于网络层（同 ``AdaptiveMembrane``）；
* **不做级联量化训练**：研究侧的整数训练需要替代梯度，那属于训练框架（torch），
  不属于这个纯 numpy 的骨架库（ADR-0002）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["TernaryMembrane"]


class TernaryMembrane:
    """有符号多值自适应阈值膜（TI-LIF）。

    Attributes:
        n_neurons: 群体规模。
        tau: 膜时间常数（ms），``alpha = exp(-1/tau)``。
        tau_adaptation: 适应时间常数（ms），``rho = exp(-1/tau_adaptation)``。
        threshold: 基阈值；同时是一次脉冲的复位步长。
        beta: 适应强度；``0`` 时退化为纯 LIF。
        n_refractory: 每次发放后的不应期步数（按步数计，与幅值无关）。
        depth: ``D``，单步脉冲的最大幅值（``>= 1``）。
        signed: 是否允许负脉冲。``False`` 且 ``depth == 1`` 时与 ``AdaptiveMembrane``
            逐位一致。
    """

    def __init__(
        self,
        n_neurons: int = 1,
        *,
        tau: float = 20.0,
        tau_adaptation: float = 500.0,
        threshold: float = 0.62,
        beta: float = 0.07,
        n_refractory: int = 2,
        depth: int = 1,
        signed: bool = True,
    ) -> None:
        if n_neurons < 1:
            raise ValueError(f"n_neurons 必须 >= 1，收到 {n_neurons}。")
        if tau <= 0 or tau_adaptation <= 0:
            raise ValueError(f"时间常数必须为正，收到 tau={tau}、tau_adaptation={tau_adaptation}。")
        if threshold <= 0:
            raise ValueError(f"threshold 必须为正，收到 {threshold}。")
        if beta < 0:
            raise ValueError(f"beta 必须 >= 0，收到 {beta}。")
        if n_refractory < 0:
            raise ValueError(f"n_refractory 必须 >= 0，收到 {n_refractory}。")
        if depth < 1:
            raise ValueError(f"depth 必须 >= 1，收到 {depth}。")

        self.n_neurons = int(n_neurons)
        self.tau = float(tau)
        self.tau_adaptation = float(tau_adaptation)
        self.threshold = float(threshold)
        self.beta = float(beta)
        self.n_refractory = int(n_refractory)
        self.depth = int(depth)
        self.signed = bool(signed)
        self.alpha = float(np.exp(-1.0 / self.tau))
        self.rho = float(np.exp(-1.0 / self.tau_adaptation))
        self.reset()

    def reset(self) -> None:
        """状态清零：膜电位、适应变量、上一步脉冲（整数）、不应期计数。"""
        shape = (self.n_neurons,)
        self.v = np.zeros(shape, dtype=np.float64)
        self.a = np.zeros(shape, dtype=np.float64)
        self.z = np.zeros(shape, dtype=np.int64)
        self.r = np.zeros(shape, dtype=np.float64)

    def forward(self, x: Any) -> np.ndarray:
        """推进一个时间步。

        Args:
            x: 输入电流，形状 ``(n_neurons,)``。可以是负数（抑制性电流）。

        Returns:
            本步脉冲，``int64`` 数组，取值在 ``[-depth, depth]``。

        Raises:
            ValueError: 形状与 ``n_neurons`` 不一致。
        """
        current = np.asarray(x, dtype=np.float64).reshape(-1)
        if current.shape != (self.n_neurons,):
            raise ValueError(f"输入形状必须是 ({self.n_neurons},)，收到 {current.shape}。")

        new_a = self.rho * self.a + np.abs(self.z)
        new_v = self.alpha * self.v + current - self.z * self.threshold
        eff = self.threshold + self.beta * new_a

        level = np.floor(np.abs(new_v) / eff)
        magnitude = np.clip(level, 1.0, float(self.depth))
        positive = new_v >= eff
        negative = new_v <= -eff if self.signed else np.zeros_like(positive)
        raw = np.where(positive, magnitude, 0.0) - np.where(negative, magnitude, 0.0)
        blocked = self.r > 0
        new_z = np.where(blocked, 0.0, raw).astype(np.int64)

        self.r = np.clip(
            self.r + self.n_refractory * np.abs(new_z) - 1.0, 0.0, float(self.n_refractory)
        )
        self.v, self.a, self.z = new_v, new_a, new_z
        return new_z

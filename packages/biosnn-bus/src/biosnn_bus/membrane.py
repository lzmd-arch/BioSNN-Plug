"""膜电位模块：一个恒等占位 + 一个可用的自适应阈值膜。

骨架库不包含任何学习规则——那是认知核心（第一阶段起）的职责，见计划书 §三。

两个实现的分工：

* :class:`PassThroughMembrane` —— 恒等，让示例插件有东西可返回，是插件作者的**最小模板**；
* :class:`AdaptiveMembrane` —— 纯 numpy 的 ALIF 神经元（自适应阈值 + 适应变量），
  **有状态**，给需要真正脉冲动力学的插件作者一个不依赖 torch 的起点。
  研究层 ``research/eprop/neurons.py`` 的 ``ALIFCell`` 与它逐项对应，差别只在权重的归属。
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["AdaptiveMembrane", "PassThroughMembrane"]


class PassThroughMembrane:
    """恒等膜模块：``forward`` 原样返回输入。

    返回类型之所以满足 :class:`~biosnn_bus.types.Membrane` 协议，是因为协议只要求
    存在 ``forward`` 方法（结构化子类型）。
    """

    def forward(self, x: Any) -> Any:
        """原样返回 ``x``。"""
        return x

    def __call__(self, x: Any) -> Any:
        return self.forward(x)

    def __repr__(self) -> str:
        return "PassThroughMembrane()"


class AdaptiveMembrane:
    """自适应阈值膜（ALIF）：有状态，逐时间步 ``forward``。

    **与恒等膜的区别**：``PassThroughMembrane`` 没有状态、原样返回输入；这个膜记住膜电位
    ``v``、适应变量 ``a``、上一步脉冲 ``z`` 与不应期计数 ``r``，调用一次推进一个时间步，
    返回**本步的脉冲**（布尔数组）。

    **它做什么、不做什么**：这里只实现神经元动力学。输入是**电流**而不是突触前脉冲——
    权重（``w_in`` / ``w_rec``）属于网络层，不属于膜。研究层的
    ``research/eprop/neurons.py`` 里的 ``ALIFCell`` 把两层合在一起（``inputs @ w_in``），
    本类的更新式与它逐项对应，差别只有“电流由谁算”。这也正是它能被插件作者直接复用的原因：
    骨架库不依赖 torch（ADR-0002），而插件需要一个能跑的参考神经元。

    更新顺序照 ``ALIFCell.step``（``research/eprop/neurons.py:214``）：

    1. 适应变量先用**上一步**的脉冲更新：``a = rho*a + z``；
    2. 膜电位：``v = alpha*v + current - z*threshold``——复位用的是**基**阈值，不是适应后阈值；
    3. 发放：``v >= threshold + beta*a``（硬阈值，没有替代梯度）；
    4. 不应期内强制不发放；``r = clamp(r + n_refractory*z - 1, 0, n_refractory)``。

    Attributes:
        n_neurons: 群体规模。输入向量的形状必须与它一致。
        tau: 膜时间常数（ms），``alpha = exp(-1/tau)``。
        tau_adaptation: 适应时间常数（ms），``rho = exp(-1/tau_adaptation)``。
        threshold: 基阈值。
        beta: 适应强度；``0`` 时退化为纯 LIF。
        n_refractory: 发放后的不应期步数。
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

        self.n_neurons = int(n_neurons)
        self.tau = float(tau)
        self.tau_adaptation = float(tau_adaptation)
        self.threshold = float(threshold)
        self.beta = float(beta)
        self.n_refractory = int(n_refractory)
        self.alpha = float(np.exp(-1.0 / self.tau))
        self.rho = float(np.exp(-1.0 / self.tau_adaptation))
        self.reset()

    def reset(self) -> None:
        """状态清零：膜电位、适应变量、上一步脉冲、不应期计数。"""
        shape = (self.n_neurons,)
        self.v = np.zeros(shape, dtype=np.float64)
        self.a = np.zeros(shape, dtype=np.float64)
        self.z = np.zeros(shape, dtype=np.float64)
        self.r = np.zeros(shape, dtype=np.float64)

    def forward(self, x: Any) -> np.ndarray:
        """推进一个时间步。

        Args:
            x: 输入电流，形状 ``(n_neurons,)``。

        Returns:
            本步脉冲，``bool`` 数组，形状 ``(n_neurons,)``。

        Raises:
            ValueError: 形状与 ``n_neurons`` 不一致。
        """
        current = np.asarray(x, dtype=np.float64).reshape(-1)
        if current.shape != (self.n_neurons,):
            raise ValueError(f"输入形状必须是 ({self.n_neurons},)，收到 {current.shape}。")

        new_a = self.rho * self.a + self.z
        new_v = self.alpha * self.v + current - self.z * self.threshold
        above = new_v >= (self.threshold + self.beta * new_a)
        refractory = self.r > 0
        new_z = np.where(refractory, False, above).astype(np.float64)
        self.r = np.clip(self.r + self.n_refractory * new_z - 1.0, 0.0, float(self.n_refractory))

        self.v, self.a, self.z = new_v, new_a, new_z
        return new_z.astype(bool)

"""TAAF：时间注意力引导的自适应融合（计划书 §2.2）。

## 这是什么

计划书 §2.2 说 TAAF 提供的是「**时间注意力引导的自适应融合**（每个时间步动态分配
重要性分数，实现时间异构脉冲特征的分层整合）」。本模块实现这个机制的一版
**可检查的**形式。

⚠️ **双通道融合是本项目自己的设计，不归因给 TAAF**（见计划书 §2.2 与 §八）。
TAAF 管的是"同一通道内多路怎么合成一路"，通道划分与它无关。

## 红线：权重必须由局部信号驱动，不允许反向传播训练

**这一条是硬约束，不是风格偏好。** 本项目的主张之一是"纯局部学习"；如果融合层的
注意力权重靠反向传播训练出来，那条声明**在融合层就被戳穿**，后面再说什么局部性
都站不住。骨架库进产品之后更要守住。

所以本实现的权重**只**来自各路自己的发放率——一个在每个神经元局部即可读出的量：

    rate_i = mean_t(data[t, i])

**没有任何 `backward`、没有可训练参数、没有 torch。** 用 Hebbian 或 rate-based
gating 也可以满足这条红线（见 ADR-0012 的备选方案），本实现选 rate-based gating，
因为它最简单且**可检查**：权重能从数据当场算出来、能写测试断言。

## 先投影、后融合（与 `ConcatenateFusion` 的不同工作点）

各路 ``spike_dim`` 不同时**无法直接求和**（`image_diff` 是 128、`mnist_pulse` 是
784）。所以本策略走**先投影、后融合**：每路先被稀疏随机投影到同一个维度，再按
注意力权重加权求和。总线（:class:`~biosnn_bus.bus.SpikeBus`）检测到策略实现了
:meth:`fuse_projected` 就走这个工作点。

投影矩阵是**固定种子的随机矩阵**，不是训练出来的——这同样服务于"无反向传播"这条
红线（见 ADR-0012 待决问题：投影层的训练规则）。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .errors import BioSNNBusError
from .types import SpikeTrain

__all__ = ["TemporalAttentionFusion"]


class TemporalAttentionFusion:
    """时间注意力引导的自适应融合。

    Args:
        temperature: softmax 的温度，必须为正。越小权重越尖（一路独大），
            越大越平（各路等权）。
        eps: 除零保护。发放率全为 0 时退化成均匀权重。

    Note:
        **注意力权重不是可训练的**，它每一步由各路的发放率现算（rate-based
        gating）。这正是"局部可计算"的含义：发放率在每个神经元上本地可读，
        不需要任何跨层或跨模态的梯度信息。
    """

    def __init__(self, *, temperature: float = 1.0, eps: float = 1e-12) -> None:
        if not temperature > 0:
            raise ValueError(f"temperature 必须为正，收到 {temperature!r}。")
        if not eps > 0:
            raise ValueError(f"eps 必须为正，收到 {eps!r}。")
        self.temperature = float(temperature)
        self.eps = float(eps)

    # ------------------------------------------------------------ 协议（可选能力）

    def fuse_projected(self, trains: Sequence[SpikeTrain]) -> SpikeTrain:
        """把**已投影到同一维度**的多路按注意力权重加权求和。

        ⚠️ 调用方（``SpikeBus``）保证：各路已对齐到同一时间网格、同通道、且
        ``n_neurons`` 相同（都等于 ``bus_dim``）。这里不再做投影。

        Returns:
            形状 ``(T, bus_dim)`` 的一路序列，``temporal_scale`` 取各路最大值。
        """
        if not trains:
            raise BioSNNBusError("TemporalAttentionFusion 至少需要一路输入。")
        channels = {t.channel for t in trains}
        if len(channels) > 1:
            raise BioSNNBusError(
                f"融合前必须先按通道分组，收到混合通道 {sorted(c.value for c in channels)}。"
            )

        shapes = {t.data.shape for t in trains}
        if len(shapes) != 1:
            raise BioSNNBusError(
                f"fuse_projected 期望各路已投影到同一维度，收到形状 {sorted(shapes)}。"
                "先投影、后融合工作点下这一步由 SpikeBus 保证；自己调用请先对齐。"
            )

        data = np.stack([np.asarray(t.data, dtype=np.float64) for t in trains], axis=0)
        weights = self.attention_weights(trains)

        # (n_trains, T, N) * (n_trains, 1, 1) -> (T, N)
        fused = np.sum(data * weights[:, None, None], axis=0)
        return SpikeTrain(
            data=fused,
            dt=trains[0].dt,
            temporal_scale=max(t.temporal_scale for t in trains),
            channel=trains[0].channel,
        )

    # ------------------------------------------------------------ 局部权重（红线所在）

    def attention_weights(self, trains: Sequence[SpikeTrain]) -> np.ndarray:
        """由各路的**发放率**现算的注意力权重（softmax），形状 ``(n_trains,)``。

        这是整条红线的落点：**权重只依赖各路自己的发放率**，一个每个神经元局部
        可读出的量。没有任何反向传播、没有可训练参数。

        Args:
            trains: 同一通道、已对齐的多路序列。

        Returns:
            形状 ``(n_trains,)`` 的非负权重，和为 1。
        """
        if not trains:
            raise BioSNNBusError("attention_weights 至少需要一路输入。")

        # 每路的平均发放率：mean over 时间步与神经元。这是该路"有多活跃"的局部量。
        rates = np.array(
            [float(np.mean(np.abs(np.asarray(t.data, dtype=np.float64)))) for t in trains]
        )

        # 全 0（或极小）时退化成均匀权重——不除零、不偏袒任何一路。
        if float(rates.max()) <= self.eps:
            return np.full(len(trains), 1.0 / len(trains), dtype=np.float64)

        logits = rates / self.temperature
        logits = logits - logits.max()  # 数值稳定
        exp = np.exp(logits)
        return exp / exp.sum()

    def __repr__(self) -> str:
        return f"TemporalAttentionFusion(temperature={self.temperature:g})"

"""图像模态示例插件：时间差分编码。

对应计划书 §2.3 模态表中的图像行——"差分编码 / DVS 事件流"，融合通道为
``semantic``。同时充当插件作者的**参考实现**：一个完整的
:class:`~biosnn_bus.plugin.ModalityPlugin` 应该长什么样，以及它的测试该怎么写
（见 ``tests/test_plugins_image_diff.py``）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..membrane import PassThroughMembrane
from ..plugin import ModalityPlugin
from ..registry import register_plugin
from ..types import SpikeTrain

__all__ = ["DiffImagePlugin"]


@register_plugin("image_diff")
class DiffImagePlugin(ModalityPlugin):
    """把灰度帧序列编码为 ON/OFF 差分脉冲。

    输入形如 ``(T, H, W)`` 的帧序列，输出形如 ``(T, 2*H*W)`` 的脉冲序列：前
    ``H*W`` 个通道是 ON（亮度上升**达到**阈值），后 ``H*W`` 个是 OFF（下降达到
    阈值）。这正是事件相机（DVS）的表示方式，也是计划书 §2.3 为图像模态选定的
    编码方案。

    第一个时间步的差分取 ``frames[0] - 0``，即假定序列从全黑开始。这让
    :meth:`decode` 成为一个良定义的逆运算——在每步变化量恰为 ``±threshold`` 的
    输入上，``decode(encode(x))`` 能精确还原 ``x``。

    Args:
        height: 帧高。
        width: 帧宽。
        threshold: 差分阈值，落在 ``(0, 1]``。变化量**达到**该值即触发事件。

    Note:
        这是**事件编码**而非**量化编码**：每个 ``(时间步, 像素)`` 最多产生一个
        脉冲，因此无法表达"一步内跳变了好几个阈值"。跳变幅度信息在编码中丢失，
        只有符号被保留。这是事件相机的固有性质，不是实现缺陷。
    """

    def __init__(self, height: int = 8, width: int = 8, threshold: float = 0.1) -> None:
        for label, value in (("height", height), ("width", width)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} 必须是正整数，收到 {value!r}。")
        if not 0 < threshold <= 1:
            raise ValueError(f"threshold 必须落在 (0, 1]，收到 {threshold!r}。")

        self.height = height
        self.width = width
        self.threshold = float(threshold)

    # ------------------------------------------------------------ 接口

    @property
    def modality_name(self) -> str:
        return "image_diff"

    @property
    def spike_dim(self) -> int:
        return self._n_pixels * 2

    @property
    def fusion_channel(self) -> str:
        # 图像走语义通道——计划书 §2.3 模态表。
        return "semantic"

    def encode(self, raw_input: Any) -> SpikeTrain:
        """``(T, H, W)`` 灰度帧序列 → ``(T, 2*H*W)`` ON/OFF 脉冲。"""
        frames = self._as_frames(raw_input)

        blank = np.zeros_like(frames[:1])
        deltas = np.diff(frames, axis=0, prepend=blank)

        on = deltas >= self.threshold
        off = deltas <= -self.threshold
        data = np.concatenate(
            [on.reshape(frames.shape[0], -1), off.reshape(frames.shape[0], -1)], axis=1
        )
        return SpikeTrain(data=data, channel=self.channel, temporal_scale=self.temporal_scale)

    def decode(self, spike_output: SpikeTrain) -> np.ndarray:
        """``(T, 2*H*W)`` 脉冲 → ``(T, H, W)`` 重建帧序列。

        重建采用"最小一致"策略：每个 ON/OFF 脉冲记作一次 ``+threshold`` /
        ``-threshold`` 的跳变，再沿时间轴累加。

        **误差性质**（逐时间步，不是全局）：设真实跳变为 ``Δ``，则

        * ``|Δ| < threshold``：不触发事件，重建跳变 ``0``，误差 ``|Δ|``；
        * ``|Δ| >= threshold``：触发一次事件，重建跳变 ``±threshold``，误差
          ``|Δ| - threshold``。

        因为一个 ``(时间步, 像素)`` 最多产生一个脉冲，**跳变幅度信息在编码中丢失，
        只有符号被保留**。所以重建误差会沿时间轴累积——连续多步同向跳变时，每一步
        都少记 ``|Δ| - threshold``，漂移不断变大。这是事件相机编码的固有性质，
        不是实现缺陷。

        正因如此，事件表示的正常用法是把脉冲序列直接喂给下游脉冲网络；本方法主要
        用于健全性检查与可视化，不是无损逆变换。
        """
        if not isinstance(spike_output, SpikeTrain):
            raise TypeError(f"decode 需要 SpikeTrain，收到 {type(spike_output).__name__}。")
        if spike_output.n_neurons != self.spike_dim:
            raise ValueError(
                f"decode 期望 spike_dim={self.spike_dim} 的输入，收到 {spike_output.n_neurons}。"
            )

        n_pixels = self._n_pixels
        on = spike_output.data[:, :n_pixels].astype(np.float64)
        off = spike_output.data[:, n_pixels:].astype(np.float64)
        deltas = (on - off) * self.threshold
        return np.cumsum(deltas, axis=0).reshape(-1, self.height, self.width)

    def get_membrane(self) -> PassThroughMembrane:
        """骨架库不含学习规则；这里返回恒等模块占位。"""
        return PassThroughMembrane()

    # ------------------------------------------------------------ 内部

    @property
    def _n_pixels(self) -> int:
        return self.height * self.width

    def _as_frames(self, raw_input: Any) -> np.ndarray:
        frames = np.asarray(raw_input, dtype=np.float64)
        expected_tail = (self.height, self.width)
        if frames.ndim != 3 or frames.shape[1:] != expected_tail:
            raise ValueError(
                f"{type(self).__name__} 期望形状 (T, {self.height}, {self.width}) 的输入，"
                f"收到 {frames.shape}。"
            )
        if frames.shape[0] < 1:
            raise ValueError("帧序列至少要有一个时间步。")
        if not np.all(np.isfinite(frames)):
            raise ValueError("帧序列含 NaN 或 Inf。")
        return frames

    def __repr__(self) -> str:
        return (
            f"DiffImagePlugin(height={self.height}, width={self.width}, "
            f"threshold={self.threshold:g})"
        )

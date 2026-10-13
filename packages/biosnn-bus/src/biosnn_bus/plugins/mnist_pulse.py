"""MNIST 脉冲编码插件：像素值作发放概率的伯努利采样。

对应计划书 §2.3 模态表的**图像**行，融合通道 ``semantic``。与
:class:`~biosnn_bus.plugins.image_diff.DiffImagePlugin` 的区别是**编码方案**：
后者是 DVS 式的 ON/OFF 事件（差分编码，每像素每步最多一个脉冲、只有符号），
本插件是**发放率编码**——像素值 ``v ∈ [0, 1]`` 就是该像素在每个时间步发放的概率，
幅度信息被编码进发放率，而不是被丢弃。

⚠️ **这是本项目自己的选择，不是任何论文的规定。** 与 `research/eprop` 的边界第 5 条
（「输入编码是本项目自己的选择」）同一条性质。W1 用它接到脉冲总线上时，这个选择是
**新的**、且与 W1 原先的 `(x/255 − 0.5) / 0.5` 连续输入不同——见
:mod:`research.ib_hebbian.bus_input` 的说明。

采样率与 ``decode`` 的关系：``T`` 个时间步上，第 ``i`` 个像素的发放率是
``rate_i = mean_t(spike[t, i])``，是 ``v_i`` 的无偏估计，故 ``decode(encode(x))``
的期望等于 ``x``，误差量级 ``O(1/sqrt(T))``。**不是**无损逆变换——
``T`` 越大越准，这正是"用时间换精度"的用法。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..membrane import PassThroughMembrane
from ..plugin import ModalityPlugin
from ..registry import register_plugin
from ..types import SpikeTrain

__all__ = ["MnistPulsePlugin"]


@register_plugin("mnist_pulse")
class MnistPulsePlugin(ModalityPlugin):
    """把 MNIST 级灰度图编码为发放率脉冲序列。

    Args:
        height: 图高（MNIST 是 28）。
        width: 图宽（MNIST 是 28）。
        n_steps: 每张图的脉冲时间步数 ``T``。``T`` 越大，``decode`` 越准，
            但下游看到的序列越长。
        seed: 伯努利采样的随机种子。**固定种子是复现的前提**（同
            ``docs/reproducibility.md``）——同一张图同一个 ``seed`` 必须给出
            同一串脉冲，否则"接线"这条链路不可复现。

    Note:
        编码是**概率性的**：同一张图跑两次会得到不同的脉冲（除非 ``seed`` 固定）。
        这与脉冲网络输入侧本就是伯努利采样的既有约定一致
        （见 ``research/eprop/tasks.py``），不是实现缺陷。
    """

    def __init__(
        self,
        height: int = 28,
        width: int = 28,
        n_steps: int = 16,
        seed: int = 0,
    ) -> None:
        for label, value in (("height", height), ("width", width), ("n_steps", n_steps)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} 必须是正整数，收到 {value!r}。")
        self.height = height
        self.width = width
        self.n_steps = n_steps
        self.seed = int(seed)

    # ------------------------------------------------------------ 接口

    @property
    def modality_name(self) -> str:
        return "mnist_pulse"

    @property
    def spike_dim(self) -> int:
        return self._n_pixels

    @property
    def fusion_channel(self) -> str:
        # 图像走语义通道——计划书 §2.3 模态表。
        return "semantic"

    @property
    def temporal_scale(self) -> float:
        # 发放率编码的典型时间尺度由 ``n_steps`` 决定，不是图像的物理时长。
        return float(self.n_steps)

    def encode(self, raw_input: Any) -> SpikeTrain:
        """``(H, W)`` 或 ``(T, H, W)`` 灰度图 → ``(T, N)`` 脉冲。

        * 输入是 ``(H, W)``：按 ``n_steps`` 展开成 ``n_steps`` 个时间步，每步独立
          伯努利采样；
        * 输入是 ``(T, H, W)``：``T`` 就是时间步数（每帧独立采样一次），此时
          ``n_steps`` 不参与。

        取值范围是 ``[0, 1]``：``0`` 永不发放，``1`` 每步必发。传入 ``[0, 255]``
        的灰度图请自己先 ``/ 255``——**不在这里猜缩放**，见下。
        """
        frames = self._as_frames(raw_input)
        if not np.all((frames >= 0) & (frames <= 1)):
            raise ValueError(
                "MnistPulsePlugin 的输入必须落在 [0, 1]；收到的值超出范围。"
                "灰度图请自己先 / 255，不要让插件猜缩放。"
            )

        # **单帧要按 n_steps 展开成 n_steps 个时间步**——否则 T=1，decode 就只剩
        # 一个时间步的均值，n_steps 这个参数形同虚设。多帧输入时 T 就是帧数。
        if frames.shape[0] == 1:
            frames = np.repeat(frames, self.n_steps, axis=0)

        # 每个像素每步独立采样。逐条建 generator（而非全局随机），
        # 这样 batch 的顺序改变不会改变同一张图的采样——同 ``sweep_results`` 的
        # 复现约定。
        n_steps = frames.shape[0]
        rng = np.random.default_rng(self.seed)
        spikes = (rng.random((n_steps, self._n_pixels)) < frames.reshape(n_steps, -1)).astype(
            np.float64
        )
        return SpikeTrain(
            data=spikes,
            channel=self.channel,
            temporal_scale=self.temporal_scale,
        )

    def decode(self, spike_output: SpikeTrain) -> np.ndarray:
        """``(T, N)`` 脉冲 → ``(H, W)`` 连续图，取值 ``[0, 1]``。

        每个像素的发放率就是它在整段序列上的脉冲占比。**不是**逐时间步还原：
        这是"用时间轴换精度"的聚合，与 :class:`DiffImagePlugin` 那种逐时间步
        还原的 ``decode`` 含义不同，两者的误差性质也不同（这里 ``O(1/sqrt(T))``
        的采样噪声，那里是随时间累积的漂移）。

        ⚠️ **这不是无损逆变换。** ``T`` 个时间步上，第 ``i`` 个像素的发放率
        ``rate_i`` 满足 ``E[rate_i] = v_i``，故 ``decode(encode(x))`` 的**期望**
        等于 ``x``；单次结果偏离 ``x`` 的量级是 ``O(1/sqrt(T))``。需要更高的还原
        精度就增大 ``n_steps``。
        """
        if not isinstance(spike_output, SpikeTrain):
            raise TypeError(f"decode 需要 SpikeTrain，收到 {type(spike_output).__name__}。")
        if spike_output.n_neurons != self.spike_dim:
            raise ValueError(
                f"decode 期望 spike_dim={self.spike_dim} 的输入，收到 {spike_output.n_neurons}。"
            )
        rates = np.asarray(spike_output.data, dtype=np.float64).mean(axis=0)
        return rates.reshape(self.height, self.width)

    def get_membrane(self) -> PassThroughMembrane:
        """骨架库不含学习规则；这里返回恒等模块占位。"""
        return PassThroughMembrane()

    # ------------------------------------------------------------ 内部

    @property
    def _n_pixels(self) -> int:
        return self.height * self.width

    def _as_frames(self, raw_input: Any) -> np.ndarray:
        frames = np.asarray(raw_input, dtype=np.float64)
        if frames.ndim == 2:
            if frames.shape != (self.height, self.width):
                raise ValueError(
                    f"{type(self).__name__} 期望形状 ({self.height}, {self.width}) 的单帧，"
                    f"收到 {frames.shape}。"
                )
            frames = frames[None, :]
        elif frames.ndim == 3:
            if frames.shape[1:] != (self.height, self.width):
                raise ValueError(
                    f"{type(self).__name__} 期望形状 (T, {self.height}, {self.width}) 的帧序列，"
                    f"收到 {frames.shape}。"
                )
        else:
            raise ValueError(
                f"{type(self).__name__} 期望 2 维或 3 维输入，收到形状 {frames.shape}。"
            )
        if frames.shape[0] < 1:
            raise ValueError("至少要有一个时间步。")
        if not np.all(np.isfinite(frames)):
            raise ValueError("输入含 NaN 或 Inf。")
        return frames

    def __repr__(self) -> str:
        return (
            f"MnistPulsePlugin(height={self.height}, width={self.width}, "
            f"n_steps={self.n_steps}, seed={self.seed})"
        )

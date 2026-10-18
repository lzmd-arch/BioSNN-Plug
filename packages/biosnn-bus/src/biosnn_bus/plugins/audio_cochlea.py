"""音频插件：耳蜗式的对数频带分解 + 发放率编码。

对应计划书 §2.3 模态表的**音频**行，融合通道 ``temporal``（音频是序列主导的模态，
与文本同通道）。

## 编码做了什么

```text
波形 (n_samples,)
  → 分帧（Hann 窗，n_steps 帧，帧间 50% 重叠）
  → 每帧 FFT → 按**对数间隔**的 n_bands 个频带聚合幅度
  → 每带按该段能量的全局最大值归一化到 [0, 1]
  → 以归一化幅度为发放概率做伯努利采样 → (n_steps, n_bands) 的 0/1 脉冲
```

**为什么是对数频带**：耳蜗的频率分辨率随频率下降，临界带（critical band）在低频更窄、
高频更宽。对数间隔是这个性质的**最粗近似**——它只保留"频带按对数排布"这一条，
不实现基底膜动力学。

⚠️ **这是本项目自己的选择，不是任何论文的规定。** 与
:class:`~biosnn_bus.plugins.mnist_pulse.MnistPulsePlugin` 的"像素值即发放概率"
同一条性质：编码方案属于本项目，改动它会改变下游看到的一切。

## 它不做什么

* **不做耳蜗的主动放大、不做内毛细胞转导**——那些需要基底膜的时变动力学。
* **不做语音识别**：本插件只负责"把波形变成频带脉冲"，认字是下游的事。
* **`decode` 不还原波形**：与文本插件同理，脉冲只保留每带的发放率，
  相位与细结构都已丢失，逆变换是病态的。

## 输入约定

* 一维浮点波形，取值任意（插件内部只用到相对能量）；多通道输入取**第一列**并在
  文档里写明，不静默平均它们。
* 采样率是构造参数（默认 16 kHz）；**它必须与波形的实际采样率一致**，插件无法自行判断。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..membrane import PassThroughMembrane
from ..plugin import ModalityPlugin
from ..registry import register_plugin
from ..types import SpikeTrain

__all__ = ["AudioCochleaPlugin", "log_spaced_bands"]

#: 计划书 §2.3 的音频行没规定带数；32 带在 16 kHz 上给出约 1/3 倍频程的分辨率。
DEFAULT_BANDS = 32
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_STEPS = 32
DEFAULT_FRAME = 512


def log_spaced_bands(n_bands: int, sample_rate: float, n_fft: int) -> np.ndarray:
    """返回 ``(n_bands + 1,)`` 的频带边界（Hz），**对数间隔**。

    边界会被夹到 ``[0, sample_rate/2]`` 内、并保证每个带至少占一个 FFT bin——
    否则低带会退化成空带，能量全被相邻带吸走。
    """
    if n_bands < 1:
        raise ValueError(f"n_bands 必须 >= 1，收到 {n_bands}。")
    if sample_rate <= 0 or n_fft < 2:
        raise ValueError(f"sample_rate 与 n_fft 必须为正，收到 {sample_rate}、{n_fft}。")

    nyquist = sample_rate / 2.0
    # 下界取两个 bin 的频率：对数间隔在 0 Hz 处没有定义
    low = max(sample_rate / n_fft * 2.0, 1e-9)
    edges = np.geomspace(low, nyquist, n_bands + 1)
    edges[0] = 0.0
    edges[-1] = nyquist

    # 强制每个带至少覆盖一个 bin 的中心频率
    bin_width = sample_rate / n_fft
    for i in range(1, len(edges)):
        if edges[i] - edges[i - 1] < bin_width:
            edges[i] = edges[i - 1] + bin_width
    return np.clip(edges, 0.0, nyquist)


@register_plugin("audio_cochlea")
class AudioCochleaPlugin(ModalityPlugin):
    """把一维波形编码为对数频带的发放率脉冲。

    Args:
        sample_rate: 波形的实际采样率（Hz）。**插件无法自行判断**，必须正确。
        n_bands: 频带数（对数间隔）。
        n_steps: 时间步数（帧数），决定 ``SpikeTrain`` 的长度 ``T``。
        frame_size: 每帧的 FFT 长度；帧间 50% 重叠。
        seed: 伯努利采样的随机种子。**固定种子是复现的前提**。

    Note:
        ``spike_dim == n_bands``，与 ``frame_size`` 无关。
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        *,
        n_bands: int = DEFAULT_BANDS,
        n_steps: int = DEFAULT_STEPS,
        frame_size: int = DEFAULT_FRAME,
        seed: int = 0,
    ) -> None:
        for label, value in (
            ("sample_rate", sample_rate),
            ("n_bands", n_bands),
            ("n_steps", n_steps),
            ("frame_size", frame_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} 必须是正整数，收到 {value!r}。")

        self.sample_rate = int(sample_rate)
        self.n_bands = int(n_bands)
        self.n_steps = int(n_steps)
        self.frame_size = int(frame_size)
        self.seed = int(seed)
        self._edges = log_spaced_bands(self.n_bands, self.sample_rate, self.frame_size)

    # ------------------------------------------------------------ 接口

    @property
    def modality_name(self) -> str:
        return "audio_cochlea"

    @property
    def spike_dim(self) -> int:
        return self.n_bands

    @property
    def fusion_channel(self) -> str:
        # 音频是序列主导的模态——与文本同走 temporal 通道（计划书 §2.3）。
        return "temporal"

    def band_energies(self, waveform: Any) -> np.ndarray:
        """``(n_samples,)`` 波形 → ``(n_steps, n_bands)`` 的归一化频带能量 ∈ [0, 1]。

        单独暴露出来的原因：测试与诊断需要看**编码前的能量**。只看脉冲会被采样噪声
        干扰——某个带为 0 是"真的没有能量"还是"概率太低没采到"，脉冲层面分不清。
        """
        samples = self._as_waveform(waveform)
        frames = self._frames(samples)
        window = np.hanning(self.frame_size)
        spectra = np.abs(np.fft.rfft(frames * window, axis=1))
        bin_hz = self.sample_rate / self.frame_size

        energies = np.empty((self.n_steps, self.n_bands), dtype=np.float64)
        for band in range(self.n_bands):
            lo = int(np.floor(self._edges[band] / bin_hz))
            hi = int(np.ceil(self._edges[band + 1] / bin_hz))
            hi = max(hi, lo + 1)  # 至少一个 bin
            energies[:, band] = spectra[:, lo:hi].mean(axis=1)

        peak = energies.max(axis=0, keepdims=True)
        # 全零信号（静音）时 peak 为 0：整段归一化后仍是 0，不产生任何脉冲。
        return np.divide(energies, peak, out=np.zeros_like(energies), where=peak > 0)

    def encode(self, raw_input: Any) -> SpikeTrain:
        """波形 → ``(n_steps, n_bands)`` 的 0/1 脉冲。"""
        probabilities = self.band_energies(raw_input)
        draws = np.random.default_rng(self.seed).random(probabilities.shape)
        data = (draws < probabilities).astype(np.float64)
        return SpikeTrain(data=data, channel=self.channel, temporal_scale=self.temporal_scale)

    def decode(self, spike_output: SpikeTrain) -> np.ndarray:
        """``(T, n_bands)`` 脉冲 → ``(T, n_bands)`` 的发放率。

        **不是**还原波形：脉冲只保留每带的发放率，相位与细结构都已丢失。
        返回值供健全性检查与下游读出使用。
        """
        if not isinstance(spike_output, SpikeTrain):
            raise TypeError(f"decode 需要 SpikeTrain，收到 {type(spike_output).__name__}。")
        if spike_output.n_neurons != self.spike_dim:
            raise ValueError(
                f"decode 期望 spike_dim={self.spike_dim} 的输入，收到 {spike_output.n_neurons}。"
            )
        return np.asarray(spike_output.data, dtype=np.float64)

    def get_membrane(self) -> PassThroughMembrane:
        return PassThroughMembrane()

    # ------------------------------------------------------------ 内部

    def _as_waveform(self, raw_input: Any) -> np.ndarray:
        array = np.asarray(raw_input, dtype=np.float64)
        if array.ndim == 2:
            # 多通道：取第一列，并**明确说明**——静默平均会让两个相位相反的通道互相抵消
            array = array[:, 0]
        if array.ndim != 1:
            raise ValueError(f"音频输入必须是一维波形或 (n, 通道) 矩阵，收到形状 {array.shape}。")
        if array.size == 0:
            raise ValueError("音频输入为空。")
        return array

    def _frames(self, samples: np.ndarray) -> np.ndarray:
        """切帧（50% 重叠）。短于一个帧的输入**补零**，长于 n_steps 帧的**截断**。"""
        hop = max(1, self.frame_size // 2)
        need = hop * (self.n_steps - 1) + self.frame_size
        if samples.size < need:
            samples = np.concatenate([samples, np.zeros(need - samples.size)])
        frames = np.stack(
            [samples[i * hop : i * hop + self.frame_size] for i in range(self.n_steps)]
        )
        return frames

    def __repr__(self) -> str:
        return (
            f"AudioCochleaPlugin(sample_rate={self.sample_rate}, n_bands={self.n_bands}, "
            f"n_steps={self.n_steps}, frame_size={self.frame_size}, seed={self.seed})"
        )

"""阶段五 M3c：z-score 标准化 top-k 群体编码（纯 stdlib，写入情景记忆用）。

计划书 §七 第五阶段任务 2：把 LLM 嵌入做 **z-score 标准化 + top-k**，再用群体编码写成
脉冲模式，供 SNN 情景记忆写入 [Savage, 2026]。本模块只做「嵌入 → 脉冲模式」这一段，
不连接任何记忆库，也不依赖 torch/numpy（包本身零依赖）。

编码步骤（写死，便于复现）：

1. 对输入向量做**总体** z-score：``z_i = (x_i − mean) / std``（``std = 0`` 时全零）；
2. 取 ``|z_i|`` 最大的 **top-k** 维（同值按下标升序，保证确定性）；
3. 每个被选维在 ``[−span, +span]`` 上铺 ``n_centers`` 个高斯中心，
   激活 ``a_j = exp(−(z_i − c_j)² / (2σ²))``；
4. 激活 ≥ ``threshold`` 记 1，否则 0；按「先被选维、后中心」的顺序展平。

输出长度恒为 ``k × n_centers``；它是**脉冲模式**，不是连续嵌入。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["ContextCode", "ZScoreTopKEncoder"]


@dataclass(frozen=True)
class ContextCode:
    """一次编码的结果。

    Attributes:
        indices: 被选中的维度下标（按 ``|z|`` 降序、同值按位置升序）。
        zscores: 对应维度的 z-score。
        activations: 展平后的群体激活，长度 ``len(indices) × n_centers``。
        spikes: 二值化后的脉冲模式，长度同上，取值 ``{0.0, 1.0}``。
    """

    indices: tuple[int, ...]
    zscores: tuple[float, ...]
    activations: tuple[float, ...]
    spikes: tuple[float, ...]


class ZScoreTopKEncoder:
    """把任意维度的嵌入编码成定长脉冲模式。

    Args:
        top_k: 选多少个维度（必须 ≥1）。
        n_centers: 每个维度的群体中心数（必须 ≥2）。
        sigma: 高斯宽度（>0）。
        span: 中心覆盖范围 ``[−span, +span]``（>0）。
        threshold: 二值化阈值，激活 ≥ 该值记 1（应在 (0, 1]）。

    Raises:
        ValueError: 参数越界，或输入为空、含非有限值。
    """

    def __init__(
        self,
        *,
        top_k: int = 16,
        n_centers: int = 8,
        sigma: float = 1.0,
        span: float = 3.0,
        threshold: float = 0.5,
    ) -> None:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError(f"top_k 必须是 ≥1 的整数，收到 {top_k!r}。")
        if not isinstance(n_centers, int) or isinstance(n_centers, bool) or n_centers < 2:
            raise ValueError(f"n_centers 必须是 ≥2 的整数，收到 {n_centers!r}。")
        if not sigma > 0:
            raise ValueError(f"sigma 必须为正，收到 {sigma!r}。")
        if not span > 0:
            raise ValueError(f"span 必须为正，收到 {span!r}。")
        if not 0 < threshold <= 1:
            raise ValueError(f"threshold 应在 (0, 1]，收到 {threshold!r}。")
        self.top_k = top_k
        self.n_centers = n_centers
        self.sigma = float(sigma)
        self.span = float(span)
        self.threshold = float(threshold)

    @property
    def dimension(self) -> int:
        """脉冲模式的长度 ``top_k × n_centers``。"""
        return self.top_k * self.n_centers

    def centers(self) -> tuple[float, ...]:
        """群体中心，``n_centers`` 个，均匀铺在 ``[−span, +span]``。"""
        if self.n_centers == 1:  # pragma: no cover - 构造器已拒绝
            return (0.0,)
        step = 2.0 * self.span / (self.n_centers - 1)
        return tuple(-self.span + step * index for index in range(self.n_centers))

    def encode(self, embedding: Sequence[float]) -> ContextCode:
        """把嵌入编码成脉冲模式。"""
        values = [float(value) for value in embedding]
        if not values:
            raise ValueError("embedding 不能为空。")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("embedding 含非有限值（nan/inf）。")

        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        if std == 0.0:
            zeros = (0.0,) * self.dimension
            return ContextCode(indices=(), zscores=(), activations=zeros, spikes=zeros)

        zscores = [(value - mean) / std for value in values]
        order = sorted(range(len(zscores)), key=lambda index: (-abs(zscores[index]), index))
        chosen = order[: min(self.top_k, len(order))]
        centers = self.centers()

        activations: list[float] = []
        spikes: list[float] = []
        for index in chosen:
            z = zscores[index]
            for center in centers:
                activation = math.exp(-((z - center) ** 2) / (2.0 * self.sigma**2))
                activations.append(activation)
                spikes.append(1.0 if activation >= self.threshold else 0.0)

        return ContextCode(
            indices=tuple(chosen),
            zscores=tuple(zscores[index] for index in chosen),
            activations=tuple(activations),
            spikes=tuple(spikes),
        )

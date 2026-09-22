"""除法归一化：分组居中、平滑方差、按方差的幂归一。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

对应 Pogodin & Latham (2020) 的 Eq. (13) 与 Eq. (17)：

    u^k_α ≡ δ/c^k_α + (1/c^k_α) Σ_{n'} (∘z^k_{αn'})²          (13)
    ∘z^k_{αn} ≡ z^k_{αn} − (1/c^k_α) Σ_{n'} z^k_{αn'}          (13, 组内居中)
    r^k_{αn} = ∘z^k_{αn} / (u^k_α)^p                            (17)

其中 ``α`` 是分组标号，``c^k_α`` 是第 ``α`` 组的神经元数，``δ`` 是平滑偏移，
``p`` 是归一化指数。

## 为什么这条不能省

论文摘要里写得很直接：

> In addition, to work on hard problems and remain biologically plausible, our update
> rules need divisive normalization, a computation done by the primary visual cortex
> and beyond.

实验部分进一步区分了两种变体（grp 与 grp+div），结论是**高斯核必须有分组 + 除法
归一化才有像样的准确率**——这正是计划书 §3.3 的「关键修正」与 §十一 列为 P0 的那条。

``p = 0.5`` 时本方案等价于标准除法归一化 [Carandini & Heeger] 与 group normalization
[Wu & He]。论文里 ``p = 0.2``（backprop / pHSIC）或 ``0.5``（其余），``δ = 1``。

## 一个容易搞错的顺序

本模块**作用在局部损失算完之后**。论文 D.1 给的每层顺序是：

    linear → batchnorm(若有) → 非线性 → pooling(若有) → 局部损失 → 除法归一化 → dropout

也就是说：**局部损失看的是未归一化的 ``z``，而下一层收到的是归一化的 ``r``**。
官方实现里 ``Layer.forward`` 的注释把这两步分别叫 pre-loss 与 post-loss，顺序一致。
把它俩调换会同时改变损失与下一层的输入，且不会报错——所以这里单独写一段说明。
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["DivisiveNormalization", "grouped_variance"]

#: 论文 D.5 用的平滑偏移。
DEFAULT_SMOOTHING_DELTA = 1.0
#: 论文 D.5 对 pHSIC / backprop 用的归一化指数。
DEFAULT_POWER = 0.2


def grouped_variance(x: torch.Tensor, n_groups: int, smoothing_delta: float) -> torch.Tensor:
    """按组计算平滑方差 ``u``（Eq. 13）。

    ``x`` 形状为 ``(batch, n_groups, group_size)``，沿最后一维计算。

    ``u = (δ + Σ (x − mean)²) / n``——即**组内先居中、再取平方和、加 δ、除以组大小**。
    注意 δ 是被 ``n`` 除过的（不是直接加在均值上），这一点照官方实现
    （``(norm(x_centered)² + δ) / n``）写，与 Eq. (13) 的 ``δ/c + (1/c)Σ(∘z)²`` 一致。

    Args:
        x: ``(..., n_groups, group_size)`` 的张量。
        n_groups: 分组数。
        smoothing_delta: 平滑偏移 δ，防止方差为零时除零。

    Returns:
        ``(..., n_groups)`` 的方差。
    """
    grouped = x.reshape(*x.shape[:-2], n_groups, -1)
    centered = grouped - grouped.mean(dim=-1, keepdim=True)
    squared = centered.pow(2).sum(dim=-1)
    return (squared + smoothing_delta) / grouped.shape[-1]


class DivisiveNormalization(nn.Module):
    """把活动按分组方差归一化（Eq. 17）。

    前向：``r = ∘z / (u)^p``，其中 ``∘z`` 是**组内居中**后的活动，``u`` 是组方差。

    与计划书 §3.3 那段伪代码的关系：

        pool = mean(h ** 2)
        h_norm = h / sqrt(pool + eps)
        out = h_norm * rrms * gain

    计划书那段是**简化示意**：它用整层方差的算术平均、固定指数 0.5、并额外乘一个
    ``rrms * gain`` 做重标定。本实现按论文 Eq. (13)/(17) 走——分组、平滑偏移 δ、
    可调指数 p，且不额外缩放（论文没有 ``rrms * gain`` 这一步，重标定由各层自己的
    学习率承担）。两者只在 p=0.5、单组、δ=0、忽略重标定时才等价。

    Attributes:
        n_groups: 分组数 ``c_k``。
        power: 归一化指数 ``p``。
        smoothing_delta: 平滑偏移 ``δ``。
        eps: 仅用于 ``u`` 的极小值保护，正常路径不参与（δ 已经承担了防除零）。
    """

    def __init__(
        self,
        n_groups: int,
        power: float = DEFAULT_POWER,
        smoothing_delta: float = DEFAULT_SMOOTHING_DELTA,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if n_groups < 1:
            raise ValueError(f"n_groups 必须 >= 1，收到 {n_groups}。")
        if not 0.0 <= power <= 1.0:
            raise ValueError(f"power 应在 [0, 1] 内，收到 {power}。")
        if smoothing_delta < 0:
            raise ValueError(f"smoothing_delta 必须 >= 0，收到 {smoothing_delta}。")

        self.n_groups = int(n_groups)
        self.power = float(power)
        self.smoothing_delta = float(smoothing_delta)
        self.eps = float(eps)

    def extra_repr(self) -> str:
        return (
            f"n_groups={self.n_groups}, power={self.power}, smoothing_delta={self.smoothing_delta}"
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``z``: ``(batch, n_features)`` → ``r``: 同形状。"""
        if z.dim() != 2:
            raise ValueError(
                f"DivisiveNormalization 需要 (batch, n_features) 的二维输入，"
                f"收到形状 {tuple(z.shape)}。"
            )
        n_features = z.shape[-1]
        if n_features % self.n_groups != 0:
            raise ValueError(
                f"{n_features} 个特征无法等分成 {self.n_groups} 组"
                f"（每组 {n_features / self.n_groups:.2f} 个）。请让 n_features 是 n_groups 的整数倍。"
            )

        grouped = z.reshape(z.shape[0], self.n_groups, -1)
        centered = grouped - grouped.mean(dim=-1, keepdim=True)  # ∘z^k_{αn}
        # u^k_α：由**已经居中**的活动算，别用 z 再算一遍——那会得到未居中的平方和。
        squared = centered.pow(2).sum(dim=-1)
        variance = (squared + self.smoothing_delta) / grouped.shape[-1]

        scale = (variance.clamp_min(self.eps) ** self.power).unsqueeze(-1)
        return (centered / scale).reshape_as(z)

"""状态编码：连续状态 → 群体活动 → 输入脉冲串。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

## 两段，别混

1. **群体编码（空间）**：``x_i = exp(−‖s − c_i‖² / (2σ²))``，把 4 维状态铺到 N 个中心上。
   这一步与 ``research/rstdp/`` 的速率版**同构**——两边的输入表征要可比，否则「脉冲化之后
   性能掉多少」这个对照就有一半差异来自编码而不是脉冲。**不做归一化**：归一化会抹掉
   「离所有中心都远」这一信息，而 CartPole 的边缘状态正是那种情形。
2. **脉冲编码（时间）**：把每个特征当成**发放概率**，在 T 步窗口内做伯努利采样。
   这是速率编码的输入侧——与 W2 的 sMNIST 输入编码同一思路（那也是本项目自己的选择，
   任何论文都没规定）。

第 2 步是这条线**与速率版的分野**：有了 T 步的脉冲串，STDP 的 Δt 才有定义。
"""

from __future__ import annotations

import torch

__all__ = ["encode_state", "population_centers", "spike_encode"]

#: CartPole 的四个状态分量（位置、速度、角度、角速度）的量级差异很大，中心在这四个轴
#: 上分别取不同的范围，免得编码被某一个分量主导。
STATE_SCALES = (2.4, 3.0, 0.21, 3.0)


def population_centers(
    n_features: int,
    *,
    n_dims: int = 4,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """铺 ``(n_features, n_dims)`` 个中心，各轴按 :data:`STATE_SCALES` 定范围。"""
    if n_features <= 0:
        raise ValueError(f"n_features 必须为正，收到 {n_features}。")
    scales = torch.tensor(STATE_SCALES[:n_dims], device=device)
    if generator is None:
        unit = torch.rand(n_features, n_dims, device=device)
    else:
        unit = torch.rand(n_features, n_dims, generator=generator, device=device)
    return (unit * 2.0 - 1.0) * scales


def encode_state(state: torch.Tensor, centers: torch.Tensor, sigma: float) -> torch.Tensor:
    """高斯群体编码：``(n_dims,)`` → ``(n_features,)``（也接受带 batch 的输入）。"""
    if sigma <= 0:
        raise ValueError(f"sigma 必须为正，收到 {sigma}。")
    squared = ((state.unsqueeze(-2) - centers) ** 2).sum(dim=-1)
    return torch.exp(-squared / (2.0 * sigma**2))


def spike_encode(
    features: torch.Tensor,
    steps: int,
    *,
    generator: torch.Generator,
    max_rate: float = 1.0,
) -> torch.Tensor:
    """把活动量当发放概率，采出 ``(steps, n_features)`` 的 0/1 脉冲串。

    Args:
        features: ``(n_features,)``，取值应落在 ``[0, 1]``；超出部分会被夹住。
        steps: 窗口长度 T。
        generator: 随机数发生器（**必须显式传**，否则不可复现）。
        max_rate: 单个时间步的最大发放概率。

    Returns:
        ``(steps, n_features)`` 的 0/1 张量。
    """
    if steps <= 0:
        raise ValueError(f"steps 必须为正，收到 {steps}。")
    if not 0.0 < max_rate <= 1.0:
        raise ValueError(f"max_rate 应在 (0, 1] 内，收到 {max_rate}。")
    probability = features.clamp(min=0.0, max=1.0) * max_rate
    draws = torch.rand(steps, *probability.shape, generator=generator, device=probability.device)
    return (draws < probability).to(probability.dtype)

"""时间维编码：连续活动 → 输入脉冲串。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

## 空间 / 时间两段，别混

1. **空间**：高斯群体编码 ``x_i = exp(−‖s − c_i‖² / (2σ²))`` 在
   :mod:`research.components.cartpole_encoding` 里。两条执行层线共用**同一份实现**，
   各传各的中心半宽（本线用 ``CARTPOLE_HALF_WIDTHS`` 分轴取，速率线统一 1.5）与
   ``sigma``（本线默认 1.0，速率线 0.5）。参数不同、实现相同，所以两边的差异是
   **写下来的**，不是两份代码漂出来的。
2. **时间**：本模块把每个特征当成**发放概率**，在 T 步窗口内做伯努利采样。
   这是速率编码的输入侧——与 W2 的 sMNIST 输入编码同一思路（那也是本项目自己的
   选择，任何论文都没规定）。

第 2 步是这条线**与速率版的分野**：有了 T 步的脉冲串，STDP 的 Δt 才有定义。
"""

from __future__ import annotations

import torch

__all__ = ["spike_encode"]


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
    # **先在生成器所在的设备上采样，再搬到特征设备**：CPU 生成器不能直接给 CUDA
    # 张量采样（实测 RuntimeError）。这样同一个 CPU 种子在 CPU/GPU 上得到同一串脉冲。
    sample_device = generator.device
    draws = torch.rand(steps, *probability.shape, generator=generator, device=sample_device)
    return (draws.to(probability.device) < probability).to(probability.dtype)

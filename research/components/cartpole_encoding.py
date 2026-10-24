"""CartPole 状态编码：速率线与脉冲线共用的唯一实现。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

## 为什么它在 `components/` 里

两条 CartPole 线（`research/rstdp/` 与 `research/rstdp_spiking/`）要回答的是
「脉冲化之后性能掉多少」。那个对照只有在**编码实现相同、参数不同**时才有意义：
实现各写一份，迟早会漂移，而漂移不会报错——它只会让两个数字不可比。
本模块就是那一份实现。

## 参数是受控变量，不是默认值

* **中心半宽** `half_widths`：速率线四个轴统一 `1.5`（历史选择）；脉冲线用
  :data:`CARTPOLE_HALF_WIDTHS`（各轴按物理范围）。**两边都写在调用点**，本模块
  不提供默认值——默认值会让新线在没人注意时继承一个它没选过的口径。
* **编码宽度** `sigma`：速率线 CLI 默认 `0.5`，脉冲线 `1.0`。同样由调用方决定。

## 公式不能改

两条线已发表的数字都基于下面两个公式；``tests/test_cartpole_encoding.py`` 用两条线各自
的历史写法当 oracle 逐位比对（``torch.equal``），改一个括号都会红。
"""

from __future__ import annotations

import torch

__all__ = ["CARTPOLE_HALF_WIDTHS", "encode_state", "population_centers"]

#: CartPole 四维状态的物理半宽（位置 / 速度 / 角度 / 角速度）。脉冲线按轴取，
#: 因为四个量的量级差两个数量级；速率线刻意不用它（统一 1.5），那是既成事实。
CARTPOLE_HALF_WIDTHS = (2.4, 3.0, 0.21, 3.0)


def population_centers(
    n_features: int,
    *,
    half_widths: tuple[float, ...],
    generator: torch.Generator,
    device: torch.device | None = None,
) -> torch.Tensor:
    """在 ``[-half_width, +half_width]`` 上均匀铺 ``(n_features, n_dims)`` 个中心。

    各轴范围由 ``half_widths`` 逐个给出——**没有默认值**，因为两条线在这里
    取的值不同，静默继承比报错更糟。

    Args:
        n_features: 中心个数。
        half_widths: 每个状态轴的中心半宽；长度即 ``n_dims``。
        generator: 随机数发生器（必须显式传，否则不可复现）。
        device: 结果搬到哪块设备；``None`` 表示留在 CPU。**先在 CPU 上采样再搬**，
            这样同一个种子在 CPU / GPU 上得到同一组中心。

    Returns:
        ``(n_features, n_dims)`` 的 float32 张量。
    """
    if n_features <= 0:
        raise ValueError(f"n_features 必须为正，收到 {n_features}。")
    widths = torch.as_tensor(tuple(half_widths), dtype=torch.float32)
    if widths.numel() == 0:
        raise ValueError("half_widths 不能为空。")
    unit = torch.rand(n_features, widths.numel(), generator=generator)
    # (u − 0.5) × (2·half_width) 是两条线历史写法的共同展开：
    # 速率线是 (u − 0.5) × 3.0，脉冲线是 (u·2 − 1) × half_width，逐位相同。
    centers = (unit - 0.5) * (2.0 * widths)
    return centers if device is None else centers.to(device)


def encode_state(state: torch.Tensor, centers: torch.Tensor, sigma: float) -> torch.Tensor:
    """高斯群体编码：``x_i = exp(−‖s − c_i‖² / (2σ²))``。

    接受 ``(n_dims,)`` 或带 batch 的输入，输出在末尾多一维 ``n_features``。
    **不做归一化**——归一化会抹掉「离所有中心都远」这一信息，而 CartPole 的
    边缘状态正是那种情形。

    Raises:
        ValueError: ``sigma`` 非正。
    """
    if not sigma > 0:
        raise ValueError(f"sigma 必须为正，收到 {sigma!r}。")
    squared = ((state.unsqueeze(-2) - centers) ** 2).sum(dim=-1)
    return torch.exp(-squared / (2.0 * sigma**2))

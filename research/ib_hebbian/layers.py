"""带局部目标的层：每层自己的优化器，梯度不跨层。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

## 「局部」在这里到底指什么——请读完再引用这一段

计划书 §3.3 说这条规则的第三因子「无需自顶向下传递即可获得」，§9 的「生物合理性」
一行要求「无全局反向传播」。本模块必须让这两句话**有据可依**，所以把边界写清楚：

**成立的部分**：每层有**自己的目标函数**与**自己的优化器**，在该层的前向过程中就地
更新参数。层的输入被显式切断梯度（``requires_grad`` 必须为 ``False``），所以**任何
梯度都不会流回上一层**——不存在从输出层一路回传到输入的全局反向传播。这正是官方
实现的做法（``assert x.requires_grad is False``，步进后 ``activation.detach_()``）。

**不成立、也不应声称的部分**：本实现用 autograd 计算**该层局部目标**对其自身权重的
梯度。autograd 在这里是**仿真手段**，不是学习机制。局部性主张的根据是 Pogodin &
Latham (2020) 由 Eq. (10) 推导出的三因子 Hebbian 形式（Eq. 12 / Eq. 18）：

    ΔW^k_{αnm} ∝ M^k_{α,ij} ( r^k_{αn,i} r^{k−1}_{m,i} − r^k_{αn,j} r^{k−1}_{m,j} )

即更新正比于「前突触活动 × 后突触活动 × 一个每对数据点一个标量的第三因子」。论文
明文说实验里用的就是批量版的 Eq. (10) 梯度（"We used the batch version of the update
rule (Eq. (10)) only to make large-scale simulations computationally feasible"），
所以本实现与论文的实现是同一条路径。

**因此**：写 README 或对外报告时，应当说「每层用局部目标更新，无跨层梯度回传，
更新规则具有三因子 Hebbian 形式（Eq. 18）」；**不应**说「完全没有反向传播」——
那会把 autograd 的角色说成不存在，而它是实实在在跑着的。

## 每层的前向顺序（照论文 D.1）

    linear → 非线性 → 局部损失（不改活动）→ [就地更新] → 切断梯度 → 除法归一化 → dropout

两个顺序细节都会静默出错，所以在这里点明：

1. **局部损失算在归一化之前的 z 上**，而下一层收到的是归一化之后的 r。调换会同时
   改变损失与下一层的输入，且不报错。
2. **更新发生在归一化之前**，所以被 detach 的是 z 而不是 r；归一化本身无参数，
   切断它不影响任何可学的东西。
"""

from __future__ import annotations

import torch
from torch import nn

from research.ib_hebbian.divisive_norm import DivisiveNormalization
from research.ib_hebbian.objective import (
    DEFAULT_GAMMA,
    DEFAULT_SIGMA,
    kernelized_bottleneck_objective,
)

__all__ = ["LocalObjectiveLayer", "grouped_activity"]

#: 论文 D.5 的平滑偏移 δ。官方 argparse 的默认值是 0.0，但 D.5 明说分组+除法归一化
#: 用的是 δ = 1，所以这里取 1.0。
DEFAULT_SMOOTHING_DELTA = 1.0
#: 论文 D.5 的归一化指数 p（官方 `--divnorm-power` 默认值一致）。
DEFAULT_DIVNORM_POWER = 0.2


def grouped_activity(
    z: torch.Tensor,
    n_groups: int,
    *,
    exponent: float,
    smoothing_delta: float = DEFAULT_SMOOTHING_DELTA,
    center: bool = True,
) -> torch.Tensor:
    """分组活动 ``v``（Eq. 14），用作核的输入。

        u^k_α = δ/c + (1/c) Σ_{n'} (∘z^k_{αn'})²
        v^k_α = (u^k_α)^q            （跨组居中：见 ``center``）

    **指数 q 必须取 ``1 − p``**，其中 p 是除法归一化的指数（Eq. 17）。这一条不是
    从论文正文猜的——官方实现的 argparse 帮助串里写着：

        --grouping-power ... 'Group as var ^ q (should be 1 - divnorm_power for
        Hebbian updates)'

    取错（例如让 q = p）不会报错，只会让核输入换一个量纲，然后安静地掉点。

    Args:
        z: ``(m, n_features)``。
        n_groups: 分组数。
        exponent: ``q = 1 − p``。
        smoothing_delta: 平滑偏移 δ。
        center: 是否跨组居中。论文 Eq. (14) 的公式里带这一步；官方实现把它做成独立
            开关 ``--center-local-loss-data``。

    Returns:
        ``(m, n_groups)`` 的分组活动。
    """
    if z.dim() != 2:
        raise ValueError(f"grouped_activity 需要 (m, n_features)，收到 {tuple(z.shape)}。")
    if z.shape[-1] % n_groups != 0:
        raise ValueError(
            f"{z.shape[-1]} 个特征无法等分成 {n_groups} 组。请让宽度是分组数的整数倍。"
        )

    grouped = z.reshape(z.shape[0], n_groups, -1)
    centered = grouped - grouped.mean(dim=-1, keepdim=True)
    variance = (centered.pow(2).sum(dim=-1) + smoothing_delta) / grouped.shape[-1]
    activity = variance.clamp_min(1e-12) ** exponent
    if center:
        activity = activity - activity.mean(dim=-1, keepdim=True)
    return activity


class LocalObjectiveLayer(nn.Module):
    """一层：线性（无偏置）→ LReLU → 局部目标更新 → 切梯度 → 除法归一化 → dropout。

    线性层**没有偏置**——论文 D.1：「None of the hidden layers had the bias term,
    but the output layer did.」

    Attributes:
        linear: 该层的权重。
        optimizer: 该层**专属**的优化器。这是「局部」的机械保证：没有共享的优化器
            就没有跨层的梯度累积。
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        *,
        n_groups: int,
        sigma: float = DEFAULT_SIGMA,
        gamma: float = DEFAULT_GAMMA,
        divnorm_power: float = DEFAULT_DIVNORM_POWER,
        smoothing_delta: float = DEFAULT_SMOOTHING_DELTA,
        learning_rate: float = 0.4,
        momentum: float = 0.95,
        weight_decay: float = 1e-7,
        negative_slope: float = 0.01,
        dropout_p: float = 0.0,
        center_activity: bool = True,
    ) -> None:
        super().__init__()
        if n_out % n_groups != 0:
            raise ValueError(f"层宽 {n_out} 无法等分成 {n_groups} 组。请让宽度是分组数的整数倍。")

        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.n_groups = int(n_groups)
        self.sigma = float(sigma)
        self.gamma = float(gamma)
        self.divnorm_power = float(divnorm_power)
        self.smoothing_delta = float(smoothing_delta)
        self.center_activity = bool(center_activity)

        self.linear = nn.Linear(n_in, n_out, bias=False)
        self.nonlinearity = nn.LeakyReLU(negative_slope)
        self.divisive_norm = DivisiveNormalization(
            n_groups=n_groups, power=divnorm_power, smoothing_delta=smoothing_delta
        )
        self.dropout = nn.Dropout(dropout_p)

        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
        )

    def extra_repr(self) -> str:
        return (
            f"n_in={self.n_in}, n_out={self.n_out}, n_groups={self.n_groups}, "
            f"sigma={self.sigma}, gamma={self.gamma}, divnorm_power={self.divnorm_power}"
        )

    @property
    def grouping_exponent(self) -> float:
        """核输入用的指数 ``q = 1 − p``（见 :func:`grouped_activity`）。"""
        return 1.0 - self.divnorm_power

    def local_objective(self, z: torch.Tensor, label_kernel: torch.Tensor) -> torch.Tensor:
        """该层的局部目标 ``pHSIC(V,V) − γ·pHSIC(Y,V)``（Eq. 9/15）。"""
        v = grouped_activity(
            z,
            self.n_groups,
            exponent=self.grouping_exponent,
            smoothing_delta=self.smoothing_delta,
            center=self.center_activity,
        )
        return kernelized_bottleneck_objective(v, label_kernel, sigma=self.sigma, gamma=self.gamma)

    def forward(
        self,
        x: torch.Tensor,
        label_kernel: torch.Tensor | None = None,
        *,
        update: bool = False,
    ) -> torch.Tensor:
        """前向 + （可选）就地更新。

        Args:
            x: ``(m, n_in)``。**必须不携带梯度**——这是局部性的机械保证，会在下面
                断言。调用方拿到的上一层输出已经被切断过梯度，所以正常路径下自然满足。
            label_kernel: ``(m, m)`` 的二值教学信号。``update=True`` 时必需。
            update: 是否在本层前向内更新权重。

        Returns:
            ``(m, n_out)`` 的归一化活动，**不带梯度**。
        """
        if update and x.requires_grad:
            raise ValueError(
                "本层的输入携带梯度，说明上一层的输出没有被切断——梯度会跨层回传，"
                "局部性假设被破坏。请检查上一层是否在更新后 detach 了输出。"
            )
        if update and label_kernel is None:
            raise ValueError("update=True 需要 label_kernel（局部目标要用到教学信号）。")

        z = self.nonlinearity(self.linear(x))

        if update:
            self.optimizer.zero_grad()
            loss = self.local_objective(z, label_kernel)
            loss.backward()
            self.optimizer.step()

        # **无条件**切断输出与计算图的关系，而不是只在更新之后切。
        # 这样「每层的梯度只到本层」这条不变量与"这一步有没有更新"无关：即使某次
        # 前向没更新（如评测），下一层也拿不到通往本层权重的路径。官方实现只在更新
        # 分支里 detach，代价是 `compute_local_loss and not update_local_loss` 那条
        # 路径上的输出仍带着图——那是一处脆弱的依赖顺序的保证，不如这里直接切掉。
        z = z.detach()

        return self.dropout(self.divisive_norm(z))

    @torch.no_grad()
    def measure_local_objective(self, x: torch.Tensor, label_kernel: torch.Tensor) -> float:
        """只测量不更新——用于记录训练过程中的局部目标值。"""
        z = self.nonlinearity(self.linear(x))
        return float(self.local_objective(z, label_kernel).item())

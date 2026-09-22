"""LIF / ALIF 神经元动力学、伪导数与脉冲函数。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

对应 Bellec et al. (2020) 的动力学方程与伪导数定义。动力学形式取自论文
（而非某个简化版）：

    v_j^t = α·v_j^{t−1} + Σ_i W_ji^rec·z_i^{t−1} + Σ_i W_ji^in·x_i^t − thr·z_j^{t−1}

    ALIF 额外一条适应轨迹： a_j^t = ρ·a_j^{t−1} + z_j^{t−1}
                            阈值 A_j^t = thr + β·a_j^t

其中 ``α = exp(−dt/τ)``、``ρ = exp(−dt/τ_a)``。

## 两个容易搞错、且都不会报错的细节

**1. 输入电流上没有 ``(1−α)`` 因子。** 有些 LIF 实现写成
``v ← α·v + (1−α)·I``；论文用的是 ``v ← α·v + I``（上面那条式子里 ``I`` 直接相加）。
两者只差一个缩放，但会改变有效阈值与不动点，进而改变结果。本模块按论文写。

**2. 复位项用基阈值，脉冲判据用适应后阈值。** ALIF 里：

    I_reset = z · thr · dt          ← 基阈值 thr
    v_scaled = (v − (thr + β·a)) / thr   ← 适应后阈值

这不是笔误：作者官方实现（``EligALIF.__call__``）就是这么写的。两处若都换成同一个阈值，
训练仍会"能跑"，所以必须对着原文/官方实现确认，而不是凭直觉统一。

## 不应期

不应期内 ``psi = 0``（见 :mod:`research.eprop.traces`），也就是该神经元此刻不产生
资格痕迹。参考实现用 ``n_refractory`` 个时间步。本模块的 ``refractory_counter`` 更新
照官方实现：

    r ← clip(r + n_ref·z − 1, 0, n_ref)

``z`` 是**归一化到每步的发放指示**（官方实现里 ``z`` 已经乘过 ``1/dt``，所以 dt=1 时
``z ∈ {0, 1}``）。

**本模块固定 ``dt = 1 ms``。** 这不是疏忽：上面那条动力学里复位项写成 ``−thr·z`` 而
没有 ``·dt``，仅当 ``dt = 1`` 时它才与官方实现的 ``−thr·z·dt`` 一致。要支持别的 ``dt``
必须把 ``dt`` 显式带进 :meth:`ALIFCell.step` 的复位项与发放率换算，那需要单独验证——
在有需求之前，宁可限制得窄一点，也不留一个悄悄算错的 ``dt`` 参数。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

__all__ = [
    "ALIFCell",
    "ALIFState",
    "pseudo_derivative",
    "spike_function",
]

#: 论文与官方实现用的默认超参。
DEFAULT_TAU = 20.0
DEFAULT_TAU_ADAPTATION = 500.0
DEFAULT_THRESHOLD = 0.62
DEFAULT_BETA = 0.07
DEFAULT_DAMPENING_FACTOR = 0.3
DEFAULT_REFRACTORY = 2


def pseudo_derivative(v_scaled: torch.Tensor, dampening_factor: float) -> torch.Tensor:
    """伪导数 ``ψ = dampening_factor · max(0, 1 − |v_scaled|)``。

    这里给的是**未除以阈值**的那个因子（官方实现里 ``psi`` 会再乘 ``1/thr``，那一步在
    :mod:`research.eprop.traces` 里做）。分开是为了让本函数就是论文里 ``ψ`` 的定义。

    Args:
        v_scaled: 相对阈值的膜电位，(v − 阈值)/阈值。阈值处为 0、静息处为 −1。
        dampening_factor: 稳定学习用的阻尼因子。
    """
    return torch.clamp(1.0 - v_scaled.abs(), min=0.0) * dampening_factor


class _SpikeFunction(torch.autograd.Function):
    """前向是 Heaviside，反向用伪导数——仅用于 BPTT 基线，e-prop 路径不经过它。"""

    @staticmethod
    def forward(ctx, v_scaled: torch.Tensor, dampening_factor: float) -> torch.Tensor:
        ctx.save_for_backward(v_scaled)
        ctx.dampening_factor = dampening_factor
        return (v_scaled > 0).to(v_scaled.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (v_scaled,) = ctx.saved_tensors
        local = pseudo_derivative(v_scaled, ctx.dampening_factor)
        return grad_output * local, None


def spike_function(v_scaled: torch.Tensor, dampening_factor: float) -> torch.Tensor:
    """脉冲生成：``z = 1[v_scaled > 0]``，反向走 :func:`pseudo_derivative`。"""
    return _SpikeFunction.apply(v_scaled, dampening_factor)


@dataclass
class ALIFState:
    """ALIF 的隐状态。

    Attributes:
        v: 膜电位，``(batch, n_rec)``。
        a: 适应变量，``(batch, n_rec)``。LIF 时恒为 0（``beta = 0``）。
        z: 上一步的脉冲，``(batch, n_rec)``，取值 ``{0, 1}``（dt = 1 ms）。
        r: 剩余不应期步数，``(batch, n_rec)``，非负。
    """

    v: torch.Tensor
    a: torch.Tensor
    z: torch.Tensor
    r: torch.Tensor

    @classmethod
    def zeros(cls, batch_size: int, n_rec: int, *, device=None, dtype=torch.float32) -> ALIFState:
        shape = (batch_size, n_rec)
        zeros = torch.zeros(shape, device=device, dtype=dtype)
        return cls(v=zeros, a=zeros.clone(), z=zeros.clone(), r=zeros.clone())

    def detach(self) -> ALIFState:
        return ALIFState(self.v.detach(), self.a.detach(), self.z.detach(), self.r.detach())


class ALIFCell(nn.Module):
    """一层 ALIF（``beta = 0`` 时退化为 LIF）循环层。

    含输入权重 ``w_in`` 与循环权重 ``w_rec``。循环权重**对角置零**——官方实现在
    ``w_rec_val`` 里把自连接切断（"Disconnect self-connection"），本模块照做，并且是在
    每次前向取用权重时置零而不是写死进参数，这样优化器更新后依然断开。

    Attributes:
        w_in: ``(n_in, n_rec)``。
        w_rec: ``(n_rec, n_rec)``，取用时对角置零。
    """

    def __init__(
        self,
        n_in: int,
        n_rec: int,
        *,
        tau: float = DEFAULT_TAU,
        tau_adaptation: float = DEFAULT_TAU_ADAPTATION,
        threshold: float = DEFAULT_THRESHOLD,
        beta: float = DEFAULT_BETA,
        dampening_factor: float = DEFAULT_DAMPENING_FACTOR,
        n_refractory: int = DEFAULT_REFRACTORY,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if tau <= 0 or tau_adaptation <= 0:
            raise ValueError(f"时间常数必须为正，收到 tau={tau}、tau_adaptation={tau_adaptation}。")
        if threshold <= 0:
            raise ValueError(f"阈值必须为正，收到 {threshold}。")
        if n_refractory < 0:
            raise ValueError(f"n_refractory 必须 >= 0，收到 {n_refractory}。")

        self.n_in = int(n_in)
        self.n_rec = int(n_rec)
        self.tau = float(tau)
        self.tau_adaptation = float(tau_adaptation)
        self.threshold = float(threshold)
        self.beta = float(beta)
        self.dampening_factor = float(dampening_factor)
        self.n_refractory = int(n_refractory)

        self.alpha = float(torch.exp(torch.tensor(-1.0 / self.tau)))
        self.rho = float(torch.exp(torch.tensor(-1.0 / self.tau_adaptation)))

        # 初始化照官方实现：randn / sqrt(扇入)。
        self.w_in = nn.Parameter(self._init((n_in, n_rec), n_in, generator))
        self.w_rec = nn.Parameter(self._init((n_rec, n_rec), n_rec, generator))

    @staticmethod
    def _init(shape: tuple[int, int], fan_in: int, generator: torch.Generator | None):
        noise = torch.randn(shape, generator=generator)
        return nn.Parameter(noise / (fan_in**0.5))

    @property
    def w_rec_effective(self) -> torch.Tensor:
        """循环权重，对角置零（无自连接）。"""
        return self.w_rec * (
            1.0 - torch.eye(self.n_rec, device=self.w_rec.device, dtype=self.w_rec.dtype)
        )

    def adaptive_threshold(self, a: torch.Tensor) -> torch.Tensor:
        """适应后阈值 ``thr + β·a``。"""
        return self.threshold + self.beta * a

    def v_scaled(self, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """``(v − 适应后阈值) / 基阈值``。阈值处为 0、静息处为 −1。"""
        return (v - self.adaptive_threshold(a)) / self.threshold

    def compute_spikes(self, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """由膜电位生成脉冲（不含不应期抑制）。"""
        return spike_function(self.v_scaled(v, a), self.dampening_factor)

    def step(self, inputs: torch.Tensor, state: ALIFState) -> tuple[torch.Tensor, ALIFState]:
        """推进一个时间步（``dt = 1 ms``）。

        Args:
            inputs: ``(batch, n_in)`` 的输入脉冲。
            state: 上一步的状态。

        Returns:
            ``(spikes, new_state)``，``spikes`` 为 ``(batch, n_rec)``。
        """
        # 适应变量先用**本步之前**的脉冲更新，再据此算本步阈值——与官方实现的顺序一致。
        new_a = self.rho * state.a + state.z

        current = inputs @ self.w_in + state.z @ self.w_rec_effective
        # 复位项用**基**阈值，不是适应后阈值。见模块 docstring 的第 2 条。
        reset = state.z * self.threshold
        new_v = self.alpha * state.v + current - reset

        # 不应期内强制不发放
        refractory = state.r > 0
        fresh = self.compute_spikes(new_v, new_a)
        new_z = torch.where(refractory, torch.zeros_like(fresh), fresh)

        new_r = state.r + self.n_refractory * new_z - 1.0
        new_r = torch.clamp(new_r, min=0.0, max=float(self.n_refractory)).detach()

        return new_z, ALIFState(v=new_v, a=new_a, z=new_z, r=new_r)

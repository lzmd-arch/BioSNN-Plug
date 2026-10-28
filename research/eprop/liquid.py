"""液态 LIF：τ 由局部发放率现算（探索项，预注册见 sweep_results/liquid_lif_prereg.md）。

**红线**：a、b、rate_decay、τ 范围都是固定常数，不是 ``nn.Parameter``，不进优化器；
本模块不调用 ``backward``。e-prop 的痕迹把 α(t) 当外部系数（冻结液态 τ 轨迹），
这一点在 :mod:`research.eprop.traces` 的 ``_eprop_gradient_varying_alpha`` 里写明。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from research.eprop.eprop import EPropLearner
from research.eprop.neurons import spike_function

__all__ = ["LiquidEPropLearner", "LiquidLIFCell", "LiquidState"]


@dataclass
class LiquidState:
    """液态 LIF 的隐状态；``rate`` 是本神经元的局部发放率。"""

    v: torch.Tensor
    a: torch.Tensor
    z: torch.Tensor
    refractory: torch.Tensor
    rate: torch.Tensor

    @classmethod
    def zeros(cls, batch_size: int, n_rec: int, *, device=None, dtype=torch.float32):
        shape = (batch_size, n_rec)
        zeros = torch.zeros(shape, device=device, dtype=dtype)
        return cls(
            v=zeros,
            a=zeros.clone(),
            z=zeros.clone(),
            refractory=zeros.clone(),
            rate=zeros.clone(),
        )


class LiquidLIFCell(nn.Module):
    """LIF，但 ``τ_i(t) = 1/(a + b·r_i(t))``，``r`` 是本神经元自己的脉冲低通。"""

    liquid = True

    def __init__(
        self,
        n_in: int,
        n_rec: int,
        *,
        threshold: float = 0.62,
        dampening_factor: float = 0.3,
        n_refractory: int = 2,
        tau_min: float = 5.0,
        tau_max: float = 20.0,
        rate_decay: float = 0.9,
        a: float = 1.0 / 20.0,
        b: float = 0.15,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 < tau_min < tau_max:
            raise ValueError(f"tau_min/tau_max 必须为正且递增，收到 {tau_min}/{tau_max}。")
        if not 0.0 <= rate_decay < 1.0:
            raise ValueError(f"rate_decay 应在 [0,1)，收到 {rate_decay}。")
        if a <= 0 or b < 0:
            raise ValueError(f"a 必须为正、b 非负，收到 a={a}, b={b}。")
        self.n_in = int(n_in)
        self.n_rec = int(n_rec)
        self.threshold = float(threshold)
        self.dampening_factor = float(dampening_factor)
        self.n_refractory = int(n_refractory)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.rate_decay = float(rate_decay)
        self.a = float(a)
        self.b = float(b)
        self.beta = 0.0
        self.rho = 1.0
        self.alpha = float(torch.exp(torch.tensor(-1.0 / self.tau_max)))
        self.w_in = nn.Parameter(self._init((n_in, n_rec), n_in, generator))
        self.w_rec = nn.Parameter(self._init((n_rec, n_rec), n_rec, generator))
        self.last_alpha: torch.Tensor | None = None

    @staticmethod
    def _init(shape: tuple[int, int], fan_in: int, generator: torch.Generator | None):
        noise = torch.randn(shape, generator=generator)
        return nn.Parameter(noise / (fan_in**0.5))

    @property
    def w_rec_effective(self) -> torch.Tensor:
        return self.w_rec * (
            1.0 - torch.eye(self.n_rec, device=self.w_rec.device, dtype=self.w_rec.dtype)
        )

    def initial_state(self, batch_size: int, *, device=None, dtype=torch.float32) -> LiquidState:
        return LiquidState.zeros(batch_size, self.n_rec, device=device, dtype=dtype)

    def v_scaled(self, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return (v - self.threshold) / self.threshold

    def tau(self, rate: torch.Tensor) -> torch.Tensor:
        return (1.0 / (self.a + self.b * rate)).clamp(self.tau_min, self.tau_max)

    def step(self, inputs: torch.Tensor, state: LiquidState) -> tuple[torch.Tensor, LiquidState]:
        """一步动力学；``rate`` 用上一步的脉冲更新（前向因果）。"""
        rate = self.rate_decay * state.rate + (1.0 - self.rate_decay) * state.z
        alpha = torch.exp(-1.0 / self.tau(rate))
        self.last_alpha = alpha.detach()

        current = inputs @ self.w_in + state.z @ self.w_rec_effective
        new_v = alpha * state.v + current - state.z * self.threshold
        refractory = state.refractory > 0
        fresh = spike_function(self.v_scaled(new_v, state.a), self.dampening_factor)
        new_z = torch.where(refractory, torch.zeros_like(fresh), fresh)
        new_refractory = torch.clamp(
            state.refractory + self.n_refractory * new_z - 1.0,
            min=0.0,
            max=float(self.n_refractory),
        ).detach()
        return new_z, LiquidState(
            v=new_v, a=torch.zeros_like(new_v), z=new_z, refractory=new_refractory, rate=rate
        )


class LiquidEPropLearner(EPropLearner):
    """把 :class:`EPropLearner` 的循环层换成液态 LIF；其余训练逻辑完全复用。"""

    def __init__(
        self,
        n_in: int,
        n_rec: int,
        n_out: int,
        *,
        tau_min: float = 5.0,
        tau_max: float = 20.0,
        rate_decay: float = 0.9,
        a: float = 1.0 / 20.0,
        b: float = 0.15,
        **kwargs,
    ) -> None:
        super().__init__(n_in, n_rec, n_out, **kwargs)
        self.cell = LiquidLIFCell(
            n_in,
            n_rec,
            threshold=kwargs.get("threshold", 0.62),
            dampening_factor=kwargs.get("dampening_factor", 0.3),
            n_refractory=kwargs.get("n_refractory", 2),
            tau_min=tau_min,
            tau_max=tau_max,
            rate_decay=rate_decay,
            a=a,
            b=b,
            generator=kwargs.get("generator"),
        )

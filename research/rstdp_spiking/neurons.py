"""脉冲版的 LIF 神经元：给 actor-critic 用的最小实现。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

## 为什么不复用 `research/eprop/neurons.py` 的 `ALIFCell`

两条原因，都不是偷懒：

1. **仓库约定：三条验证线只共享 `research/common/`，互不 import。** 脉冲版 W3 与 e-prop
   是两个主题，跨主题 import 会把两条线的依赖缠在一起。
2. **两者要的东西不一样。** `ALIFCell` 带 `v_scaled` 归一化、不应期计数、以及**伪导数**
   `spike_function`——伪导数是给 **BPTT 对照**用的。而脉冲版 W3 是**纯局部**的：没有反向
   传播，所以一个伪导数都不需要。硬复用会把 BPTT 的机械结构带进一条不需要它的线里。

要不要把「脉冲神经元」提升到 `research/common/`，是一个真问题。**本阶段不裁决**——
等第三阶段真的把这条线做起来、有了第二个消费者时再定；现在只有 W2 与这里两个使用者，
而这里还没跑出任何结果，提前抽到 `common/` 等于把一次没有证据的重构写进共用层。
（计划书 §七 第三阶段 · 执行层脉冲化；修订记录 ⑪。）

## 动力学

    v ← α·v + I − ϑ·z_prev      α = exp(−dt/τ)
    z = 1[v ≥ ϑ]

复位用**减法**（`− ϑ·z_prev`）而不是清零，这是 LIF 的标准写法，也让膜电位在连续发放时
自然地被压低。不应期用**上次发放之后的步数**实现。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = ["LIFLayer", "LIFState"]


@dataclass
class LIFState:
    """一层 LIF 的状态。所有张量形状都是 ``(batch, n_neurons)``。"""

    v: torch.Tensor
    z: torch.Tensor
    r: torch.Tensor

    @classmethod
    def zeros(cls, batch: int, n_neurons: int, *, device=None, dtype=torch.float32) -> LIFState:
        shape = (batch, n_neurons)
        return cls(
            v=torch.zeros(shape, device=device, dtype=dtype),
            z=torch.zeros(shape, device=device, dtype=dtype),
            r=torch.zeros(shape, device=device, dtype=dtype),
        )

    def detach(self) -> LIFState:
        """切断计算图。纯局部学习不需要它，但评测时省内存。"""
        return LIFState(self.v.detach(), self.z.detach(), self.r.detach())


class LIFLayer(torch.nn.Module):
    """一层全连接 LIF。**前向全程不建计算图**——这条线不用反向传播。

    Args:
        n_in: 输入维数。
        n_out: 神经元个数。
        tau: 膜时间常数（毫秒）。
        threshold: 发放阈值 ``ϑ``。
        n_refractory: 发放后强制静默的步数。
        baseline_period: 无输入时的静息发放周期（步）。``None`` 表示偏置取 0——**只在
            你确定输入自己能驱动起发放时才这么用**，否则这一层会完全沉默且不报错。
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        *,
        tau: float = 20.0,
        threshold: float = 1.0,
        n_refractory: int = 2,
        dt: float = 1.0,
        baseline_period: int | None = 5,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if tau <= 0 or dt <= 0:
            raise ValueError(f"时间常数与步长必须为正，收到 tau={tau}、dt={dt}。")
        if threshold <= 0:
            raise ValueError(f"阈值必须为正，收到 {threshold}。")
        if n_refractory < 0:
            raise ValueError(f"不应期不能为负，收到 {n_refractory}。")

        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.tau = float(tau)
        self.threshold = float(threshold)
        self.n_refractory = int(n_refractory)
        self.alpha = math.exp(-dt / tau)

        # 初始化。**两件事都要做对，否则这个层要么全不发放、要么全发放，两种都学不动**：
        #
        # 1. 突触权重按 fan-in 缩放，且**正负均衡**——它们只负责调制发放时刻；
        # 2. 偏置提供一个**基线驱动**，让静息发放率不为零。这一条不能省：只有随机正负权重时，
        #    各输入项的电流互相抵消、期望为零，膜电位永远到不了阈值——**层会完全沉默**，
        #    而沉默是不报错的（痕迹恒为零、梯度恒为零，看起来只是"学不动"）。
        #    ``baseline_period`` 就是想让神经元在**没有输入**时多少步发放一次。
        bound = 1.0 / math.sqrt(max(n_in, 1))
        if generator is None:
            w = torch.empty(n_in, n_out).uniform_(-bound, bound)
        else:
            w = torch.empty(n_in, n_out).uniform_(-bound, bound, generator=generator)
        self.weight = torch.nn.Parameter(w)
        if baseline_period is None:
            baseline_bias = 0.0
        else:
            if baseline_period <= 1:
                raise ValueError(f"baseline_period 必须 > 1，收到 {baseline_period}。")
            # v_n = b·(1−α^n)/(1−α) 达到阈值时 n = baseline_period，反解出 b。
            baseline_bias = threshold * (1.0 - self.alpha) / (1.0 - self.alpha**baseline_period)
        self.bias = torch.nn.Parameter(torch.full((n_out,), baseline_bias))

    @torch.no_grad()
    def step(self, inputs: torch.Tensor, state: LIFState) -> tuple[torch.Tensor, LIFState]:
        """推进一个时间步。

        Args:
            inputs: ``(batch, n_in)`` 的输入（脉冲或模拟量）。
            state: 上一步的状态。

        Returns:
            ``(spikes, new_state)``，``spikes`` 为 ``(batch, n_out)`` 的 0/1 张量。
        """
        current = inputs @ self.weight + self.bias
        # 复位用减法；不应期内同样要复位，所以 ``reset`` 只看上一步发没发。
        new_v = self.alpha * state.v + current - self.threshold * state.z

        fresh = (new_v >= self.threshold).to(new_v.dtype)
        in_refractory = state.r > 0
        new_z = torch.where(in_refractory, torch.zeros_like(fresh), fresh)

        new_r = torch.clamp(state.r + self.n_refractory * new_z - 1.0, min=0.0)
        return new_z, LIFState(v=new_v, z=new_z, r=new_r)

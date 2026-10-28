"""事件驱动 e-prop 的最小实现：LIF（β=0）、最后一步损失、固定稀疏 fan-in。

**边界（先写死）**：

* 只支持 ``beta=0``（LIF）与 ``loss_timesteps="last"``。在这两个条件下，e-prop 的
  ε_v 不含后突触下标，梯度可写成
  ``outer(filtered_pre, L_j·ψ_j)``；因此不需要 O(N²) 的痕迹，只需要 O(N·k) 的稀疏
  权重与 O(N) 的滤波脉冲。
* 这不是完整的 Korcsák-Gorzo 事件驱动实现：前向/读出仍按时间步推进；省掉的是
  **O(N²) 的资格痕迹**，权重也只存固定 fan-in。它的用途是先把 10k–100k 的规模
  可行性做出来，再谈 500k。
* 稀疏 mask 固定；不做学习型连接；不做 BPTT。
"""

from __future__ import annotations

import torch
from torch import nn

from research.eprop.neurons import spike_function
from research.eprop.traces import exp_convolve, refractory_mask

__all__ = ["EventDrivenEProp", "SparseLIFState"]


class SparseLIFState:
    def __init__(self, v: torch.Tensor, z: torch.Tensor, refractory: torch.Tensor) -> None:
        self.v = v
        self.z = z
        self.refractory = refractory

    @classmethod
    def zeros(cls, batch: int, n_rec: int, *, device=None, dtype=torch.float32):
        zeros = torch.zeros(batch, n_rec, device=device, dtype=dtype)
        return cls(v=zeros.clone(), z=zeros.clone(), refractory=zeros.clone())


class EventDrivenEProp(nn.Module):
    """固定 fan-in 的 LIF + 最后一步 e-prop。"""

    def __init__(
        self,
        n_in: int,
        n_rec: int,
        n_out: int,
        *,
        fan_in: int = 32,
        tau: float = 20.0,
        threshold: float = 0.62,
        dampening_factor: float = 0.3,
        n_refractory: int = 2,
        learning_rate_in: float = 1e-3,
        learning_rate_rec: float = 1e-3,
        learning_rate_out: float = 1e-2,
        decay_out: float = 0.95,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if not 1 <= fan_in <= n_rec:
            raise ValueError(f"fan_in 必须在 [1, {n_rec}]，收到 {fan_in}。")
        self.n_in, self.n_rec, self.n_out = int(n_in), int(n_rec), int(n_out)
        self.fan_in = int(fan_in)
        self.threshold = float(threshold)
        self.dampening_factor = float(dampening_factor)
        self.n_refractory = int(n_refractory)
        self.alpha = float(torch.exp(torch.tensor(-1.0 / tau)))
        self.learning_rate_in = float(learning_rate_in)
        self.learning_rate_rec = float(learning_rate_rec)
        self.learning_rate_out = float(learning_rate_out)
        self.decay_out = float(decay_out)

        gen = generator or torch.Generator()
        self.w_in = nn.Parameter(torch.randn(n_in, n_rec, generator=gen) / (n_in**0.5))
        self.w_out = nn.Parameter(torch.randn(n_rec, n_out, generator=gen) / (n_rec**0.5))
        # 固定 fan-in：每个后神经元选 fan_in 个不同的前神经元（不含自身）。
        indices = []
        values = []
        for post in range(n_rec):
            candidates = torch.randperm(n_rec - 1, generator=gen)[:fan_in]
            candidates = candidates + (candidates >= post).long()
            indices.append(candidates)
            values.append(torch.randn(fan_in, generator=gen) / (fan_in**0.5))
        self.register_buffer("pre_index", torch.stack(indices), persistent=False)  # (N,k)
        self.w_rec = nn.Parameter(torch.stack(values))  # (N,k)

    def _trace(self, values: torch.Tensor) -> torch.Tensor:
        """ε_v 递推：``ε_t = α·ε_{t−1} + z_t``（没有 1−α；那是读出滤波）。"""
        running = torch.zeros_like(values[0])
        out = []
        for value in values:
            running = self.alpha * running + value
            out.append(running)
        return torch.stack(out)

    def _recurrent_current(self, z_previous: torch.Tensor) -> torch.Tensor:
        current = torch.zeros_like(z_previous)
        for slot in range(self.fan_in):
            current += z_previous[:, self.pre_index[:, slot]] * self.w_rec[:, slot]
        return current

    def _v_scaled(self, v: torch.Tensor) -> torch.Tensor:
        return (v - self.threshold) / self.threshold

    @torch.no_grad()
    def simulate(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        steps, batch = inputs.shape[0], inputs.shape[1]
        state = SparseLIFState.zeros(batch, self.n_rec, device=inputs.device, dtype=inputs.dtype)
        spikes, scaled = [], []
        for t in range(steps):
            current = inputs[t] @ self.w_in + self._recurrent_current(state.z)
            new_v = self.alpha * state.v + current - state.z * self.threshold
            refractory = state.refractory > 0
            fresh = spike_function(self._v_scaled(new_v), self.dampening_factor)
            new_z = torch.where(refractory, torch.zeros_like(fresh), fresh)
            new_refractory = torch.clamp(
                state.refractory + self.n_refractory * new_z - 1.0,
                min=0.0,
                max=float(self.n_refractory),
            ).detach()
            state = SparseLIFState(v=new_v, z=new_z, refractory=new_refractory)
            spikes.append(new_z)
            scaled.append(self._v_scaled(new_v))
        return torch.stack(spikes), torch.stack(scaled)

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        spikes, _ = self.simulate(inputs)
        filtered = exp_convolve(spikes, self.decay_out)
        return (filtered[-1] @ self.w_out).argmax(dim=-1)

    @torch.no_grad()
    def update(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """一次在线更新；损失只取最后一步（本类只实现这一支）。"""
        batch = inputs.shape[1]
        spikes, scaled = self.simulate(inputs)
        filtered = exp_convolve(spikes, self.decay_out)
        logits = filtered[-1] @ self.w_out
        loss = float(torch.nn.functional.cross_entropy(logits, targets).item())

        probabilities = torch.softmax(logits, dim=-1)
        one_hot = torch.zeros_like(probabilities)
        one_hot.scatter_(1, targets.reshape(-1, 1), 1.0)
        output_gradient = (probabilities - one_hot) / batch  # dE/dy at last step
        signal = output_gradient @ self.w_out.T  # (batch, n_rec)
        refractory = refractory_mask(spikes, self.n_refractory)[-1]
        psi_full = (self.dampening_factor / self.threshold) * torch.clamp(
            1.0 - scaled[-1].abs(), min=0.0
        )
        psi = torch.where(refractory, torch.zeros_like(psi_full), psi_full)
        post = signal * psi  # (batch, n_rec)

        z_previous = torch.cat([torch.zeros_like(spikes[:1]), spikes[:-1]])
        input_trace = self._trace(inputs)[-1]  # (batch, n_in)
        rec_trace = self._trace(z_previous)[-1]  # (batch, n_rec)

        grad_in = torch.einsum("bi,bj->ij", input_trace, post)
        grad_rec = torch.zeros_like(self.w_rec)
        for slot in range(self.fan_in):
            grad_rec[:, slot] = torch.einsum(
                "bi,bi->i", rec_trace[:, self.pre_index[:, slot]], post
            )
        self.w_in.add_(grad_in, alpha=-self.learning_rate_in)
        self.w_rec.add_(grad_rec, alpha=-self.learning_rate_rec)
        self.w_out.add_(-self.learning_rate_out * (filtered[-1].T @ output_gradient))
        return loss

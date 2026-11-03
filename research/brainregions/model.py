"""线 A：脑区特化的 e-prop（判据见 `prereg-brain-region-specialization.md`）。

与 :class:`~research.eprop.EventDrivenEProp` 的差别只有两处：

1. **阈值与膜时间常数可以逐神经元不同**（按区分组）；
2. **`pre_index` 按"区内偏置"采样**（默认 24/32 取自本区）。

其余——`w_in` / `w_out`、读出、e-prop 更新式、`fan_in`——完全继承，一行没改。
输入侧的资格痕迹仍用标量 α（输入维与神经元维不是一回事），只有**循环侧**的痕迹用逐神经元 α。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from research.eprop import (
    EventDrivenEProp,
    SparseLIFState,
    exp_convolve,
    refractory_mask,
    spike_function,
)

__all__ = ["RegionEProp", "RegionSpec"]


@dataclass(frozen=True)
class RegionSpec:
    """一个脑区：多少个神经元、什么时间常数、什么阈值。"""

    name: str
    count: int
    tau: float
    threshold: float


class RegionEProp(EventDrivenEProp):
    """按脑区分组的循环层。``regions`` 的 ``count`` 之和必须等于 ``n_rec``。"""

    def __init__(
        self,
        n_in: int,
        n_rec: int,
        n_out: int,
        *,
        fan_in: int,
        regions: list[RegionSpec],
        intra_fraction: float = 0.75,
        generator: torch.Generator | None = None,
        **kwargs,
    ) -> None:
        if sum(region.count for region in regions) != n_rec:
            raise ValueError(f"regions 的 count 之和必须等于 n_rec={n_rec}。")
        if not 0.0 <= intra_fraction <= 1.0:
            raise ValueError(f"intra_fraction 必须落在 [0, 1]，收到 {intra_fraction!r}。")
        super().__init__(n_in, n_rec, n_out, fan_in=fan_in, generator=generator, **kwargs)

        alphas, thresholds, region_of, starts, sizes = [], [], [], [], []
        start = 0
        for index, region in enumerate(regions):
            alphas += [float(torch.exp(torch.tensor(-1.0 / region.tau)))] * region.count
            thresholds += [region.threshold] * region.count
            region_of += [index] * region.count
            starts.append(start)
            sizes.append(region.count)
            start += region.count
        self.register_buffer("alpha_vec", torch.tensor(alphas))
        self.register_buffer("threshold_vec", torch.tensor(thresholds))

        gen = generator or torch.Generator()
        intra = round(fan_in * intra_fraction)
        post = torch.arange(n_rec).unsqueeze(1)
        region_index = torch.tensor(region_of)
        lo = torch.tensor(starts)[region_index].unsqueeze(1)
        size = torch.tensor(sizes)[region_index].unsqueeze(1)
        draw = torch.rand((n_rec, intra), generator=gen)
        offset = torch.floor(draw * (size - 1).float()).long()
        offset = offset + (offset >= (post - lo)).long()
        within = lo + offset
        inter = torch.randint(0, n_rec - 1, (n_rec, fan_in - intra), generator=gen)
        inter = inter + (inter >= post).long()
        self.pre_index = torch.cat([within, inter], dim=1)

    def _v_scaled(self, v: torch.Tensor) -> torch.Tensor:
        return (v - self.threshold_vec) / self.threshold_vec

    def _trace_rec(self, values: torch.Tensor) -> torch.Tensor:
        """循环侧的资格痕迹：每个神经元用自己的 α。"""
        running = torch.zeros_like(values[0])
        out = []
        for value in values:
            running = self.alpha_vec * running + value
            out.append(running)
        return torch.stack(out)

    @torch.no_grad()
    def simulate(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        steps, batch = inputs.shape[0], inputs.shape[1]
        state = SparseLIFState.zeros(batch, self.n_rec, device=inputs.device, dtype=inputs.dtype)
        spikes, scaled = [], []
        for t in range(steps):
            current = inputs[t] @ self.w_in + self._recurrent_current(state.z)
            new_v = self.alpha_vec * state.v + current - state.z * self.threshold_vec
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
    def update(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        batch = inputs.shape[1]
        spikes, scaled = self.simulate(inputs)
        filtered = exp_convolve(spikes, self.decay_out)
        logits = filtered[-1] @ self.w_out
        loss = float(torch.nn.functional.cross_entropy(logits, targets).item())

        probabilities = torch.softmax(logits, dim=-1)
        one_hot = torch.zeros_like(probabilities)
        one_hot.scatter_(1, targets.reshape(-1, 1), 1.0)
        output_gradient = (probabilities - one_hot) / batch
        signal = output_gradient @ self.w_out.T
        refractory = refractory_mask(spikes, self.n_refractory)[-1]
        # 与父类保持同一个舍入：父类是 float64 的标量除法再乘 float32 张量，
        # 这里若直接做 float32 除法，最后一位可能不同（守卫测试会红）。
        ratio = (
            torch.tensor(self.dampening_factor, dtype=torch.float64) / self.threshold_vec.double()
        ).to(self.threshold_vec.dtype)
        psi_full = ratio * torch.clamp(1.0 - scaled[-1].abs(), min=0.0)
        psi = torch.where(refractory, torch.zeros_like(psi_full), psi_full)
        post = signal * psi

        z_previous = torch.cat([torch.zeros_like(spikes[:1]), spikes[:-1]])
        input_trace = self._trace(inputs)[-1]
        rec_trace = self._trace_rec(z_previous)[-1]

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

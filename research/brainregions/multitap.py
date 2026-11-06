"""线 A v3：多抽头读出（判据见 `prereg-a3-multitap.md`）。

A2 把天花板定位到了读出：只读最后一步 0.759，同一组特征换闭式解 0.772，
而**全 28 步**换闭式解 0.869——信息在时序里，我们只读了一帧。

这个类的改动**只有读出**：滤波发放在 ``k`` 个等距时间点各取一份、拼接成
``k × n_rec`` 的特征，每个抽头用同一套 e-prop 规则更新（学习信号各取各的、
资格痕迹取各自时刻的）。``k = 1`` 时与 :class:`RegionEProp` 逐位一致——守卫测试盯着这一条。
"""

from __future__ import annotations

import torch
from torch import nn

from research.brainregions.model import RegionEProp
from research.eprop import exp_convolve, refractory_mask

__all__ = ["MultiTapEProp", "tap_indices"]


def tap_indices(steps: int, taps: int) -> list[int]:
    """``[0, steps-1]`` 上等距取 ``taps`` 个下标；整数运算，避免银行家舍入。"""
    if taps < 1:
        raise ValueError(f"taps 必须 ≥1，收到 {taps!r}。")
    if taps == 1:
        return [steps - 1]
    return [(steps - 1) * index // (taps - 1) for index in range(taps)]


class MultiTapEProp(RegionEProp):
    """读出改成 ``k`` 个等距抽头的拼接；其余（循环层、e-prop 规则）全部继承。"""

    def __init__(self, n_in: int, n_rec: int, n_out: int, *, taps: int = 1, **kwargs) -> None:
        super().__init__(n_in, n_rec, n_out, **kwargs)
        self.taps = int(taps)
        if self.taps > 1:
            # 抽头维度的初始化：与父类同样的幅度（1/√总维度），使初始 logits 的尺度不变。
            gen = kwargs.get("generator") or torch.Generator()
            width = self.taps * self.n_rec
            self.w_out = nn.Parameter(torch.randn(width, n_out, generator=gen) / (width**0.5))

    # -- 读出 ---------------------------------------------------------------

    def _features(self, filtered: torch.Tensor) -> torch.Tensor:
        """``(T, batch, N)`` 的滤波发放 → ``(batch, taps·N)``。"""
        index = tap_indices(filtered.shape[0], self.taps)
        return torch.cat([filtered[point] for point in index], dim=1)

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        spikes, _ = self.simulate(inputs)
        filtered = exp_convolve(spikes, self.decay_out)
        return (self._features(filtered) @ self.w_out).argmax(dim=-1)

    # -- 更新 ---------------------------------------------------------------

    @torch.no_grad()
    def update(
        self, inputs: torch.Tensor, targets: torch.Tensor, *, learn_recurrent: bool = True
    ) -> float:
        """每个抽头做一遍"最后一步"的 e-prop，梯度按抽头求和。

        ``learn_recurrent=False`` 时冻结 ``w_rec``（A9 用它做"去掉递推"的对照）。
        """
        batch = inputs.shape[1]
        spikes, scaled = self.simulate(inputs)
        filtered = exp_convolve(spikes, self.decay_out)
        logits = self._features(filtered) @ self.w_out
        loss = float(torch.nn.functional.cross_entropy(logits, targets).item())

        probabilities = torch.softmax(logits, dim=-1)
        one_hot = torch.zeros_like(probabilities)
        one_hot.scatter_(1, targets.reshape(-1, 1), 1.0)
        output_gradient = (probabilities - one_hot) / batch  # (batch, n_out)

        index = torch.as_tensor(tap_indices(spikes.shape[0], self.taps), device=inputs.device)
        refractory = refractory_mask(spikes, self.n_refractory)[index]  # (k, batch, N)
        ratio = (
            torch.tensor(self.dampening_factor, dtype=torch.float64) / self.threshold_vec.double()
        ).to(self.threshold_vec.dtype)
        psi_full = ratio * torch.clamp(1.0 - scaled[index].abs(), min=0.0)
        psi = torch.where(refractory, torch.zeros_like(psi_full), psi_full)

        # 每个抽头一条学习信号：只有它自己那块 w_out 参与。
        tap_gradient = (output_gradient @ self.w_out.T).reshape(batch, self.taps, self.n_rec)
        post = tap_gradient.permute(1, 0, 2) * psi  # (k, batch, N)

        z_previous = torch.cat([torch.zeros_like(spikes[:1]), spikes[:-1]])
        input_trace = self._trace(inputs)[index]  # (k, batch, n_in)
        rec_trace = self._trace_rec(z_previous)[index]  # (k, batch, N)

        grad_in = torch.einsum("kbi,kbj->ij", input_trace, post)
        grad_rec = torch.zeros_like(self.w_rec)
        for slot in range(self.fan_in):
            grad_rec[:, slot] = torch.einsum(
                "kbi,kbi->i", rec_trace[:, :, self.pre_index[:, slot]], post
            )
        self.w_in.add_(grad_in, alpha=-self.learning_rate_in)
        if learn_recurrent:
            self.w_rec.add_(grad_rec, alpha=-self.learning_rate_rec)
        self.w_out.add_(-self.learning_rate_out * (self._features(filtered).T @ output_gradient))
        return loss

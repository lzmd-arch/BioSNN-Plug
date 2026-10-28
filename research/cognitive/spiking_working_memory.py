"""多时间尺度工作记忆 v5：LIF/ALIF 脉冲群体 + 脉冲计数读出。

预注册见 ``sweep_results/working_memory_v5_prereg.md``。每个 τ 一组 n_dims 个 ALIF 单元，
输入权重为单位阵、无循环权重；读数取 60 步脉冲计数，局部 RLS 读出。
"""

from __future__ import annotations

import numpy as np
import torch

from research.eprop import ALIFCell, ALIFState

__all__ = ["TAUS", "SpikingMultiTimescale", "SpikingPopulationTimescale"]

TAUS = (2, 5, 10, 20, 50, 100, 200)
N_DIMS = 4
CUE_A = (0, 4)
CUE_B = (20, 24)
TEST_STEP = 60


class SpikingMultiTimescale:
    """每个 τ 一组 ALIF 单元；身份输入、无循环；返回脉冲计数。"""

    def __init__(
        self,
        *,
        taus: tuple[int, ...] = TAUS,
        n_dims: int = N_DIMS,
        beta: float = 0.0,
        threshold: float = 1.0,
        seed: int = 0,
        readout: str = "counts",
        membrane_window: int = 10,
    ) -> None:
        if readout not in ("counts", "membrane"):
            raise ValueError("readout 只能是 counts / membrane。")
        self.readout = readout
        self.membrane_window = int(membrane_window)
        self.n_dims = int(n_dims)
        self.taus = tuple(taus)
        generator = torch.Generator().manual_seed(seed)
        self.cells = []
        for tau in self.taus:
            cell = ALIFCell(
                n_dims,
                n_dims,
                tau=float(tau),
                beta=beta,
                threshold=threshold,
                generator=generator,
            )
            with torch.no_grad():
                cell.w_in.copy_(torch.eye(n_dims))
                cell.w_rec.zero_()
            for parameter in cell.parameters():
                parameter.requires_grad_(False)
            self.cells.append(cell)

    @torch.no_grad()
    def run(
        self,
        cue_a: np.ndarray,
        cue_b: np.ndarray,
        *,
        generator: torch.Generator,
        test_step: int = TEST_STEP,
    ) -> np.ndarray:
        cue_a_t = torch.as_tensor(cue_a, dtype=torch.float32)
        cue_b_t = torch.as_tensor(cue_b, dtype=torch.float32)
        counts = torch.zeros(len(self.taus), self.n_dims)
        membrane = torch.zeros(len(self.taus), self.n_dims)
        states = [ALIFState.zeros(1, self.n_dims) for _ in self.cells]
        for step in range(test_step):
            probability = torch.zeros(self.n_dims)
            if CUE_A[0] <= step <= CUE_A[1]:
                probability = ((cue_a_t + 3.0) / 6.0).clamp(0.0, 1.0)
            elif CUE_B[0] <= step <= CUE_B[1]:
                probability = ((cue_b_t + 3.0) / 6.0).clamp(0.0, 1.0)
            inputs = (torch.rand(self.n_dims, generator=generator) < probability).float()
            for index, cell in enumerate(self.cells):
                spikes, states[index] = cell.step(inputs.unsqueeze(0), states[index])
                counts[index] += spikes.squeeze(0)
                if step >= test_step - self.membrane_window:
                    membrane[index] += states[index].v.squeeze(0)
        if self.readout == "membrane":
            span = min(self.membrane_window, test_step)
            return (membrane / max(span, 1)).reshape(-1).numpy()
        return counts.reshape(-1).numpy()


class SpikingPopulationTimescale:
    """每个 τ 一组独立传入通道的 ALIF 群体；读出为群体膜电位均值。

    v5/v6 每个 cue 每个维度只有**一个**伯努利通道（5 个样本），线性解码上界约 0.38。
    这里每个 (τ, 维度) 用 ``population`` 个独立通道（同一 p、独立采样），每个通道驱动
    一个 ALIF 单元；单元间无循环权重，输入权重按 ``gain_range`` 对数间隔给不同灵敏度。
    读出取最后 ``membrane_window`` 步、每个 (τ, 维度) 的群体膜电位均值，得到与 v5/v6
    相同的 ``len(taus)·n_dims`` 维状态（population 维已平均掉）。
    """

    def __init__(
        self,
        *,
        taus: tuple[int, ...] = TAUS,
        n_dims: int = N_DIMS,
        population: int = 32,
        gain_range: tuple[float, float] = (0.5, 4.0),
        beta: float = 0.0,
        threshold: float = 1.0,
        seed: int = 0,
        membrane_window: int = 10,
        readout: str = "post",
    ) -> None:
        if population < 1:
            raise ValueError(f"population 必须 >= 1，收到 {population}。")
        if not 0.0 < gain_range[0] <= gain_range[1]:
            raise ValueError(f"gain_range 必须为正且递增，收到 {gain_range}。")
        if readout not in ("post", "prereset", "trace"):
            raise ValueError(f"readout 只能是 post / prereset / trace，收到 {readout!r}。")
        self.readout = readout
        self.n_dims = int(n_dims)
        self.population = int(population)
        self.taus = tuple(taus)
        self.membrane_window = int(membrane_window)
        self.gains = torch.as_tensor(
            np.geomspace(gain_range[0], gain_range[1], self.population), dtype=torch.float32
        )
        generator = torch.Generator().manual_seed(seed)
        self.cells = []
        width = self.n_dims * self.population
        for tau in self.taus:
            cell = ALIFCell(
                width,
                width,
                tau=float(tau),
                beta=beta,
                threshold=threshold,
                generator=generator,
            )
            with torch.no_grad():
                cell.w_in.zero_()
                index = torch.arange(width)
                cell.w_in[index, index] = self.gains.repeat(self.n_dims)
                cell.w_rec.zero_()
            for parameter in cell.parameters():
                parameter.requires_grad_(False)
            self.cells.append(cell)
        self.last_spike_rates = np.zeros(len(self.taus))
        self.last_membrane_max = 0.0

    @torch.no_grad()
    def run(
        self,
        cue_a: np.ndarray,
        cue_b: np.ndarray,
        *,
        generator: torch.Generator,
        test_step: int = TEST_STEP,
    ) -> np.ndarray:
        cue_a_t = torch.as_tensor(cue_a, dtype=torch.float32)
        cue_b_t = torch.as_tensor(cue_b, dtype=torch.float32)
        probability_a = ((cue_a_t + 3.0) / 6.0).clamp(0.0, 1.0)
        probability_b = ((cue_b_t + 3.0) / 6.0).clamp(0.0, 1.0)
        width = self.n_dims * self.population
        membrane = torch.zeros(len(self.taus), self.n_dims)
        spikes = torch.zeros(len(self.taus))
        peak = 0.0
        states = [ALIFState.zeros(1, width) for _ in self.cells]
        traces = [torch.zeros(1, width) for _ in self.cells]
        for step in range(test_step):
            if CUE_A[0] <= step <= CUE_A[1]:
                probability = probability_a
            elif CUE_B[0] <= step <= CUE_B[1]:
                probability = probability_b
            else:
                probability = torch.zeros(self.n_dims)
            expanded = probability.repeat_interleave(self.population)
            inputs = (torch.rand(width, generator=generator) < expanded).float().unsqueeze(0)
            for index, cell in enumerate(self.cells):
                previous = states[index].z
                fired, states[index] = cell.step(inputs, states[index])
                spikes[index] += fired.sum()
                if step >= test_step - self.membrane_window:
                    if self.readout == "trace":
                        # v 含 −thr·z_{t−1} 的复位；加上 thr·trace_{t−1} 得到无复位的
                        # 泄漏积分 u_t = Σ α^{t−s} I_s。trace 与膜电位同 τ。
                        value = states[index].v + cell.threshold * traces[index]
                    elif self.readout == "prereset":
                        # 复位项用基阈值；把上一步的复位加回去即复位前膜电位。
                        value = states[index].v + cell.threshold * previous
                    else:
                        value = states[index].v
                    membrane[index] += (
                        value.squeeze(0).reshape(self.n_dims, self.population).mean(dim=1)
                    )
                    peak = max(peak, float(states[index].v.max()))
                traces[index] = cell.alpha * traces[index] + fired
        self.last_spike_rates = (spikes / (test_step * width)).numpy()
        self.last_membrane_max = peak
        span = min(self.membrane_window, test_step)
        return (membrane / max(span, 1)).reshape(-1).numpy()

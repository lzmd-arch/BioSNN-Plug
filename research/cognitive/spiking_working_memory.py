"""多时间尺度工作记忆 v5：LIF/ALIF 脉冲群体 + 脉冲计数读出。

预注册见 ``sweep_results/working_memory_v5_prereg.md``。每个 τ 一组 n_dims 个 ALIF 单元，
输入权重为单位阵、无循环权重；读数取 60 步脉冲计数，局部 RLS 读出。
"""

from __future__ import annotations

import numpy as np
import torch

from research.eprop import ALIFCell, ALIFState

__all__ = ["TAUS", "SpikingMultiTimescale"]

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

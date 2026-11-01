"""SpikeBus 的输出电流喂进 AdaptiveMembrane：总线 → 膜 → 脉冲。

这是 ``biosnn_bus`` 里唯一"能跑出脉冲"的完整路径：总线给电流，ALIF 膜给动力学。
**总线本身不跑神经元**（它只输出输入电流），所以这两步必须由调用方接起来——本文件
就是那条接线的最小示例。

跑法：``python examples/spiking_membrane.py``（无 GPU、无网络、无 torch）。
"""

from __future__ import annotations

import numpy as np
from biosnn_bus import AdaptiveMembrane, ModalityPlugin, SpikeBus, SpikeTrain

BUS_DIM = 64
STEPS = 32


class RampPlugin(ModalityPlugin):
    """把一条递增斜坡编码成脉冲：越到后面发放越密。"""

    def __init__(self, *, steps: int = STEPS, dim: int = 8, dt: float = 1.0) -> None:
        self._steps = steps
        self._dim = dim
        self._dt = dt

    @property
    def modality_name(self) -> str:
        return "ramp"

    @property
    def spike_dim(self) -> int:
        return self._dim

    @property
    def dt(self) -> float:
        return self._dt

    def encode(self, raw_input: np.ndarray) -> SpikeTrain:
        data = np.asarray(raw_input, dtype=np.float64).reshape(self._steps, self._dim)
        return SpikeTrain(data=data, dt=self._dt)

    def decode(self, spike_output: SpikeTrain) -> np.ndarray:
        return spike_output.rates

    def get_membrane(self) -> AdaptiveMembrane:
        return AdaptiveMembrane(self._dim)


def main() -> dict:
    """跑一遍并把可核查的数返回（测试也用它）。"""
    bus = SpikeBus(bus_dim=BUS_DIM, seed=0)
    bus.register(RampPlugin())

    ramp = np.linspace(0.0, 1.0, STEPS * 8).reshape(STEPS, 8)
    currents = bus.step({"ramp": ramp})  # (T, BUS_DIM) 的输入电流
    assert currents.data.shape == (STEPS, BUS_DIM)

    membrane = AdaptiveMembrane(BUS_DIM, beta=0.07)
    spikes = np.array([membrane.forward(row) for row in currents.data])

    fired = int(spikes.sum())
    return {
        "steps": STEPS,
        "neurons": BUS_DIM,
        "spikes": fired,
        "active_neurons": int(spikes.any(axis=0).sum()),
    }


if __name__ == "__main__":
    result = main()
    print(
        f"{result['steps']} 步 × {result['neurons']} 个 ALIF 神经元："
        f"共 {result['spikes']} 次发放，{result['active_neurons']} 个神经元至少发放一次"
    )

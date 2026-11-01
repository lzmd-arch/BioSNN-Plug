"""自适应阈值膜（ALIF）：协议、与 torch 参考实现的对齐、以及几个行为性质。

对齐的那组数不是手算的——是用研究层 `research/eprop/neurons.py` 的 `ALIFCell`
在同一组输入上跑出来的（`w_in` 取单位阵、`w_rec` 取零，于是它的 "inputs @ w_in"
就等于本类的输入电流）。torch 只在生成这组数时用过一次，**不是本包的依赖**。
"""

from __future__ import annotations

import numpy as np
import pytest
from biosnn_bus import AdaptiveMembrane, Membrane, PassThroughMembrane
from biosnn_bus.membrane import AdaptiveMembrane as MembraneFromModule

#: 研究层 ALIFCell 在同一组输入上的输出（0/1 逐时间步、逐神经元）。
_TORCH_INPUTS = [
    [0.9, 0.1, 0.0],
    [0.9, 0.7, 0.0],
    [0.0, 0.7, 0.9],
    [0.0, 0.0, 0.9],
    [0.5, 0.5, 0.5],
    [1.2, 0.0, 0.0],
]
_TORCH_SPIKES = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 0.0, 0.0],
]


def _run(membrane: AdaptiveMembrane, inputs) -> list[list[int]]:
    return [[int(s) for s in membrane.forward(np.array(step, dtype=float))] for step in inputs]


class TestProtocol:
    def test_satisfies_membrane_protocol(self):
        assert isinstance(AdaptiveMembrane(4), Membrane)

    def test_passthrough_still_satisfies_it(self):
        assert isinstance(PassThroughMembrane(), Membrane)

    def test_exported_from_package_root(self):
        assert AdaptiveMembrane is MembraneFromModule


class TestTorchAlignment:
    def test_matches_the_research_cell_step_for_step(self):
        membrane = AdaptiveMembrane(
            3, tau=20.0, tau_adaptation=500.0, threshold=0.62, beta=0.07, n_refractory=2
        )
        assert _run(membrane, _TORCH_INPUTS) == [[int(v) for v in row] for row in _TORCH_SPIKES]


class TestBehaviour:
    def test_beta_zero_makes_adaptation_irrelevant(self):
        """beta=0 时适应变量照样累积（a = rho*a + z），但**不影响发放**。

        所以判据不是"a 恒为 0"，而是"换一个适应时间常数，脉冲序列一个字都不变"。
        """
        fast = AdaptiveMembrane(3, beta=0.0, tau_adaptation=50.0, n_refractory=0)
        slow = AdaptiveMembrane(3, beta=0.0, tau_adaptation=5000.0, n_refractory=0)
        assert _run(fast, _TORCH_INPUTS) == _run(slow, _TORCH_INPUTS)

    def test_stronger_adaptation_spikes_less(self):
        weak = AdaptiveMembrane(1, beta=0.0, n_refractory=0)
        strong = AdaptiveMembrane(1, beta=2.0, n_refractory=0)
        train = [[0.7]] * 40
        weak_spikes = sum(sum(row) for row in _run(weak, train))
        strong_spikes = sum(sum(row) for row in _run(strong, train))
        assert strong_spikes < weak_spikes

    def test_refractory_suppresses_the_next_steps(self):
        plain = AdaptiveMembrane(1, n_refractory=0)
        blocked = AdaptiveMembrane(1, n_refractory=3)
        train = [[1.0]] * 5
        assert sum(sum(row) for row in _run(plain, train)) > sum(
            sum(row) for row in _run(blocked, train)
        )

    def test_reset_returns_to_the_initial_state(self):
        membrane = AdaptiveMembrane(2)
        _run(membrane, [[1.0, 1.0]] * 3)
        membrane.reset()
        assert np.all(membrane.v == 0.0) and np.all(membrane.a == 0.0)
        fresh = AdaptiveMembrane(2)
        assert _run(membrane, [[1.0, 0.0]] * 3) == _run(fresh, [[1.0, 0.0]] * 3)

    def test_wrong_input_shape_is_refused(self):
        with pytest.raises(ValueError, match="输入形状"):
            AdaptiveMembrane(3).forward(np.zeros(2))

    def test_illegal_parameters_are_refused(self):
        with pytest.raises(ValueError):
            AdaptiveMembrane(1, tau=0.0)
        with pytest.raises(ValueError):
            AdaptiveMembrane(0)

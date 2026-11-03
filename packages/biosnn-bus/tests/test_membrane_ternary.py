"""TI-LIF：与 ALIF 的退化一致、有符号脉冲的方向与幅值、以及可复现性。

**那个退化测试是本文件的核心。** TI-LIF 号称是 ALIF 的超集；如果 ``depth=1,
signed=False`` 与 ``AdaptiveMembrane`` 有任何一步不同，就说明两套动力学各写各的，
"超集"只是文档里的说法。这里用 200 步随机电流逐步比对脉冲与两个内部状态。
"""

from __future__ import annotations

import numpy as np
import pytest
from biosnn_bus import AdaptiveMembrane, Membrane, TernaryMembrane


def _run(membrane, inputs) -> list[list[int]]:
    return [[int(s) for s in membrane.forward(np.array(step, dtype=float))] for step in inputs]


def _random_inputs(seed: int, steps: int, n: int) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    return [list(rng.normal(0.0, 1.0, size=n)) for _ in range(steps)]


class TestProtocol:
    def test_satisfies_membrane_protocol(self):
        assert isinstance(TernaryMembrane(4), Membrane)

    def test_exported_from_package_root(self):
        from biosnn_bus.ternary import TernaryMembrane as FromModule

        assert TernaryMembrane is FromModule


class TestReductionToAlif:
    def test_depth_one_unsigned_is_the_alif_membrane_step_for_step(self):
        inputs = _random_inputs(seed=3, steps=200, n=5)
        alif = AdaptiveMembrane(5)
        ternary = TernaryMembrane(5, depth=1, signed=False)
        assert _run(ternary, inputs) == _run(alif, inputs)
        # 状态也应当一致：只比脉冲的话，"轨迹不同但恰好同发放"会被放过。
        assert np.allclose(ternary.v, alif.v)
        assert np.allclose(ternary.a, alif.a)
        assert np.allclose(ternary.r, alif.r)

    def test_unsigned_never_emits_negative_spikes(self):
        inputs = [[-5.0]] * 20
        assert all(row[0] >= 0 for row in _run(TernaryMembrane(1, signed=False), inputs))


class TestSignedSpikes:
    def test_negative_current_produces_negative_spikes(self):
        spikes = _run(TernaryMembrane(1, signed=True), [[-5.0]] * 5)
        assert any(row[0] < 0 for row in spikes), spikes

    def test_magnitude_never_exceeds_depth(self):
        spikes = _run(
            TernaryMembrane(1, depth=2, signed=True), [[9.0, -9.0][i % 2] for i in range(30)]
        )
        flat = [abs(row[0]) for row in spikes]
        assert max(flat) <= 2, spikes

    def test_a_single_strong_step_can_carry_more_than_one_level(self):
        """一步的幅值随膜电位增长——这正是二值脉冲表达不了的那个量。"""
        spikes = _run(TernaryMembrane(1, depth=3, signed=True, n_refractory=0), [[9.0]])
        assert spikes[0][0] >= 2, spikes

    def test_depth_one_still_gives_only_unit_spikes(self):
        spikes = _run(TernaryMembrane(1, depth=1, signed=True), [[9.0]] * 10)
        assert all(abs(row[0]) <= 1 for row in spikes)


class TestBehaviour:
    def test_reproducible_for_the_same_inputs(self):
        inputs = _random_inputs(seed=11, steps=50, n=3)
        assert _run(TernaryMembrane(3), inputs) == _run(TernaryMembrane(3), inputs)

    def test_firing_rate_is_monotone_in_input_strength(self):
        def total(level: float) -> int:
            spikes = _run(TernaryMembrane(4, depth=2, signed=False), [[level] * 4] * 40)
            return sum(sum(row) for row in spikes)

        totals = [total(level) for level in (0.2, 0.5, 1.0, 2.0)]
        assert totals == sorted(totals), totals
        assert totals[-1] > totals[0]

    def test_refractory_suppresses_spikes(self):
        plain = _run(TernaryMembrane(1, n_refractory=0), [[1.0]] * 6)
        blocked = _run(TernaryMembrane(1, n_refractory=3), [[1.0]] * 6)
        assert sum(abs(row[0]) for row in blocked) < sum(abs(row[0]) for row in plain)

    def test_shape_is_checked(self):
        with pytest.raises(ValueError):
            TernaryMembrane(3).forward(np.zeros(2))

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"depth": 0}, "depth"),
            ({"n_neurons": 0}, "n_neurons"),
            ({"threshold": 0.0}, "threshold"),
        ],
    )
    def test_arguments_are_validated(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            TernaryMembrane(**kwargs)

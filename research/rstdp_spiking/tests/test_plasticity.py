"""脉冲版 W3 的可塑性测试。

**这一组测试的职责比一般的单元测试重**：这条线存在的理由就是「把速率版丢掉的 Δt 做出来」。
所以这里逐条钉住「Δt 真的在起作用」，而不是只测「函数能跑不报错」。

1. 相差 Δt 步的一对 pre/post，贡献**恰好**是 ``exp(−Δt/τ)``；
2. pre-before-post → LTP（正），post-before-pre → LTD（负），**同时发放 → 0**；
3. ``τ_+`` / ``τ_−`` 是**活的参数**——改它结果就变（速率版下它们根本不是自由参数）；
4. LTD 不会因为少写一个负号而变成第二个 LTP；
5. 跨 batch 求和；形状与参数校验。
"""

from __future__ import annotations

import math

import pytest
import torch

from research.rstdp_spiking.plasticity import (
    EligibilityTrace,
    STDPWindow,
    pair_based_stdp,
)


def _spikes(times: dict[int, list[int]], steps: int, n: int) -> torch.Tensor:
    """把 ``{时刻: [神经元下标]}`` 变成 ``(steps, 1, n)`` 的脉冲张量。"""
    out = torch.zeros(steps, 1, n)
    for t, neurons in times.items():
        for j in neurons:
            out[t, 0, j] = 1.0
    return out


class TestDeltaTDependence:
    def test_contribution_is_exactly_the_exponential_window(self):
        """pre 在 0、post 在 Δt → Δw == A+ · exp(−Δt/τ+)，**逐个 Δt 都对**。"""
        tau = 10.0
        window = STDPWindow(tau_plus=tau, tau_minus=tau, a_plus=1.0, a_minus=0.0)
        for delta_t in range(1, 8):
            pre = _spikes({0: [0]}, delta_t + 1, 1)
            post = _spikes({delta_t: [0]}, delta_t + 1, 1)
            got = float(pair_based_stdp(pre, post, window)[0, 0])
            expected = math.exp(-delta_t / tau)
            assert got == pytest.approx(expected, rel=1e-5), f"Δt={delta_t}"

    def test_ltd_side_is_the_mirror_image(self):
        """post 在前、pre 在后 Δt 步 → Δw == −A− · exp(−Δt/τ−)。"""
        tau = 10.0
        window = STDPWindow(tau_plus=tau, tau_minus=tau, a_plus=0.0, a_minus=1.0)
        for delta_t in range(1, 8):
            post = _spikes({0: [0]}, delta_t + 1, 1)
            pre = _spikes({delta_t: [0]}, delta_t + 1, 1)
            got = float(pair_based_stdp(pre, post, window)[0, 0])
            assert got == pytest.approx(-math.exp(-delta_t / tau), rel=1e-5), f"Δt={delta_t}"

    def test_simultaneous_spikes_contribute_nothing(self):
        """**严格先后**才计分——同时发放的那一对两侧都不进。这正是「先后」的字面意思。"""
        window = STDPWindow(tau_plus=10.0, tau_minus=10.0, a_plus=1.0, a_minus=1.0)
        same = _spikes({3: [0]}, 6, 1)
        assert float(pair_based_stdp(same, same, window)[0, 0]) == pytest.approx(0.0, abs=1e-7)

    def test_order_determines_the_sign(self):
        window = STDPWindow(tau_plus=10.0, tau_minus=10.0)
        pre_first = pair_based_stdp(_spikes({2: [0]}, 8, 1), _spikes({5: [0]}, 8, 1), window)
        post_first = pair_based_stdp(_spikes({5: [0]}, 8, 1), _spikes({2: [0]}, 8, 1), window)
        assert float(pre_first[0, 0]) > 0 > float(post_first[0, 0])


class TestTauIsAlive:
    def test_tau_plus_actually_changes_the_result(self):
        """**这条是整条线存在的理由**：速率型下窗口形状无从体现，这里它必须起作用。

        同一个 Δt，τ+ 越小衰减越快 → 权重越小。
        """
        pre, post = _spikes({0: [0]}, 20, 1), _spikes({10: [0]}, 20, 1)
        fast = float(pair_based_stdp(pre, post, STDPWindow(tau_plus=5.0, a_minus=0.0))[0, 0])
        slow = float(pair_based_stdp(pre, post, STDPWindow(tau_plus=50.0, a_minus=0.0))[0, 0])
        assert fast < slow
        assert fast == pytest.approx(math.exp(-10 / 5.0), rel=1e-5)
        assert slow == pytest.approx(math.exp(-10 / 50.0), rel=1e-5)

    def test_plus_and_minus_windows_are_independent(self):
        pre, post = _spikes({1: [0]}, 20, 1), _spikes({6: [0]}, 20, 1)
        window = STDPWindow(tau_plus=3.0, tau_minus=30.0, a_plus=1.0, a_minus=1.0)
        got = float(pair_based_stdp(pre, post, window)[0, 0])
        # 只走 LTP 侧（pre 在前），所以 τ− 不该有任何影响。
        assert got == pytest.approx(math.exp(-5 / 3.0), rel=1e-5)


class TestLtdIsNotASecondLtp:
    def test_negative_amplitude_is_rejected_rather_than_silently_flipped(self):
        """符号由实现承担。让调用方传负数，就是在邀请「LTD 静默变成第二个 LTP」。"""
        with pytest.raises(ValueError, match="幅度必须非负"):
            STDPWindow(a_minus=-1.0)

    def test_turning_off_a_minus_removes_all_depression(self):
        window = STDPWindow(tau_plus=10.0, tau_minus=10.0, a_plus=0.0, a_minus=0.0)
        pre, post = _spikes({1: [0]}, 10, 1), _spikes({4: [0]}, 10, 1)
        assert float(pair_based_stdp(pre, post, window)[0, 0]) == pytest.approx(0.0)


class TestShapesAndValidation:
    def test_sums_over_batch(self):
        window = STDPWindow(tau_plus=10.0, a_minus=0.0)
        single = _spikes({0: [0]}, 6, 1)
        post = _spikes({2: [0]}, 6, 1)
        one = float(pair_based_stdp(single, post, window)[0, 0])
        two = torch.cat([single, single], dim=1)
        assert float(pair_based_stdp(two, torch.cat([post, post], dim=1), window)[0, 0]) == (
            pytest.approx(2 * one)
        )

    def test_population_shape(self):
        window = STDPWindow()
        result = pair_based_stdp(torch.zeros(5, 3, 4), torch.zeros(5, 3, 2), window)
        assert result.shape == (4, 2)

    def test_mismatched_steps_is_an_error(self):
        with pytest.raises(ValueError, match="时间步数不一致"):
            pair_based_stdp(torch.zeros(5, 1, 2), torch.zeros(4, 1, 2), STDPWindow())

    def test_mismatched_batch_is_an_error(self):
        with pytest.raises(ValueError, match="batch 不一致"):
            pair_based_stdp(torch.zeros(5, 1, 2), torch.zeros(5, 3, 2), STDPWindow())

    def test_non_positive_tau_is_rejected(self):
        with pytest.raises(ValueError, match="时间常数必须为正"):
            STDPWindow(tau_plus=0.0)


class TestEligibilityTrace:
    def test_accumulates_with_decay(self):
        trace = EligibilityTrace(2, 2, decay=0.5)
        trace.accumulate(torch.ones(2, 2))
        trace.accumulate(torch.ones(2, 2))
        assert torch.allclose(trace.value, torch.full((2, 2), 1.5))

    def test_reset_clears_it(self):
        trace = EligibilityTrace(2, 2)
        trace.accumulate(torch.ones(2, 2))
        trace.reset()
        assert float(trace.value.abs().sum()) == 0.0

    def test_shape_mismatch_is_an_error(self):
        trace = EligibilityTrace(2, 3)
        with pytest.raises(ValueError, match="形状不符"):
            trace.accumulate(torch.ones(3, 2))

    def test_invalid_decay_is_rejected(self):
        with pytest.raises(ValueError, match="decay"):
            EligibilityTrace(2, 2, decay=1.0)

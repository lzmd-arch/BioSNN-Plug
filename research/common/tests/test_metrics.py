"""``research.common.metrics`` 的测试。

指标口径是三条验证线共用的，所以这里测的不只是"算得对"，还有**定义是否按文档
写的那样**：活跃神经元是"至少发放过一次"，稀疏度是密度的补数。
"""

from __future__ import annotations

import numpy as np
import pytest

from research.common.metrics import (
    DEAD_NEURON_THRESHOLD,
    active_neuron_count,
    active_neuron_fraction,
    spike_density,
    spike_sparsity,
    summarize,
)


class TestActiveNeurons:
    def test_counts_neurons_that_fired_at_least_once(self):
        # 4 个神经元：0 号从不发放，1/2 号发放，3 号只在最后一步发放
        spikes = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=bool,
        )
        assert active_neuron_count(spikes) == 3
        assert active_neuron_fraction(spikes) == pytest.approx(0.75)

    def test_firing_once_is_enough(self):
        """选"至少一次"而不是发放率阈值：判据是"这个神经元还活着吗"。"""
        spikes = np.zeros((100, 2), dtype=bool)
        spikes[99, 1] = True
        assert active_neuron_count(spikes) == 1
        assert active_neuron_fraction(spikes) == pytest.approx(0.5)

    def test_all_dead_gives_zero(self):
        assert active_neuron_fraction(np.zeros((10, 5), dtype=bool)) == 0.0

    def test_all_alive_gives_one(self):
        assert active_neuron_fraction(np.ones((10, 5), dtype=bool)) == 1.0

    def test_float_rates_use_positive_as_fired(self):
        """浮点输入是发放率或膜电位代理，用 > 0 判定，不是 != 0。

        膜电位代理可能带极小负值；用 != 0 会把它们全算成发放。
        """
        rates = np.array([[0.0, -1e-9, 0.3], [0.0, -2.0, 0.0]])
        assert active_neuron_fraction(rates) == pytest.approx(1 / 3)

    def test_accepts_an_object_with_a_data_attribute(self):
        """骨架库的 ``SpikeTrain`` 长这样（``.data``），不该为它单独写一条路径。"""

        class FakeTrain:
            def __init__(self, data):
                self.data = data

        spikes = np.array([[1, 0], [0, 1]], dtype=bool)
        assert active_neuron_fraction(FakeTrain(spikes)) == 1.0


class TestSparsity:
    def test_sparsity_is_the_complement_of_density(self):
        spikes = np.array([[1, 0, 0, 0], [0, 0, 0, 0]], dtype=bool)
        assert spike_density(spikes) == pytest.approx(0.125)
        assert spike_sparsity(spikes) == pytest.approx(0.875)

    def test_all_zero_is_fully_sparse(self):
        assert spike_sparsity(np.zeros((4, 4), dtype=bool)) == 1.0

    def test_all_one_is_not_sparse_at_all(self):
        assert spike_sparsity(np.ones((4, 4), dtype=bool)) == 0.0


class TestShapeValidation:
    def test_rejects_one_dimensional_input(self):
        with pytest.raises(ValueError, match=r"\(T, N\)"):
            active_neuron_fraction(np.zeros(5, dtype=bool))

    def test_rejects_three_dimensional_input(self):
        with pytest.raises(ValueError, match=r"\(T, N\)"):
            active_neuron_fraction(np.zeros((2, 3, 4), dtype=bool))

    def test_error_message_explains_the_single_timestep_case(self):
        with pytest.raises(ValueError, match="只有一个时间步"):
            spike_density(np.zeros(7, dtype=bool))


class TestSummarize:
    def test_reports_every_field_the_record_needs(self):
        spikes = np.zeros((10, 4), dtype=bool)
        spikes[:, :3] = True  # 3 个神经元每步都发放，1 个从不发放
        stats = summarize(spikes)

        assert stats["n_steps"] == 10
        assert stats["n_neurons"] == 4
        assert stats["active_neurons"] == 3
        assert stats["active_neuron_fraction"] == pytest.approx(0.75)
        assert stats["spike_density"] == pytest.approx(0.75)
        assert stats["spike_sparsity"] == pytest.approx(0.25)

    def test_threshold_matches_the_project_plan(self):
        """§3.1：活跃神经元比例低于 60% 触发阈值调整。这个数不该被随手改掉。"""
        assert DEAD_NEURON_THRESHOLD == 0.60

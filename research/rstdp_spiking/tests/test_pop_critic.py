"""群体 Critic 的测试：活动范围、值的组成、行归一化、TD 误差的终止处理。"""

from __future__ import annotations

import torch

from research.rstdp_spiking.pop_critic import PopulationCritic, td_error


class TestTDError:
    def test_terminates_without_bootstrap(self):
        assert td_error(1.0, 0.5, 99.0, terminated=True, gamma=0.9) == 0.5

    def test_bootstraps_otherwise(self):
        assert abs(td_error(1.0, 0.5, 2.0, terminated=False, gamma=0.5) - 1.5) < 1e-9


class TestPopulationCritic:
    def test_rates_are_bounded(self):
        critic = PopulationCritic(6, n_units=8, generator=torch.Generator().manual_seed(0))
        y = critic.rates(torch.ones(6))
        assert float(y.min()) > 0.0 and float(y.max()) < 1.0

    def test_value_is_readout_dot_rates_plus_bias(self):
        critic = PopulationCritic(4, n_units=5, generator=torch.Generator().manual_seed(0))
        x = torch.tensor([0.1, 0.2, 0.3, 0.4])
        expected = float(torch.dot(critic.readout, critic.rates(x)) + critic.out_bias)
        assert abs(critic.value(x) - expected) < 1e-6

    def test_rows_stay_normalised_after_update(self):
        critic = PopulationCritic(
            4, n_units=6, learning_rate=1.0, generator=torch.Generator().manual_seed(0)
        )
        x = torch.tensor([0.5, -0.5, 0.5, -0.5])
        critic.update(x, 1.0, x, terminated=True)
        norms = critic.weights.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_trace_resets_but_weights_do_not(self):
        critic = PopulationCritic(
            4, n_units=6, learning_rate=1.0, generator=torch.Generator().manual_seed(0)
        )
        x = torch.tensor([1.0, 0.0, 0.0, 0.0])
        critic.update(x, 1.0, x, terminated=True)
        before = critic.weights.clone()
        critic.reset_trace()
        assert float(critic.trace.abs().sum()) == 0.0
        assert torch.equal(critic.weights, before)


class TestAblationSwitches:
    """两处「速率版有、精简移植漏了」的机制：gradient 后因子与有符号读出。"""

    def test_gradient_post_factor_uses_the_sigmoid_derivative(self):
        critic = PopulationCritic(
            4, n_units=6, trace_post_factor="gradient", generator=torch.Generator().manual_seed(0)
        )
        y = critic.rates(torch.ones(4))
        assert torch.allclose(critic._post_factor(y), y * (1 - y))

    def test_rate_post_factor_passes_the_rate_through(self):
        critic = PopulationCritic(
            4, n_units=6, trace_post_factor="rate", generator=torch.Generator().manual_seed(0)
        )
        y = critic.rates(torch.ones(4))
        assert torch.allclose(critic._post_factor(y), y)

    def test_signed_readout_requires_gradient(self):
        """与速率版同一条约束：有符号读出配 rate 后因子会让一半单元的更新方向反过来。"""
        import pytest

        with pytest.raises(ValueError, match="gradient"):
            PopulationCritic(
                4,
                n_units=6,
                readout="random",
                trace_post_factor="rate",
                generator=torch.Generator().manual_seed(0),
            )

    def test_signed_readout_has_both_signs(self):
        critic = PopulationCritic(
            4,
            n_units=16,
            readout="random",
            trace_post_factor="gradient",
            generator=torch.Generator().manual_seed(0),
        )
        assert float(critic.readout.min()) < 0 < float(critic.readout.max())

    def test_signed_readout_can_compute_a_value(self):
        """回归：读出曾是 float64 而活动是 float32，torch.dot 直接报 dtype 错。

        这条断言的价值就在"真的算一次值"——只检查符号的测试放过了那个 bug。
        """
        critic = PopulationCritic(
            4,
            n_units=8,
            readout="random",
            trace_post_factor="gradient",
            generator=torch.Generator().manual_seed(0),
        )
        value = critic.value(torch.ones(4))
        assert value == value  # not NaN

    def test_uniform_readout_is_positive(self):
        critic = PopulationCritic(4, n_units=8, generator=torch.Generator().manual_seed(0))
        assert float(critic.readout.min()) > 0

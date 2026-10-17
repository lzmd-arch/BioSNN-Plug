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

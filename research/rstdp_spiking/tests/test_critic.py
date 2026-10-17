"""线性 TD(0) Critic 的测试：δ 的定义、终止状态的引导、权重方向。"""

from __future__ import annotations

import torch

from research.rstdp_spiking.critic import LinearCritic


class TestLinearCritic:
    def test_zero_weights_value_is_zero(self):
        critic = LinearCritic(3)
        assert critic.value(torch.tensor([1.0, 2.0, 3.0])) == 0.0

    def test_delta_without_bootstrap_at_termination(self):
        """

        终止状态不加 gamma*V(s')——加进去会让最后一格永远学不到负值。
        """
        critic = LinearCritic(2, gamma=0.9, learning_rate=0.1)
        x = torch.tensor([1.0, 0.0])
        delta = critic.update(x, 1.0, torch.tensor([0.0, 1.0]), terminated=True)
        assert delta == 1.0

    def test_delta_bootstraps_when_not_terminated(self):
        critic = LinearCritic(2, gamma=0.5, learning_rate=0.1)
        critic.weights = torch.tensor([1.0, 0.0])
        delta = critic.update(
            torch.tensor([1.0, 0.0]), 0.0, torch.tensor([0.0, 2.0]), terminated=False
        )
        # V(s)=1, V(s')=0, gamma*V(s')=0 → δ = -1
        assert abs(delta + 1.0) < 1e-6

    def test_weights_move_toward_positive_delta(self):
        critic = LinearCritic(2, gamma=0.9, learning_rate=0.5)
        x = torch.tensor([1.0, 0.0])
        critic.update(x, 1.0, torch.zeros(2), terminated=True)
        assert critic.weights[0] > 0.0

    def test_rejects_bad_gamma(self):
        import pytest

        with pytest.raises(ValueError, match="gamma"):
            LinearCritic(2, gamma=1.5)

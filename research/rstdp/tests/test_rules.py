"""``research.rstdp.td_ltp`` 与 ``research.rstdp.rstdp`` 的测试。

重心：两条规则的三因子结构（资格痕迹 × 全局标量）、权重归一化的不变量、
以及"探索步不该强化 Actor"这条容易写错的接线。
"""

from __future__ import annotations

import pytest
import torch

from research.rstdp.rstdp import RSTDPActor
from research.rstdp.td_ltp import TDLCritic, td_error


def make_features(n: int = 5, seed: int = 0) -> torch.Tensor:
    return torch.rand(n, generator=torch.Generator().manual_seed(seed))


class TestTDLCritic:
    def test_value_is_a_scaled_linear_readout(self):
        critic = TDLCritic(4, value_scale=2.0, weight_norm=None)
        with torch.no_grad():
            critic.weights.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        assert critic.value(torch.tensor([1.0, 1.0, 1.0, 1.0])).item() == pytest.approx(20.0)

    def test_weight_norm_is_an_invariant(self):
        """计划书 §3.2：权重归一化防止突触动态失控。不归一化时实测权重会发散成 NaN。"""
        critic = TDLCritic(6, learning_rate=1.0, weight_norm=2.0)
        features = make_features(6)
        for step in range(30):
            critic.update(features, critic.value(features), td_error=1.0 if step % 3 else -1.0)
        assert critic.weights.norm().item() == pytest.approx(2.0, rel=1e-5)

    def test_unnormalised_weights_can_run_away(self):
        """这条是那个反馈回路的回归测试：``e = x·V`` 且 ``V = w·x``，不归一化就发散。"""
        critic = TDLCritic(6, learning_rate=1.0, weight_norm=None)
        features = make_features(6)
        for _ in range(60):
            critic.update(features, critic.value(features), td_error=1.0)
        assert critic.weights.abs().max().item() > 1e3, "不归一化时应当爆炸，否则这条测试失去意义"

    def test_trace_accumulates_pre_times_value(self):
        """``e ← λ·e + x·V``，V 是后突触活动。"""
        critic = TDLCritic(3, trace_decay=0.5)
        with torch.no_grad():
            critic.weights.copy_(torch.tensor([1.0, 0.0, 0.0]))
        features = torch.tensor([2.0, 0.0, 0.0])
        value = critic.value(features)  # = 2.0
        critic.update(features, value, td_error=0.0)
        # e = 0.5·0 + 2·2 = 4
        assert critic.trace[0].item() == pytest.approx(4.0)

    def test_weight_update_is_learning_rate_times_delta_times_trace(self):
        """三因子：``Δw = η·δ·e``——δ 是全局标量，乘在痕迹上。

        **关掉权重归一化**：归一化会重标定权重，把要验的那条等式盖掉。归一化本身由
        :meth:`test_weight_norm_is_an_invariant` 单独验。
        """
        critic = TDLCritic(3, learning_rate=0.1, trace_decay=0.0, weight_norm=None, value_scale=1.0)
        with torch.no_grad():
            critic.weights.copy_(torch.tensor([1.0, 0.0, 0.0]))
        features = torch.tensor([2.0, 0.0, 0.0])
        value = critic.value(features)
        before = critic.weights.clone()
        critic.update(features, value, td_error=0.5)
        # trace = 2·2 = 4；Δw = 0.1·0.5·4 = 0.2
        assert (critic.weights - before)[0].item() == pytest.approx(0.2)

    def test_zero_initialisation_would_lock_the_critic(self):
        """**零点锁死**：把权重压成 0，痕迹与更新量都会恒为 0，Critic 永远学不动。

        痕迹是 ``e ← λ·e + x·V``，被 Critic 自己的输出门控。这条测试是那个真 bug 的
        回归测试——当初用零初始化，critic 在 CartPole 上完全没学。
        """
        critic = TDLCritic(5, learning_rate=1.0, weight_norm=None)
        with torch.no_grad():
            critic.weights.zero_()
        features = make_features(5)
        for _ in range(10):
            critic.update(features, critic.value(features), td_error=1.0)
        assert critic.trace.abs().sum().item() == 0.0
        assert critic.weights.abs().sum().item() == 0.0

    def test_default_initialisation_is_nonzero(self):
        critic = TDLCritic(5)
        assert critic.weights.abs().sum().item() > 0

    def test_a_global_scalar_scales_every_synapse_alike(self):
        """第三因子对所有突触相同——这正是"非局部量只有一个标量"的可测形式。"""
        torch.manual_seed(0)
        critic = TDLCritic(5, learning_rate=1.0, trace_decay=0.9, weight_norm=None, value_scale=1.0)
        features = make_features(5)
        value = critic.value(features)
        critic.update(features, value, td_error=0.0)  # 先把痕迹建立起来
        before = critic.weights.clone()
        critic.update(features, value, td_error=1.0)
        change = critic.weights - before
        # Δw = η·δ·e → 每个分量的变化都正比于 trace，比例常数相同
        ratio = change / critic.trace
        nonzero = critic.trace.abs() > 1e-9
        assert torch.allclose(ratio[nonzero], torch.full_like(ratio[nonzero], ratio[nonzero][0]))

    def test_zero_delta_leaves_weights_untouched(self):
        """δ = 0（预测准确）时不该有任何更新——这是 Critic 收敛的标志。"""
        critic = TDLCritic(4, learning_rate=1.0)
        features = make_features(4)
        before = critic.weights.clone()
        critic.update(features, critic.value(features), td_error=0.0)
        torch.testing.assert_close(critic.weights, before)

    def test_reset_trace_zeroes_it(self):
        critic = TDLCritic(3)
        critic.update(make_features(3), torch.tensor(1.0), td_error=1.0)
        assert critic.trace.abs().sum().item() > 0
        critic.reset_trace()
        assert critic.trace.abs().sum().item() == 0

    def test_rejects_a_bad_trace_decay(self):
        with pytest.raises(ValueError, match="trace_decay"):
            TDLCritic(3, trace_decay=1.0)


class TestTDError:
    def test_matches_the_definition(self):
        """δ = r + γ·V(s') − V(s)。"""
        delta = td_error(1.0, torch.tensor(0.5), torch.tensor(0.7), discount=0.9)
        assert delta == pytest.approx(1.0 + 0.9 * 0.7 - 0.5)

    def test_terminal_next_value_of_zero(self):
        assert td_error(1.0, torch.tensor(0.5), torch.tensor(0.0), discount=0.99) == pytest.approx(
            0.5
        )

    def test_discount_of_zero_ignores_the_next_value(self):
        a = td_error(1.0, torch.tensor(0.5), torch.tensor(9.0), discount=0.0)
        b = td_error(1.0, torch.tensor(0.5), torch.tensor(0.0), discount=0.0)
        assert a == pytest.approx(b)


class TestRSTDPActor:
    def test_trace_uses_the_chosen_action_indicator(self):
        """后突触活动取被选中动作的指示量：只有那一列的痕迹在长。"""
        actor = RSTDPActor(3, 2, trace_decay=0.0, normalize=False)
        actor.update(torch.tensor([1.0, 2.0, 3.0]), action=1, success_signal=0.0)
        assert actor.trace[:, 0].abs().sum().item() == 0
        assert torch.allclose(actor.trace[:, 1], torch.tensor([1.0, 2.0, 3.0]))

    def test_weight_update_is_success_signal_times_trace(self):
        actor = RSTDPActor(3, 2, learning_rate=0.5, trace_decay=0.0, normalize=False)
        with torch.no_grad():
            actor.weights.zero_()
        actor.update(torch.tensor([2.0, 0.0, 0.0]), action=0, success_signal=1.0)
        # trace[:,0] = [2,0,0]；Δw = 0.5·1·[2,0,0]
        assert actor.weights[0, 0].item() == pytest.approx(1.0)

    def test_negative_success_signal_weakens_the_synapse(self):
        actor = RSTDPActor(3, 2, learning_rate=0.5, trace_decay=0.0, normalize=False)
        with torch.no_grad():
            actor.weights.zero_()
        actor.update(torch.tensor([2.0, 0.0, 0.0]), action=0, success_signal=-1.0)
        assert actor.weights[0, 0].item() == pytest.approx(-1.0)

    def test_l1_normalisation_is_an_invariant(self):
        """每个动作单元的入权重 L1 范数恒为 ``weight_norm``——这是"权重总和恒定"。"""
        actor = RSTDPActor(6, 2, learning_rate=1.0, weight_norm=2.0)
        features = make_features(6)
        for step in range(20):
            actor.update(features, action=step % 2, success_signal=1.0 if step % 3 else -1.0)
            column_sums = actor.weights.abs().sum(dim=0)
            torch.testing.assert_close(
                column_sums, torch.full_like(column_sums, 2.0), rtol=1e-5, atol=1e-6
            )

    def test_normalisation_can_be_switched_off(self):
        actor = RSTDPActor(6, 2, learning_rate=1.0, normalize=False)
        before = actor.weights.abs().sum(dim=0).clone()
        for _ in range(20):
            actor.update(make_features(6), action=0, success_signal=1.0)
        assert not torch.allclose(before, actor.weights.abs().sum(dim=0))

    def test_select_action_returns_the_argmax_without_exploration(self):
        actor = RSTDPActor(3, 2, normalize=False)
        with torch.no_grad():
            actor.weights.copy_(torch.tensor([[1.0, -1.0], [1.0, -1.0], [1.0, -1.0]]))
        generator = torch.Generator().manual_seed(0)
        action, explored = actor.select_action(
            torch.tensor([1.0, 1.0, 1.0]), generator=generator, return_explored=True
        )
        assert action == 0
        assert explored is False

    def test_select_action_reports_exploration(self):
        """``explored`` 必须由选择处报出——外面无法复现那条随机数。"""
        actor = RSTDPActor(3, 2, normalize=False)
        generator = torch.Generator().manual_seed(0)
        explored_count = sum(
            actor.select_action(
                torch.tensor([1.0, 1.0, 1.0]),
                generator=generator,
                exploration=1.0,
                return_explored=True,
            )[1]
            for _ in range(20)
        )
        assert explored_count == 20, "exploration=1.0 时每一步都该是探索步"

    def test_full_exploration_never_uses_the_argmax(self):
        actor = RSTDPActor(3, 2, normalize=False)
        with torch.no_grad():
            actor.weights.copy_(torch.tensor([[10.0, -10.0]] * 3))
        generator = torch.Generator().manual_seed(1)
        actions = {
            actor.select_action(torch.tensor([1.0, 1.0, 1.0]), generator=generator, exploration=1.0)
            for _ in range(30)
        }
        assert actions == {0, 1}, "全探索时两个动作都该被选到"

    def test_reset_trace_zeroes_it(self):
        actor = RSTDPActor(3, 2)
        actor.update(make_features(3), action=0, success_signal=1.0)
        assert actor.trace.abs().sum().item() > 0
        actor.reset_trace()
        assert actor.trace.abs().sum().item() == 0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [({"trace_decay": 1.0}, "trace_decay"), ({"weight_norm": 0.0}, "weight_norm")],
    )
    def test_rejects_invalid_hyperparameters(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            RSTDPActor(3, 2, **kwargs)

"""``research.rstdp.td_ltp`` 与 ``research.rstdp.rstdp`` 的测试。

重心：两条规则的三因子结构（资格痕迹 × 全局标量）、权重归一化的不变量、
以及"探索步不该强化 Actor"这条容易写错的接线。
"""

from __future__ import annotations

import pytest
import torch

from research.rstdp.rstdp import RSTDPActor
from research.rstdp.td_ltp import PopulationCritic, TDLCritic, td_error


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


def _bump(generator: torch.Generator, n: int = 64, k: int = 5) -> torch.Tensor:
    """模拟真实的状态编码：少数几个局域凸起，而不是稠密的随机向量。"""
    vector = torch.zeros(n)
    index = torch.randint(0, n, (k,), generator=generator)
    vector[index] = torch.rand(k, generator=generator)
    return vector


class TestPopulationCritic:
    """群体 + 固定读出的 Critic（论文结构）。

    最要紧的三条：
    :meth:`test_does_not_diverge_at_a_large_learning_rate`、
    :meth:`test_rows_stay_normalised_so_the_readout_owns_the_scale`、
    :meth:`test_has_no_zero_lock`——它们是单单元版那两个病（正值反馈发散、归一化夺走尺度
    自由度）与零点锁死的回归测试。
    """

    @staticmethod
    def _features(n: int = 5, seed: int = 0) -> torch.Tensor:
        return torch.rand(n, generator=torch.Generator().manual_seed(seed))

    def test_rates_are_bounded_and_non_negative(self):
        critic = PopulationCritic(5, n_units=7)
        rates = critic.rates(self._features())
        assert rates.shape == (7,)
        assert (rates > 0).all() and (rates < 1).all()

    def test_value_is_the_fixed_readout_of_the_rates(self):
        critic = PopulationCritic(5, n_units=7, value_scale=14.0)
        features = self._features()
        expected = float(critic.readout @ critic.rates(features))
        assert critic.value(features).item() == pytest.approx(expected)
        assert critic.readout.sum().item() == pytest.approx(14.0)

    def test_readout_is_fixed(self):
        """读出是**固定**的——它不参与学习，也就不会把尺度问题带回给 Critic。"""
        critic = PopulationCritic(5, n_units=7, value_scale=14.0)
        before = critic.readout.clone()
        for _ in range(10):
            critic.update(self._features(), critic.value(self._features()), td_error=1.0)
        torch.testing.assert_close(critic.readout, before)

    def test_trace_uses_the_units_own_rate_not_the_value(self):
        """关键结构差别：第二因子是**单元自己的发放率**，不是值。

        单单元版把两者当成同一个量，于是 ``e = x·V`` 与 ``V = w·x`` 构成正值反馈。
        这里 ``y_j ∈ (0,1)`` 有界，回路断开。
        """
        critic = PopulationCritic(3, n_units=2, trace_decay=0.0)
        with torch.no_grad():
            critic.weights.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        features = torch.tensor([1.0, 2.0, 0.0])
        critic.update(features, critic.value(features), td_error=0.0)
        rates = critic.rates(features)
        torch.testing.assert_close(critic.trace[0], rates[0] * features)
        torch.testing.assert_close(critic.trace[1], rates[1] * features)

    def test_rows_stay_normalised_so_the_readout_owns_the_scale(self):
        """每个单元的权重行恒为单位 L2 范数——尺度由固定读出承担，不由权重承担。"""
        critic = PopulationCritic(6, n_units=5, learning_rate=1.0)
        features = self._features(6)
        for step in range(40):
            critic.update(features, critic.value(features), td_error=1.0 if step % 3 else -1.0)
        norms = critic.weights.norm(dim=1)
        torch.testing.assert_close(norms, torch.ones(5), rtol=1e-5, atol=1e-6)

    def test_does_not_diverge_at_a_large_learning_rate(self):
        """回归测试：单单元版在这个学习率下权重会发散成 NaN。

        原因是 ``e = x·V`` 且 ``V = w·x`` 的正值反馈。群体版的第二因子有界，所以不发散。
        """
        critic = PopulationCritic(8, n_units=8, learning_rate=1.0)
        features = self._features(8)
        for _ in range(200):
            critic.update(features, critic.value(features), td_error=1.0)
        assert torch.isfinite(critic.weights).all(), "群体版不该发散"
        assert critic.weights.abs().max().item() <= 1.0 + 1e-5, "归一化后每行范数为 1"

    def test_has_no_zero_lock(self):
        """零锁死的回归测试：sigmoid 在 0 处是 0.5，不是 0，所以痕迹不会恒为零。"""
        critic = PopulationCritic(5, n_units=4, learning_rate=1.0)
        features = self._features()
        assert (critic.rates(features) > 0).all()
        critic.update(features, critic.value(features), td_error=1.0)
        assert critic.trace.abs().sum().item() > 0

    def test_init_directions_are_honoured(self):
        """传入的偏好方向必须被真的用上（归一化后存为权重）。"""
        generator = torch.Generator().manual_seed(1)
        directions = torch.stack([_bump(generator) for _ in range(4)])
        critic = PopulationCritic(64, n_units=4, init_directions=directions)
        expected = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
        torch.testing.assert_close(critic.weights, expected, rtol=1e-5, atol=1e-7)

    def test_rejects_mismatched_init_directions(self):
        with pytest.raises(ValueError, match="init_directions"):
            PopulationCritic(8, n_units=4, init_directions=torch.rand(3, 8))

    def test_reset_trace_zeroes_it(self):
        critic = PopulationCritic(3, n_units=2)
        critic.update(self._features(3), critic.value(self._features(3)), td_error=1.0)
        assert critic.trace.abs().sum().item() > 0
        critic.reset_trace()
        assert critic.trace.abs().sum().item() == 0

    def test_rates_are_scale_invariant(self):
        """发放率只看活动的**方向**，不看模长——否则固定的 ``bias`` 在不同状态下含义不同。

        实测过：不做归一化时 ``V`` 恒等于其上限（所有单元饱和）。这条把那个失败模式钉住。
        """
        critic = PopulationCritic(6, n_units=4, bias=0.5, gain=8.0)
        features = self._features(6)
        scaled = features * 25.0
        torch.testing.assert_close(
            critic.rates(features), critic.rates(scaled), rtol=1e-5, atol=1e-7
        )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"trace_decay": 1.0}, "trace_decay"),
            ({"n_units": 0}, "n_units"),
            ({"gain": 0.0}, "gain"),
            ({"value_scale": -1.0}, "value_scale"),
        ],
    )
    def test_rejects_invalid_hyperparameters(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            PopulationCritic(4, **kwargs)


class TestCliDefaultsMatchTheClass:
    """回归测试：CLI 默认值必须与 :class:`CartPoleAgent` 的构造默认值一致。

    起因是一个真实踩到的坑：类的 ``actor_learning_rate`` 默认 1e-2、
    ``critic_value_scale`` 默认 60.0，而 argparse 里写的是 3e-3 与 200.0。
    直接构造 agent 的人会**静默**拿到与 CLI 不同的行为——两个独立的消融臂第一版脚本
    都因此对不上基线。两份默认值就一定会漂移，所以现在只有一份（类的签名）。
    """

    def test_every_shared_parameter_agrees(self):
        from research.rstdp.cartpole import CLI_TO_AGENT_PARAM, agent_default, build_parser

        parser = build_parser()
        cli = {action.dest: action.default for action in parser._actions}

        mismatched = []
        for cli_name, agent_param in CLI_TO_AGENT_PARAM.items():
            if cli[cli_name] != agent_default(agent_param):
                mismatched.append(
                    f"{cli_name}: CLI={cli[cli_name]!r} vs 类={agent_default(agent_param)!r}"
                )
        assert not mismatched, "CLI 与类默认值不一致：" + "；".join(mismatched)

    def test_the_two_parameters_that_actually_drifted(self):
        """把当初漂移的那两个钉死——它们是这条测试存在的理由。"""
        from research.rstdp.cartpole import build_parser

        parser = build_parser()
        cli = {action.dest: action.default for action in parser._actions}
        assert cli["actor_learning_rate"] == pytest.approx(3e-3)
        assert cli["critic_value_scale"] == pytest.approx(200.0)

    def test_actor_normalize_is_reachable_from_the_cli(self):
        """归一化此前**没有 CLI 出口**，消融只能另写脚本——现在能直接开关。"""
        from research.rstdp.cartpole import build_parser

        parser = build_parser()
        assert parser.parse_args([]).actor_normalize is True
        assert parser.parse_args(["--no-actor-normalize"]).actor_normalize is False

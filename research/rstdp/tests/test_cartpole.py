"""``research.rstdp.cartpole`` 的测试。

重心是**集成处**：终止与截断的自举、状态编码，以及新旋钮真的传到了 agent。
这些是单元测试覆盖不到的地方——规则本身在 ``test_rules.py`` 里验。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from research.rstdp.cartpole import CartPoleAgent, encode_state, run_episode


def _agent(n_features: int = 4, seed: int = 0) -> CartPoleAgent:
    """两次调用得到**逐位相同**的 agent——测试要拿一份没被更新过的做参照。"""
    return CartPoleAgent(n_features, critic_units=3, generator=torch.Generator().manual_seed(seed))


class TestEncodeState:
    def test_is_a_gaussian_bump_centred_on_the_closest_centre(self):
        centers = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        features = encode_state(torch.zeros(4), centers, 0.5)
        assert features[0].item() == pytest.approx(1.0)
        assert features[1].item() == pytest.approx(0.13533528, rel=1e-6)

    def test_closer_centres_fire_harder(self):
        centers = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
        features = encode_state(torch.zeros(4), centers, 0.5)
        assert features[0] > features[1] > features[2]

    def test_does_not_normalise(self):
        """远离所有中心时活动就该趋近于 0——归一化会抹掉这条信息。

        这正是模块 docstring 里那句「**不做归一化**」的可测形式：归一化之后
        「离所有中心都远」和「贴着一个中心」会得到同一个向量。
        """
        features = encode_state(torch.full((4,), 10.0), torch.zeros(3, 4), 0.5)
        assert features.abs().max().item() < 1e-50

    def test_accepts_a_batch_of_states(self):
        centers = torch.zeros(3, 4)
        batch = torch.zeros(2, 4)
        assert encode_state(batch, centers, 0.5).shape == (2, 3)


class TestTerminationVersusTruncation:
    """回归测试：终止与截断的自举必须不同。

    起因是一个真实踩到的坑：``run_episode`` 过去传的是 ``terminated or truncated``，
    于是撞上 ``EPISODE_LIMIT`` 的回合被当成「终止」、``V(s')`` 被置零，最后一步拿到
    ``δ = 1 − V ≈ −98``。**策略越好，这一下打得越狠**，而实测到的「峰值 500 步 → 崩到
    14 步」正是这个形状。
    """

    @staticmethod
    def _features() -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor([1.0, 0.5, 0.25, 0.0]), torch.tensor([0.0, 1.0, 0.5, 0.25])

    def test_termination_zeroes_the_bootstrap(self):
        features, next_features = self._features()
        value = float(_agent().critic.value(features))
        delta, _ = _agent().learn_step(features, 0, 1.0, next_features, terminated=True)
        assert delta == pytest.approx(1.0 - value)

    def test_truncation_bootstraps_off_the_next_state(self):
        features, next_features = self._features()
        reference = _agent()
        value = float(reference.critic.value(features))
        next_value = float(reference.critic.value(next_features))
        delta, _ = _agent().learn_step(
            features, 0, 1.0, next_features, terminated=False, truncated=True
        )
        assert delta == pytest.approx(1.0 + 0.99 * next_value - value)
        assert delta > 1.0 - value, "截断自举后 δ 应当明显高于置零那条路径"

    def test_a_terminal_state_after_truncation_still_zeroes(self):
        """两个标志同时为真时以**终止**为准——否则真结束的回合会漏掉自举截断。"""
        features, next_features = self._features()
        value = float(_agent().critic.value(features))
        delta, _ = _agent().learn_step(
            features, 0, 1.0, next_features, terminated=True, truncated=True
        )
        assert delta == pytest.approx(1.0 - value)


class _StubEnv:
    """只实现 ``run_episode`` 用到的两个方法，好把终止/截断的时机捏在手里。

    用桩而不是真的 ``gymnasium``：真的 CartPole 要跑满 500 步才会截断，这个测试
    不该花那个时间，也不该依赖 gymnasium 的版本行为。
    """

    def __init__(self, *, done_after: int, truncated: bool) -> None:
        self.done_after = done_after
        self.truncated = truncated
        self.steps = 0

    def reset(self):
        self.steps = 0
        return [0.1, 0.0, 0.05, 0.0], {}

    def step(self, action: int):
        self.steps += 1
        done = self.steps >= self.done_after
        return (
            [0.1, 0.0, 0.05, 0.0],
            1.0,
            done and not self.truncated,
            done and self.truncated,
            {},
        )


class _RecordingAgent(CartPoleAgent):
    """记下 ``learn_step`` 收到的标志——集成处写错的话，这里看得见。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[dict[str, bool]] = []

    def learn_step(self, features, action, reward, next_features, **kwargs):
        self.calls.append(dict(kwargs))
        return super().learn_step(features, action, reward, next_features, **kwargs)


class TestRunEpisodeWiring:
    @pytest.mark.parametrize(
        ("truncated", "expected"),
        [
            (False, {"terminated": True, "truncated": False}),
            (True, {"terminated": False, "truncated": True}),
        ],
    )
    def test_the_last_step_reports_the_right_flag(self, truncated, expected):
        env = _StubEnv(done_after=3, truncated=truncated)
        agent = _RecordingAgent(4, critic_units=2, generator=torch.Generator().manual_seed(0))
        steps, _ = run_episode(
            env,
            agent,
            torch.zeros(4, 4),
            0.5,
            generator=torch.Generator().manual_seed(0),
            exploration=0.0,
            learn=True,
        )
        assert steps == 3
        assert len(agent.calls) == 3
        assert agent.calls[-1] == {**expected, "exploring": False}
        assert all(not call["terminated"] and not call["truncated"] for call in agent.calls[:-1]), (
            "回合中间的步两个标志都该是假"
        )

    def test_no_learning_means_no_updates(self):
        agent = _RecordingAgent(4, critic_units=2, generator=torch.Generator().manual_seed(0))
        run_episode(
            _StubEnv(done_after=3, truncated=False),
            agent,
            torch.zeros(4, 4),
            0.5,
            generator=torch.Generator().manual_seed(0),
            exploration=0.0,
            learn=False,
        )
        assert agent.calls == []


class TestNewOptionsReachTheCritic:
    """CLI/`run_trial` 的新旋钮必须真的传到 Critic 上——不然扫描扫的是空气。"""

    def test_critic_output_bias_defaults_to_zero(self):
        """默认 0 时 ``value()`` 逐位不变——新选项不许改变既有数字。"""
        agent = _agent()
        assert agent.critic.out_bias.item() == 0.0
        assert agent.critic.bias_learning_rate is None

    def test_output_bias_is_applied_to_the_value(self):
        reference = _agent()
        shifted = CartPoleAgent(
            4,
            critic_units=3,
            critic_output_bias=-30.0,
            generator=torch.Generator().manual_seed(0),
        )
        features = torch.tensor([1.0, 0.5, 0.25, 0.0])
        assert shifted.critic.value(features).item() == pytest.approx(
            reference.critic.value(features).item() - 30.0
        )

    def test_bias_learning_rate_moves_the_bias_during_a_step(self):
        agent = CartPoleAgent(
            4,
            critic_units=3,
            critic_bias_learning_rate=0.5,
            generator=torch.Generator().manual_seed(0),
        )
        before = agent.critic.out_bias.item()
        agent.learn_step(
            torch.tensor([1.0, 0.5, 0.25, 0.0]),
            0,
            1.0,
            torch.tensor([0.0, 1.0, 0.5, 0.25]),
            terminated=True,
        )
        assert agent.critic.out_bias.item() != before

    def test_the_single_unit_critic_also_takes_the_bias(self):
        """两个类的签名要保持一致——诊断与 CLI 会按同一套参数名传值。"""
        agent = CartPoleAgent(
            4,
            critic_kind="single",
            critic_output_bias=-30.0,
            generator=torch.Generator().manual_seed(0),
        )
        assert agent.critic.out_bias.item() == pytest.approx(-30.0)


class TestGreedyScoreIsActuallyGreedy:
    """``_greedy_score`` 在 boltzmann 下必须真的返回 argmax。

    回归测试：``select_action`` 的 boltzmann 分支不看 ``exploration``，所以只传
    ``exploration=0.0`` 会得到一个**随机**策略——而那个函数同时供途中曲线与末次验收使用。
    """

    @staticmethod
    def _agent() -> CartPoleAgent:
        agent = CartPoleAgent(
            4,
            critic_units=3,
            actor_action_sampling="boltzmann",
            actor_logit_scale=0.5,
            generator=torch.Generator().manual_seed(0),
        )
        return agent

    def test_the_evaluation_path_passes_greedy_true(self):
        """``_greedy_score`` **必须**把 ``greedy=True`` 传下去——这才是那个缺陷的接线处。

        不去比两次评测的结果相不相等：``run_episode`` 每次都 ``env.reset()``，初始状态来自
        环境自己的随机流，所以两次评测本来就会不同——那个断言测不出想测的东西（第一版就是
        这么写的，它失败得毫无信息）。
        """
        gymnasium = pytest.importorskip("gymnasium")
        from research.rstdp.cartpole import _greedy_score

        class _Recorder(CartPoleAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.seen: list[bool] = []

            def behave(self, features, **kwargs):
                self.seen.append(bool(kwargs.get("greedy", False)))
                return super().behave(features, **kwargs)

        env = gymnasium.make("CartPole-v1")
        agent = _Recorder(
            4,
            critic_units=3,
            actor_action_sampling="boltzmann",
            actor_logit_scale=0.5,
            generator=torch.Generator().manual_seed(0),
        )
        _greedy_score(env, agent, torch.zeros(4, 4), 0.5, torch.Generator().manual_seed(0), 3)
        env.close()
        assert agent.seen, "一局都没跑起来"
        assert all(agent.seen), "评测路径必须每一步都传 greedy=True"

    def test_an_untrained_boltzmann_agent_does_not_get_lucky(self):
        """未训练的 boltzmann 智能体应当接近随机策略的 ~9–10 步，而不是偶然更高。"""
        pytest.importorskip("gymnasium")
        from research.rstdp.cartpole import _greedy_score

        gymnasium = pytest.importorskip("gymnasium")
        env = gymnasium.make("CartPole-v1")
        score = _greedy_score(
            env, self._agent(), torch.zeros(4, 4), 0.5, torch.Generator().manual_seed(0), 10
        )
        env.close()
        assert score < 40.0, "贪心评测下未训练的策略不该有高分；分数高说明还在采样"


class TestLogitScaleAnnealing:
    """Boltzmann 逆温度的退火：``logit_scale`` 应当在训练过程中从起点走到终点。

    动机是实测出来的双峰：恒定 logit 的两端各丢一半——近均匀（logit 5）学得起来但钝
    （中位数 141.4），较尖锐（logit 20）上限高（留出集 177.3、最大 434）但**约十分之一的
    种子从头到尾学不起来**（峰值只有 10–12 步）。退火先钝后尖，两头都要。
    """

    @staticmethod
    def _run(**kwargs):
        pytest.importorskip("gymnasium")
        from research.rstdp.cartpole import run_trial

        _, context = run_trial(
            0,
            episodes=kwargs.pop("episodes", 6),
            n_features=16,
            actor_action_sampling="boltzmann",
            actor_logit_scale=1.0,
            actor_trace_center="sampling",
            device=torch.device("cpu"),
            return_context=True,
            **kwargs,
        )
        return context.agent.actor.logit_scale

    def test_no_schedule_leaves_the_constant_alone(self):
        """**两端都显式给 ``None``** 才是不退火——默认现在是退火（ADR-0009）。"""
        assert self._run(logit_scale_start=None, logit_scale_end=None) == pytest.approx(1.0)

    def test_the_schedule_ends_at_the_end_value(self):
        """最后一个回合结束后应当落在 ``logit_scale_end`` 上（线性 progress 走到 1）。"""
        assert self._run(logit_scale_start=2.0, logit_scale_end=20.0) == pytest.approx(20.0)

    def test_the_schedule_interpolates_in_log_space(self):
        """中途的值应当落在**几何**插值上，而不是算术插值。

        ``logit_scale`` 是温度的倒数，线性插值它等于对温度做双曲退火（前期过慢、后期过快）。
        **必须用 ``observer`` 在训练途中取**：循环最后一轮会把 progress 推到 1.0，
        跑完之后读到的永远是终点值。
        """
        import math

        from research.rstdp.cartpole import run_trial

        pytest.importorskip("gymnasium")
        seen: dict[int, float] = {}

        def observer(episode, agent, centers):
            seen[episode] = agent.actor.logit_scale

        run_trial(
            0,
            episodes=5,
            n_features=16,
            actor_action_sampling="boltzmann",
            actor_logit_scale=1.0,
            actor_trace_center="sampling",
            logit_scale_start=1.0,
            logit_scale_end=100.0,
            observer=observer,
            observer_every=4,
            device=torch.device("cpu"),
        )
        assert 4 in seen, "observer 没被调用"
        assert seen[4] == pytest.approx(math.exp(0.8 * math.log(100.0)), rel=1e-6)
        assert seen[4] < 50.0, "几何插值应当明显低于算术中点 50.5"


class TestStructuralOptionsReachTheActor:
    """结构性那一组选项必须真的传到 Actor 上。"""

    def test_defaults_are_the_accepted_configuration(self):
        """默认值就是 ADR-0009 采纳的合格配置——**不再是 ε-贪心**。"""
        actor = _agent().actor
        assert actor.action_sampling == "boltzmann"
        assert actor.trace_center == "sampling"
        assert actor.logit_scale == 20.0

    def test_the_old_epsilon_greedy_scheme_is_still_reachable(self):
        """旧的 ε-贪心方案仍可显式选回——账里那一整批对照就是它。"""
        agent = CartPoleAgent(
            4,
            critic_units=3,
            actor_action_sampling="epsilon_greedy",
            actor_trace_center="none",
            actor_logit_scale=1.0,
            generator=torch.Generator().manual_seed(0),
        )
        assert agent.actor.action_sampling == "epsilon_greedy"
        assert agent.actor.trace_center == "none"

    def test_run_trial_defaults_match_the_accepted_configuration(self):
        """``run_trial`` 的两个调度默认值也要与 ADR-0009 一致。"""
        import inspect

        from research.rstdp.cartpole import run_trial

        parameters = inspect.signature(run_trial).parameters
        # 退火起点是 **1.0**：验收那批（`sweep_results/step13_*`、`step14_*`）显式传的就是 1，
        # 而默认值一度是 2——于是「默认运行即达标配置」这句话不成立。改成 1 之后，
        # `--seed 45` 重跑给出 232.1 步，与验收批次里的 232.1 逐字一致（见 ADR-0009 的复核注记）。
        assert parameters["actor_lr_final_fraction"].default == pytest.approx(0.01)
        assert parameters["logit_scale_start"].default == pytest.approx(1.0)
        assert parameters["logit_scale_end"].default == pytest.approx(40.0)

        # **CLI 的默认值也必须跟上**：这两个不是 agent 构造参数，走的是独立字面量，
        # 所以它们会与 run_trial 的默认值漂移——实测就漏过一次（CLI 还是 None ⇒ 默认不退火，
        # 于是 `--seed 0` 跑出的是恒定 logit 20 的塌陷配置）。
        from research.rstdp.cartpole import build_parser

        cli = {action.dest: action.default for action in build_parser()._actions}
        assert cli["logit_scale_start"] == pytest.approx(1.0)
        assert cli["logit_scale_end"] == pytest.approx(40.0)

    def test_boltzmann_and_centering_reach_the_actor(self):
        agent = CartPoleAgent(
            4,
            critic_units=3,
            actor_action_sampling="boltzmann",
            actor_logit_scale=3.0,
            actor_trace_center="sampling",
            generator=torch.Generator().manual_seed(0),
        )
        assert agent.actor.action_sampling == "boltzmann"
        assert agent.actor.logit_scale == 3.0
        assert agent.actor.trace_center == "sampling"

    def test_the_agent_updates_the_actor_on_every_boltzmann_step(self):
        """Boltzmann 下没有「探索步」，所以每一步都更新 Actor。"""
        agent = CartPoleAgent(
            4,
            critic_units=3,
            actor_action_sampling="boltzmann",
            actor_logit_scale=2.0,
            generator=torch.Generator().manual_seed(0),
        )
        features = torch.tensor([1.0, 0.5, 0.25, 0.0])
        next_features = torch.tensor([0.0, 1.0, 0.5, 0.25])
        before = agent.actor.trace.clone()
        agent.learn_step(features, 0, 1.0, next_features, terminated=True)
        assert not torch.allclose(agent.actor.trace, before)


class TestCriticOperatingPoint:
    """``gain`` / 阈值决定 ``V`` 的值域——默认那一组把 V 顶在 8 以上，而真值要低得多。"""

    def test_gain_and_threshold_reach_the_critic(self):
        agent = CartPoleAgent(
            4,
            critic_units=3,
            critic_gain=4.0,
            critic_threshold=0.9,
            generator=torch.Generator().manual_seed(0),
        )
        assert agent.critic.gain == 4.0
        assert agent.critic.bias == 0.9

    def test_defaults_are_unchanged(self):
        agent = _agent()
        assert agent.critic.gain == 8.0
        assert agent.critic.bias == 0.4

    def test_a_higher_threshold_lowers_the_value_floor(self):
        """调高阈值才让 ``V`` 落得到低段——真值在临死那一步约 1，而默认下界是 7.83。"""
        low = CartPoleAgent(
            4, critic_units=3, critic_threshold=0.4, generator=torch.Generator().manual_seed(0)
        )
        high = CartPoleAgent(
            4, critic_units=3, critic_threshold=1.0, generator=torch.Generator().manual_seed(0)
        )
        assert high.critic.value_floor() < low.critic.value_floor()

    def test_a_higher_threshold_makes_the_units_sparser(self):
        agent = _agent()
        sparse = CartPoleAgent(
            4, critic_units=32, critic_threshold=1.0, generator=torch.Generator().manual_seed(0)
        )
        features = torch.tensor([1.0, 0.5, 0.25, 0.0])
        assert float(sparse.critic.rates(features).mean()) < float(
            agent.critic.rates(features).mean()
        )


class TestSignalClip:
    """裁剪只作用于 Actor 的成功信号，Critic 仍然学真正的 δ。"""

    @staticmethod
    def _features() -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor([1.0, 0.5, 0.25, 0.0]), torch.tensor([0.0, 1.0, 0.5, 0.25])

    def _agent(self, **kwargs) -> CartPoleAgent:
        # 偏置把 V 抬到 100 附近，于是终止步的 δ ≈ −100，远超任何合理的裁剪阈值
        return CartPoleAgent(
            4,
            critic_units=3,
            critic_output_bias=100.0,
            generator=torch.Generator().manual_seed(0),
            **kwargs,
        )

    def test_clip_bounds_the_signal_but_not_the_delta(self):
        features, next_features = self._features()
        delta, signal = self._agent(actor_signal_clip=1.0).learn_step(
            features, 0, 1.0, next_features, terminated=True
        )
        assert delta < -50.0, "δ 本身没有被裁——它是 Critic 的回归目标"
        assert signal == pytest.approx(-1.0)

    def test_clip_keeps_the_sign(self):
        features, next_features = self._features()
        _, signal = self._agent(actor_signal_clip=1.0).learn_step(
            features, 0, 1.0, next_features, terminated=True
        )
        assert signal < 0

    def test_no_clip_leaves_the_signal_equal_to_the_delta(self):
        features, next_features = self._features()
        delta, signal = self._agent().learn_step(features, 0, 1.0, next_features, terminated=True)
        assert signal == pytest.approx(delta)

    def test_a_clip_above_the_delta_changes_nothing(self):
        features, next_features = self._features()
        delta, signal = self._agent(actor_signal_clip=1e6).learn_step(
            features, 0, 1.0, next_features, terminated=True
        )
        assert signal == pytest.approx(delta)


class TestCriticDirectionPool:
    """``build_agent_inputs`` 的采样池大小曾经**静默地**给 ``critic_units`` 封了顶。"""

    @staticmethod
    def _inputs(critic_units: int, seed: int = 0):
        pytest.importorskip("gymnasium")
        from research.common.seeding import SeedBook
        from research.rstdp.cartpole import build_agent_inputs

        return build_agent_inputs(
            SeedBook(base=seed),
            n_features=64,
            encoding_sigma=0.5,
            critic_units=critic_units,
            device=torch.device("cpu"),
        )

    def test_more_units_than_sampled_states_is_supported(self):
        """池子只有 ~850–960 个状态，而 ``critic_units`` 可以比它大。

        实测踩到过：``critic_units=1024`` 抛「init_directions 的形状应为 (1024, 64)，收到
        (892, 64)」——默认的 64 让这个上限一直没露出来。缺采样时按有放回取，而不是报错。
        """
        centers, directions = self._inputs(1024)
        assert centers.shape == (64, 4)
        assert directions.shape == (1024, 64)

    def test_the_pool_is_reused_rather_than_padded_with_zeros(self):
        """有放回取出来的方向必须都是**真状态**的编码，不能混进零向量。"""
        _, directions = self._inputs(1024)
        assert float(directions.norm(dim=1).min()) > 0.0, "不该有零方向"

    def test_the_within_pool_path_is_deterministic(self):
        """够用时走无放回采样，同一本书两次调用逐位相同。"""
        _, first = self._inputs(64)
        _, second = self._inputs(64)
        torch.testing.assert_close(first, second)


class TestLocalityIsPinned:
    """「纯局部」不能只是文档里的一句话——这里把它钉成断言。

    W2（e-prop）那条线**已经**有这组断言了，见 ``research/eprop/tests/test_eprop.py``：
    ``update()`` 之后 ``w_rec`` / ``w_in`` / ``w_out`` 的 ``.grad`` 必须是 ``None``、
    张量上不能有 ``grad_fn``；而 BPTT **基线**必须**有**梯度（那正是它作为对照的作用）。
    W3 此前没有这个加固，所以「无全局反向传播」在本条目里只是承诺。补上。
    """

    @staticmethod
    def _step() -> CartPoleAgent:
        agent = _agent()
        agent.learn_step(
            torch.tensor([1.0, 0.5, 0.25, 0.0]),
            0,
            1.0,
            torch.tensor([0.0, 1.0, 0.5, 0.25]),
            terminated=True,
        )
        return agent

    def test_learn_step_leaves_no_gradients_anywhere(self):
        """走一步学习之后，参与学习的每个张量都不该带计算图、也不该有梯度。"""
        agent = self._step()
        tensors = {
            "actor.weights": agent.actor.weights,
            "actor.trace": agent.actor.trace,
            "critic.weights": agent.critic.weights,
            "critic.readout": agent.critic.readout,
            "critic.out_bias": agent.critic.out_bias,
        }
        for name, tensor in tensors.items():
            assert tensor.grad_fn is None, f"{name} 带着计算图——学习路径上不该有 autograd"
            assert tensor.grad is None, f"{name} 上出现了梯度"

    def test_a_whole_episode_leaves_no_gradients(self):
        """整回合也不留——不只单步。"""
        agent = _agent()
        run_episode(
            _StubEnv(done_after=4, truncated=False),
            agent,
            torch.zeros(4, 4),
            0.5,
            generator=torch.Generator().manual_seed(0),
            exploration=0.0,
            learn=True,
        )
        for name in ("weights", "trace"):
            tensor = getattr(agent.actor, name)
            assert tensor.grad_fn is None and tensor.grad is None, f"actor.{name} 不干净"
            tensor = getattr(agent.critic, name)
            assert tensor.grad_fn is None and tensor.grad is None, f"critic.{name} 不干净"

    @pytest.mark.parametrize("forbidden", [".backward(", "torch.optim", "torch.autograd"])
    def test_the_source_never_reaches_for_autograd(self, forbidden):
        """源码级检查：学习路径上的任何模块都不出现这三样东西。

        这比「运行时没留下梯度」更强——它防的是「今天因为没人调用所以恰好没触发」。
        断言的粒度是**代码行**（先剥掉 ``#`` 之后的注释），所以文档里说明「不用 autograd」
        不会误报。

        豁免 ``capacity_probe.py``：它是**诊断**模块，用 Adam 拟合一个线性策略来测该编码的
        容量上限，明确标注「只诊断，不产生验收数字」。它与学习规则不是一回事。
        """
        package = Path(__file__).resolve().parents[1]
        exempt = {"capacity_probe.py"}
        offenders = []
        for path in sorted(package.glob("*.py")):
            if path.name in exempt:
                continue
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                code = line.split("#", 1)[0]
                if forbidden in code:
                    offenders.append(f"{path.name}:{lineno}  {forbidden}")
        assert not offenders, "学习路径上出现了 autograd 的痕迹：" + "；".join(offenders)


class TestIdenticalConstruction:
    def test_agent_can_be_constructed_twice_identically(self):
        """上头的成对比较全靠这条：同一个种子两次构造必须逐位相同。"""
        torch.testing.assert_close(_agent().actor.weights, _agent().actor.weights)
        torch.testing.assert_close(_agent().critic.weights, _agent().critic.weights)

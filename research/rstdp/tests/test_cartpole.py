"""``research.rstdp.cartpole`` 的测试。

重心是**集成处**：终止与截断的自举、状态编码，以及新旋钮真的传到了 agent。
这些是单元测试覆盖不到的地方——规则本身在 ``test_rules.py`` 里验。
"""

from __future__ import annotations

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


class TestIdenticalConstruction:
    def test_agent_can_be_constructed_twice_identically(self):
        """上头的成对比较全靠这条：同一个种子两次构造必须逐位相同。"""
        torch.testing.assert_close(_agent().actor.weights, _agent().actor.weights)
        torch.testing.assert_close(_agent().critic.weights, _agent().critic.weights)

"""``research.rstdp.signal_probe`` 的测试。

重心是**能算出闭式答案的那几条**：折扣回报、解释方差、切空间投影。真环境那部分用桩环境
验算法，用真 CartPole 验接线。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from research.rstdp.rstdp import RSTDPActor
from research.rstdp.signal_probe import (
    _subsample,
    action_value,
    explained_variance,
    greedy_policy,
    probe_states,
    project_to_l1_tangent,
    rollout_return,
    state_value,
)


class _ChainEnv:
    """确定性桩环境：状态第一维是计数器，走到 ``terminate_at`` 就结束。

    用桩而不是真 CartPole：真环境里「从某状态起还能活几步」得手算才知道，而这个测试要验的
    是**折扣回报的算法**本身——桩能给出闭式答案（``1 + γ + γ² + …``）。

    **终止条件写成依赖状态而不是依赖步数**：``_place`` 会把 ``_elapsed_steps`` 归零
    （探针要从探针状态重新计时），按步数终止的话「把状态摆到别处」就完全不起作用，
    测出来的东西也就不是要验的东西了。真 CartPole 的终止同样只看状态。
    """

    def __init__(self, *, terminate_at: float, truncated: bool = False, step: float = 1.0) -> None:
        self.terminate_at = terminate_at
        self.truncated = truncated
        self.increment = step
        self.unwrapped = self
        self.state = np.zeros(4, dtype=np.float64)
        self.steps_beyond_terminated = None
        self._elapsed_steps = 0

    def reset(self):
        self.state = np.zeros(4, dtype=np.float64)
        self.steps_beyond_terminated = None
        self._elapsed_steps = 0
        return self.state.astype(np.float32), {}

    def step(self, action: int):
        self.state = self.state + np.array([self.increment, 0.0, 0.0, 0.0])
        self._elapsed_steps += 1
        done = self.state[0] >= self.terminate_at
        return (
            self.state.astype(np.float32),
            1.0,
            done and not self.truncated,
            done and self.truncated,
            {},
        )


class _TwoAction(_ChainEnv):
    """动作 0 每步走 1，动作 1 每步走 2——于是「哪个动作更好」是已知的。"""

    def step(self, action: int):
        self.increment = 1.0 + float(action)
        return super().step(action)


def _stay(state: torch.Tensor) -> int:
    return 0


class TestRolloutReturn:
    def test_matches_the_closed_form_when_the_episode_terminates(self):
        """活 3 步 ⇒ ``1 + γ + γ²``。"""
        value = state_value(_ChainEnv(terminate_at=3), np.zeros(4), _stay, gamma=0.99)
        assert value == pytest.approx(1 + 0.99 + 0.99**2)

    def test_truncation_gives_the_same_return(self):
        """截断与终止在**回报**上没有区别——区别只在自举，那是 Actor 那侧的事。"""
        terminated = state_value(_ChainEnv(terminate_at=3), np.zeros(4), _stay, gamma=0.99)
        truncated = state_value(
            _ChainEnv(terminate_at=3, truncated=True), np.zeros(4), _stay, gamma=0.99
        )
        assert truncated == pytest.approx(terminated)

    def test_max_steps_caps_the_rollout(self):
        value = rollout_return(
            _ChainEnv(terminate_at=99), np.zeros(4), _stay, gamma=0.99, max_steps=4
        )
        assert value == pytest.approx(1 + 0.99 + 0.99**2 + 0.99**3)

    def test_the_state_is_placed_before_rolling_out(self):
        """从计数器已经走了一段的状态出发，剩余步数就少——这验的是 ``_place`` 真的生效。"""
        env = _ChainEnv(terminate_at=3)
        env.state = np.array([1.0, 0.0, 0.0, 0.0])
        value = state_value(env, np.array([1.0, 0.0, 0.0, 0.0]), _stay, gamma=0.99)
        assert value == pytest.approx(1 + 0.99)

    def test_action_value_forces_the_first_action(self):
        """强制走策略自己会选的那个动作，应当正好还原成 ``V``；走另一个则不同。"""
        env = _TwoAction(terminate_at=3)
        value = state_value(env, np.zeros(4), _stay, gamma=0.99)
        assert value == pytest.approx(1 + 0.99 + 0.99**2)
        assert action_value(env, np.zeros(4), 0, _stay, gamma=0.99) == pytest.approx(value)
        # 强制走动作 1：一步就冲到 2（还没到 3），之后只剩一步
        assert action_value(env, np.zeros(4), 1, _stay, gamma=0.99) == pytest.approx(1 + 0.99)
        assert action_value(env, np.zeros(4), 1, _stay, gamma=0.99) < value

    def test_rejects_a_non_classic_control_env(self):
        class _Opaque:
            pass

        with pytest.raises(TypeError, match="CartPole"):
            state_value(_Opaque(), np.zeros(4), _stay)


class TestExplainedVariance:
    def test_a_perfect_predictor_scores_one(self):
        true = np.array([1.0, 2.0, 3.0, 4.0])
        assert explained_variance(true, true) == pytest.approx(1.0)

    def test_a_constant_predictor_scores_zero(self):
        true = np.array([1.0, 2.0, 3.0, 4.0])
        assert explained_variance(np.zeros(4), true) == pytest.approx(0.0)

    def test_a_predictor_worse_than_the_mean_scores_negative(self):
        true = np.array([1.0, 2.0, 3.0, 4.0])
        assert explained_variance(-true, true) < 0

    def test_no_variance_in_the_truth_is_undefined_not_zero(self):
        """无定义与「很差」必须分得开，否则「真值没方差」会被读成「Critic 很差」。"""
        assert np.isnan(explained_variance(np.array([1.0, 2.0, 3.0]), np.ones(3)))


class TestProjectToL1Tangent:
    def test_the_result_is_orthogonal_to_the_sign_direction(self):
        """切条件 ``Σ sign(w_i)·v_i = 0``——这是「投影对了」的定义。"""
        rng = np.random.default_rng(0)
        weights = rng.normal(size=10)
        direction = rng.normal(size=10)
        projected = project_to_l1_tangent(direction, weights)
        assert float(np.dot(np.sign(weights), projected)) == pytest.approx(0.0, abs=1e-12)

    def test_a_purely_radial_direction_projects_to_zero(self):
        """纯径向的分量会被 L1 归一化抹掉，所以它不该参与方向比较。"""
        weights = np.array([1.0, -2.0, 3.0])
        projected = project_to_l1_tangent(5.0 * np.sign(weights), weights)
        assert np.allclose(projected, 0.0, atol=1e-12)

    def test_projection_is_idempotent(self):
        rng = np.random.default_rng(1)
        weights = rng.normal(size=8)
        once = project_to_l1_tangent(rng.normal(size=8), weights)
        assert np.allclose(project_to_l1_tangent(once, weights), once)

    def test_zero_weights_do_not_blow_up(self):
        """权重里出现 0 时 ``sign`` 是 0，会把那个分量留在投影里——那也要是有限值。"""
        projected = project_to_l1_tangent(np.array([1.0, 2.0]), np.array([0.0, 1.0]))
        assert np.all(np.isfinite(projected))


class TestGreedyPolicy:
    def test_takes_the_argmax_of_the_scores(self):
        actor = RSTDPActor(4, 2, normalize=False)
        with torch.no_grad():
            actor.weights.copy_(torch.tensor([[1.0, -1.0]] * 4))
        policy = greedy_policy(actor, torch.zeros(4, 4), 0.5)
        assert policy(torch.zeros(4)) == 0

    def test_prefers_the_other_action_when_the_weights_are_flipped(self):
        actor = RSTDPActor(4, 2, normalize=False)
        with torch.no_grad():
            actor.weights.copy_(torch.tensor([[-1.0, 1.0]] * 4))
        policy = greedy_policy(actor, torch.zeros(4, 4), 0.5)
        assert policy(torch.zeros(4)) == 1

    def test_uses_the_encoding_not_the_raw_state(self):
        """策略作用在**编码后**的活动上——两个中心，状态靠近哪个就走哪边。"""
        centers = torch.tensor([[0.0, 0.0, 0.0, 0.0], [10.0, 10.0, 10.0, 10.0]])
        actor = RSTDPActor(2, 2, normalize=False)
        with torch.no_grad():
            actor.weights.copy_(torch.tensor([[1.0, -1.0], [-1.0, 1.0]]))
        policy = greedy_policy(actor, centers, 0.5)
        assert policy(torch.zeros(4)) == 0, "贴着中心 0 的状态该选动作 0"
        assert policy(torch.full((4,), 10.0)) == 1, "贴着中心 1 的状态该选动作 1"


class TestSubsample:
    def test_leaves_short_input_alone(self):
        states = np.arange(6).reshape(6, 1)
        assert len(_subsample(states, 10)) == 6

    def test_takes_the_requested_number_and_keeps_the_endpoints(self):
        states = np.arange(100).reshape(100, 1)
        picked = _subsample(states, 7)
        assert len(picked) == 7
        assert picked[0, 0] == 0 and picked[-1, 0] == 99

    def test_is_deterministic(self):
        states = np.arange(50).reshape(50, 1)
        assert np.array_equal(_subsample(states, 5), _subsample(states, 5))


class TestAgainstRealCartPole:
    """真环境的接线：解析值拿不到，但「值随状态变差而下降」是可以断言的。"""

    @pytest.fixture
    def env(self):
        gymnasium = pytest.importorskip("gymnasium")
        environment = gymnasium.make("CartPole-v1")
        yield environment
        environment.close()

    def test_probe_states_are_four_dimensional_and_non_empty(self, env):
        from research.rstdp.capacity_probe import heuristic_action

        occupied, danger = probe_states(
            env, lambda state: heuristic_action(state), episodes=1, seed=0
        )
        assert occupied.shape[1] == 4 and len(occupied) > 0
        assert danger.shape[1] == 4 and 0 < len(danger) <= 20

    def test_a_good_policy_is_worth_more_than_a_bad_one(self, env):
        from research.rstdp.capacity_probe import heuristic_action

        state = np.array([0.0, 0.0, 0.0, 0.0])
        good = state_value(env, state, lambda s: heuristic_action(s))
        always_right = state_value(env, state, lambda s: 1)
        assert good > always_right, "启发式应当明显优于「一直往右推」"
        assert good > 90.0, "近零状态下的启发式应当接近满分回报（γ=0.99、500 步上限下约 99）"

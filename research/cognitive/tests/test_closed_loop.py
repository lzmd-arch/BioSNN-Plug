"""P3 闭环想象的守卫：策略、采集、闭环长度与指标有限。"""

from __future__ import annotations

import numpy as np

from research.cognitive.run_world_model_closed_loop import (
    closed_loop,
    collect_episodes,
    open_loop,
    policy_action,
)
from research.cognitive.world_model import RidgeActionModel


def test_policy_action_balances_the_pole():
    assert policy_action(np.array([0.0, 0.0, 0.1, 0.0]), np.random.default_rng(0), 0.0) == 1
    assert policy_action(np.array([0.0, 0.0, -0.1, 0.0]), np.random.default_rng(0), 0.0) == 0


def test_collect_produces_terminations_with_noise():
    episodes = collect_episodes(20, seed=0, noise=0.3)
    assert len(episodes) == 20
    assert all(episode.states.shape[0] >= 1 for episode in episodes)
    assert any(bool(episode.terminated.any()) for episode in episodes)


def test_closed_loop_runs_and_metrics_are_finite():
    train = collect_episodes(10, seed=0, noise=0.3)
    test = collect_episodes(5, seed=1, noise=0.3)
    states = np.concatenate([episode.states for episode in train])
    actions = np.concatenate([episode.actions for episode in train])
    next_states = np.concatenate([episode.next_states for episode in train])
    terminated = np.concatenate([episode.terminated for episode in train])
    scale = np.var(np.concatenate([states, next_states]), axis=0)
    model = RidgeActionModel(quadratic=True)
    model.fit(states, actions, next_states, terminated)

    loop = closed_loop(model, test[0], scale=scale, horizon=20, policy_seed=0)
    assert len(loop.nmse) >= 1
    assert np.all(np.isfinite(loop.nmse))
    assert len(open_loop(model, test[0], scale=scale, horizon=10)) == min(
        10, test[0].states.shape[0]
    )

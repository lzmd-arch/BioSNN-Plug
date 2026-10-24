"""世界模型打底零件的测试：契约、动作条件化、两项评估量。"""

from __future__ import annotations

import numpy as np
import pytest

from research.cognitive.run_world_model import Transitions, collect_transitions, evaluate
from research.cognitive.world_model import (
    CopyLastModel,
    MeanDeltaModel,
    PerActionMeanModel,
    RidgeActionModel,
    action_sensitivity,
    auroc,
    per_dim_nmse,
)


def _linear_data(n: int = 400, seed: int = 0) -> tuple[np.ndarray, ...]:
    """构造一份**已知动作交互**的线性动力学。"""
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(n, 4))
    actions = rng.integers(0, 2, size=n)
    delta = np.empty((n, 4))
    delta[:, 0] = 0.1 * states[:, 1] + 0.5 * actions
    delta[:, 1] = 0.2 * actions - 0.1 * states[:, 0]
    delta[:, 2] = states[:, 3]
    delta[:, 3] = -0.3 * actions
    next_states = states + delta
    terminated = (np.abs(next_states[:, 2]) > 1.0).astype(float)
    return states, actions, next_states, terminated


class TestBaselines:
    def test_copy_last_predicts_no_change_and_a_constant_rate(self) -> None:
        model = CopyLastModel()
        states, actions, next_states, terminated = _linear_data()
        model.fit(states, actions, next_states, terminated)

        prediction = model.predict(states, actions)

        assert np.array_equal(prediction.delta, np.zeros_like(states))
        assert np.allclose(prediction.termination, terminated.mean())
        assert action_sensitivity(model, states, np.ones(4)) == 0.0

    def test_mean_delta_predicts_the_global_mean(self) -> None:
        model = MeanDeltaModel()
        states, actions, next_states, terminated = _linear_data()
        model.fit(states, actions, next_states, terminated)

        prediction = model.predict(states[:3], actions[:3])

        assert prediction.delta.shape == (3, 4)
        assert np.allclose(prediction.delta[0], (next_states - states).mean(axis=0))

    def test_per_action_means_differ_when_the_data_says_so(self) -> None:
        states, actions, next_states, terminated = _linear_data()
        model = PerActionMeanModel()
        model.fit(states, actions, next_states, terminated)

        zero = model.predict(states[:1], np.zeros(1, dtype=int)).delta
        one = model.predict(states[:1], np.ones(1, dtype=int)).delta

        assert not np.allclose(zero, one)
        assert action_sensitivity(model, states[:20], np.ones(4)) > 0.0

    def test_ridge_learns_the_action_interaction(self) -> None:
        """在已知线性数据上，岭回归应当把交互项学出来，而不是只学平均差。"""
        states, actions, next_states, terminated = _linear_data(n=800)
        model = RidgeActionModel(ridge=1e-6)
        model.fit(states, actions, next_states, terminated)

        prediction = model.predict(states, actions)
        rmse = np.sqrt(((prediction.delta - (next_states - states)) ** 2).mean())

        assert rmse < 0.05
        assert action_sensitivity(model, states, np.ones(4)) > 0.1

    def test_ridge_rejects_bad_actions_and_shapes(self) -> None:
        model = RidgeActionModel()
        states, actions, next_states, terminated = _linear_data(n=10)
        with pytest.raises(ValueError):
            model.predict(states, np.full(10, 2))
        with pytest.raises(ValueError):
            model.fit(states, actions[:5], next_states, terminated)


class TestMetrics:
    def test_perfect_prediction_has_zero_nmse(self) -> None:
        delta = np.ones((10, 4))
        assert np.allclose(per_dim_nmse(delta, delta, np.ones(4)), 0.0)

    def test_nmse_uses_the_given_scale(self) -> None:
        predicted = np.zeros((10, 4))
        true = np.ones((10, 4))
        assert np.allclose(per_dim_nmse(predicted, true, np.full(4, 4.0)), 0.25)

    def test_auroc_perfect_separation_is_one(self) -> None:
        assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0

    def test_auroc_reversed_is_zero(self) -> None:
        assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) == 0.0

    def test_auroc_ties_are_half(self) -> None:
        assert auroc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 0, 1, 1])) == 0.5

    def test_auroc_single_class_is_nan(self) -> None:
        assert np.isnan(auroc(np.array([0.1, 0.2]), np.array([1, 1])))


class TestRunner:
    def test_collect_transitions_keeps_episode_boundaries(self) -> None:
        data = collect_transitions(2, seed=0)
        assert data.states.shape[1] == 4
        assert data.states.shape[0] == data.next_states.shape[0]
        assert set(np.unique(data.episode)) == {0, 1}

    def test_evaluate_returns_the_three_reported_quantities(self) -> None:
        states, actions, next_states, terminated = _linear_data(n=600, seed=1)
        data = Transitions(
            states=states,
            actions=actions,
            next_states=next_states,
            terminated=terminated.astype(bool),
            episode=np.arange(states.shape[0]) // 100,
        )
        train = data.subset(data.episode < 4)
        test = data.subset(data.episode >= 4)

        result = evaluate(RidgeActionModel(ridge=1e-6), train, test)

        assert set(result) == {
            "nmse",
            "nmse_per_dim",
            "termination_auroc",
            "action_sensitivity",
        }
        assert result["nmse"] >= 0.0

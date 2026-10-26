"""D2 线性策略适配器的守卫：动作 = argmax(W·x)、痕迹居中、信号方向、退火与快照。"""

from __future__ import annotations

import math

import numpy as np
import torch

from research.rstdp_spiking.actor import SpikingActor
from research.rstdp_spiking.cartpole import train
from research.rstdp_spiking.linear_policy import LinearRatePolicy


def _policy(**kwargs) -> LinearRatePolicy:
    kwargs.setdefault("normalize", False)
    kwargs.setdefault("logit_scale", 2.0)
    kwargs.setdefault("generator", torch.Generator().manual_seed(0))
    return LinearRatePolicy(16, 2, **kwargs)


class TestLinearPolicy:
    def test_greedy_is_argmax_of_linear_scores(self):
        policy = _policy()
        features = torch.rand(16, generator=torch.Generator().manual_seed(1))
        chosen = policy.step(features, generator=torch.Generator().manual_seed(2), greedy=True)

        assert chosen == int((features @ policy.inner.weights).argmax().item())

    def test_trace_is_centered_by_the_action_distribution(self):
        policy = _policy(trace_decay=0.5)
        features = torch.rand(16, generator=torch.Generator().manual_seed(3))
        probabilities = policy.inner.action_probabilities(features)
        chosen = policy.step(features, generator=torch.Generator().manual_seed(4), greedy=True)
        indicator = torch.zeros(2)
        indicator[chosen] = 1.0

        assert torch.allclose(
            policy.inner.trace,
            torch.outer(features, indicator - probabilities),
            atol=1e-7,
        )

    def test_signal_moves_along_the_trace(self):
        policy = _policy(learning_rate=0.5)
        features = torch.rand(16, generator=torch.Generator().manual_seed(5))
        policy.step(features, generator=torch.Generator().manual_seed(6), greedy=True)
        before = policy.inner.weights.clone()
        trace = policy.inner.trace.clone()
        policy.apply_success_signal(2.0)

        assert torch.allclose(policy.inner.weights - before, 1.0 * trace, atol=1e-7)

    def test_state_dict_roundtrip(self):
        policy = _policy()
        state = policy.state_dict()
        other = _policy()
        other.load_state_dict(state)

        assert torch.equal(policy.inner.weights, other.inner.weights)
        assert torch.equal(other.inner.weights, other.inner.weights_deployed)


class TestAnnealing:
    def test_log_space_anneal_reaches_the_expected_last_value(self):
        result = train(
            seed=0,
            episodes=3,
            eval_every=3,
            eval_episodes=1,
            n_features=16,
            sigma=1.0,
            steps=5,
            learning_rate=1e-3,
            baseline_decay=0.05,
            critic_kind="ema",
            critic_rate=1e-2,
            gamma=0.99,
            warmup=0,
            device=torch.device("cpu"),
            policy="linear",
            logit_scale_start=1.0,
            logit_scale_end=40.0,
            keep_actor=True,
        )

        assert np.isclose(result.actor.logit_scale, math.exp(math.log(40.0) * 2 / 3))


class TestSnapshotApi:
    def test_spiking_actor_state_roundtrip(self):
        actor = SpikingActor(16, 2, steps=5, generator=torch.Generator().manual_seed(7))
        state = {key: value.clone() for key, value in actor.state_dict().items()}
        with torch.no_grad():
            actor.layer.weight.add_(1.0)
        actor.load_state_dict(state)

        for key, value in actor.state_dict().items():
            assert torch.equal(value, state[key])

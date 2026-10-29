"""M3i 脉冲触发机制的守卫测试。"""

from __future__ import annotations

import pytest
from biosnn_mcp import SpikingTrigger


def _triggers(trigger: SpikingTrigger, saliences) -> list[int]:
    return [index for index, value in enumerate(saliences) if trigger.process(value)]


def test_background_does_not_trigger():
    trigger = SpikingTrigger(seed=0)
    assert _triggers(trigger, [0.0] * 1000) == []


def test_strong_salience_triggers_in_small_population():
    """小群体（无侧向抑制干扰）下，持续显著性应触发；这是机制接线的证据。"""
    trigger = SpikingTrigger(n_neurons=8, seed=0)
    assert _triggers(trigger, [0.0] * 2 + [1.0] * 20) != []


def test_deterministic_with_seed():
    first = SpikingTrigger(seed=11)
    second = SpikingTrigger(seed=11)
    stream = [0.0] * 50 + [0.5] * 100 + [0.0] * 50
    assert _triggers(first, stream) == _triggers(second, stream)


def test_cooldown_spacing():
    """两次显著性起始相隔 60 步，触发也应相隔 ≥ cooldown（持续显著性只会在起始触发）。"""
    trigger = SpikingTrigger(n_neurons=8, seed=0, cooldown=50)
    stream = [1.0] * 20 + [0.0] * 40 + [1.0] * 20
    triggers = _triggers(trigger, stream)
    assert len(triggers) >= 2
    assert triggers[1] - triggers[0] >= 50


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_neurons": 1},
        {"tau": 0.0},
        {"threshold": 0.0},
        {"noise_sigma": -0.1},
        {"lateral_eta": -0.1},
        {"n_hits": 0},
        {"cooldown": -1},
    ],
)
def test_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        SpikingTrigger(**kwargs)


def test_rejects_bad_salience():
    trigger = SpikingTrigger(n_neurons=4)
    for value in (-0.1, 1.1):
        with pytest.raises(ValueError):
            trigger.process(value)

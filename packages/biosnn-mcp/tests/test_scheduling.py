"""M3g 调度策略的守卫测试。"""

from __future__ import annotations

import pytest

from biosnn_mcp import SchedulingPolicy


def test_low_score_triggers_after_cooldown():
    policy = SchedulingPolicy(score_threshold=0.25, cooldown=1.0)
    assert policy.decide(0.0, 1) is True
    assert policy.decide(0.0, 0) is False


def test_high_score_never_triggers():
    assert SchedulingPolicy().decide(0.9, 5) is False


def test_cooldown_rounds_and_floor():
    assert SchedulingPolicy(cooldown=0.0).cooldown_turns == 1
    assert SchedulingPolicy(cooldown=2.4).cooldown_turns == 2
    assert SchedulingPolicy(cooldown=2.6).cooldown_turns == 3


@pytest.mark.parametrize(
    "kwargs", [{"score_threshold": -0.1}, {"score_threshold": 1.1}, {"cooldown": -0.1}, {"cooldown": 6.0}]
)
def test_rejects_out_of_range(kwargs):
    with pytest.raises(ValueError):
        SchedulingPolicy(**kwargs)


def test_rejects_bad_inputs():
    policy = SchedulingPolicy()
    with pytest.raises(ValueError):
        policy.decide(-0.1, 1)
    with pytest.raises(ValueError):
        policy.decide(0.1, -1)

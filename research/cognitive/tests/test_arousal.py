"""唤醒通道的行为测试。三条是这个组件的全部契约：有界、可复现、只调增益。"""

from __future__ import annotations

import pytest

from research.cognitive.arousal import ArousalChannel, ArousalConfig


def test_starts_at_baseline() -> None:
    channel = ArousalChannel(ArousalConfig(baseline=0.25))
    assert channel.arousal == pytest.approx(0.25)


def test_high_input_raises_and_silence_returns_toward_baseline() -> None:
    channel = ArousalChannel(ArousalConfig(baseline=0.2, input_gain=0.8, firing_gain=0.8, tau=0.5))

    after_high = channel.observe(input_strength=1.0, firing_rate=1.0)
    assert after_high > 0.2

    for _ in range(20):
        after_silence = channel.observe(input_strength=0.0, firing_rate=0.0)
    assert after_silence < after_high
    assert after_silence == pytest.approx(0.0, abs=1e-3)


def test_state_is_bounded_for_extreme_inputs() -> None:
    channel = ArousalChannel(ArousalConfig(lo=0.1, hi=0.9, tau=1.0))
    for _ in range(10):
        channel.observe(input_strength=99.0, firing_rate=99.0)
    assert channel.arousal == pytest.approx(0.9)

    for _ in range(10):
        channel.observe(input_strength=-99.0, firing_rate=-99.0)
    assert channel.arousal == pytest.approx(0.1)


def test_trajectory_is_deterministic() -> None:
    """无随机性：同一串输入必须给出同一串状态。"""

    def run() -> list[float]:
        channel = ArousalChannel()
        return [channel.observe(input_strength=s, firing_rate=s / 2) for s in (0.1, 0.9, 0.4, 0.0)]

    assert run() == run()


def test_reset_returns_to_baseline() -> None:
    channel = ArousalChannel()
    channel.observe(input_strength=1.0, firing_rate=1.0)
    channel.reset()
    assert channel.arousal == pytest.approx(channel.config.baseline)


class TestGain:
    """增益是它**唯一**的作用方式——不能改权重，只能乘在这里。"""

    def test_identity_at_baseline(self) -> None:
        channel = ArousalChannel(ArousalConfig(baseline=0.5, lo=0.0, hi=1.0, gain_span=0.5))
        assert channel.gain(10.0) == pytest.approx(10.0)

    def test_monotone_in_arousal(self) -> None:
        low = ArousalChannel(ArousalConfig(baseline=0.2, lo=0.0, hi=1.0, tau=1.0, gain_span=0.5))
        low.observe(input_strength=0.2, firing_rate=0.2)
        high = ArousalChannel(ArousalConfig(baseline=0.2, lo=0.0, hi=1.0, tau=1.0, gain_span=0.5))
        high.observe(input_strength=1.0, firing_rate=1.0)
        assert high.gain(10.0) > low.gain(10.0)

    def test_span_bounds_the_modulation(self) -> None:
        channel = ArousalChannel(
            ArousalConfig(baseline=0.0, lo=0.0, hi=1.0, tau=1.0, gain_span=0.5)
        )
        channel.observe(input_strength=1.0, firing_rate=1.0)
        assert channel.gain(10.0) == pytest.approx(15.0)  # 1 + span = 1.5 倍


class TestConfig:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"lo": 0.5, "hi": 0.4},
            {"baseline": 1.5, "lo": 0.0, "hi": 1.0},
            {"tau": 0.0},
            {"tau": 1.5},
            {"input_gain": -1.0},
            {"gain_span": 2.0},
        ],
    )
    def test_rejects_bad_values(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            ArousalConfig(**kwargs)

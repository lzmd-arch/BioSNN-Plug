"""唤醒通道接进策略温度：默认路径逐位不变，接上后只改采样尺度。"""

from __future__ import annotations

import pytest
import torch

from research.cognitive import ArousalChannel, ArousalConfig
from research.rstdp_spiking.actor import SpikingActor


def make_actor(*, arousal: ArousalChannel | None = None, logit_scale: float = 2.0) -> SpikingActor:
    return SpikingActor(
        n_features=8,
        n_actions=2,
        steps=3,
        logit_scale=logit_scale,
        arousal=arousal,
        generator=torch.Generator().manual_seed(0),
    )


def test_default_path_is_unchanged() -> None:
    """``arousal=None`` 时尺度必须**逐位等于** ``logit_scale``——既有的数不能被动到。"""
    actor = make_actor(logit_scale=2.0)
    assert actor.sampling_scale() == 2.0


def test_high_arousal_flattens_the_policy() -> None:
    """唤醒越高 → 尺度越小 → 分布越平（探索更多）。"""
    channel = ArousalChannel(ArousalConfig(baseline=0.2, lo=0.0, hi=1.0, tau=1.0, gain_span=0.5))
    actor = make_actor(arousal=channel, logit_scale=2.0)
    calm = actor.sampling_scale()

    channel.observe(input_strength=1.0, firing_rate=1.0)
    aroused = actor.sampling_scale()

    assert aroused < calm
    assert aroused == pytest.approx(2.0 / channel.gain(1.0))


def test_low_arousal_sharpens_the_policy() -> None:
    channel = ArousalChannel(ArousalConfig(baseline=0.8, lo=0.0, hi=1.0, tau=1.0, gain_span=0.5))
    actor = make_actor(arousal=channel, logit_scale=2.0)
    calm = actor.sampling_scale()

    channel.observe(input_strength=0.0, firing_rate=0.0)
    drowsy = actor.sampling_scale()

    assert drowsy > calm


def test_actor_never_writes_to_the_channel() -> None:
    """演员只**读**唤醒值：``step`` 不许改状态（更新是调用方的事）。"""
    channel = ArousalChannel()
    channel.observe(input_strength=0.5, firing_rate=0.5)
    before = channel.arousal

    actor = make_actor(arousal=channel)
    actor.step(
        torch.zeros(8),
        generator=torch.Generator().manual_seed(1),
        learn_trace=False,
    )

    assert channel.arousal == before


def test_greedy_is_unaffected_by_arousal() -> None:
    """贪心评测必须与唤醒无关——否则评测口径会被状态污染。"""
    features = torch.zeros(8)
    plain = make_actor(logit_scale=2.0)
    channel = ArousalChannel(ArousalConfig(baseline=0.2, tau=1.0, gain_span=0.5))
    channel.observe(input_strength=1.0, firing_rate=1.0)
    aroused = make_actor(arousal=channel, logit_scale=2.0)

    a = plain.step(features, generator=torch.Generator().manual_seed(7), greedy=True)
    b = aroused.step(features, generator=torch.Generator().manual_seed(7), greedy=True)
    assert a == b

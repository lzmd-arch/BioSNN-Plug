"""符号一致化痕迹的守卫（预注册 rstdp_spiking_actor_fix_prereg.md）。

盯三件事：两相非负、只有被选动作列被写、成功信号的符号决定更新方向。
默认的 pair-stdp 路径由 test_actor / test_plasticity 里的既有断言覆盖。
"""

from __future__ import annotations

import pytest
import torch

from research.rstdp_spiking.actor import SpikingActor
from research.rstdp_spiking.plasticity import (
    STDPWindow,
    build_trace,
    ltp_ltd,
    pair_based_stdp,
)


def _spikes(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    pre = (torch.rand(25, 3, 16, generator=generator) < 0.1).float()
    post = (torch.rand(25, 3, 2, generator=generator) < 0.05).float()
    return pre, post


class TestTwoPhases:
    def test_ltp_ltd_are_nonnegative_and_match_the_signed_rule(self):
        pre, post = _spikes()
        window = STDPWindow()
        ltp, ltd = ltp_ltd(pre, post, window)

        assert torch.all(ltp >= 0) and torch.all(ltd >= 0)
        assert torch.equal(pair_based_stdp(pre, post, window), ltp - ltd)

    def test_unknown_rule_is_rejected(self):
        with pytest.raises(ValueError):
            build_trace("signed-magic", 4, 2)


class TestSignConsistentTrace:
    def test_only_the_selected_column_is_written(self):
        pre, post = _spikes()
        trace = build_trace("sign-consistent", 16, 2, decay=0.9)
        trace.accumulate_window(pre, post, STDPWindow(), 1)

        assert float(trace.positive[:, 0].abs().sum()) == 0.0
        assert float(trace.negative[:, 0].abs().sum()) == 0.0
        assert float(trace.positive[:, 1].sum()) > 0.0

    def test_positive_signal_uses_only_the_ltp_phase(self):
        pre, post = _spikes(seed=1)
        trace = build_trace("sign-consistent", 16, 2)
        trace.accumulate_window(pre, post, STDPWindow(), 0)
        weights = torch.zeros(16, 2)
        trace.apply(weights, 1.0, 1.0)

        assert torch.equal(weights[:, 0], trace.positive[:, 0])
        assert float(weights[:, 1].abs().sum()) == 0.0
        assert float(weights[:, 0].sum()) > 0.0

    def test_negative_signal_uses_only_the_ltd_phase(self):
        pre, post = _spikes(seed=2)
        trace = build_trace("sign-consistent", 16, 2)
        trace.accumulate_window(pre, post, STDPWindow(), 1)
        weights = torch.zeros(16, 2)
        trace.apply(weights, 1.0, -2.0)

        assert torch.all(weights[:, 1] <= 0)
        assert float(weights[:, 1].sum()) < 0.0
        assert float(weights[:, 0].abs().sum()) == 0.0

    def test_zero_signal_does_nothing(self):
        pre, post = _spikes(seed=3)
        trace = build_trace("sign-consistent", 16, 2)
        trace.accumulate_window(pre, post, STDPWindow(), 0)
        weights = torch.zeros(16, 2)
        trace.apply(weights, 1.0, 0.0)

        assert float(weights.abs().sum()) == 0.0

    def test_decay_applies_to_columns_not_written_this_step(self):
        pre, post = _spikes(seed=4)
        trace = build_trace("sign-consistent", 16, 2, decay=0.5)
        trace.accumulate_window(pre, post, STDPWindow(), 0)
        kept = trace.positive[:, 0].clone()
        trace.accumulate_window(pre, post, STDPWindow(), 1)

        assert torch.allclose(trace.positive[:, 0], 0.5 * kept)


class TestActorWiring:
    def test_step_writes_the_action_it_chose(self):
        """``step`` 必须把**自己选的动作**交给痕迹，而不是随手一列。"""
        features = torch.rand(16, generator=torch.Generator().manual_seed(8))
        actor = SpikingActor(
            16,
            2,
            steps=8,
            trace_rule="sign-consistent",
            normalize=False,
            generator=torch.Generator().manual_seed(9),
        )
        probe = SpikingActor(
            16,
            2,
            steps=8,
            trace_rule="sign-consistent",
            normalize=False,
            generator=torch.Generator().manual_seed(9),
        )
        chosen = actor.step(features, generator=torch.Generator().manual_seed(11), greedy=True)
        _, post = probe.window_spikes(features, generator=torch.Generator().manual_seed(11))
        expected = int(post.sum(dim=0).squeeze(0).argmax().item())

        assert chosen == expected
        other = 1 - chosen
        assert float(actor.trace.positive[:, other].abs().sum()) == 0.0
        assert float(actor.trace.negative[:, other].abs().sum()) == 0.0
        assert float(actor.trace.positive[:, chosen].sum()) > 0.0

    def test_apply_success_signal_goes_through_the_trace_object(self):
        actor = SpikingActor(
            16,
            2,
            steps=8,
            trace_rule="sign-consistent",
            normalize=False,
            learning_rate=0.5,
            generator=torch.Generator().manual_seed(10),
        )
        generator = torch.Generator().manual_seed(12)
        actor.step(torch.rand(16, generator=generator), generator=generator)
        before = actor.layer.weight.clone()
        expected = actor.trace.positive.clone()
        actor.apply_success_signal(2.0)

        assert torch.allclose(actor.layer.weight - before, 1.0 * expected, atol=1e-7)
        assert torch.all(actor.layer.weight - before >= 0)

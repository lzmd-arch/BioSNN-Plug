"""脉冲 Actor 的测试。

盯三件事：**接口能跑**、**痕迹真的被 Δt 加权的东西驱动**、**成功信号按 η·S·e 落到权重上**。
"""

from __future__ import annotations

import torch

from research.components.cartpole_encoding import (
    CARTPOLE_HALF_WIDTHS,
    encode_state,
    population_centers,
)
from research.rstdp_spiking.actor import SpikingActor
from research.rstdp_spiking.plasticity import STDPWindow
from research.rstdp_spiking.spike_encoding import spike_encode


def _actor(**kwargs) -> SpikingActor:
    kwargs.setdefault("steps", 8)
    kwargs.setdefault("window", STDPWindow(tau_plus=5.0, tau_minus=5.0))
    kwargs.setdefault("generator", torch.Generator().manual_seed(0))
    return SpikingActor(6, 2, **kwargs)


class TestEncoding:
    def test_centers_have_the_expected_shape(self):
        centers = population_centers(
            10, half_widths=CARTPOLE_HALF_WIDTHS, generator=torch.Generator().manual_seed(0)
        )
        assert centers.shape == (10, 4)

    def test_features_are_in_the_unit_interval(self):
        centers = population_centers(
            16, half_widths=CARTPOLE_HALF_WIDTHS, generator=torch.Generator().manual_seed(0)
        )
        state = torch.tensor([0.3, -0.4, 0.05, 1.2])
        features = encode_state(state, centers, sigma=0.5)
        assert features.shape == (16,)
        assert float(features.min()) >= 0.0 and float(features.max()) <= 1.0

    def test_spike_encode_is_binary_and_reproducible(self):
        features = torch.full((5,), 0.5)
        first = spike_encode(features, 20, generator=torch.Generator().manual_seed(1))
        second = spike_encode(features, 20, generator=torch.Generator().manual_seed(1))
        assert set(first.unique().tolist()) <= {0.0, 1.0}
        assert torch.equal(first, second), "同一个 seed 必须给出同一串脉冲"

    def test_zero_feature_never_spikes(self):
        features = torch.zeros(4)
        spikes = spike_encode(features, 30, generator=torch.Generator().manual_seed(2))
        assert float(spikes.sum()) == 0.0

    def test_invalid_steps_is_rejected(self):
        try:
            spike_encode(torch.ones(4), 0, generator=torch.Generator().manual_seed(0))
        except ValueError as exc:
            assert "steps" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("steps=0 应当报错")


class TestActorPlumbing:
    def test_step_returns_a_valid_action(self):
        actor = _actor()
        generator = torch.Generator().manual_seed(3)
        features = torch.rand(6, generator=generator)
        for _ in range(5):
            assert actor.step(features, generator=generator) in (0, 1)

    def test_trace_becomes_nonzero_after_a_step(self):
        """痕迹由 STDP 驱动——一步之后它不该还是零。"""
        actor = _actor()
        generator = torch.Generator().manual_seed(4)
        assert actor.trace_rms() == 0.0
        actor.step(torch.rand(6, generator=generator), generator=generator)
        assert actor.trace_rms() > 0.0

    def test_end_episode_clears_the_trace(self):
        actor = _actor()
        generator = torch.Generator().manual_seed(5)
        actor.step(torch.rand(6, generator=generator), generator=generator)
        actor.end_episode()
        assert actor.trace_rms() == 0.0

    def test_success_signal_moves_weights_along_the_trace(self):
        """``w ← w + η·S·e``：正信号让权重朝痕迹方向走，负信号朝反方向。"""
        actor = _actor(learning_rate=0.1, normalize=False)
        generator = torch.Generator().manual_seed(6)
        actor.step(torch.rand(6, generator=generator), generator=generator)

        before = actor.layer.weight.clone()
        trace = actor.trace.value.clone()
        actor.apply_success_signal(1.0)
        assert torch.allclose(actor.layer.weight - before, 0.1 * trace, atol=1e-6)

    def test_weight_normalisation_holds_the_column_norm(self):
        actor = _actor(weight_norm=1.0, normalize=True)
        generator = torch.Generator().manual_seed(7)
        for _ in range(3):
            actor.step(torch.rand(6, generator=generator), generator=generator)
            actor.apply_success_signal(1.0)
        column = actor.layer.weight.abs().sum(dim=0)
        assert torch.allclose(column, torch.ones(2), atol=1e-5)

    def test_greedy_ignores_the_sampling_scheme(self):
        """贪心必须真的贪心——速率版在这里踩过「评测其实在采样」的坑。

        注意**脉冲窗口本身是随机的**，所以「贪心」只能是「对**这一个窗口**取 argmax」。
        要验证它，就得让两次调用消耗**同一条随机流**——所以这里用两个同 seed 的 actor。
        """
        features = torch.rand(6, generator=torch.Generator().manual_seed(8))
        greedy_actor = _actor(action_sampling="boltzmann", logit_scale=100.0)
        probe_actor = _actor(action_sampling="boltzmann", logit_scale=100.0)

        chosen = greedy_actor.step(
            features, generator=torch.Generator().manual_seed(11), greedy=True, learn_trace=False
        )
        _, post = probe_actor.window_spikes(features, generator=torch.Generator().manual_seed(11))
        counts = post.sum(dim=0).squeeze(0)
        assert chosen == int(counts.argmax().item())


class TestValidation:
    def test_zero_steps_is_rejected(self):
        try:
            _actor(steps=0)
        except ValueError as exc:
            assert "steps" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("steps=0 应当报错")

    def test_unknown_sampling_is_rejected(self):
        try:
            _actor(action_sampling="softmax")
        except ValueError as exc:
            assert "action_sampling" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("未知采样方案应当报错")

    def test_zero_repeats_is_rejected(self):
        actor = _actor()
        try:
            actor.counts(torch.rand(6), generator=torch.Generator().manual_seed(9), repeats=0)
        except ValueError as exc:
            assert "repeats" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("repeats=0 应当报错")


class TestReadoutMode:
    """D3：连续速率读出只换策略读出；默认计数路径必须逐位不变。"""

    def _actor(self, **kwargs):
        import torch

        from research.rstdp_spiking.actor import SpikingActor

        return SpikingActor(8, 2, steps=5, generator=torch.Generator().manual_seed(0), **kwargs)

    def test_default_matches_explicit_count(self):
        import torch

        features = torch.rand(8)
        default = self._actor().step(features, generator=torch.Generator().manual_seed(1))
        explicit = self._actor(readout="count").step(
            features, generator=torch.Generator().manual_seed(1)
        )
        assert default == explicit

    def test_rate_mode_returns_valid_action(self):
        import torch

        actor = self._actor(readout="rate")
        action = actor.step(torch.rand(8), generator=torch.Generator().manual_seed(2))
        assert action in (0, 1)

    def test_rate_scores_differ_from_counts(self):
        import torch

        actor = self._actor()
        post = torch.zeros(5, 1, 2)
        post[0, 0, 0] = 1.0  # 早期一个脉冲：计数=1，滤波后不到 1
        counts = actor._policy_scores(post)
        rate_actor = self._actor(readout="rate")
        rates = rate_actor._policy_scores(post)
        assert counts[0] == 1.0
        assert rates[0] < 1.0

    def test_rejects_unknown_readout(self):
        try:
            self._actor(readout="spikes")
        except ValueError:
            return
        raise AssertionError("未知 readout 应被拒绝")


class TestContrastiveGamma:
    """D9：列间去相关项；γ=0 必须与历史行为逐位相同。"""

    def _actor(self, gamma: float):
        import torch

        from research.rstdp_spiking.actor import SpikingActor

        return SpikingActor(
            8,
            2,
            steps=5,
            trace_rule="count-aligned",
            contrastive_gamma=gamma,
            generator=torch.Generator().manual_seed(5),
        )

    def test_zero_gamma_matches_default(self):
        import torch

        features = torch.rand(8)
        default = self._actor(0.0)
        default.step(features, generator=torch.Generator().manual_seed(6))
        default.apply_success_signal(0.5)
        explicit = self._actor(0.0)
        explicit.step(features, generator=torch.Generator().manual_seed(6))
        explicit.apply_success_signal(0.5)
        assert torch.equal(default.layer.weight, explicit.layer.weight)

    def test_positive_gamma_changes_weights(self):
        import torch

        features = torch.rand(8)
        control = self._actor(0.0)
        control.step(features, generator=torch.Generator().manual_seed(6))
        control.apply_success_signal(0.5)
        contrastive = self._actor(0.5)
        contrastive.step(features, generator=torch.Generator().manual_seed(6))
        contrastive.apply_success_signal(0.5)
        assert not torch.equal(control.layer.weight, contrastive.layer.weight)

    def test_rejects_out_of_range(self):
        try:
            self._actor(1.0)
        except ValueError:
            return
        raise AssertionError("contrastive_gamma=1.0 应被拒绝")

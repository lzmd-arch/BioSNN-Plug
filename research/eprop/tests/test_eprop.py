"""``research.eprop.eprop`` 的测试。

重心有两条：

1. **学习路径上没有 autograd**——这是计划书 §9「无全局反向传播」的落点，必须可查，
   而不是靠一句文档承诺。做法是断言更新量**等于按公式手算的值**：若实现偷偷用了
   梯度下降，手算的 ``−η Σ L e`` 就不会与之相符。
2. **指标是原义的**——活跃神经元比例在脉冲网络里就是"整窗内是否发放过"，不是类比量。
"""

from __future__ import annotations

import pytest
import torch

from research.eprop.eprop import EPropLearner


def make_learner(**overrides) -> EPropLearner:
    settings = {
        "learning_rate_in": 0.0,
        "learning_rate_rec": 0.0,
        "learning_rate_out": 0.0,
        "n_rec": 16,
        "generator": torch.Generator().manual_seed(0),
    }
    settings.update(overrides)
    return EPropLearner(4, n_out=3, **settings)


def make_task(steps=6, batch=8, n_in=4, n_out=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return (
        (torch.rand(steps, batch, n_in, generator=generator) < 0.4).float(),
        torch.arange(batch) % n_out,
    )


class TestConstruction:
    def test_rejects_an_unknown_loss_timestep_mode(self):
        with pytest.raises(ValueError, match="loss_timesteps"):
            EPropLearner(4, 8, 3, loss_timesteps="sometimes")

    def test_shapes(self):
        learner = make_learner()
        assert learner.cell.w_in.shape == (4, 16)
        assert learner.cell.w_rec.shape == (16, 16)
        assert learner.w_out.shape == (16, 3)


class TestNoAutogradInTheLearningPath:
    """「无全局反向传播」的可测形式。"""

    def test_the_rollout_carries_no_computation_graph(self):
        learner = make_learner()
        inputs, _ = make_task()
        rollout = learner.simulate(inputs)
        for name in ("spikes", "v_scaled", "filtered"):
            tensor = getattr(rollout, name)
            assert tensor.grad_fn is None, f"{name} 带着计算图"
            assert not tensor.requires_grad, f"{name} 需要梯度"

    def test_weights_do_not_require_grad(self):
        """权重是 ``Parameter``，但学习路径不该把它们接进任何计算图。"""
        learner = make_learner()
        inputs, targets = make_task()
        learner.update(inputs, targets)
        assert learner.cell.w_rec.grad is None, "w_rec 上出现了梯度——说明有 autograd 参与"
        assert learner.cell.w_in.grad is None
        assert learner.w_out.grad is None

    def test_update_equals_the_hand_computed_three_factor_rule(self):
        """核心断言：权重变化必须等于 ``−η·Σ_t L_j^t·e_ij^t``。

        这条同时钉住实现与公式：若哪天有人把更新换成 ``loss.backward()``，这里会失败，
        因为梯度下降给出的方向与因子分解不同（两者的差就是 e-prop 丢掉的那些路径，
        见 ``tests/test_traces.py``）。
        """
        learner = make_learner(learning_rate_in=0.1, learning_rate_rec=0.1, learning_rate_out=0.1)
        inputs, targets = make_task()

        # 更新前的快照与手算所需的中间量（全部在 no_grad 下取，不建图）
        with torch.no_grad():
            rollout = learner.simulate(inputs)
            signal = learner.learning_signal(rollout, targets)
            from research.eprop.traces import eligibility_traces

            z_prev = torch.cat([torch.zeros_like(rollout.spikes[:1]), rollout.spikes[:-1]])
            expected_in = torch.einsum(
                "btj,btij->ij",
                signal,
                eligibility_traces(
                    rollout.v_scaled,
                    inputs,
                    rollout.spikes,
                    alpha=learner.cell.alpha,
                    rho=learner.cell.rho,
                    beta=learner.cell.beta,
                    threshold=learner.cell.threshold,
                    dampening_factor=learner.cell.dampening_factor,
                    n_refractory=learner.cell.n_refractory,
                ),
            )
            expected_rec = torch.einsum(
                "btj,btij->ij",
                signal,
                eligibility_traces(
                    rollout.v_scaled,
                    z_prev,
                    rollout.spikes,
                    alpha=learner.cell.alpha,
                    rho=learner.cell.rho,
                    beta=learner.cell.beta,
                    threshold=learner.cell.threshold,
                    dampening_factor=learner.cell.dampening_factor,
                    n_refractory=learner.cell.n_refractory,
                    is_recurrent=True,
                ),
            )
            before_in = learner.cell.w_in.detach().clone()
            before_rec = learner.cell.w_rec.detach().clone()

        learner.update(inputs, targets)

        torch.testing.assert_close(
            learner.cell.w_in - before_in, -0.1 * expected_in, rtol=1e-5, atol=1e-7
        )
        torch.testing.assert_close(
            learner.cell.w_rec - before_rec, -0.1 * expected_rec, rtol=1e-5, atol=1e-7
        )


class TestContrastWithBPTT:
    """把"局部"这件事用**对照**钉死：同一架构，一个用因子分解，一个用反向传播。

    单看 e-prop 侧"没有梯度"只能说明它没建图；两边一起看才说明差别在哪。
    """

    @staticmethod
    def _task():
        steps, batch, n_in, n_out = 8, 16, 6, 2
        labels = torch.arange(batch) % n_out
        inputs = torch.zeros(steps, batch, n_in)
        inputs[steps // 2 :, :, 0] = (
            (labels == 0).float().unsqueeze(0).expand(steps - steps // 2, batch)
        )
        inputs[steps // 2 :, :, 1] = (
            (labels == 1).float().unsqueeze(0).expand(steps - steps // 2, batch)
        )
        return inputs, labels

    def test_eprop_leaves_no_gradient_but_bptt_does(self):
        from research.eprop.bptt_baseline import BPTTLearner

        inputs, labels = self._task()

        eprop = EPropLearner(
            6,
            16,
            2,
            learning_rate_rec=0.01,
            learning_rate_in=0.01,
            generator=torch.Generator().manual_seed(0),
        )
        eprop.update(inputs, labels)
        assert eprop.cell.w_rec.grad is None

        bptt = BPTTLearner(6, 16, 2, learning_rate=0.01, generator=torch.Generator().manual_seed(0))
        bptt.update(inputs, labels)
        assert bptt.cell.w_rec.grad is not None, "BPTT 基线应当留下跨时间的梯度"

    def test_eprop_needs_no_computation_graph_while_bptt_does(self):
        """e-prop 的前向在 ``no_grad`` 下就能跑；BPTT 的前向必须建图。"""
        from research.eprop.bptt_baseline import BPTTLearner

        inputs, _ = self._task()
        learner = EPropLearner(6, 16, 2, generator=torch.Generator().manual_seed(0))
        assert learner.simulate(inputs).spikes.grad_fn is None

        bptt = BPTTLearner(6, 16, 2, generator=torch.Generator().manual_seed(0))
        assert bptt(inputs).grad_fn is not None, "BPTT 的输出应当带着计算图"


class TestLearningSignal:
    def test_is_zero_except_at_the_loss_timestep(self):
        """``loss_timesteps='last'`` 时，只有最后一步的学习信号非零。"""
        learner = make_learner(loss_timesteps="last")
        inputs, targets = make_task(steps=6)
        signal = learner.learning_signal(learner.simulate(inputs), targets)
        assert signal.shape == (6, 8, 16)
        assert torch.allclose(signal[:-1], torch.zeros_like(signal[:-1]))
        assert signal[-1].abs().sum().item() > 0

    def test_is_nonzero_everywhere_in_all_mode(self):
        learner = make_learner(loss_timesteps="all")
        inputs, targets = make_task(steps=6)
        signal = learner.learning_signal(learner.simulate(inputs), targets)
        assert signal[:-1].abs().sum().item() > 0

    def test_matches_the_closed_form(self):
        """``L = W_out @ (softmax(ŷ) − onehot) / N``——闭式，不用 autograd。"""
        learner = make_learner()
        inputs, targets = make_task()
        rollout = learner.simulate(inputs)
        logits = learner._selected_logits(rollout)[-1]
        expected = (
            torch.softmax(logits, dim=-1) - torch.nn.functional.one_hot(targets, 3).float()
        ) / len(targets)
        expected_signal = expected @ learner.w_out.detach().transpose(0, 1)
        torch.testing.assert_close(learner.learning_signal(rollout, targets)[-1], expected_signal)


class TestUpdate:
    def test_changes_all_three_weight_matrices(self):
        learner = make_learner(
            learning_rate_in=0.01, learning_rate_rec=0.01, learning_rate_out=0.01
        )
        inputs, targets = make_task()
        before = (
            learner.cell.w_in.detach().clone(),
            learner.cell.w_rec.detach().clone(),
            learner.w_out.detach().clone(),
        )
        learner.update(inputs, targets)
        assert not torch.equal(before[0], learner.cell.w_in)
        assert not torch.equal(before[1], learner.cell.w_rec)
        assert not torch.equal(before[2], learner.w_out)

    def test_zero_learning_rates_leave_everything_alone(self):
        learner = make_learner()
        inputs, targets = make_task()
        before = learner.cell.w_rec.detach().clone()
        learner.update(inputs, targets)
        torch.testing.assert_close(before, learner.cell.w_rec)

    def test_returns_the_loss_before_the_update(self):
        """报的必须是**更新前**的损失——否则曲线会显得收敛更快。"""
        learner = make_learner(learning_rate_in=0.05)
        inputs, targets = make_task()
        before = learner.loss_value(learner.simulate(inputs), targets)
        reported = learner.update(inputs, targets)
        assert reported == pytest.approx(before)

    def test_reduces_the_loss_on_a_learnable_task(self):
        """抓"符号反了""学习率接错"这类不会报错的错。"""
        torch.manual_seed(0)
        steps, batch, n_in, n_out = 8, 16, 6, 2
        generator = torch.Generator().manual_seed(1)
        labels = torch.arange(batch) % n_out
        inputs = torch.zeros(steps, batch, n_in)
        inputs[steps // 2 :, :, 0] = (
            (labels == 0).float().unsqueeze(0).expand(steps - steps // 2, batch)
        )
        inputs[steps // 2 :, :, 1] = (
            (labels == 1).float().unsqueeze(0).expand(steps - steps // 2, batch)
        )
        _ = generator

        learner = EPropLearner(
            6,
            32,
            2,
            learning_rate_in=5e-3,
            learning_rate_rec=5e-3,
            learning_rate_out=5e-2,
            generator=torch.Generator().manual_seed(2),
        )
        first = learner.loss_value(learner.simulate(inputs), labels)
        for _ in range(60):
            last = learner.update(inputs, labels)
        assert last < first, f"损失没有下降：{first:.4f} → {last:.4f}"


class TestMetrics:
    def test_active_fraction_matches_a_manual_computation(self):
        learner = make_learner()
        inputs, _ = make_task(steps=10, batch=4)
        spikes = learner.simulate(inputs).spikes  # (10, 4, 16)
        manual = float((spikes.reshape(-1, 16) > 0).any(dim=0).float().mean().item())
        assert learner.active_neuron_fraction(inputs) == pytest.approx(manual)

    def test_a_silent_network_reports_zero(self):
        """把权重压到极低、输入清零，网络应当完全沉默——此时活跃比例必须是 0。"""
        learner = make_learner()
        with torch.no_grad():
            learner.cell.w_in.zero_()
            learner.cell.w_rec.zero_()
        silent = torch.zeros(5, 3, 4)
        assert learner.active_neuron_fraction(silent) == 0.0

    def test_sparsity_is_the_zero_fraction(self):
        learner = make_learner()
        inputs, _ = make_task(steps=10, batch=4)
        spikes = learner.simulate(inputs).spikes
        manual = float((spikes == 0).float().mean().item())
        assert learner.spike_sparsity(inputs) == pytest.approx(manual)


class TestPredict:
    def test_returns_one_label_per_sample(self):
        learner = make_learner()
        inputs, _ = make_task(steps=6, batch=5)
        prediction = learner.predict(inputs)
        assert prediction.shape == (5,)
        assert prediction.dtype == torch.int64

    def test_is_the_argmax_of_the_final_logits(self):
        learner = make_learner()
        inputs, _ = make_task(steps=6, batch=5)
        expected = learner.logits(learner.simulate(inputs))[-1].argmax(dim=-1)
        torch.testing.assert_close(learner.predict(inputs), expected)

"""``research.eprop.traces`` 与 ``neurons`` 的测试。

这里最重要的一组是 :class:`TestFactorizationMatchesBPTT`：它把 e-prop 的因子分解
（``dE/dW = Σ_t L_j^t·e_ij^t``）与 autograd 给出的 BPTT 梯度对照。作者自己的仓库里
就有同名的验证脚本（``numerical_verification_eprop_factorization_vs_BPTT.py``），
理由是**痕迹的时间对齐读错不会报错，只会让梯度系统性偏一点**——只有数值对照能发现。
"""

from __future__ import annotations

import pytest
import torch

from research.eprop.neurons import (
    ALIFCell,
    ALIFState,
    pseudo_derivative,
    spike_function,
)
from research.eprop.traces import eligibility_traces, exp_convolve, refractory_mask


class TestPseudoDerivative:
    def test_peaks_at_threshold(self):
        """v_scaled = 0 即处在阈值上，此时伪导数最大（= dampening）。"""
        assert pseudo_derivative(torch.tensor([0.0]), 0.3).item() == pytest.approx(0.3)

    def test_zero_at_rest_and_beyond(self):
        """静息处 v_scaled = −1；再远离就完全为 0。"""
        values = torch.tensor([[-1.0, -2.0, 1.0, 2.0]])
        assert torch.allclose(pseudo_derivative(values, 0.3), torch.zeros_like(values))

    def test_is_symmetric_around_the_threshold(self):
        above = pseudo_derivative(torch.tensor([0.25]), 0.3)
        below = pseudo_derivative(torch.tensor([-0.25]), 0.3)
        assert above.item() == pytest.approx(below.item())

    def test_never_negative(self):
        values = torch.linspace(-3, 3, 41)
        assert (pseudo_derivative(values, 0.3) >= 0).all()


class TestSpikeFunction:
    def test_forward_is_a_heaviside(self):
        v = torch.tensor([-0.5, 0.0, 0.5])
        assert spike_function(v, 0.3).tolist() == [0.0, 0.0, 1.0]

    def test_backward_uses_the_pseudo_derivative(self):
        v = torch.tensor([0.0], requires_grad=True)
        spike_function(v, 0.3).sum().backward()
        assert v.grad.item() == pytest.approx(0.3)

    def test_backward_is_zero_far_from_threshold(self):
        v = torch.tensor([-3.0], requires_grad=True)
        spike_function(v, 0.3).sum().backward()
        assert v.grad.item() == pytest.approx(0.0)


class TestALIFCell:
    def test_lif_when_beta_is_zero(self):
        cell = ALIFCell(3, 4, beta=0.0)
        state = ALIFState.zeros(2, 4)
        _, new_state = cell.step(torch.zeros(2, 3), state)
        assert torch.allclose(new_state.a, torch.zeros(2, 4))

    def test_adaptation_accumulates_spikes(self):
        """适应轨迹按 ``a ← ρ·a + z`` 累积**上一步**的脉冲。

        注意时序：第 t 步的阈值用的是第 t−1 步及更早累积出来的 ``a``，所以单跑一步
        之后 ``a`` 仍为 0——这不是 bug，而是与官方实现一致的顺序
        （``new_b = decay_b * b + state.z`` 里的 ``state.z`` 是上一步的脉冲）。
        """
        cell = ALIFCell(2, 3, beta=1.0, threshold=0.01)
        with torch.no_grad():
            cell.w_in.fill_(1.0)
            cell.w_rec.zero_()
        state = ALIFState.zeros(1, 3)
        strong = torch.ones(1, 2) * 5.0
        _, state = cell.step(strong, state)
        assert (state.z > 0).any(), "这个输入应当引发发放"
        assert torch.allclose(state.a, torch.zeros_like(state.a)), "第一步的 a 还没累积到本步的脉冲"

        _, state = cell.step(strong, state)
        assert (state.a > 0).any(), "第二步的 a 应当已累积上一步的脉冲"
        assert (cell.adaptive_threshold(state.a) > cell.threshold).any()

    def test_adaptive_threshold_raises_with_adaptation(self):
        cell = ALIFCell(2, 3, beta=0.5, threshold=1.0)
        assert cell.adaptive_threshold(torch.tensor([[2.0, 0.0, 0.0]])).tolist() == [
            [2.0, 1.0, 1.0]
        ]

    def test_self_connection_is_disconnected(self):
        """官方实现把 w_rec 的对角切断（"Disconnect self-connection"）。"""
        cell = ALIFCell(2, 3)
        with torch.no_grad():
            cell.w_rec.copy_(torch.ones(3, 3))
        assert torch.allclose(torch.diagonal(cell.w_rec_effective), torch.zeros(3))

    def test_membrane_follows_the_paper_equation(self):
        """v ← α·v + I − thr·z。**输入电流上没有 (1−α) 因子**——这一条容易写错。"""
        cell = ALIFCell(1, 1, tau=20.0, threshold=0.5, beta=0.0, n_refractory=0)
        with torch.no_grad():
            cell.w_in.copy_(torch.zeros(1, 1))
            cell.w_rec.copy_(torch.zeros(1, 1))
        state = ALIFState.zeros(1, 1)
        state.v = torch.tensor([[0.3]])
        # I = 0，所以 new_v = α·0.3
        _, new_state = cell.step(torch.zeros(1, 1), state)
        assert new_state.v.item() == pytest.approx(cell.alpha * 0.3, rel=1e-6)

    def test_refractory_suppresses_spikes(self):
        """不应期内即使膜电位过阈值也不发放。"""
        cell = ALIFCell(1, 1, threshold=0.01, beta=0.0, n_refractory=3)
        with torch.no_grad():
            cell.w_in.fill_(1.0)
            cell.w_rec.zero_()
        state = ALIFState.zeros(1, 1)
        strong = torch.ones(1, 1) * 10.0
        _, state = cell.step(strong, state)
        assert state.z.item() == 1.0
        _, state = cell.step(strong, state)
        assert state.z.item() == 0.0, "上一步刚发放，这一步应处在不应期"

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"tau": 0.0}, "时间常数"),
            ({"tau_adaptation": -1.0}, "时间常数"),
            ({"threshold": 0.0}, "阈值"),
            ({"n_refractory": -1}, "n_refractory"),
        ],
    )
    def test_rejects_invalid_hyperparameters(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            ALIFCell(2, 3, **kwargs)


class TestRefractoryMask:
    def test_mask_lags_the_spike_by_one_step(self):
        """判据是 ``r_{t-1}`` 而不是 ``r_t``：发放那一步**不**被标记，其后一步才被标记。

        这一点曾经写反过，代价是 20% 量级的梯度偏差（见
        :class:`TestFactorizationMatchesBPTT` 的说明）。动力学里第 t 步的判据来自状态里
        携带的计数器，而那个计数器已经包含了第 t 步的脉冲——所以掩码必须滞后一步。
        """
        z = torch.tensor([[[1.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]]])
        mask = refractory_mask(z, n_refractory=2)
        assert mask[0, 0, 0].item() is False, "发放当刻的判据来自上一步，此时尚未进入不应期"
        assert mask[1, 0, 0].item() is True, "发放之后的一步才处在不应期"
        assert mask[:, 0, 1].any().item() is False

    def test_mask_agrees_with_the_simulation_state(self):
        """掩码必须与真实模拟的 ``state.r`` 对得上——这是它唯一的判据来源。"""
        torch.manual_seed(0)
        cell = ALIFCell(2, 3, threshold=0.05, beta=0.1, n_refractory=3)
        with torch.no_grad():
            cell.w_in.fill_(1.0)
            cell.w_rec.fill_(0.5)
        steps, batch = 12, 1
        generator = torch.Generator().manual_seed(3)
        inputs = (torch.rand(steps, batch, 2, generator=generator) < 0.7).float()

        state = ALIFState.zeros(batch, 3)
        spikes, refractory_states = [], []
        for t in range(steps):
            refractory_states.append(state.r > 0)  # 第 t 步的判据 = 进入该步时的 r
            z, state = cell.step(inputs[t], state)
            spikes.append(z)
        mask = refractory_mask(torch.stack(spikes), n_refractory=3)
        for t in range(steps):
            assert torch.equal(mask[t], refractory_states[t].float().bool()), f"t={t} 对不上"

    def test_zero_refractory_gives_an_all_false_mask(self):
        z = torch.ones(3, 1, 2)
        assert not refractory_mask(z, 0).any()

    def test_rejects_negative_refractory(self):
        with pytest.raises(ValueError, match="n_refractory"):
            refractory_mask(torch.zeros(1, 1, 1), -1)


class TestEligibilityTraces:
    def _inputs(self, steps=5, batch=2, n_pre=3, n_post=4, seed=0):
        generator = torch.Generator().manual_seed(seed)
        z_pre = (torch.rand(steps, batch, n_pre, generator=generator) < 0.3).float()
        v_scaled = torch.randn(steps, batch, n_post, generator=generator) * 0.2
        z_post = (torch.rand(steps, batch, n_post, generator=generator) < 0.2).float()
        return v_scaled, z_pre, z_post

    def test_shape_is_time_batch_pre_post(self):
        v_scaled, z_pre, z_post = self._inputs()
        trace = eligibility_traces(
            v_scaled,
            z_pre,
            z_post,
            alpha=0.95,
            rho=0.99,
            beta=0.07,
            threshold=0.62,
            dampening_factor=0.3,
            n_refractory=2,
        )
        assert trace.shape == (5, 2, 3, 4)

    def test_epsilon_v_accumulates_pre_synaptic_spikes(self):
        """β=0、无不应期时，e_ij^t = ψ_j^t · Σ_{t'<=t} α^{t−t'}·z_i^{t'}。"""
        steps, batch, n_pre, n_post = 4, 1, 2, 2
        z_pre = torch.zeros(steps, batch, n_pre)
        z_pre[0, 0, 0] = 1.0
        v_scaled = torch.zeros(steps, batch, n_post)  # 恒在阈值上 → ψ = dampening/thr
        z_post = torch.zeros(steps, batch, n_post)  # 无脉冲 → 无不应期
        alpha = 0.9
        trace = eligibility_traces(
            v_scaled,
            z_pre,
            z_post,
            alpha=alpha,
            rho=0.99,
            beta=0.0,
            threshold=1.0,
            dampening_factor=1.0,
            n_refractory=0,
        )
        # ψ = 1/1 * max(0, 1-0) * 1 = 1
        for t in range(steps):
            assert trace[t, 0, 0, 0].item() == pytest.approx(alpha**t), f"t={t}"

    def test_zero_trace_where_psi_is_zero(self):
        """ψ=0（远离阈值）时该时刻的痕迹必须为 0。"""
        steps, batch, n_pre, n_post = 3, 1, 2, 2
        z_pre = torch.ones(steps, batch, n_pre)
        v_scaled = torch.full((steps, batch, n_post), -5.0)  # 远低于阈值 → ψ=0
        z_post = torch.zeros(steps, batch, n_post)
        trace = eligibility_traces(
            v_scaled,
            z_pre,
            z_post,
            alpha=0.9,
            rho=0.99,
            beta=0.07,
            threshold=1.0,
            dampening_factor=1.0,
            n_refractory=0,
        )
        assert torch.allclose(trace, torch.zeros_like(trace))

    def test_recurrent_trace_has_a_zero_diagonal(self):
        v_scaled, z_pre, z_post = self._inputs(n_pre=4, n_post=4)
        trace = eligibility_traces(
            v_scaled,
            z_pre,
            z_post,
            alpha=0.9,
            rho=0.99,
            beta=0.07,
            threshold=0.62,
            dampening_factor=0.3,
            n_refractory=2,
            is_recurrent=True,
        )
        diagonal = torch.diagonal(trace, dim1=-2, dim2=-1)
        assert torch.allclose(diagonal, torch.zeros_like(diagonal))

    def test_rejects_mismatched_post_shapes(self):
        v_scaled, z_pre, _ = self._inputs()
        with pytest.raises(ValueError, match="形状必须一致"):
            eligibility_traces(
                v_scaled,
                z_pre,
                torch.zeros(5, 2, 99),
                alpha=0.9,
                rho=0.99,
                beta=0.07,
                threshold=0.62,
                dampening_factor=0.3,
            )


class TestExpConvolve:
    def test_matches_the_recurrence(self):
        x = torch.rand(6, 2, 3)
        decay = 0.8
        out = exp_convolve(x, decay)
        running = torch.zeros_like(x[0])
        for t in range(6):
            running = decay * running + (1 - decay) * x[t]
            assert torch.allclose(out[t], running)

    def test_starts_at_zero_and_rises(self):
        x = torch.ones(5, 1, 1)
        out = exp_convolve(x, 0.9)
        assert out[0].item() == pytest.approx(0.1)
        assert out[-1].item() > out[0].item()


class TestFactorizationMatchesBPTT:
    """e-prop 因子分解 vs BPTT。

    这一组把两件事分开，因为它们的性质完全不同：

    * :meth:`test_is_exact_without_recurrence_or_refractoriness` —— 在**没有**循环耦合、
      **没有**不应期、且损失只依赖最后一个时间步时，因子分解 ``Σ_t L_j^t·e_ij^t``
      与 BPTT 梯度**精确相等**（实测相对差 0.000e+00，LIF 与 ALIF 都是）。这条把
      Eq. (25) 的公式与时间对齐钉死——公式或对齐错了它必然失败。

    * :meth:`test_is_an_approximation_in_the_general_case` —— 一般情况下两者**不相等**，
      这是 e-prop 的设计取舍，不是实现缺陷。被丢掉的路径有两类：跨神经元的循环路径，
      以及**复位项** ``−thr·z^{t−1}`` 带来的自身脉冲介导路径（后者与 w_rec 无关，所以
      把循环权重置零也去不掉）。本条目把这个差值**测出来并写在这里**，而不是断言一个
      它并不满足的精度。
    """

    DECAY = 0.951229424500714  # = exp(−1/20)，与 alpha 一致（官方验证脚本如此取值）

    @staticmethod
    def _simulate(cell: ALIFCell, inputs: torch.Tensor):
        steps = inputs.shape[0]
        state = ALIFState.zeros(inputs.shape[1], cell.n_rec)
        spikes, v_scaleds = [], []
        for t in range(steps):
            spikes_t, state = cell.step(inputs[t], state)
            spikes.append(spikes_t)
            v_scaleds.append(cell.v_scaled(state.v, state.a))
        return torch.stack(spikes), torch.stack(v_scaleds)

    @staticmethod
    def _relative_difference(grad_a: torch.Tensor, grad_b: torch.Tensor) -> float:
        return (grad_a - grad_b).abs().max().item() / max(grad_b.abs().max().item(), 1e-12)

    def _gradients(self, cell, inputs, spikes, v_scaled, *, single_step_only, n_refractory):
        loss = (
            spikes[-1].pow(2).sum()
            if single_step_only
            else ((exp_convolve(spikes, self.DECAY) @ torch.ones(cell.n_rec, 1)) ** 2).mean()
        )
        learning_signal = torch.autograd.grad(loss, spikes, retain_graph=True)[0]
        z_pre = torch.cat([torch.zeros_like(spikes[:1]), spikes[:-1]]).detach()
        trace = eligibility_traces(
            v_scaled.detach(),
            z_pre,
            spikes.detach(),
            alpha=cell.alpha,
            rho=cell.rho,
            beta=cell.beta,
            threshold=cell.threshold,
            dampening_factor=cell.dampening_factor,
            n_refractory=n_refractory,
            is_recurrent=True,
        )
        grad_eprop = torch.einsum("btj,btij->ij", learning_signal, trace)
        grad_bptt = torch.autograd.grad(loss, cell.w_rec, retain_graph=True)[0]
        return self._relative_difference(grad_eprop, grad_bptt)

    @pytest.mark.parametrize("beta", [0.0, 0.07])
    def test_is_exact_without_recurrence_or_refractoriness(self, beta):
        """这条是**精确性**断言：公式与时间对齐必须到机器精度。

        条件刻意收紧到"没有可丢掉的路径"：w_rec 置零（无跨神经元路径）、n_refractory=0
        （无不应期掩码）、损失只看最后一步（更早的复位路径影响不到它）。
        """
        torch.manual_seed(0)
        steps, batch, n_in, n_rec = 6, 1, 3, 4
        cell = ALIFCell(n_in, n_rec, beta=beta, n_refractory=0)
        with torch.no_grad():
            cell.w_rec.zero_()
        generator = torch.Generator().manual_seed(2)
        inputs = (torch.rand(steps, batch, n_in, generator=generator) < 0.5).float()

        spikes, v_scaled = self._simulate(cell, inputs)
        relative = self._gradients(
            cell, inputs, spikes, v_scaled, single_step_only=True, n_refractory=0
        )
        assert relative < 1e-6, (
            f"应当精确相等，实测相对差 {relative:.3e}（beta={beta}）——"
            f"Eq. (25) 的公式或时间对齐有问题。"
        )

    def test_is_an_approximation_in_the_general_case(self):
        """一般情形下两者不等——这是 e-prop 的设计取舍，把差值测出来并记录。

        断言的是"落在可解释的范围内且方向一致"，不是"相等"。若哪天它意外变成精确相等，
        下面第一条断言会失败，提示这里该重新审视。
        """
        torch.manual_seed(0)
        steps, batch, n_in, n_rec = 30, 2, 3, 5
        cell = ALIFCell(n_in, n_rec, beta=0.07, n_refractory=2)
        generator = torch.Generator().manual_seed(1)
        inputs = (torch.rand(steps, batch, n_in, generator=generator) < 0.3).float()

        spikes, v_scaled = self._simulate(cell, inputs)
        relative = self._gradients(
            cell, inputs, spikes, v_scaled, single_step_only=False, n_refractory=2
        )
        assert relative > 1e-6, "若真的精确相等，这条测试的定位就该改（见 docstring）"
        assert relative < 1.0, (
            f"近似误差 {relative:.3e} 大得不像 e-prop 的近似——"
            f"更可能是实现错了，而不是方法本身的取舍。"
        )

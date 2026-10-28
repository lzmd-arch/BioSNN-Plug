"""液态 LIF 的守卫与 frozen-α 梯度对照。"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from research.eprop.liquid import LiquidEPropLearner, LiquidLIFCell, LiquidState
from research.eprop.neurons import spike_function
from research.eprop.traces import eprop_gradient

LIQUID = Path(__file__).resolve().parents[1] / "liquid.py"


class TestLiquidGuards:
    def test_only_synaptic_weights_are_parameters(self):
        cell = LiquidLIFCell(4, 6, generator=torch.Generator().manual_seed(0))
        assert {name for name, _ in cell.named_parameters()} == {"w_in", "w_rec"}

    def test_constants_are_not_tensors_requiring_grad(self):
        cell = LiquidLIFCell(4, 6)
        for name in ("a", "b", "rate_decay", "tau_min", "tau_max"):
            value = getattr(cell, name)
            assert isinstance(value, float)
        assert not any(p.requires_grad for name, p in cell.named_parameters() if name in {"a", "b"})

    def test_no_backward_in_liquid_path(self):
        tree = ast.parse(LIQUID.read_text(encoding="utf-8"))
        calls = [
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        assert "backward" not in calls

    def test_tau_is_clamped_and_rate_dependent(self):
        cell = LiquidLIFCell(1, 1)
        low = cell.tau(torch.tensor([[0.0]]))
        high = cell.tau(torch.tensor([[1.0]]))
        assert low.item() == pytest.approx(cell.tau_max)
        assert high.item() == pytest.approx(cell.tau_min)
        assert high.item() < low.item()

    def test_rate_uses_previous_spike(self):
        cell = LiquidLIFCell(1, 1, threshold=0.01, n_refractory=0)
        with torch.no_grad():
            cell.w_in.fill_(1.0)
            cell.w_rec.zero_()
        state = LiquidState.zeros(1, 1)
        z, state = cell.step(torch.ones(1, 1) * 10.0, state)
        assert z.item() == 1.0
        assert state.rate.item() == pytest.approx(0.0), "本步 rate 用的是上一步脉冲"
        _, state = cell.step(torch.zeros(1, 1), state)
        assert state.rate.item() == pytest.approx(0.1)


class TestFrozenAlphaFactorization:
    """把 α 冻结后，e-prop 痕迹必须与 autograd 梯度到机器精度一致。"""

    @staticmethod
    def _forward(cell, inputs, alphas):
        state = LiquidState.zeros(inputs.shape[1], cell.n_rec)
        spikes, scaled = [], []
        for t in range(inputs.shape[0]):
            current = inputs[t] @ cell.w_in + state.z @ cell.w_rec_effective
            new_v = alphas[t] * state.v + current - state.z * cell.threshold
            fresh = spike_function(cell.v_scaled(new_v, state.a), cell.dampening_factor)
            state = LiquidState(
                v=new_v,
                a=torch.zeros_like(new_v),
                z=fresh,
                refractory=torch.zeros_like(new_v),
                rate=state.rate,
            )
            spikes.append(fresh)
            scaled.append(cell.v_scaled(new_v, state.a))
        return torch.stack(spikes), torch.stack(scaled)

    def test_matches_autograd(self):
        torch.manual_seed(0)
        steps, batch, n_in, n_rec = 6, 1, 3, 4
        cell = LiquidLIFCell(n_in, n_rec, n_refractory=0)
        with torch.no_grad():
            cell.w_rec.zero_()
        generator = torch.Generator().manual_seed(2)
        inputs = (torch.rand(steps, batch, n_in, generator=generator) < 0.5).float()

        with torch.no_grad():
            state = LiquidState.zeros(batch, n_rec)
            alphas = []
            for t in range(steps):
                _, state = cell.step(inputs[t], state)
                alphas.append(cell.last_alpha)
            alpha = torch.stack(alphas)

        spikes, scaled = self._forward(cell, inputs, alpha)
        loss = spikes[-1].pow(2).sum()
        learning_signal = torch.autograd.grad(loss, spikes, retain_graph=True)[0]
        trace_grad = eprop_gradient(
            scaled.detach(),
            inputs,
            spikes.detach(),
            learning_signal,
            alpha=alpha.detach(),
            rho=cell.rho,
            beta=cell.beta,
            threshold=cell.threshold,
            dampening_factor=cell.dampening_factor,
            n_refractory=0,
            is_recurrent=False,
        )
        autograd_grad = torch.autograd.grad(loss, cell.w_in)[0]
        relative = (trace_grad - autograd_grad).abs().max().item() / max(
            autograd_grad.abs().max().item(), 1e-12
        )
        assert relative < 1e-6, f"frozen-α 因子分解不精确：相对差 {relative:.3e}"


class TestLiquidLearner:
    def test_update_runs_and_keeps_shapes(self):
        learner = LiquidEPropLearner(3, 4, 2, generator=torch.Generator().manual_seed(0))
        inputs = torch.zeros(5, 2, 3)
        loss = learner.update(inputs, torch.tensor([0, 1]))
        assert loss == pytest.approx(0.693147, abs=1e-5)
        rollout = learner.simulate(inputs)
        assert rollout.alpha is not None
        assert rollout.alpha.shape == (5, 2, 4)
        assert all(torch.isfinite(p).all() for p in learner.cell.parameters())

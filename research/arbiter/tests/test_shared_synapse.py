"""Tests for wiring the arbiter into both research lines over one shared synapse.

These are the tests that make `research/arbiter/` more than parts: they assert the two
rules' updates reach the **same** synapses, that the transpose conventions are aligned
correctly, and that ES arbitration works on that real connection.

The shape guard matters most: `ib_hebbian` stores `(n_shared, n_in)` (PyTorch's
`(out, in)`) while `eprop` stores `(n_in, n_shared)`. Comparing one with the other's
transpose without noticing reports a meaningless cosine -- exactly the failure
`align_update` exists to prevent.
"""

from __future__ import annotations

import pytest
import torch

from research.arbiter.es_arbiter import ESArbiter
from research.arbiter.shared_synapse import (
    SharedSynapseConfig,
    SharedSynapseHarness,
    StepRecord,
)

N_IN, N_SHARED = 32, 8


@pytest.fixture
def harness() -> SharedSynapseHarness:
    torch.manual_seed(0)
    return SharedSynapseHarness(
        SharedSynapseConfig(n_in=N_IN, n_shared=N_SHARED),
        arbiter=ESArbiter(alpha=0.5, seed=0),
    )


class TestSharing:
    def test_both_sides_hold_the_same_numbers(self, harness):
        """ "Same synapse" holds numerically, which is what the criterion asks for."""
        w_eprop = harness.eprop.cell.w_in
        w_hebb = harness.hebbian.layers[0].linear.weight
        assert tuple(w_eprop.shape) == (N_IN, N_SHARED)
        assert tuple(w_hebb.shape) == (N_SHARED, N_IN)
        assert torch.allclose(w_eprop, w_hebb.T)

    def test_sync_copies_into_both_sides(self, harness):
        harness.shared_weight.data.add_(1.0)
        harness.sync_from_shared()
        assert torch.allclose(harness.eprop.cell.w_in, harness.shared_weight)
        assert torch.allclose(harness.hebbian.layers[0].linear.weight, harness.shared_weight.T)

    def test_shared_weight_convention_is_eprop(self, harness):
        assert tuple(harness.shared_weight.shape) == (N_IN, N_SHARED)


class TestAlignment:
    def test_eprop_update_passes_through(self, harness):
        u = torch.zeros(N_IN, N_SHARED)
        assert harness.align_update(u, source="eprop") is u

    def test_hebbian_update_is_transposed(self, harness):
        u = torch.arange(N_SHARED * N_IN, dtype=torch.float32).reshape(N_SHARED, N_IN)
        assert torch.equal(harness.align_update(u, source="hebbian"), u.T)

    def test_wrong_convention_is_rejected(self, harness):
        """Handing the eprop shape to the hebbian side must raise, not silently compare."""
        with pytest.raises(ValueError, match="hebbian 的更新应为"):
            harness.align_update(torch.zeros(N_IN, N_SHARED), source="hebbian")

    def test_unknown_source_is_rejected(self, harness):
        with pytest.raises(ValueError, match="source"):
            harness.align_update(torch.zeros(N_IN, N_SHARED), source="other")


class TestStep:
    def test_step_reports_conflict_and_retention(self, harness):
        torch.manual_seed(1)
        u_e = torch.randn(N_IN, N_SHARED)
        u_h = torch.randn(N_SHARED, N_IN)
        record = harness.step(u_e, u_h)
        assert isinstance(record, StepRecord)
        assert record.conflict.n_synapses == N_IN
        assert record.retention == harness.arbiter.alpha

    def test_step_writes_the_blend_back_to_both_sides(self, harness):
        torch.manual_seed(2)
        u_e = torch.ones(N_IN, N_SHARED)
        u_h = torch.zeros(N_SHARED, N_IN)  # hebbian proposes nothing
        harness.arbiter.alpha = 0.0  # blend == eprop
        harness.step(u_e, u_h)
        assert torch.allclose(harness.eprop.cell.w_in, harness.hebbian.layers[0].linear.weight.T)


class TestArbitrationOnRealConnection:
    def test_es_reduces_events_while_keeping_some_of_b(self, harness):
        """The phase-two bar, but on a real shared synapse -- and not by turning B off."""
        torch.manual_seed(3)
        updates_e = [torch.randn(N_IN, N_SHARED) for _ in range(10)]
        updates_h = [(-u).T for u in updates_e]  # the two rules oppose each other

        result = harness.fit_arbiter(updates_e, updates_h, steps=16, min_retention=0.3)
        assert result.events_before == 10, "the trace must start fully in conflict"
        assert result.reduction >= 0.5, f"降幅 {result.reduction:.0%} 未达阶段二的 50%"
        assert harness.arbiter.alpha >= 0.3, "min_retention must floor alpha -- no trivial fix"

    def test_unconstrained_es_drifts_to_the_trivial_answer(self, harness):
        """Without min_retention, ES turns rule B off. That is the symptom, not the fix."""
        torch.manual_seed(4)
        updates_e = [torch.randn(N_IN, N_SHARED) for _ in range(10)]
        updates_h = [(-u).T for u in updates_e]
        harness.fit_arbiter(updates_e, updates_h, steps=16)
        assert harness.arbiter.alpha < 0.5, "unconstrained ES should drift toward alpha -> 0"

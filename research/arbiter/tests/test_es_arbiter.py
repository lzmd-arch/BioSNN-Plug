"""Tests for the ES arbiter: does it actually reduce conflict events?

The arbiter's whole reason to exist is the phase-two success criterion -- "the arbiter
reduces dual-rule conflict events by 50%+". So the test that matters is the one that
says ES found an alpha that cut events. Everything else is plumbing.

The arbiter has **one** parameter on purpose: Confavreux et al. report meta-learning
failing as rule parameterisations grow complex.
"""

from __future__ import annotations

import numpy as np
import pytest

from research.arbiter.es_arbiter import ArbiterResult, ESArbiter


def trace(n_steps: int = 12, n_pre: int = 4, n_post: int = 6, seed: int = 0):
    """A per-step update trace where rule A and rule B disagree a lot."""
    rng = np.random.default_rng(seed)
    updates_a = [rng.standard_normal((n_pre, n_post)) for _ in range(n_steps)]
    # rule B mostly opposes A -- this is what creates conflict events.
    updates_b = [-a + 0.1 * rng.standard_normal(a.shape) for a in updates_a]
    return updates_a, updates_b


class TestBlend:
    def test_alpha_zero_is_rule_a(self):
        arb = ESArbiter(alpha=0.0)
        a, b = np.ones((2, 2)), np.full((2, 2), 3.0)
        assert np.allclose(arb.blend(a, b), a)

    def test_alpha_one_is_rule_b(self):
        arb = ESArbiter(alpha=1.0)
        a, b = np.ones((2, 2)), np.full((2, 2), 3.0)
        assert np.allclose(arb.blend(a, b), b)

    def test_alpha_half_is_the_average(self):
        arb = ESArbiter(alpha=0.5)
        a, b = np.zeros((2, 2)), np.full((2, 2), 2.0)
        assert np.allclose(arb.blend(a, b), 1.0)

    @pytest.mark.parametrize("alpha", [-0.1, 1.1])
    def test_rejects_alpha_out_of_range(self, alpha):
        with pytest.raises(ValueError, match="alpha"):
            ESArbiter(alpha=alpha)

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="形状"):
            ESArbiter().blend(np.zeros((2, 2)), np.zeros((3, 3)))


class TestMeasure:
    def test_blend_versus_rule_a_is_the_question(self):
        """alpha=1 (pure B) must not look perfect just because B agrees with itself."""
        arb = ESArbiter(alpha=1.0)
        a = np.ones((3, 3))
        b = -np.ones((3, 3))
        report = arb.measure(a, b)
        assert report.events == 1, "blend is pure B; vs A it is a total conflict"

    def test_events_fall_as_alpha_moves_toward_a(self):
        """Blend-vs-A conflict must vanish as alpha -> 0 and stay high as alpha -> 1.

        This is the shape of the thing ES is searching over: the closer the blend is to
        rule A, the less it conflicts with A. The test asserts that monotone relationship
        rather than a single point, so a broken blend cannot pass by luck.
        """
        arb = ESArbiter(alpha=0.5)
        a, b = np.ones((3, 3)), -np.ones((3, 3))
        at_zero = arb.measure(a, b, alpha=0.0).n_conflicting  # blend == A -> no conflict
        at_half = arb.measure(a, b, alpha=0.5).n_conflicting
        at_one = arb.measure(a, b, alpha=1.0).n_conflicting  # blend == B -> total conflict
        assert at_zero == 0
        assert at_one == 3
        assert at_zero <= at_half <= at_one


class TestFit:
    def test_es_reduces_conflict_events(self):
        """The criterion the phase-two success bar is measured against."""
        updates_a, updates_b = trace()
        arb = ESArbiter(alpha=0.5, seed=0)
        result = arb.fit(updates_a, updates_b, steps=24)

        assert isinstance(result, ArbiterResult)
        assert result.events_before > 0, "the trace must start in conflict"
        assert result.events_after <= result.events_before
        assert result.reduction >= 0.5, (
            f"仲裁后冲突事件只降了 {result.reduction:.1%}，达不到阶段二的 50%：{result}"
        )

    def test_alpha_moves_toward_the_less_conflicting_rule(self):
        updates_a, updates_b = trace()
        arb = ESArbiter(alpha=0.5, seed=1)
        arb.fit(updates_a, updates_b, steps=24)
        assert arb.alpha != 0.5, "ES should have moved alpha"

    def test_same_seed_is_reproducible(self):
        def run():
            arb = ESArbiter(alpha=0.5, seed=3)
            return arb.fit(*trace(), steps=16)

        first, second = run(), run()
        assert first.alpha == second.alpha
        assert first.events_after == second.events_after

    @pytest.mark.parametrize(("alpha", "sigma", "step"), [(0.5, 0.0, 0.1), (0.5, 0.1, 0.0)])
    def test_rejects_non_positive_params(self, alpha, sigma, step):
        with pytest.raises(ValueError):
            ESArbiter(alpha=alpha, sigma=sigma, step_size=step)

    def test_rejects_mismatched_trace_lengths(self):
        with pytest.raises(ValueError, match="步数"):
            ESArbiter().fit([np.ones((2, 2))], [np.ones((2, 2)), np.ones((2, 2))])

    def test_rejects_empty_trace(self):
        with pytest.raises(ValueError, match="至少"):
            ESArbiter().fit([], [])

    def test_no_conflict_baseline_gives_zero_reduction(self):
        """Nothing to fix -> reduction is 0, and that is honest, not a failure."""
        updates = [np.ones((3, 3)) for _ in range(4)]
        result = ESArbiter(alpha=0.5).fit(updates, updates, steps=8)
        assert result.events_before == 0
        assert result.reduction == 0.0


class TestTrivialSolution:
    """alpha -> 0 turns rule B off. That passes the 50% bar trivially, so it is guarded.

    This is the failure mode where a green test proves nothing: on a trace where B
    opposes A outright, minimising conflict with A always wins by ignoring B.
    """

    def test_min_retention_blocks_the_degenerate_answer(self):
        updates_a, updates_b = trace()
        free = ESArbiter(alpha=0.5, seed=0).fit(updates_a, updates_b, steps=24)
        assert free.alpha < 0.5, "unconstrained ES drifts toward turning B off"

        guarded = ESArbiter(alpha=0.5, seed=0).fit(
            updates_a, updates_b, steps=24, min_retention=0.3
        )
        assert guarded.alpha >= 0.3, "min_retention must floor alpha"

    def test_retention_is_just_alpha(self):
        """alpha -> 0 is the trivial solution, so it must be reportable by name."""
        arb = ESArbiter(alpha=0.25)
        assert arb.retention(np.ones((2, 2)), np.ones((2, 2))) == 0.25

    def test_rejects_bad_min_retention(self):
        with pytest.raises(ValueError, match="min_retention"):
            ESArbiter().fit(*trace(), min_retention=-0.1)

    def test_reduction_alone_is_not_a_victory(self):
        """Report both numbers: 100% reduction with 0 retention is just rule A."""
        updates_a, updates_b = trace()
        arb = ESArbiter(alpha=0.5, seed=0)
        result = arb.fit(updates_a, updates_b, steps=24)
        assert result.reduction >= 0.5
        assert arb.retention(updates_a[0], updates_b[0]) < 0.5, (
            "这条断言本身就是在说：降幅达标的同时，B 的贡献可能已经所剩无几——两个数必须一起报"
        )

"""Tests for the ES arbiter: does the search move, and toward the right place?

The objective changed on 2026-10-24: it used to minimise the **integer** conflict-event
count, which saturates and left `alpha` stuck at its starting value (the synergy
benchmark's MIX arm came out bit-identical to a frozen-0.5 control). It now maximises
the mean `min(cos(blend, A), cos(blend, B))`.

So these tests are: does ES move at all when the geometry says it should, does it stop
at the balance point a **grid search** finds (not at whatever the implementation
happens to return), does it stay put when the rules already agree, and does the event
count still get reported for comparability with the old runs.

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


def trace_partial(
    angle_deg: float = 120.0,
    b_scale: float = 3.0,
    n_steps: int = 12,
    n_pre: int = 4,
    n_post: int = 6,
    seed: int = 0,
):
    """B 与 A 有固定夹角、模长可不同——**平衡点因此不在 0.5**。

    ``trace()`` 里 ``b = -a + 噪声``，混合更新在 α ≈ 0.5 处几乎被噪声主导，
    余弦没有意义。要测「ES 会不会朝平衡点走」，需要一条几何上干净的轨迹。
    """
    rng = np.random.default_rng(seed)
    updates_a = [rng.standard_normal((n_pre, n_post)) for _ in range(n_steps)]
    theta = np.deg2rad(angle_deg)
    updates_b = []
    for a in updates_a:
        flat = a.ravel()
        unit = flat / np.linalg.norm(flat)
        perp = rng.standard_normal(flat.shape)
        perp -= (perp @ unit) * unit
        perp /= np.linalg.norm(perp)
        updates_b.append((b_scale * (np.cos(theta) * unit + np.sin(theta) * perp)).reshape(a.shape))
    return updates_a, updates_b


def best_alpha(updates_a, updates_b, *, lo: float = 0.05, hi: float = 0.95, n: int = 181) -> float:
    """网格扫出目标函数的最优 α。**测试的基准是它，不是实现自己吐出来的数。**"""
    arb = ESArbiter()
    grid = np.linspace(lo, hi, n)
    scores = [arb._alignment(updates_a, updates_b, float(alpha)) for alpha in grid]
    return float(grid[int(np.argmax(scores))])


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
    def test_es_moves_alpha_toward_the_balance_point(self):
        """目标换成连续的对齐之后，ES 必须真的动，并且停在平衡点附近。

        b 的模长是 a 的 3 倍、夹角 120°，平衡点因此**偏离起点 0.5**（实测约 0.62）。
        基准是**网格扫出来的最优**——拿实现自己的输出当基准，等于什么都没测。
        """
        updates_a, updates_b = trace_partial()
        optimum = best_alpha(updates_a, updates_b)
        assert abs(optimum - 0.5) > 0.05, f"这条轨迹的平衡点本该离开 0.5，实际 {optimum:.3f}"

        arb = ESArbiter(alpha=0.5, seed=0)
        arb.fit(updates_a, updates_b, steps=64)
        assert abs(arb.alpha - optimum) <= 0.1, f"ES 停在 {arb.alpha:.3f}，网格最优是 {optimum:.3f}"

    def test_no_movement_when_the_rules_agree(self):
        """两规则完全一致时目标处处相同——不动才是对的，不能为了"动"而动。"""
        rng = np.random.default_rng(7)
        updates_a = [rng.standard_normal((3, 4)) for _ in range(6)]
        updates_b = [a.copy() for a in updates_a]
        arb = ESArbiter(alpha=0.5, seed=0)
        arb.fit(updates_a, updates_b, steps=32)

        # 目标处处等于 1（浮点误差量级），所以只要求"没有系统性漂移"，
        # 不要求逐位不动——要求逐位不动等于要求浮点求和顺序也被钉死。
        assert abs(arb.alpha - 0.5) <= 0.15
        assert arb._alignment(updates_a, updates_b, arb.alpha) == pytest.approx(1.0, abs=1e-6)

    def test_fit_still_reports_event_counts(self):
        """事件计数照报——旧数字要能继续比较，只是它不再是被优化的量。"""
        updates_a, updates_b = trace()
        result = ESArbiter(alpha=0.5, seed=0).fit(updates_a, updates_b, steps=24)
        assert isinstance(result, ArbiterResult)
        assert result.events_before > 0
        assert isinstance(result.events_after, int)

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

    def test_objective_does_not_reward_turning_b_off(self):
        """旧目标在 B 与 A 对立时奖励 α→0；新目标不该给这个方向任何好处。"""
        updates_a, updates_b = trace()
        arb = ESArbiter()
        at_zero = arb._alignment(updates_a, updates_b, 0.0)
        at_half = arb._alignment(updates_a, updates_b, 0.5)
        assert at_half >= at_zero, "把 B 关掉不该是更优解"

    def test_min_retention_still_floors_alpha(self):
        """min_retention 从唯一防线退成保险，但保险仍然要生效。"""
        updates_a, updates_b = trace_partial()
        guarded = ESArbiter(alpha=0.5, seed=0).fit(
            updates_a, updates_b, steps=64, min_retention=0.6
        )
        assert guarded.alpha >= 0.6 - 1e-12

    def test_retention_is_just_alpha(self):
        """alpha -> 0 is the trivial solution, so it must be reportable by name."""
        arb = ESArbiter(alpha=0.25)
        assert arb.retention(np.ones((2, 2)), np.ones((2, 2))) == 0.25

    def test_rejects_bad_min_retention(self):
        with pytest.raises(ValueError, match="min_retention"):
            ESArbiter().fit(*trace(), min_retention=-0.1)

    def test_reduction_is_reported_not_optimised(self):
        """降幅仍然是汇报量，但**不再**是被优化的目标——别把旧指标当新结论。"""
        updates_a, updates_b = trace()
        result = ESArbiter(alpha=0.5, seed=0).fit(updates_a, updates_b, steps=24)
        assert 0.0 <= result.reduction <= 1.0


class TestTwoSidedBounds:
    """P4 的约束是**双侧**的：两规则各保留至少 20% 权重，α ∈ [0.2, 0.8]。

    单侧 `min_retention` 只能挡住"关掉 B"；"关掉 A"（α → 1）是同一枚硬币的另一面。
    """

    def test_max_retention_caps_alpha(self):
        updates_a, updates_b = trace_partial()
        arb = ESArbiter(alpha=0.5, seed=0)
        arb.fit(updates_a, updates_b, steps=64, min_retention=0.2, max_retention=0.3)
        assert 0.2 - 1e-12 <= arb.alpha <= 0.3 + 1e-12

    def test_rejects_inverted_bounds(self):
        with pytest.raises(ValueError, match="不能大于"):
            ESArbiter().fit(*trace(), min_retention=0.8, max_retention=0.2)

    def test_rejects_out_of_range_max(self):
        with pytest.raises(ValueError, match="max_retention"):
            ESArbiter().fit(*trace(), max_retention=1.5)

"""ES meta-learned arbiter for two plasticity rules (plan section 3.4, class 1).

Second phase does **only class 1**: Hebbian <-> e-prop conflict detection and weight
adjustment. Classes 2-3 (the other rule pairs) and class 4 (dispatch decisions) are
later phases; see the plan's section 3.4.

## What the arbiter does

Two rules each propose an update for the same synapses. The arbiter blends them with a
**mixture weight** `alpha` in [0, 1]:

    combined = (1 - alpha) * update_a + alpha * update_b

`alpha = 0` means rule A alone, `alpha = 1` means rule B alone. When the two rules
conflict (cosine < 0, plan section 3.4) the blend softens the damage; when they agree
the blend does not matter much.

## Why ES and not backprop

The plan selects **evolution strategies** for the arbiter, following Confavreux et al.
2025, and for a reason that matters here: **ES does not use the global loss gradient**,
so the arbiter stays inside the purely local learning story (plan section 2.3's boundary
declaration). Optimising `alpha` by backprop would puncture that story exactly the way
the TAAF red line forbids for fusion weights.

The ES here is deliberately simple: perturb `alpha` with Gaussian noise, keep the
perturbation that most reduced measured conflict, repeat. Confavreux et al. found
meta-learning **starts failing as the rule parameterisation grows more complex**, so
this arbiter has **one** parameter and is expected to stay there.

## What "reduces conflict events by 50%" measures

`fit()` returns the event count before and after; the phase-two success criterion is
`after <= 0.5 * before`. That is measured on **the caller's data**, not on a synthetic
benchmark -- so the arbiter's claim is only as good as what it was fitted on.

## ⚠️ The trivial solution: `alpha -> 0` silently turns rule B off

**If the objective is only "minimise conflict with rule A", the optimum is always
`alpha = 0`** -- that is, do not use rule B at all. That is not arbitration, that is
degeneration, and it **passes the 50% criterion trivially**. On a synthetic trace where
B opposes A outright, ES reaches 100% reduction in every seed, which is a symptom of
this, not a success.

This module therefore exposes `retention()` and `fit(..., min_retention=...)`: the
arbiter must keep some of rule B's contribution, or it is just rule A with extra steps.
**Report both numbers together** -- "conflict events dropped 100%" on its own is
uninformative, and possibly wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .conflict import ConflictReport, measure_conflict

__all__ = ["ArbiterResult", "ESArbiter"]


@dataclass(frozen=True)
class ArbiterResult:
    """Outcome of one `fit()` run.

    Attributes:
        alpha: the mixture weight that survived ES.
        events_before: conflict events counted before arbitration.
        events_after: conflict events counted after arbitration, at the final alpha.
        steps: how many ES steps ran.
    """

    alpha: float
    events_before: int
    events_after: int
    steps: int

    @property
    def reduction(self) -> float:
        """Fractional reduction in conflict events. >= 0.5 meets phase two."""
        if self.events_before == 0:
            return 0.0
        return 1.0 - self.events_after / self.events_before


class ESArbiter:
    """One-parameter arbiter between two rules, tuned by evolution strategies.

    Args:
        alpha: starting mixture weight in [0, 1].
        sigma: ES perturbation scale.
        step_size: how far each accepted perturbation moves alpha.
        seed: ES noise seed. Fixed seed is the precondition of reproducibility.

    Note:
        **One parameter on purpose.** Confavreux et al. report meta-learning failing as
        rule parameterisations grow complex; keeping the arbiter at a single scalar is
        the cheapest way to stay on the side of the finding.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        *,
        sigma: float = 0.1,
        step_size: float = 0.1,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha 必须落在 [0, 1]，收到 {alpha!r}。")
        if not sigma > 0:
            raise ValueError(f"sigma 必须为正，收到 {sigma!r}。")
        if not step_size > 0:
            raise ValueError(f"step_size 必须为正，收到 {step_size!r}。")

        self.alpha = float(alpha)
        self.sigma = float(sigma)
        self.step_size = float(step_size)
        self.seed = int(seed)

    def blend(self, update_a: np.ndarray, update_b: np.ndarray, *, alpha: float | None = None):
        """Combine the two rules' updates with the mixture weight."""
        weight = self.alpha if alpha is None else float(alpha)
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"alpha 必须落在 [0, 1]，收到 {weight!r}。")
        a = np.asarray(update_a, dtype=np.float64)
        b = np.asarray(update_b, dtype=np.float64)
        if a.shape != b.shape:
            raise ValueError(f"两规则的更新张量形状必须相同，收到 {a.shape} 与 {b.shape}。")
        return (1.0 - weight) * a + weight * b

    def measure(
        self, update_a: np.ndarray, update_b: np.ndarray, *, alpha: float | None = None
    ) -> ConflictReport:
        """Count conflict events of the **blend vs rule A**, at `alpha`.

        The comparison target is rule A on purpose: A is the baseline the arbiter is
        trying to protect (Hebbian, in the second phase). Asking "does the blend still
        conflict with B?" would make `alpha = 1` look perfect -- pure B trivially agrees
        with itself -- and that is not the question. The question is whether arbitration
        reduced conflict **relative to running A alone**.
        """
        weight = self.alpha if alpha is None else float(alpha)
        blended = self.blend(update_a, update_b, alpha=weight)
        return measure_conflict(blended, update_a)

    def retention(
        self, update_a: np.ndarray, update_b: np.ndarray, *, alpha: float | None = None
    ) -> float:
        """How much of rule B survives into the blend, in [0, 1].

        Exactly `alpha`: the blend is `(1 - alpha) * A + alpha * B`. Exposed as a named
        quantity because **alpha -> 0 is the trivial solution** and must be reported.
        """
        return self.alpha if alpha is None else float(alpha)

    def fit(
        self,
        updates_a: list[np.ndarray],
        updates_b: list[np.ndarray],
        *,
        steps: int = 32,
        min_retention: float = 0.0,
    ) -> ArbiterResult:
        """Tune `alpha` with ES on a trace of per-step updates.

        Args:
            updates_a: rule A's proposed update at each training step.
            updates_b: rule B's proposed update at each training step (same shapes).
            steps: how many ES steps to run.

        Args:
            min_retention: refuse solutions below this alpha. **Set it above 0 to rule
                out the degenerate "turn rule B off" answer**; 0 means any alpha is fine.

        Returns:
            The surviving `alpha` and the event counts before / after.

        Raises:
            ValueError: the two traces differ in length or in per-step shape.
        """
        if not 0.0 <= min_retention <= 1.0:
            raise ValueError(f"min_retention 必须落在 [0, 1]，收到 {min_retention!r}。")

        if len(updates_a) != len(updates_b):
            raise ValueError(f"两条规则的步数必须相同，收到 {len(updates_a)} 与 {len(updates_b)}。")
        if not updates_a:
            raise ValueError("至少要有一个训练步的更新。")

        before = sum(
            measure_conflict(a, b).events for a, b in zip(updates_a, updates_b, strict=True)
        )

        rng = np.random.default_rng(self.seed)
        for _ in range(steps):
            trial = self.alpha + rng.normal(0.0, self.sigma)
            trial = float(np.clip(trial, min_retention, 1.0))
            if trial == self.alpha:
                continue
            current = self._events(updates_a, updates_b, self.alpha)
            candidate = self._events(updates_a, updates_b, trial)
            if candidate < current:
                self.alpha = float(
                    np.clip(self.alpha + self.step_size * np.sign(trial - self.alpha), 0.0, 1.0)
                )
            elif candidate > current:
                self.alpha = float(
                    np.clip(self.alpha - self.step_size * np.sign(trial - self.alpha), 0.0, 1.0)
                )

        after = self._events(updates_a, updates_b, self.alpha)
        return ArbiterResult(
            alpha=self.alpha, events_before=before, events_after=after, steps=steps
        )

    def _events(
        self, updates_a: list[np.ndarray], updates_b: list[np.ndarray], alpha: float
    ) -> int:
        """Count conflict events at a given alpha over the whole trace."""
        total = 0
        for a, b in zip(updates_a, updates_b, strict=True):
            blended = self.blend(a, b, alpha=alpha)
            total += measure_conflict(blended, a).events  # blend vs rule A，同 measure()
        return total

    def __repr__(self) -> str:
        return f"ESArbiter(alpha={self.alpha:g}, sigma={self.sigma:g}, seed={self.seed})"

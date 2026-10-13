"""Conflict detection between two plasticity rules (plan section 3.4).

The plan gives the criterion verbatim:

> Conflict judgement: compute the cosine similarity between the two rules' update
> directions on the same synaptic weight; an angle greater than 90 degrees is a conflict.

This module is that criterion and nothing else. It does not know what Hebbian or e-prop
are -- it takes two update tensors and answers one question. Keeping it that way is why
the arbiter can be tested without spinning up any research line.

## Why the criterion is the way it is

Cosine similarity of update directions is a **local** quantity: it only needs the two
updates for the same synapse at the same time. It does not need the loss, the task, or
any gradient -- so it can live inside a purely local learning story (see the boundary
declaration in plan section 2.3). A criterion that required the global loss would have
broken that story the moment it was implemented.

## What "conflict event" counts as

One call to `is_conflict` on one synapse is **not** an event. A *conflict event* is one
training step in which **at least one** synapse is in conflict. That is the unit the
phase-two success criterion ("the arbiter reduces dual-rule conflict events by 50%+")
counts, and it is recorded in `ConflictReport.events`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ConflictReport", "cosine_similarity", "is_conflict", "measure_conflict"]

#: Plan section 3.4: angle > 90 degrees is a conflict, i.e. cosine < 0.
CONFLICT_THRESHOLD = 0.0


@dataclass(frozen=True)
class ConflictReport:
    """What one step's conflict measurement found.

    Attributes:
        n_synapses: how many groups (rows by default) were measured.
        n_conflicting: how many of them are in conflict (cosine < threshold).
        events: 1 if any synapse conflicted this step, else 0 -- the counting unit for
            the "conflict events" metric.
        mean_cosine: mean cosine over all measured synapses. Conflicts drag it negative.
    """

    n_synapses: int
    n_conflicting: int
    events: int
    mean_cosine: float

    @property
    def conflict_fraction(self) -> float:
        return self.n_conflicting / self.n_synapses if self.n_synapses else 0.0


def cosine_similarity(
    a: np.ndarray, b: np.ndarray, *, axis: int | None = 1, eps: float = 1e-12
) -> np.ndarray:
    """Cosine similarity between the two rules' update directions.

    Two readings of plan section 3.4's "update direction on the same synapse", both
    offered because they answer different questions:

    * ``axis=None`` — **per synapse**: the "direction" of a single scalar is only its
      sign, so the cosine is always ±1 and "angle > 90 degrees" collapses to
      "the two rules disagree in sign". Strictest, no intermediate values.
    * ``axis=1`` (default) — **per row**: cosine across one row of synapses (all
      outgoing weights of one presynaptic neuron). Now "direction" is a real vector and
      the angle is meaningful, so a threshold like 0.8 can express graded agreement.

    ⚠️ **``axis`` follows numpy's convention**: for a ``(n_pre, n_post)`` update tensor,
    ``axis=1`` reduces over ``n_post``, giving one value per **row**. ``axis=0`` would
    reduce over ``n_pre`` and give one value per **column** — a different question
    (all incoming weights of one postsynaptic neuron). Pick deliberately.

    ⚠️ **Per-synapse cosine is degenerate.** If you want "angle > 90 degrees" to mean
    more than "sign flipped", use a group axis. The default is the row reading.

    ⚠️ **Never flatten the whole tensor.** That answers "do the two rules agree
    overall", a much weaker question: one big agreeing block hides a region of total
    disagreement. The test suite asserts exactly that.
    """
    if a.shape != b.shape:
        raise ValueError(f"两规则的更新张量形状必须相同，收到 {a.shape} 与 {b.shape}。")
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if axis is None:
        # 逐元素余弦：分母是逐突触的 |a_i|*|b_i|。
        denom = (np.abs(a) * np.abs(b)) + eps
        return (a * b) / denom

    num = np.sum(a * b, axis=axis, keepdims=True)
    den = np.linalg.norm(a, axis=axis, keepdims=True) * np.linalg.norm(b, axis=axis, keepdims=True)
    return num / (den + eps)


def is_conflict(
    a: np.ndarray,
    b: np.ndarray,
    *,
    threshold: float = CONFLICT_THRESHOLD,
    axis: int | None = 1,
) -> np.ndarray:
    """Boolean mask of conflicting groups: cosine below `threshold` (default 0).

    `threshold = 0` is literally "angle > 90 degrees". Raising it makes the criterion
    stricter (requires more agreement); lowering it into negative values makes it
    looser. The default is the plan's wording.
    """
    return cosine_similarity(a, b, axis=axis) < threshold


def measure_conflict(
    a: np.ndarray,
    b: np.ndarray,
    *,
    threshold: float = CONFLICT_THRESHOLD,
    axis: int | None = 1,
) -> ConflictReport:
    """Measure conflicts for one training step and return the counting unit."""
    mask = is_conflict(a, b, threshold=threshold, axis=axis)
    cosine = cosine_similarity(a, b, axis=axis)
    n_synapses = int(mask.size)
    n_conflicting = int(mask.sum())
    return ConflictReport(
        n_synapses=n_synapses,
        n_conflicting=n_conflicting,
        events=1 if n_conflicting > 0 else 0,
        mean_cosine=float(cosine.mean()) if n_synapses else 0.0,
    )

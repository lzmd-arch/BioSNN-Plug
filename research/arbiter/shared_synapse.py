"""Wire the arbiter into both research lines: one shared input weight.

This is the piece that turns `conflict.py` / `es_arbiter.py` from parts into a
connection. Until now `research/ib_hebbian/` and `research/eprop/` were validated
independently and **no code path fed both rules' updates into the same arbiter** --
the same problem as "a green pure layer with no wiring" (see `memterm-pure-layer-not-wired`).

## Why this module exists and what it does not change

Plan section 3.4's criterion says "the two rules' update directions **on the same
synapse**". Measured fact: the two lines share **no** synapses -- their weight shapes
have an empty intersection (`ib_hebbian` is a feed-forward MLP at (1024, 784) /
(1024, 1024); `eprop` is a single recurrent ALIF at (784, 256) / (256, 256)). See
plan revision note 28.

Rather than resize either validated network (which would invalidate the accepted
numbers 0.9779 and 0.7764), this module gives both rules **one shared input weight**:

* the shared matrix is `(n_in, n_shared)` in `eprop`'s convention;
* `ib_hebbian`'s first layer stores the **transpose** of the same numbers -- the two
  frameworks disagree on `(in, out)` vs `(out, in)`, so a single `nn.Parameter` cannot
  be shared by pointer;
* each step **copies the numbers both ways**, so "same synapse" holds **numerically**,
  which is what the criterion asks for.

The two downstream networks stay independent and keep their accepted widths. Only the
input->hidden mapping is shared, which is exactly where the two rules would compete.

## What the arbiter blends

Two rules each propose an update for that shared matrix. They are aligned to
`(n_in, n_shared)` before comparison (`ib_hebbian`'s update is transposed), then
`ESArbiter.blend` mixes them by `alpha` and the result is written back to both sides.

**Report conflict events and retention together.** `alpha -> 0` is the trivial solution
that switches one rule off; see `es_arbiter.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .conflict import ConflictReport, measure_conflict
from .es_arbiter import ArbiterResult, ESArbiter

__all__ = ["SharedSynapseConfig", "SharedSynapseHarness", "StepRecord"]


@dataclass(frozen=True)
class SharedSynapseConfig:
    """Shape of the shared input mapping and of both downstreams.

    Attributes:
        n_in: input dimension (784 for MNIST).
        n_shared: width of the shared input->hidden mapping. **Must be divisible by
            `n_groups`** -- `IBHebbianPerceptron` groups its units, so a width that is
            not a multiple of `n_groups` raises at construction.
        n_classes: readout classes.
        n_groups: `ib_hebbian`'s grouping factor. Defaults to `None`, which picks the
            largest divisor of `n_shared` up to 16.
    """

    n_in: int = 784
    n_shared: int = 256
    n_classes: int = 10
    n_groups: int | None = None

    @property
    def groups(self) -> int:
        """The grouping factor actually used, guaranteed to divide `n_shared`."""
        if self.n_groups is not None:
            if self.n_shared % self.n_groups != 0:
                raise ValueError(f"n_shared={self.n_shared} 无法被 n_groups={self.n_groups} 整除。")
            return self.n_groups
        for candidate in range(min(16, self.n_shared), 0, -1):
            if self.n_shared % candidate == 0:
                return candidate
        return 1


@dataclass(frozen=True)
class StepRecord:
    """One training step through the shared synapse."""

    conflict: ConflictReport
    alpha: float
    retention: float


class SharedSynapseHarness:
    """Run both rules over one shared input weight and arbitrate their updates.

    Args:
        config: dimensions of the shared mapping.
        arbiter: the ES arbiter. Its `alpha` is the mixture weight.
        seed: weight-initialisation seed. Fixed seed is the precondition of
            reproducibility.
    """

    def __init__(
        self,
        config: SharedSynapseConfig | None = None,
        *,
        arbiter: ESArbiter | None = None,
        seed: int = 0,
    ) -> None:
        from research.eprop import EPropLearner
        from research.ib_hebbian import IBHebbianPerceptron

        self.config = config or SharedSynapseConfig()
        self.arbiter = arbiter or ESArbiter(alpha=0.5, seed=seed)
        self.seed = int(seed)

        cfg = self.config
        self.eprop = EPropLearner(n_in=cfg.n_in, n_rec=cfg.n_shared, n_out=cfg.n_classes)
        self.hebbian = IBHebbianPerceptron(
            cfg.n_in,
            width=cfg.n_shared,
            n_layers=1,
            n_classes=cfg.n_classes,
            n_groups=cfg.groups,
        )
        self._shared = self.eprop.cell.w_in.detach().clone()
        self.sync_from_shared()

    # ------------------------------------------------------------ 数值共享

    @property
    def shared_weight(self) -> torch.Tensor:
        """The shared input->hidden matrix, in `(n_in, n_shared)` convention."""
        return self._shared

    def sync_from_shared(self) -> None:
        """Copy the shared numbers into both networks (transpose for ib_hebbian)."""
        self.eprop.cell.w_in.data.copy_(self._shared)
        self.hebbian.layers[0].linear.weight.data.copy_(self._shared.T)

    def align_update(self, update: torch.Tensor, *, source: str) -> torch.Tensor:
        """Align a rule's update to `(n_in, n_shared)`.

        `eprop` already uses that convention; `ib_hebbian`'s is `(n_shared, n_in)`,
        so its update is transposed here. **This is where the two frameworks' storage
        conventions meet** -- getting it backwards silently compares a matrix with its
        own transpose and reports a meaningless cosine.
        """
        tensor = torch.as_tensor(update, dtype=self._shared.dtype)
        if source == "eprop":
            if tuple(tensor.shape) != tuple(self._shared.shape):
                raise ValueError(
                    f"eprop 的更新应为 {tuple(self._shared.shape)}，收到 {tuple(tensor.shape)}。"
                )
            return tensor
        if source == "hebbian":
            if tuple(tensor.shape) != tuple(self._shared.T.shape):
                raise ValueError(
                    f"hebbian 的更新应为 {tuple(self._shared.T.shape)}，收到 {tuple(tensor.shape)}。"
                )
            return tensor.T
        raise ValueError(f"source 必须是 'eprop' 或 'hebbian'，收到 {source!r}。")

    # ------------------------------------------------------------ 一步训练

    def step(
        self,
        update_eprop: torch.Tensor,
        update_hebbian: torch.Tensor,
    ) -> StepRecord:
        """One step: align, measure conflict, blend, write back to both sides."""
        a = self.align_update(update_eprop, source="eprop")
        b = self.align_update(update_hebbian, source="hebbian")

        conflict = measure_conflict(a.detach().cpu().numpy(), b.detach().cpu().numpy())

        blended = self.arbiter.blend(a.detach().cpu().numpy(), b.detach().cpu().numpy())
        self._shared = torch.as_tensor(blended, dtype=self._shared.dtype)
        self.sync_from_shared()

        return StepRecord(
            conflict=conflict,
            alpha=self.arbiter.alpha,
            retention=self.arbiter.retention(a, b),
        )

    def fit_arbiter(
        self,
        updates_eprop: list[torch.Tensor],
        updates_hebbian: list[torch.Tensor],
        *,
        steps: int = 32,
        min_retention: float = 0.0,
        max_retention: float = 1.0,
    ) -> ArbiterResult:
        """Tune `alpha` on a trace of both rules' updates, then keep the survivor."""
        aligned_a = [
            self.align_update(u, source="eprop").detach().cpu().numpy() for u in updates_eprop
        ]
        aligned_b = [
            self.align_update(u, source="hebbian").detach().cpu().numpy() for u in updates_hebbian
        ]
        result = self.arbiter.fit(
            aligned_a,
            aligned_b,
            steps=steps,
            min_retention=min_retention,
            max_retention=max_retention,
        )
        self.sync_from_shared()
        return result

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"SharedSynapseHarness(n_in={cfg.n_in}, n_shared={cfg.n_shared}, "
            f"alpha={self.arbiter.alpha:g})"
        )

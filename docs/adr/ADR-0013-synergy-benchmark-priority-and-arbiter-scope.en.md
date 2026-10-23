# ADR-0013: the synergy benchmark becomes P0, the shared-interface plan is rejected, three-way mixing stays in phase five

[中文](ADR-0013-synergy-benchmark-priority-and-arbiter-scope.md) · [日本語](ADR-0013-synergy-benchmark-priority-and-arbiter-scope.ja.md)

- **Status**: Accepted
- **Date**: 2026-10-18 (ruling) · 2026-10-24 (recorded)
- **Related**: plan §12.7 (external references); [`sweep_results/arbiter_synergy_task_spec.md`](../../sweep_results/arbiter_synergy_task_spec.md) (task spec and criterion evolution); [research/arbiter](../../research/arbiter/README.en.md) (implementation and results); [ADR-0010](ADR-0010-eprop-quadratic-storage-and-trace-propagation.en.md)

## Context

The P4 paper-skeleton discussion put two proposals on the table that would change the roadmap:

1. **A shared interface**: let Hebbian and e-prop cooperate through an explicit shared interface rather than sharing one weight tensor;
2. **Move priority forward**: pull phase five's three-way mixing and scheduling dimension into the current phase.

Both touch the semantic boundary of the word "synergy". Without pinning it down, every later
"synergy" number mixes two different mechanisms with different falsifiability.

## Decision

**① The shared-interface plan is rejected.** It is a semantic distortion: it measures a bridge
between two representations, not two rules cooperating on the same set of synapses. It survives
**only as a clearly labelled counterfactual arm**, and must never be presented alongside the
native-point results.

**② Priority inverted: the synergy benchmark is P0.** The benchmark's point is the **native point
of Hebbian ↔ e-prop** — the weight the two lines already share (input → hidden). The original
criterion was "each alone fails, only the mixture works". The e-prop ↔ R-STDP counterfactual arm
comes **after** the benchmark, and is designed once benchmark results exist.

**③ Three-way mixing and the scheduling dimension stay in phase five.** They are not pulled
forward because the benchmark needs them.

**④ Retained**: the test infrastructure that passes 43 tests stays; the per-row cosine and
retention **joint reporting** of conflict adjudication stays.

## Consequences

- **The native point's shape is set by the task**: sMNIST feeds one row of pixels per step, so the
  shared weight is `28 → 256`, not static MNIST's `784 → 256`. This is not a smaller model; it is
  what the task's form dictates.
- **Criterion evolution must leave a trace**: the original criterion demanded "H alone fails,
  E alone fails", and measurement refuted it — pure Hebbian lifted the Fisher ratio from 0.309 to
  0.472. On 2026-10-18 it was replaced by the **synergy-margin criterion**
  (`Fisher(MIX) − max(Fisher(H), Fisher(E)) ≥ δ = 0.05`), written **before any MIX number was seen**.
  Changing a criterion without recording it in an ADR leaves later readers unable to explain why
  the conclusion changed.
- **The counterfactual arm does not disappear, it gets labelled**: the shared-interface route can
  still be run, results must be written as counterfactual, and it may not be used to answer
  "does synergy hold".
- **Phase five's boundary stays clean**: three-way mixing and the scheduling dimension appear in
  no experiment of the current phase.

⚠️ This ADR fixes the rulings and the boundaries that follow; it adds no empirical claim. The
benchmark's own numbers live in
[`sweep_results/arbiter_synergy_task_spec.md`](../../sweep_results/arbiter_synergy_task_spec.md)
and the results directory.

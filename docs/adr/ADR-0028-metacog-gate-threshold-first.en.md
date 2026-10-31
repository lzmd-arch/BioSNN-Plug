# ADR-0028: Metacognitive gate — implement the real mechanism; the product path starts with a measured threshold

[中文](ADR-0028-metacog-gate-threshold-first.md) · [日本語](ADR-0028-metacog-gate-threshold-first.ja.md)

- **Status**: Accepted
- **Date**: 2026-11-01
- **Related**: Metacog v8–v11, [ADR-0021](ADR-0021-criterion-death-conditions.en.md),
  `BioSNN-Plug_项目计划书_v6.3_amend1.md`, Axo's `docs/prereg-metacog-v1.md`

## Context

An external integration brief described the "metacognitive gate" as an **already validated
512-neuron SNN component** ("MG v11 verified; the three-dimensional gate passes OOD"). The
facts, after checking both repositories:

- MG v11 is a **kNN(5) density gate** over three readout-residual scalars
  `[max_errors, errors, state_norm]` on a 4-dimensional episodic-memory task. No spikes, no
  ALIF, no e-prop, and no weight file was ever produced.
- Its own README states that "all four families pass" covers four **synthetic input
  transforms** only — it is not a general OOD proof.
- `biosnn_bus`'s `SpikeBus` is still a skeleton (no `state()`, no layers).

Axo implemented the **real** mechanism (the three scalars mapped onto recall-ledger
features `top_score` / `mean_topk` / `density`) and evaluated it offline on a copy of the
217-entry real library: gate AUROC **1.000**, while a single-feature `top_score` threshold
also scored **1.000** (hard positives 0.996 vs 0.996). The pre-registered criterion
(≥ +0.05 increment) **failed**, so the gate was not wired.

## Decision

1. **Implement the mechanism, not the name.** Any request to "integrate MG v11 / a
   metacognitive SNN" is checked against the real
   `research/cognitive/run_metacognitive_gate_v11.py` first; proposals that presuppose a
   non-existent component are rewritten.
2. **The product path starts with a measured threshold.** "Do not inject weak memories" is
   implemented as one fused-score threshold in the recall layer, calibrated on the real
   library (novel topics' top score maxes at 0.163; positives' p5 is 0.211 → threshold 0.20).
3. **Every new gate must show an increment**: AUROC ≥ best single-feature baseline + 0.05
   before wiring; otherwise it is treated as a trivial solution (same discipline as
   [ADR-0021](ADR-0021-criterion-death-conditions.en.md)).
4. **The spiking version is scheduled, not implemented.** Entry conditions (bus has layers /
   a labelled natural-language OOD set exists / the rate-based version has shipped and been
   measured) are recorded in `BioSNN-Plug_项目计划书_v6.3_amend1.md`.

## Consequences

- The research-layer MG v11 stays in the repository as a component that works on its own
  benchmark; the product does not use it.
- One less opaque layer; the threshold is explainable, revertible, and backed by a
  published distribution curve.
- Cost: in the product, "metacognition" degrades into "a cutoff". That is the only claim the
  current evidence supports.

## Alternatives

- **Wire the SNN gate as briefed**: the component does not exist; it would amount to starting
  research from scratch and skipping validation. Rejected.
- **Wire the three-feature gate (no better than a threshold)**: identical AUROC, more
  complexity and more knobs. Rejected.
- **No filtering at all**: novel topics still inject ~60 weakly related memories per turn — a
  real product wound. Rejected.

# ADR-0030: The main line shifts to spiking SoS

[中文](ADR-0030-mainline-shift-to-spiking-sos.md) · [日本語](ADR-0030-mainline-shift-to-spiking-sos.ja.md)

- **Status**: Accepted
- **Date**: 2026-11-02
- **Related**: [ADR-0029](ADR-0029-research-line-archival.en.md), [stage-four status](../stage4_status.en.md), [Paper C design draft](../../paper/c/DESIGN.md)

## Context

W3's spiking actor has hit **the same failure shape** for a dozen-plus rounds: the readout is limited
by noise and quantisation together, the direction of the difference is there but its relative size is
only 0.5%-7.3%, and neither changing the readout nor the threshold enlarges it (archived in ADR-0029).
Pouring more compute into "learn the policy directly" would treat "tweak it again" as a method.

Stage four left two usable pieces, but **neither is a policy**:

- the world model (AUROC 1.0 / FP 0.0009) is a **one-step dynamics predictor**: given a state and an
  action it predicts the next state;
- working memory (0.9990) can hold the current state, but **supports neither branching nor reset**.

The gap between them is "**use prediction to find action sequences**". SoS (Gandhi et al., CoLM 2024,
arXiv 2404.03683) supplies the evidence: teaching a model the **search process** (exploration, dead
ends, backtracking, retrying) builds search ability better than teaching it optimal trajectories.

The user decided on 2026-11-02 that spiking SoS becomes the new main line.

## Decision

1. **The main line shifts to spiking SoS.** Compute and attention go there first; Paper C is demoted
   to a non-main line.
2. **Three prerequisites land first; no training runs before they do**:
   - the world model emits an **uncertainty estimate** (predictive variance), not just a point prediction;
   - **R-STDP's reward is redefined**: from an external success signal to **the reduction of prediction error**;
   - a **multi-trajectory experimental frame**: N parallel trajectories plus reset/backtrack interfaces.
3. Until those land, the only permitted outputs are: **an experiment design, a pre-registration, and the
   skeleton of a minimal environment (grid/maze)**. The pre-registration must be written into
   `research/sos/`, criteria on disk first.

## Re-evaluation conditions (written down now)

| Condition | Conclusion |
| :--- | :--- |
| Success rate **< 80%** on a simple grid/maze | Fall back, no further investment, archive as a negative result |
| Search steps **do not adapt to task difficulty** | Same -- it means a fixed policy was learned, not search |
| H1 fails (process reward does not beat outcome reward) | Record the negative result and re-evaluate the line |

Any trigger follows ADR-0021: open a new pre-registration or a new ADR; never edit an old criterion.

## Consequences

- Compute moves from Paper C and the scale experiment to SoS;
- Paper C's design draft (`paper/c/DESIGN.md`) is kept but **not advanced** -- its prerequisites are the
  40 annotations and recall coverage, neither of which is on the main line now;
- Cost: **the risk of SoS failing is high** -- W3 already failed on this line once. The reason to accept
  it: SoS changes the **learning signal** (process versus outcome), not a parameter. Parameter-level
  failures are already pinned down by ADR-0029;
- Cost: SoS needs new parts (uncertainty, multiple trajectories), so there will be no publishable
  numbers in the short term.

## Alternatives

- **Keep training the spiking actor directly**: archived in ADR-0029 after a dozen-plus rounds of the
  same failure shape. Rejected.
- **Make Paper C the main line**: LoCoMo evidence coverage@8 is only 0.439, so an end-to-end comparison
  would mostly measure the answerer rather than the memory system. Rejected as a main-line candidate
  (the design draft is kept).
- **Do not do SoS**: the user has decided. Rejected.

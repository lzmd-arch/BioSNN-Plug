# ADR-0029: Archiving two research lines, and the boundary of "paused"

[中文](ADR-0029-research-line-archival.md) · [日本語](ADR-0029-research-line-archival.ja.md)

- **Status**: Accepted
- **Date**: 2026-11-02
- **Related**: [ADR-0018](ADR-0018-task-aware-cl-gate.en.md), [ADR-0019](ADR-0019-stage-four-entry-decoupled-from-relearn.en.md), [ADR-0021](ADR-0021-criterion-death-conditions.en.md), [stage-four status](../stage4_status.en.md), [trigger criterion v2](../../sweep_results/phase5_trigger_v2/README.md)

## Context

Three research lines have each run for many rounds and each sits in a "one more tweak might do it"
state. No document says **which line is finished and which continues**, so compute and attention are
allocated by inertia: whichever line was touched most recently keeps burning.

Where the three lines stand (all with pre-registrations and results on disk):

1. **Pure CIL (class-incremental learning without a task oracle)**: pure CIL ACC **8.76%**, AF
   **0.290**; the best task inference is **0.5855 / 0.1285**. ADR-0018 changed the CL gate to
   task-aware, which then passes (oracle per-task **0.8195 / AF 0.0**), and ADR-0019 decoupled the
   stage-four entry from that metric. **No criterion hangs on pure CIL any more.**
2. **W3's spiking actor-critic (R-STDP + TD-LTP)**: ten rounds of diagnosis (D10-D15) concluded that
   the readout is limited by noise and quantisation together -- the direction of the difference is
   there, its size is only **0.5%-7.3%**; changing the readout or the threshold does not enlarge it,
   and only K-window averaging helps, with a gain proportional to **K^0.6**. The reservoir actor v3
   (frozen reservoir plus two readouts) has H1/H2 medians of **10.0 / 12.0** over 10 seeds against a
   criterion of 200: **0/10**.
3. **The phase-five trigger**: M3l-M3p kept showing "swap the corpus and the advantage disappears".
   Criterion v2 separated the failure: **false-trigger control holds** (mean false-trigger rate over
   500 random splits is **0.100 = alpha**), and **the failure sits in the scoring function** (mean hit
   rate **0.385**, only **9%** of splits reach 0.50). Criterion v3 is now testing something more
   basic: M3p's target turns are scripted fact sentences, so surface cues alone may solve the corpus.

## Decision

1. **Pure CIL is archived.** No new pure-CIL experiments, no further compute. The CL acceptance
   wording is ADR-0018's.
2. **W3's spiking actor-critic is archived.** No more training on that line. The rate-based execution
   layer stays as it is -- what is archived is "policy learning through a spiking readout", not the
   execution layer.
3. **The phase-five trigger is not archived; it is marked "paused".** Its criterion is sound and its
   false-trigger control holds; what is broken is the score and (possibly) the corpus. Until v3
   reports, no new score is tried (that needs its own pre-registration) and the line is not declared
   dead.

## Re-opening conditions (written down now, not judged after the fact)

| Line | What evidence would void the archive |
| :--- | :--- |
| Pure CIL | A long sequence with **real task boundaries** (not a hand-segmented benchmark) where task inference reaches **0.8** or above |
| W3 spiking actor | A scale experiment (5k-8k neurons) showing the readout limit comes from **scale**, not from rate-based readout as such |
| Trigger | A corpus whose **target turns are not scripted**, with some score family reaching **T-2 >= 0.50** |

All three require a **new pre-registration**; old results may not be reinterpreted as a pass (ADR-0021).

## Consequences

- Compute comes out of two dead lines and goes to Paper C and the scale experiment;
- **Archiving is not deleting evidence**: every pre-registration, result and negative result stays in
  the repository and can be cited again;
- Cost: if someone really does solve the readout limit or obtains real task boundaries, these
  archives will not revive on their own -- someone has to open a new pre-registration per the table
  above;
- Cost: the trigger only being "paused" means it keeps a little attention until v3 reports. That is
  deliberate: v2 has just shown the criterion itself is intact, and archiving now would throw away a
  part that can still be fixed.

## Alternatives

- **Archive all three, trigger included**: v3 has not run, and v2 has already localised the failure
  to the score. Archiving now would substitute "results are bad" for "we understand why", which this
  repository has consistently refused. Rejected.
- **Archive none, keep iterating**: the first two have each hit the **same failure shape** for ten-plus
  rounds (readout limit, unreachable criterion); more investment treats "tweak it again" as a method.
  Rejected.
- **Time-based expiry** (e.g. "archive after three months without progress"): expiry without an
  evidence condition is just laziness. Rejected.

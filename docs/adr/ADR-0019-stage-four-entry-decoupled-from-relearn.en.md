# ADR-0019: Decouple the phase-4 entry from "relearn 3×"; record 3× as unmet and open

[中文](ADR-0019-stage-four-entry-decoupled-from-relearn.md) · [日本語](ADR-0019-stage-four-entry-decoupled-from-relearn.ja.md)

- **Status**: Accepted
- **Date**: 2026-10-26
- **Related**: [ADR-0015](ADR-0015-world-model-as-phase-four-component.en.md),
  [ADR-0018](ADR-0018-task-aware-cl-gate.en.md);
  [`relearn_3x/README.md`](../../sweep_results/relearn_3x/README.md),
  [`relearn_3x/formal_g20/README.md`](../../sweep_results/relearn_3x/formal_g20/README.md),
  [`relearn_3x/calibration_v3/README.md`](../../sweep_results/relearn_3x/calibration_v3/README.md)

## Context

The plan made phase-3 acceptance the entry to phase 4. One acceptance item is "after a rule
change, relearning is at least 3× faster than training from scratch." By 2026-10-26 this had
three pre-registered rounds on CartPole with the rate-based R-STDP agent:

| Round | Rule | warm initial <150 | median ratio | Verdict |
| :--- | :--- | :---: | ---: | :--- |
| v1 | gravity 14 | 1/10 | 1.03 | rule too weak (9/10 already pass) |
| v2 formal | gravity 20 | 2/10 | 0.957 | still too weak (8/10 already pass) |
| v3 calibration | pole 0.3 / 0.2, gravity 30 | 1/5 / 2/5 / 3/5 | ≈1.06 | all three candidates fail; Phase A median 186.3, pilot batch unusable |

v2's exploratory threshold scan (250/300/350) gave median ratios 0.95 / 0.78 / 0.88 — no 3×
either; it is not a pre-registered result and only shows that changing the threshold does not
fix it. The old policy is robust to all three rule families, while scratch always learns
within 800 episodes; making the rule harsher also makes scratch harder to learn.

Meanwhile the CL gate is task-aware CIL per [ADR-0018](ADR-0018-task-aware-cl-gate.en.md)
and passes (oracle 0.8195 / 0.0, constructive), and the world-model groundwork (ADR-0015 P1/P2)
is under way. **Keeping phase 4 fully blocked by a criterion that three rounds failed to reach
costs more than it buys.**

## Decision

1. **Decouple the phase-4 entry from "relearn 3×"**: phase 4 may start once the **task-aware
   CL gate (ADR-0018)** passes; it no longer waits for 3×.
2. **3× is recorded as unmet, not passed.** Papers, plans and external material must state
   "3× unmet (see ADR-0019)"; starting phase 4 must not be written as full phase-3 acceptance.
3. **3× becomes an open phase-4 research item**: practical relearning (warm keeps the final
   exploration temperature and learning rate, scratch uses the full schedule) or a different
   task / meta-learning may be tried. These are **new claims** ("practical relearning speed" /
   new task) and each needs its own pre-registration; they must not be mixed with the original 3×.
4. **All phase-3 results stay as they are**: the pure-CIL negatives, the task-inference boundary,
   and the three relearning negatives are not rewritten; this ADR changes only the **phase-4
   entry condition**.
5. Phase 4 starts with the world-model groundwork per ADR-0015; neurogenesis / scale expansion /
   metacognitive gating continue under their own pre-registrations.

## Consequences

- **Phase 4 can start**, no longer locked behind an unmet criterion;
- **Cost**: phase 3 is recorded as **partially accepted** — CL (task-aware) passes, relearn 3×
  is unmet. This is a **governance decision, not a measurement**; all external material must say so;
- 3× remains an open risk: if practical relearning in phase 4 also fails to show 3×, the metric
  may need redefinition or retirement in a later review;
- In the original "consolidation → AF → phase 3 → phase 4" chain, the AF part already changed
  under ADR-0018; this ADR changes the other item (relearning).

## Alternatives

- **Keep blocking**: the most conservative option, but it locks phase 4 behind a criterion three
  rounds failed to reach; whether "the boundary is measured" or "the gate must pass" is a
  governance choice.
- **Change the gate to practical relearning and wait for it**: possibly the only remaining path
  to a real 3×, but it changes the comparison (arms no longer differ only in initial weights)
  and may still fail; hence it is an open phase-4 item, not a blocking entry.
- **Change the task/algorithm and redo 3×**: a new research line, the most expensive; kept open
  rather than blocking the entry.
- **Write 3× as passed**: rejected — three pre-registered rounds do not support it.

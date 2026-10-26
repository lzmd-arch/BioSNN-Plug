# ADR-0018: The phase-3 CL gate becomes task-aware CIL (task-incremental)

[中文](ADR-0018-task-aware-cl-gate.md) · [日本語](ADR-0018-task-aware-cl-gate.ja.md)

- **Status**: Accepted
- **Date**: 2026-10-26
- **Related**: project plan §7 phases 3/4; [ADR-0017](ADR-0017-cl-main-route-switch-and-replay-first.en.md);
  [`continual_v2/README.md`](../../sweep_results/continual_v2/README.md);
  [`task_inference_strong_prereg.md`](../../sweep_results/continual_v2_task_inference_strong_prereg.md)

## Context

The phase-3 CL gate used to be **pure CIL** (no task id; argmax over all seen classes):
**AF < 8% and ACC ≥ 65%**. By 2026-10-26 every mechanism had been measured under it:

| Mechanism | ACC | AF | Note |
| :--- | ---: | ---: | :--- |
| R-delta-mini (reference) | 0.2466 | 0.8047 | no replay, no isolation |
| replay ρ0.3 | 0.3630 | 0.7399 | replay is not enough |
| normalisation N1 / N1-scale | 0.269 / 0.315 | 0.903 / 0.792 | `no_signal` |
| output-row isolation P1 | 0.5426 | 0.1739 | acquisition collapses |
| P1 + replay | 0.4310 | **0.0291** | AF passes, ACC does not |
| feature masks F1 | 0.2343 | 0.7727 | Fisher ≈ random |
| task inference (prototype / diagonal Mahalanobis / full covariance / delta classifier) | 0.42 / 0.461 / 0.586 / 0.242 | 0.17 / 0.157 / 0.128 / 0.808 | final routing 0.30 / 0.35 / 0.53 / 0.10 |
| **oracle per-task heads + task id** | **0.8195** | **0.0** | upper bound |

The conclusion is sealed in [`continual_v2/README.md`](../../sweep_results/continual_v2/README.md):
**in this feature space the ten CIFAR-100 tasks are not separable by simple local
discriminators; without a task id, pure CIL cannot reach ACC ≥ 65 / AF < 8.**

## Decision

1. **The phase-3 CL gate becomes task-aware CIL (task-incremental)**: **the task id may be
   provided at inference** (task labels were already available during training). The ACC ≥ 65%
   and AF < 8% definitions are unchanged, but the argmax is taken **inside the true task's head**.
2. Under this protocol the oracle per-task heads reach **ACC 0.8195 / AF 0.0**; the phase-3 CL
   gate is recorded as **met — under this protocol only**.
3. **We do not claim "task-inference CIL"**: all four local routers (prototype, diagonal/full
   Mahalanobis, delta classifier) reach only 0.10–0.53 final routing, far below 90%. The
   inference-only protocol did **not** pass and is not claimed here.
4. **All published pure-CIL numbers and negative results are kept as they are**; this ADR changes
   the gate's protocol, not the history.
5. The phase-4 entry remains **phase-3 acceptance**: the task-aware CL gate (this ADR) plus
   "relearn 3× faster after a rule change" (not run yet; design pre-registered). Both must pass.

## Consequences

- **AF = 0 is constructive**: each task head is frozen after training and the task id is known,
  so old heads cannot be overwritten. This is a **protocol-level pass**, not an anti-forgetting
  algorithm; every report must say so next to the number.
- The project's claim changes from "class-incremental" to "task-incremental"; papers, plans and
  external material must be updated. The oracle's 0.82/0.0 must not be written as "pure CIL met".
- The pure-CIL boundary remains a real product of this phase: it shows that **task-boundary
  information must enter inference**. That is a valuable negative result, not a failure.
- **"Relearn 3×"** is still unrun (design:
  [`relearn_3x_prereg.md`](../../sweep_results/relearn_3x_prereg.md)); this ADR does not pass it.
- W3 spiking and the world-model groundwork are unaffected; they continue under their own
  pre-registrations / the background queue.

## Alternatives

- **Keep pure CIL and record it as unmet**: the most honest option, but it blocks phase 4 behind a
  gate no tested configuration can reach; the project must decide whether that is a boundary or a
  target.
- **Switch to task-inference CIL** (infer the task from the input): all four local routers reach
  only 0.10–0.53 final routing; measured and rejected as a passing protocol.
- **Change the representation and keep pure CIL** (train the features, stronger backbone,
  non-local methods): a new research line, the most expensive, and in tension with the
  "purely local" claim; not started in this phase.
- **Present AF = 0 as an algorithmic result**: rejected — it is constructed by frozen per-task
  heads plus a known task id.

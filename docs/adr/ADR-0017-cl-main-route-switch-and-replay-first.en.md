# ADR-0017: Switching the main CL route to Route R, replay first

[中文](ADR-0017-cl-main-route-switch-and-replay-first.md) · [日本語](ADR-0017-cl-main-route-switch-and-replay-first.ja.md)

- **Status**: Accepted
- **Date**: 2026-10-25
- **Related**: plan §7 phases 3–4, §5.2; [ADR-0015](ADR-0015-world-model-as-phase-four-component.en.md); [prereg v2](../../sweep_results/continual_v2_replay_prereg.md); [formal_cil](../../sweep_results/continual_cifar100/formal_cil/README.md)

## Context

The formal phase-3 continual-learning run (W1 features + delta readout, pre-registered CIL protocol)
gives **ACC 0.0876 / AF 0.290**. On the same features, the **oracle upper bound with task id is only
0.322**. The failure is catastrophic forgetting (the final row has zeros for every old task), but the
deeper limit is the **representation**: even with task id the 65% gate is out of reach, and replay
cannot fix features that do not separate 100 classes.

The exploratory Route R (frozen ResNet18 avgpool + class prototypes) reported ACC 0.658 / AF 0.127,
but those numbers came from the **old cumulative-class-set matrix**, not the CIL protocol. They must
be re-measured before they can serve as a baseline.

## Decision

1. **Switch the main CL research route to Route R** (frozen strong representation + anti-forgetting).
   **Route W stays as the declared original route** and continues to be reported as-is; nothing is
   deleted or rewritten.
2. Order mechanisms by an **attribution chain**: **replay first**, then normalisation, then parameter
   isolation; **one mechanism at a time**.
3. **Redefine the phase-4 entry as "Route R + replay: AF < 8% and ACC ≥ 65%"** (CIL protocol). The
   concrete numbers are derived from the Step-0 re-measured baseline and dated in the results README —
   **the old 0.658/0.127 is not carried over**.
4. **Split phase 3**: 3a (closed today, sealed as-is) / 3b (replay arm; the "3× relearning" design
   comes after replay results; the spiking W3 diagnosis spends GPU only after its thresholds are
   written down).
5. Old results and old preregistrations are not edited. Together with this ADR and its date, they are
   the evidence that the route change was not made to pass a gate.

## Consequences

- Latent replay must be implemented (fixed per-class feature buffer, ρ ∈ {0.1, 0.3}) and the Route R
  baseline must be **re-measured first**;
- The schedules for "3× relearning" and the spiking diagnosis do not move, but they wait for replay
  results / written thresholds respectively;
- Phase-4 components (world model, neurogenesis) remain gated by phase-3 acceptance; the world-model
  groundwork (one-step dynamics, rollout, Acrobot) does not change the gate.

## Alternatives

- **Normalisation first**: it would contaminate the replay attribution and is a generic trick—starting
  with it gives up the answer to "which mechanism actually helps";
- **Parameter isolation (EWC family) first**: in tension with P4 §2.4 (regularisation cannot handle
  conflicts without task boundaries); one paper cannot cite it and use it at the same time;
- **Stay on Route W**: the oracle upper bound is 32.2%; the 65% gate fails at the representation
  layer, and replay cannot fix features.

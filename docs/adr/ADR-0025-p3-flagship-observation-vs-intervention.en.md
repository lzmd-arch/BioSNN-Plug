# ADR-0025: make "observation learning vs intervention learning" the phase-4 flagship problem

[中文](ADR-0025-p3-flagship-observation-vs-intervention.md) · [日本語](ADR-0025-p3-flagship-observation-vs-intervention.ja.md)

- **Status**: accepted
- **Date**: 2026-10-29
- **Related**: world model P3 v2-v6, [stage-4 status](../stage4_status.en.md), ADR-0015

## Context

Two P3 results form a sharp contrast: with **open-loop** fixed actions, termination prediction reaches AUROC **0.999996** (FP 0.000504, hit 0.998); with **closed-loop** receding-horizon v6, AUROC **1.0** and termination-step error 1.14e-5; but **500-step pure imagination** reaches only **0.63-0.67**, with the root cause located in action feedback and float32 bit-level sensitivity (structural, not tuning).

This contrast is not "a component is good or bad" but the **signature of two learning settings**: under observation learning (external action sequences fixed) the model is near-perfect; under intervention learning (the model's own actions enter the feedback loop) error accumulates. It is the sharpest and most falsifiable question in phase four.

## Decision

1. Make "**observation learning vs intervention learning**" the phase-4 flagship: every later world-model experiment must state which side it belongs to and report the difference between the two.
2. Keep the 500-step pure-imagination boundary **as is**; do not try to bypass it with threshold or precision tuning. Crossing it needs a new mechanism (e.g. action discretization or error reparameterization) and a new preregistration.
3. Other phase-4 components (WM / metacognition / episodic) are supporting parts, no longer flagships on their own.

## Consequences

- research focuses on one falsifiable difference;
- the boundary is written as a "signature", not a "defect", so it is not misread as undertraining;
- cost: the flagship will not produce a "ready" verdict soon and may stay open for a long time.

## Alternatives

- **Keep tuning the pure-imagination loop**: the root cause is float32 and action flipping, proven unreachable; rejected.
- **Drop P3**: would discard phase four's sharpest contrast; rejected.

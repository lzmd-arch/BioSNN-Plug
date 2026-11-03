# ADR-0033: Archiving the capacity axis -- more neurons and more connections both buy no accuracy

[中文](ADR-0033-capacity-axis-archival.md) · [日本語](ADR-0033-capacity-axis-archival.ja.md)

- **Status**: Accepted
- **Date**: 2026-11-03
- **Related**: [ADR-0031](ADR-0031-mainline-to-scale.en.md) (main line moved to the scale experiment),
  [ADR-0032](ADR-0032-sel-research-track.en.md), `sweep_results/scale_s2` ... `scale_s9`

## Context

ADR-0031 moved the main line to the scale experiment with a single question: **how does accuracy
grow with size, and at what cost.**

S2-S9 answered it along two axes:

| Axis | Evidence | Result |
| :--- | :--- | :--- |
| **More neurons** | S2: 1k -> 8k is +14.6pp; S6: 8k -> 32k is only +0.5pp (5 epochs); S5: 64k vs 16k is **-0.3pp**; S8 on SHD at 10 epochs: 8k **0.338** vs 32k **0.348** | nothing beyond 8k |
| **More connections** | S9 at 8k neurons, fan-in 32 -> 128: **0.774 -> 0.766 (-0.8pp)** at **3.2x** the time | nothing either |

S4 also corrected the earlier memory claim: **the 8 GB "ceiling" came from the evaluation batch**;
one training update at 64k neurons costs a few hundred MB. So the saturation is **not a resource
limit** -- the pipeline's capacity itself is used up.

**Addendum (2026-11-04, A3)**: everything above was measured **through the last-step readout**.
On the **same 8,192 neurons**, line A's A3 changed only the readout -- one time point to 28, with the
same local rule -- and moved sMNIST from **0.759 to 0.917**. So "capacity is used up" must be read as
"used up **under the last-step readout**". The decision below is **unchanged** -- more neurons and more
connections still buy nothing, and no scale run needs repeating; but **0.83 must not be quoted as this
pipeline's capability ceiling**. Source: `sweep_results/brainregions_a3/README.md`.

S6 adds the mechanism: **more training lets the small network catch up** -- at 2 epochs 32k leads 8k
by 3.1pp, at 5 epochs by 0.5pp. Part of "scale helps" was "bigger learns faster", not "bigger learns
better".

## Decision

1. **The capacity axis is archived.** Neither more neurons nor more connections is treated as a way
   to raise accuracy, and neither gets compute any more.
2. **Keep the current operating point**: **8,192 recurrent neurons with fan-in 32**, recorded as the
   pipeline's default in the benchmark.
3. **The scale experiment's assets stay citable**: the slippery-maze benchmark (`benchmarks/maze/`),
   chunked evaluation (`--predict-chunk`), the runner's `--dataset`, and every S2-S9 result directory.

## Re-opening conditions

| Condition | Note |
| :--- | :--- |
| A change in **organisation** | Region specialisation, heterogeneous neuron parameters and the like may move the capacity curve -- that needs a new pre-registration, not this ADR as a shield |
| **Training-budget** evidence | At 5/10 epochs the slope is "the gap is shrinking"; if 250 epochs reverse it, this ADR needs an amendment |
| A **new task family** | If a task appears where 8k clearly underfits (training loss still high), the capacity axis is live again for that task |

All three require a **new pre-registration**; old results may not be reinterpreted as a pass.

## Consequences

- Compute moves from widening to organisation and delivery;
- One reusable conclusion: **the capacity knee of this event-driven pipeline sits near 8k neurons**;
  beyond it time grows linearly and accuracy does not;
- Cost: **"we are simply not large enough" can no longer explain any failure.** Every miss beyond 8k
  must be attributed to another axis (organisation, task, learning signal);
- Cost: the default is pinned at 8k, so anyone running 32k must **say why** instead of assuming
  bigger is better.

## Alternatives

- **Keep adding neurons**: rejected by S5 (64k, no gain) and S8 (SHD, +1.0pp).
- **Keep raising fan-in**: rejected by S9 (-0.8pp at 3.2x time).
- **Swap tasks and retry capacity**: S7/S8 already did this with SHD, with the same +1.0pp. Rejected.
- **Write the saturation down as a resource limit**: S4 showed it is not (memory is set by the
  evaluation batch). Rejected -- that would be a false attribution.

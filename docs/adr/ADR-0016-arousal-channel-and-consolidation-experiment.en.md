# ADR-0016: the arousal channel becomes a component, and the consolidation experiment is scheduled

[中文](ADR-0016-arousal-channel-and-consolidation-experiment.md) · [日本語](ADR-0016-arousal-channel-and-consolidation-experiment.ja.md)

- **Status**: Accepted
- **Date**: 2026-10-24
- **Related**: plan §3.4 (dynamics axis), §七 (phases); [ADR-0015](ADR-0015-world-model-as-phase-four-component.en.md) (same batch); `research/arbiter` (the retention-discount item)

## Context

The dynamics-axis audit of 2026-10-24 produced three findings; this ADR settles two of them.

**① Missing: arousal has no component.** The plan lists hormones, affect, consolidation and
instinct on the dynamics axis, but **arousal** appears only as an adjective --- never a component,
a state variable or an interface. The same hole holds **instinct**: "genetically encoded wiring
rules" exists as an idea with no name, and therefore cannot be cited or accepted.

**② Inverted: consolidation runs in production and has zero research validation.** The terminal
Axo's resident `axo.busd` process consolidates ("dreams") every day, while this project has **no
experiment** measuring what consolidation buys. That is the dangerous state: **a component looks
validated because it is running**, while its benefit has never been measured.

## Decision

**① The arousal channel becomes a formal component of the dynamics axis.**

| Item | Definition |
| :--- | :--- |
| State variable | A scalar `arousal`, updated from input intensity and recent firing statistics, **never trained by backpropagation** |
| Where it acts | It modulates the **gain** of other components (consolidation frequency, exploration temperature, wake threshold); it does not touch weights directly |
| Relation to "instinct" | Instinct is named as the channel's **sub-item**: the initial values and wiring rules that exist **without learning** |
| What it is not | It does not claim equivalence to any biological construct; it is this project's own controller variable |

**② A consolidation comparison experiment is scheduled** (half a GPU-day, as a rehearsal for phase five):

| Item | Value |
| :--- | :--- |
| Question | Does consolidation improve **delayed recall**, with versus without? |
| Pre-registered metrics | Recall difference after N delayed rounds (paired, per seed), plus the cost of consolidation (time/energy proxy) |
| Controls | No consolidation; consolidation with shuffled order (rules out "just ran it again") |
| Claim boundary | Holds for the current task and implementation only; no extrapolation to biological sleep |

## Consequences

- **"The daemon is running" no longer counts as validation.** Production runs a product; research
  needs numbers. Both use the same `run_consolidation`, but **only an experiment can validate it**.
- Once arousal is a component, every state-dependent gain must be written as a function of that
  scalar rather than as magic constants scattered through the code.
- The consolidation experiment shares ADR-0015's discipline: pre-register the metric, then look.
- This closes the two ❌ entries in the plan's dynamics-axis audit (arousal unnamed, instinct
  unnamed). The `retention = 0.3` discount remains open in `research/arbiter` under ADR-0014's wording.

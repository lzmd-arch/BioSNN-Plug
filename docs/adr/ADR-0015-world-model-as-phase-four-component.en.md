# ADR-0015: the world model becomes a formal phase-four component

[中文](ADR-0015-world-model-as-phase-four-component.md) · [日本語](ADR-0015-world-model-as-phase-four-component.ja.md)

- **Status**: Accepted (component **registered**; the schedule does not move)
- **Date**: 2026-10-24
- **Related**: plan §2.2 (cognitive core), §七 (phases); [ADR-0016](ADR-0016-arousal-channel-and-consolidation-experiment.en.md) (same batch); [research/eprop](../../research/eprop/README.en.md)

## Context

**The plan has no world-model component.** An audit on 2026-10-24 walked the four nearest candidates
and none of them is one:

| Existing | What it is | What it lacks |
| :--- | :--- | :--- |
| Working-memory RSNN | Maintenance of the **current** input | No prediction of "what if the action differed" |
| Predictive E-prop | An error signal with no predicted object | Predicts its own state, not environment dynamics |
| Episodic memory | What happened | "Was", not "would be" |
| Metacognitive gating | Monitoring uncertainty | Produces no candidate future |

Against a Dreamer-style standard a world model needs at least four parts: **action-conditioned
dynamics**, an **imagination/planning loop**, **quality metrics**, and an **explicit prediction
target**. The project has none of the four.

**Why it was absent, and why it is needed now.** Its absence was not an oversight: phases one to
three ask whether a rule can learn, and that question needs no predictor. It is needed now because
the dynamics-axis audit found the axis *thick but uneven* --- R-STDP and the critic are validated
while the **across-time** components (consolidation, arousal, world model) are unimplemented or
unnamed. The world model is the most expensive of them and the one that gives the others something
to measure: without it, "did consolidation improve prediction" is not even a well-formed question.

## Decision

**① The world model is a formal phase-four component**, inheriting Predictive E-prop's error-signal
mechanism and supplying the object it predicts. **The current schedule does not move**: phases one
to three keep their tasks and acceptance criteria verbatim.

**② The definition is narrowed to four checkable items**; missing one means there is no world model:

| Item | Acceptance form |
| :--- | :--- |
| Action-conditioned dynamics | Given state and action, predict the next state; a different action must change the prediction |
| An explicit prediction target | State (or a low-dimensional projection) and reward, reported separately |
| Quality metric | A pre-registered prediction error compared against a no-prediction baseline |
| Imagination/planning loop | Late phase four; a **one-step** forward pass is acceptable to start |

**③ Boundaries fixed**: the world model answers "if", episodic memory answers "was", working memory
answers "now". Sharing representations is fine; **sharing vocabulary is not** --- documents and code
must not blur the three.

## Consequences

- The phase-four task table gains this row with the four acceptances above; **until then it is a
  registration, not a schedule item**.
- The quality metric must be pre-registered **before any number exists**, or "the world model helps"
  cannot be separated from "the predictor fit the training distribution".
- Same discipline as [ADR-0014](ADR-0014-adaptive-arbitration-unmeasurable-and-fit-retained.en.md):
  **only measurable things qualify for the component table**; unmeasurable ones stay in the
  exploration list.

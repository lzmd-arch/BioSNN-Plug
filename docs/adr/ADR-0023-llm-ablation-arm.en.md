# ADR-0023: every LLM-collaboration claim needs ablation arms

[中文](ADR-0023-llm-ablation-arm.md) · [日本語](ADR-0023-llm-ablation-arm.ja.md)

- **Status**: accepted
- **Date**: 2026-10-29
- **Related**: phase-5 M3a-M3k, [ADR-0024](ADR-0024-stub-llm-probe.en.md)

## Context

This project bets on "mechanism replacing substrate": learning and memory live in the SNN's dynamics and the LLM is only an external reasoning engine. But as soon as an LLM is attached it quietly brings the substrate back - answer quality belongs mostly to the model, not the SNN. Existing phase-5 results (M3b: completion 0.975 vs baseline 0.350) have no LLM vs no-LLM ablation on the same task, so they cannot say what the SNN contributed.

## Decision

Every phase-5 collaboration claim must report **three arms** together:

1. **SNN-only**: no LLM, only memory retrieval and rules;
2. **SNN + stub-LLM**: the LLM returns placeholder answers, separating the "call pipeline" from "model content";
3. **SNN + real-LLM**: a real model.

Single-arm numbers (especially real-LLM-only) do not enter acceptance.

## Consequences

- about 3x cost, but attribution becomes possible: pipeline contribution and model contribution are separated;
- M3b's 0.975 only counts as collaboration evidence after a three-arm rerun;
- cost: more real-LLM calls and spend.

## Alternatives

- **Real-LLM only**: the current practice, no attribution; rejected.
- **Stub only**: verifies the pipeline but not the model's value; equally incomplete.

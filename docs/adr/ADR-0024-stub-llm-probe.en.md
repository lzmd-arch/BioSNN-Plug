# ADR-0024: probe autonomous triggering with a stub LLM first

[中文](ADR-0024-stub-llm-probe.md) · [日本語](ADR-0024-stub-llm-probe.ja.md)

- **Status**: accepted
- **Date**: 2026-10-29
- **Related**: [ADR-0023](ADR-0023-llm-ablation-arm.en.md), phase-5 M2/M3b/M3g-M3k

## Context

The most falsifiable claim of this project is "the SNN decides by itself when to call the LLM", but it sits late in a 12-month roadmap. The two pieces of evidence we have do not support treating it as done:

- triggering in M2/M3b is a **rule gate** (retrieval-score threshold), not learned; M3b's "first trigger at turn 3, accuracy 1.000" is structural;
- M3g's ES scheduling passed every criterion but was a trivial solution (stuck at initialization); M3h got a non-trivial pass only with identifiable data; M3i-M3k took the STDP lateral trigger from 1-2/4 to 4/4 on a synthetic stream, still protocol level.

Waiting for a real LLM and a product form keeps this claim untestable.

## Decision

Before any real LLM, run a **stub-LLM probe**: an adapter that only returns placeholder answers, to obtain measurable evidence on three things:

1. **scheduling**: whether the ES-learned call policy strictly beats the rule gate (M3h already gives an identifiable protocol);
2. **triggering**: whether the spiking population triggers when it should and stays quiet when it should not (M3k's ROC-style evaluation);
3. **cost**: whether call count and latency are acceptable.

Probe conclusions are written only as "pipeline and scheduling hold/do not hold", **not** as "autonomous triggering capability holds".

## Consequences

- the most falsifiable claim moves from the end of the roadmap to a few weeks;
- real LLMs are attached only after the probe passes, so model quality cannot hide scheduling defects;
- cost: the probe cannot validate open-domain dialogue, only structure and cost.

## Alternatives

- **Go straight to a real LLM (the M3b route)**: already done; the first three criteria are structural and attribution is impossible; rejected as an endpoint.
- **Keep only the rule gate**: cheapest, but "autonomous" becomes meaningless; rejected.

# ADR-0021: write the death condition of a criterion first

[中文](ADR-0021-criterion-death-conditions.md) · [日本語](ADR-0021-criterion-death-conditions.ja.md)

- **Status**: accepted
- **Date**: 2026-10-29
- **Related**: plan section 12.4, [ADR-0022](ADR-0022-restatement-dashboard.en.md), every preregistration of phases 3-5

## Context

This repository has restated criteria three times: cross-modal retrieval moved from R@1 to recovery rate; TAAF's "no fine-tuning" standard was dissolved by ADR-0012; the arbitration criterion was replaced by the P4 synergy benchmark. Each restatement had a good reason, but **no rule constrained restatement itself** - it could happen after seeing results, so a "criterion" could always become an post-hoc explanation.

The phase-3 measurement audit (ledger 148) exposed another kind of death: the criterion did not change, but the **measurement** (best vs last evaluation) did, and the conclusion flipped. A criterion's death condition must be written down together with the criterion.

## Decision

1. Every preregistration must contain a "**death conditions**" section: under what evidence the criterion is void or needs restating, and what restating requires (a new preregistration, a recorded reason, a link to the old document).
2. Death conditions must be written **before seeing data**; adding them afterwards does not count.
3. Restating may not edit the old preregistration's criteria; only a new document may supersede it.
4. Every restatement is logged in the dashboard of [ADR-0022](ADR-0022-restatement-dashboard.en.md).

## Consequences

- Restating becomes slower and more expensive, but can no longer happen silently;
- negative results are harder to dodge by "changing the criterion";
- cost: early experiments have no death conditions; they can only be added in a new round, not retroactively.

## Alternatives

- **Forbid all restatement**: freezes obviously wrong criteria (e.g. v4's self-contradictory R2/MSE pair); rejected.
- **Rely on the author's discipline**: has failed three times; rejected.

# ADR-0022: the criterion-restatement dashboard

[中文](ADR-0022-restatement-dashboard.md) · [日本語](ADR-0022-restatement-dashboard.ja.md)

- **Status**: accepted
- **Date**: 2026-10-29
- **Related**: [ADR-0021](ADR-0021-criterion-death-conditions.en.md), plan ledger 148

## Context

Criterion restatements are scattered across the plan, result READMEs and ADRs; nobody can answer at a glance "how many times was this line restated, and why". ADR-0021 requires logging each restatement, but logging needs a single entry point, otherwise the log itself drifts.

## Decision

The dashboard at the end of this document is the **only** register for restatements. Each restatement appends one row: date, criterion, from -> to, reason, link (ADR/preregistration/ledger). Old rows are never edited; corrections append new rows.

## Dashboard

| Date | Criterion | From -> to | Reason | Link |
| :--- | :--- | :--- | :--- | :--- |
| 2026-10-26 | cross-modal retrieval | R@1 >= 50% -> recovery rate >= 50% | on the synthetic set R@1 and recovery rate measure different things | plan ledger 35 |
| 2026-10-25 | TAAF needs no fine-tuning | pass as written -> dissolved (not achieved) | the locality standard conflicted with the code contract and was untestable | [ADR-0012](ADR-0012-taaf-locality-and-bus-output-contract.en.md) |
| 2026-10-26 | arbiter reduces conflicts by >=50% | -> P4 synergy benchmark (static mix +0.179) | conflict-event counting was not reproducible; measure mixing gain instead | [ADR-0014](ADR-0014-adaptive-arbitration-unmeasurable-and-fit-retained.en.md), ledger 49 |
| 2026-10-29 | phase-3 "best evaluation" | best_steps -> last/final-three | taking the max over 32 points treats a peak as the level | ledger 148, `stage3_sustained_audit.md` |

## Consequences

- restatements become countable; "which time is this" no longer depends on memory;
- cost: one logging step per restatement; a missed entry can be spotted by a reviewer (CI does not enforce it yet);
- the dashboard does not judge whether a restatement was right; it only records that it happened.

## Alternatives

- **Keep entries scattered across READMEs**: already happened, no global view; rejected.
- **Use issues/PRs**: single-author repo, no PR flow; rejected.

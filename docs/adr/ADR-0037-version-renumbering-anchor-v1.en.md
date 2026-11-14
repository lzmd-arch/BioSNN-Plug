# ADR-0037: Renumbering — the old V1 becomes V0.5 (archived); the persona/memory core becomes V1

[中文](ADR-0037-version-renumbering-anchor-v1.md) · [日本語](ADR-0037-version-renumbering-anchor-v1.ja.md)

- **Status**: Accepted
- **Date**: 2026-11-15

## Context

Before 2026-11-14, "V1" meant **Semprea-4.5B**: Qwen 4B (frozen) plus a 0.5B local-rule fast
half, fused at the **output-distribution** level. That version is frozen and left three
negative results that do not move: four token-level fusion operators all failed to beat the
temperature control; the spiking half was negative on generation, encoding, discrimination and
sequence (0/543, 0.015, 0.763, 0.562); and an additive conditioning bias was zero twice on a
real backbone (S0).

The direction then changed: fusion moved from *output distributions* to *persona and memory*.
The local side stopped competing as a generator and became an **anchor** (identity, memory,
behavioural constraints), leaving generation to the cloud. That line has positive results:
pool-state decoding 0.64–1.0, readout-level continual learning AF 0.80→0.22, crisis gate recall
0.996 at FPR 0.11.

That created a naming collision: two unrelated things both called V1.

## Decision

1. The old Semprea-4.5B is renumbered **V0.5**, **archived and no longer published externally**.
   Its results and negative findings are kept; citations must say `V0.5`.
2. The persona/memory core becomes **V1**, named **Semprea-Anchor-0.5B** (Chinese: 锚核).
3. In documents written before this ADR, "V2" reads as "V1" and "V1" reads as "V0.5".
   Historical documents are **not** rewritten; only new documents and indexes use the new numbers.
4. `dist/semprea-4.5b-v1.0/` and `dist/v1.0/` stay where they are, each with an `ARCHIVED.md`
   pointing at this ADR.

## Consequences

- Externally: "V0.5 is output-level fusion, archived; none of the four operators beat the
  temperature control".
- **V1's criteria are no longer token-level gains** but I-1…I-7 in `prereg-persona-in-weights.md`.
- The package name `persona-core` does not change — it names a mechanism, not a version.

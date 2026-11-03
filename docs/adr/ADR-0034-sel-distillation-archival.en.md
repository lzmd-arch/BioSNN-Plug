# ADR-0034: Archiving the SEL spiking-distillation path — keep the concept, stop TI-LIF + CWSSA + SSM + KD

[中文](ADR-0034-sel-distillation-archival.md) · [日本語](ADR-0034-sel-distillation-archival.ja.md)

- **Status**: Accepted
- **Date**: 2026-11-04
- **Related**: [ADR-0029](ADR-0029-research-line-archival.en.md) (pure CIL / spiking actor),
  [ADR-0031](ADR-0031-mainline-to-scale.en.md) (spiking SoS archived),
  [ADR-0032](ADR-0032-sel-research-track.en.md) (SEL opened),
  [ADR-0033](ADR-0033-capacity-axis-archival.en.md) (capacity axis archived),
  `research/sel/prereg_p4.md`, `research/sel/README.en.md`

## Context

ADR-0032 opened SEL as a research item on the path **TI-LIF + CWSSA + selective SSM +
distillation + online R-STDP**. P0–P4 have now run:

| Stage | Result |
| :--- | :--- |
| P0 TI-LIF | `biosnn_bus.TernaryMembrane` 0.6.0; `D=1, signed=False` bit-identical to `AdaptiveMembrane` |
| P1 recall | negative: ALIF / TI-LIF hit@1 0.012 vs lexical baseline 0.259 |
| G1 gating | negative: TI-LIF AUROC 0.410 vs lexical baseline 0.825 |
| P2 teacher | template hit T1 58% passes; T4 style dispersion 0.88x (0.5B), 0.76x (1.5B), 0.99x (3B), all short of 0.5x |
| P3 data | 391 pairs (301/44/46), D1–D4 pass; teacher downgraded to Phase 1 (1.5B GGUF) |
| P4 student | 5 seeds x 10k steps: P4-1/P4-2 pass, **P4-3 fails** |

P4 five-seed medians (both arms CE-only, not KD):

| Arm | test top-1 | test CE | median seconds |
| :--- | ---: | ---: | ---: |
| bigram (denominator) | 0.6457 | -- | -- |
| GRU(256) | 0.8831 [0.8795, 0.8846] | 0.5203 [0.4686, 0.5540] | 83 |
| ALIF(512) | 0.8685 [0.8634, 0.8729] | **0.4594** [0.4304, 0.5518] | 620 |
| TI-LIF(512, D=3) | 0.8641 [0.8532, 0.8685] | 0.5274 [0.4987, 0.8482] | 946 |

P4-1 (>= 0.7457) and P4-2 (>= GRU - 5pp = 0.8331) pass; **P4-3 fails: TI-LIF CE is
1.148x ALIF** (criterion <= 0.90x), and 4 of 5 seeds are worse in a same-seed
comparison. P4-4 is N/A because no tokenizer alignment exists (`prereg_p4_amend1.md`);
P4-5 generation acceptance has not been run; P4-6 passes on cost.

**Separate the concept from the path.** The product pain point — "keep the tone when
the backend model changes" — is still real; TI-LIF/ALIF are numerically correct neuron
implementations; what failed is the efficiency of **TI-LIF + CWSSA + SSM + KD under
8 GB VRAM, a CPU teacher, and no character/token alignment**.

## Decision

1. **Stop the SEL spiking-distillation path.** After P4, do not start P5 online R-STDP;
   do not implement CWSSA/SSM; do not add KD. P3 data is not used for distillation.
2. **The product falls back to Phase 2 (template + model mixed rendering).** Do not wire
   the spiking expression layer into the product. Phase 2 remains off by default; T4 did
   not pass, so neither docs nor UI may claim "tone is now unified".
3. **Keep P3's 391 teacher pairs; they may be reused as SFT corpus.** That is data reuse,
   not distillation, and requires a new data card and training preregistration. The data
   still does not enter the repository.
4. **The only active research line is Line A (brain-region specialization).** Its
   preregistration is locked (`research/brainregions/prereg-brain-region-specialization.md`);
   the first 2-arm x 3-seed run started on 2026-11-04 and results are pending. ALIF-only SEL is a long-term backup only: no preregistration now,
   no compute now.
5. **Adjust Paper C's narrative to the archived results**: §4.1 spiking SoS archived;
   §4.2 capacity axis archived; §4.3 Line A preregistration; §5 SEL as "an attempt at a
   spiking expression layer and a reflection on the technical path", not as "an
   implementation".

Four hard boundaries have now been reached: **pure CIL (ADR-0029), spiking SoS
(ADR-0031), capacity axis (ADR-0033), SEL spiking distillation (this ADR)**.

## Reopen conditions

Any of the following requires a **new preregistration**; this ADR is not a shield:

- **ALIF-only SEL**: drop CWSSA/SSM/KD and keep only an ALIF expression layer; first
  answer teacher quality, data provenance, and evaluation protocol;
- **A tokenizer alignment layer**: only then can token-level teacher distributions be
  marginalised onto a character-level student for KD;
- **A different working point**: >8 GB VRAM, or a smaller teacher/student pair that lets
  P4-3's efficiency ratio be measured under a fair budget;
- **A new product requirement**: a measurable tone-consistency requirement appears and
  Phase 2's T4 still fails.

## Consequences

- No more training compute for SEL; P5 cancelled; `research/sel/` keeps code,
  preregistrations, and results for reproduction;
- The product stays on Phase 2, default off; no weights released, nothing wired in;
- P3 data may be reused for SFT, but only with a new data-use note and preregistration;
- Paper C §5 is written as a reflection chapter, so "the concept holds" cannot be
  smuggled in as "the path succeeded";
- ADR-0032 opened the item; this ADR closes it: **concept kept, path stopped**.

## Alternatives

- **Explain P4-3 away as "train a few more steps"**: rejected. The 5-seed median is
  stable and the preregistration fixed <= 0.90x; more steps have no preregistered basis;
- **Change the T4 ruler to make Phase 2 green**: rejected. T4 was fixed before the run;
  changing it afterwards is claim drift;
- **Use P3 data for SFT and call SEL a success**: rejected. SFT is a different path and
  needs its own preregistration and evaluation; it cannot inherit SEL's conclusion;
- **Treat the recall/gating negatives as SEL's failure**: rejected. The tasks differ;
  SEL's stopping point is P4-3, not P1/G1.

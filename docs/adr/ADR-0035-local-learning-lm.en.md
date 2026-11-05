# ADR-0035: Pretraining a language model with local learning rules — reopened as a new branch, M0 first

[中文](ADR-0035-local-learning-lm.md) · [日本語](ADR-0035-local-learning-lm.ja.md)

- **Status**: Accepted
- **Date**: 2026-11-06
- **Related**: [ADR-0034](ADR-0034-sel-distillation-archival.en.md) (SEL archival and its reopening
  conditions), [ADR-0029](ADR-0029-research-line-archival.en.md) (research-line archival),
  [ADR-0033](ADR-0033-capacity-axis-archival.en.md) (capacity-axis archival),
  `research/sel/prereg-lm-eprop.md` (M0 criteria)

## Context

The user's goal is "a model that can at least hold a casual conversation". Three findings reframed
the question:

1. **This repository has never trained a language model with local rules.** The spiking LM in P4
   (`research/sel/student.py`) uses `.backward()` (surrogate gradients + BPTT); the real local-rule
   implementation (`research/eprop/event_driven.py`) only supports classification with one label per
   sequence. So "can local learning pretrain" has **neither a positive nor a negative example**.
2. **P4's CE 0.5274 is a BPTT result, not a local-learning starting point.** A local model starts at
   CE = `ln V`. Treating 0.5274 as a starting point mistakes the bar for the baseline.
3. **One of ADR-0034's reopening conditions is now met**: "a tokenizer alignment layer that can
   marginalise the token-level teacher distribution down to a character-level student" —
   `research/sel/probe_logprobs.py` measured teacher top-5 coverage of **0.967**, so first-character
   marginalisation is arithmetically feasible.

Line A (A1–A8) offers indirect support: on this pipeline the bottleneck for classification was the
**readout**, not the learning rule (changing the readout was worth 14–42pp; changing organisation was
worth 0.3pp). That does not extrapolate directly to language models, but it is enough to show that
"local rules cannot learn" is not an established prior.

## Decision

**Open a new branch: next-token pretraining with local learning rules.** It is not a revival of the
SEL distillation path — SEL's TI-LIF + CWSSA + SSM + KD remain stopped and the product stays on
Phase 2.

Four hard boundaries:

1. **M0 first, criteria written down first.** Only if both M0-1 (top-1 ≥ 0.70) and M0-2 (the
   recurrent weights are actually learning) in `research/sel/prereg-lm-eprop.md` hold may M1 start.
   Otherwise the branch stops under its kill criterion.
2. **Do not touch the product.** No recall integration, no gating, no published weights, no change to
   Axo's default backend.
3. **Data does not leave this machine.** Real conversation text is used for local training only; the
   cloud receives synthetic topics/SIR, never conversation text. Data and checkpoints live in
   `D:\AxoData\sel\lm\` and are not committed.
4. **Do not claim it can chat.** The target is narrow-domain, single- to few-turn, in-persona text.
   General conversation needs 5–7 orders of magnitude more data and compute and is capped by an
   architectural limit (no attention ⇒ no in-context learning), which is outside this ADR.

Technical essentials (details in `research/sel/prereg-lm-eprop.md`):

- **LIF with per-region τ, never ALIF** (`traces.py`'s `epsilon_a` is O(B·N²) — 4.3 GB at N=8192);
- write **streaming traces**; do not reuse `eprop_gradient` (it allocates O(N²) unconditionally);
- **chunked training = carrying state across chunks** (v / z / refractory / input trace / recurrent
  trace / tap filters are all carried; reset only at session boundaries). **Chunk length sets the
  update cadence; memory is set by τ** — that is the fundamental difference from TBPTT.

## Alternatives

- **Resume SEL distillation**: rejected. P4-3 failed (TI-LIF CE is 1.148× ALIF, bar ≤ 0.90×), and the
  8 GB working point has not changed, so ADR-0034's reopening conditions are not met.
- **Fine-tune an open LLM (QLoRA)**: technically feasible, but that is a different route (not local
  learning) and conflicts with the project's premise. The user explicitly chose to stay with local
  rules; this ADR does not cover it.
- **Pretrain a general base model from scratch**: rejected. It is 6–8 orders of magnitude away —
  neither this machine nor this project can do it.
- **Skip M0 and go straight to scale**: rejected. Scaling without evidence that local rules can learn
  next-token turns "unverified" into "one more negative result".

## Consequences

**Upside**: turns "can it" into a 0.2 GPU-hour experiment; either outcome yields a citable boundary.

**Cost**: M0 uses only 391 samples (46 test items, top-1 SE ≈ 0.9pp), enough to detect a signal but not
to measure its size. M1–M5 rise to 50–200 GPU-hours.

**On failure**: if M0 fails, record the negative result, archive the branch, and add a note here that
the kill criterion took effect. Do not retune and rerun, and do not retry with another seed.

## Addendum

(To be appended once M0 reports.)


**2026-11-06, update: M0b also fails; the branch is confirmed stopped.**

M0b only raised the hidden learning rates (`lr_in` 1e-3, `lr_rec` 1e-3), everything else unchanged.
Results in `sweep_results/lm_eprop_m0b/README.md`:

- the hidden layer did move: `w_rec` relative change **4.976%** (M0: 0.070%, a 70× difference); but against
  the pre-registered 5% bar **M0b-0 formally fails by 0.024pp** — under the rule that means "the experiment
  is invalid", no conclusion is drawn from it, and the learning rate is not raised again to rerun;
- with that caveat, **learning the recurrent weights made things worse**: CE 1.861 vs 1.741 frozen (**+6.9%**),
  top-1 0.516 vs 0.566;
- both variants **lose to a character bigram** (CE 1.423 / top-1 0.646).

**Conclusion**: at 1,024 neurons, 2,000 updates and 301 templated samples, per-step local learning does not
beat a second-order character bigram, and **training the recurrent weights is harmful**. This negative is
stronger than M0's: M0's "no difference" could be read as "not measured", M0b's cannot.

**The branch stops**; both pre-registered kill criteria have been executed. Reopening needs **new data and a
new scale** (the M3 rung), not more tuning of this working point.

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


**2026-11-06, further correction: the M0/M0b kill is not supported; the branch should be reassessed.**

S0 (`sweep_results/lm_eprop_s0/README.md`) measured three arms, changing only where the recurrence comes from:

| Arm | CE | top-1 | recurrent / input current |
| :--- | ---: | ---: | ---: |
| no recurrence | **1.139** | **0.717** | 0× |
| frozen random recurrence | 1.861 | 0.516 | 2.16× |
| frozen diagonal SSM | 2.639 | 0.352 | 7.36× |

**The ordering is perfectly monotone in "recurrent current / input current".** The recurrent arms of the first
two rounds carried 2.16× the input current from initialisation, i.e. they were handicapped before the test
began — so **"local learning loses to bigram" does not hold**. Without recurrence, **a feedforward LIF with a
locally-trained readout beats bigram for the first time (1.139 / 0.717 vs 1.423 / 0.646)**.

**Disposition**: the "stop" conclusion of this ADR is **withdrawn** (the numbers stand, the inference does
not). Reopening does not need new data — it needs a **new pre-registration** that writes "recurrent current /
input current ∈ [0.8, 1.3]" as a precondition and asks again, fairly, whether recurrence or SSM helps.
That corrects an implementation defect; it is not tuning a result green.

## ADR-0035-c: the M0/M0b conclusions are marked "evidence contaminated"

**The M0/M0b stop decision rests on incomplete evidence.** Their recurrent arms carried 2.16× the input
current at initialisation, letting internal dynamics dominate the e-prop learning signal. Therefore:

- the M0/M0b **numbers** stand and remain valid (reproducible);
- the M0/M0b **inferences** ("local learning loses to a character bigram", "training the recurrent weights
  is harmful") are marked **"evidence contaminated"** and must not be cited;
- whether to reopen the branch is decided by S1's pre-registration, not by this ADR.


**2026-11-06, third round: S1 answers "does controlled recurrence help?" — no.**

Recurrent current calibrated to 0.5× and 1.0× of the input (power-law calibration, achieved 0.556 / 1.021),
everything else as in S0:

| Arm | CE | top-1 |
| :--- | ---: | ---: |
| no recurrence | **1.139** | **0.717** |
| rec@0.5 (frozen) | 1.231 | 0.698 |
| rec@1.0 (frozen) | 1.330 | 0.668 |
| rec@0.5 (learned, `w_rec` moved 7.2%) | 1.213 | 0.687 |

- **the monotonic trend persists in the controlled range** (0× → 0.717, 0.56× → 0.698, 1.02× → 0.668);
- **learning the recurrent weights still does not help** — and this round rules out "drowning" as the
  explanation, since the current was calibrated;
- **zero-input firing is 0 in all four arms**, so this is not self-sustained oscillation: it is a large but
  uninformative current diluting the token signal.

**Conclusion**: at this working point **a feedforward LIF with a locally-trained readout is the best
configuration**; any recurrent current is a drag. The SSM branch closes with it: the structural prior neither
helped while drowned nor needs a fair re-test, because **even the simplest recurrence is not worth keeping**.

**Aside**: the A-line `EventDrivenEProp` (sMNIST, N=8192, fan-in 32) has a recurrent/input current ratio of
**1.106**, the same regime that cost 4.9pp here. The A-line's **relative** conclusions are unaffected (both arms share
the recurrence). **A9, run the same day, refutes the corollary**: removing the recurrence gives a median of
0.904 vs 0.899, only +0.5pp (per seed −1.3 / +1.7 / +1.2pp). **So the "recurrent-current slope" does not
transfer across tasks**, and the A-line's absolute numbers are not depressed by the recurrence.


**2026-11-06, fourth round: S2 confirms across five seeds — 0.717 was not luck.**

Feedforward LIF + local readout, five seeds (everything else identical to S1's `none` arm):

| Seed | 0 | 1 | 2 | 3 | 4 | Mean | std |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| top-1 | 0.7166 | 0.7137 | 0.7115 | 0.7312 | 0.7400 | **0.7226** | **0.0124** |
| CE | 1.139 | 1.195 | 1.222 | 1.152 | 1.156 | **1.173** | — |

- **all five criteria pass**: mean 0.7226 ≥ 0.70, std 0.0124 ≤ 0.02, **every seed beats bigram by
  6.6–9.4pp**, mean CE 1.173 < 1.423;
- the spread is only **2.85pp**, much tighter than A3's three seeds (4.4pp).

**This is the branch's first seed-robust positive result**: a purely local learning rule (per-step e-prop +
NLMS readout, no autograd anywhere) reliably beats a second-order character bigram on this corpus, and it
**needs no recurrence at all** (S1 showed adding one only hurts).

**Boundaries**: character level, templated corpus, one layer, N=1,024, 2,000 updates, 46 test items; this does
not extrapolate to "can hold a conversation".


**2026-11-06, fifth round: R0 — switching to a real corpus, all six folds lose.**

The corpus goes from SEL's templated pairs to **real conversations** (265 pairs / 34,183 chars / 1,246 unique
characters, outside the repo, not committed), leave-one-session-out over six folds, everything else as in S2:

| Corpus | unique chars | bigram top-1 | model top-1 | delta |
| :--- | ---: | ---: | ---: | ---: |
| SEL templated (S2) | 274 | 0.646 | **0.7226** | **+7.7pp** |
| **real conversations (R0)** | **986-1,244** | **0.184** | **0.080** | **-10.4pp** |

**All four criteria fail**: 0/6 folds win, mean delta -10.37pp, and training loss only falls to 0.57-0.63 of its
initial value.

**Conclusion**: S2's 0.7226 **does not transfer to real text** — that was a low-entropy templated corpus. At this
budget (118-252 training pairs per fold, 2,000 updates, N=1,024) the feedforward local-rule model **cannot even
match a character bigram**. It does learn (0.080 is far above the 1/1,200 chance line), just far too little.

**One thing this cannot separate**: "too little data" from "too shallow an architecture". The planned next step
is **synthetic expansion** (thousands to tens of thousands of pairs) under the same criteria; **no tuning of N or
tau** before that — that would be moving the goalposts.


**2026-11-06, sixth round: R1 difficulty ladder — only the easiest rung is valid.**

Three synthetic corpora from the local `qwen3-4b` (3,000 pairs each; only four knobs vary: topics, personas,
temperature, turns):

| Rung | bigram | model | delta | folds won | CE tail/head |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **E** | 0.4765 | **0.4973** | **+2.08pp** | **6/6** | 0.50-0.54 (converged) |
| M | 0.3463 | 0.2390 | -10.72pp | 0/6 | **0.71-0.73 (underfit)** |
| H | 0.2919 | 0.1752 | -11.67pp | 0/6 | **0.79-0.82 (underfit)** |

**R1-2 (every rung converges) fails, so M/H are "not judgeable", not "losing".** The root cause is **my design
error**: S2 trained 301 samples for 2,000 updates (106 epochs); R1 trained 2,500 per fold for the same 2,000
updates (**12.8 epochs**) -- the corpus grew 8x but the update count did not.

A suspicion was also ruled out: the M/H collapse is **not** vocabulary truncation (top-800 covers 97%+ of characters).

**Three valid points now exist**: bigram 0.646 -> **+7.7pp** (S2), 0.4765 -> **+2.1pp** (R1-E), 0.184 -> **-10.4pp**
(R0). **Monotone across three points**, but the 0.35-0.30 stretch is still empty.

**Next is R1b: align the update count to epochs** (~20,000 updates; three folds brings it to 1.5-2 hours).
**No changes to N or tau before that** -- that would mistake "not trained enough" for "architecture too small".


**2026-11-06, seventh round: R1b — train long enough and R1's "M/H lose" disappears.**

Only the update count changed, from 2,000 to **16,000** (about 103 epochs, matching S2's 106), three rungs x
three folds:

| Rung | bigram | model | **delta** | R1 (2k) | CE tail/head |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **E** | 0.4766 | 0.6123 | **+13.58pp** | +2.1pp | 0.32 (converged) |
| **M** | 0.3505 | 0.3397 | **-1.08pp** | -10.7pp | 0.51 (converged) |
| H | 0.2933 | 0.2542 | -3.92pp | -11.7pp | 0.62 (still slightly underfit) |

**Same corpus, same architecture, same hyperparameters -- only the training budget changed, and the three deltas
rose by 11.5 / 9.6 / 7.8pp.** So **R1's "M/H lose by 10-12pp" is void**; it was a 12.8-epoch result.

**The crossover sits between bigram 0.35 and 0.48**: a 13.6pp win at 0.48, a tie (-1.1pp) at 0.35. H remains
not judgeable (R1b-0 fails).

**One open item**: R0 (real corpus) also ran at the 2,000-update level. There is no direct evidence of underfitting
(CE ratio 0.57-0.63, 128-271 epochs), but it was **never re-run epoch-aligned** -- so whether its -10.4pp is a
real loss or underfitting is unresolved. This is the only loose end.


**2026-11-07, eighth round: R0b is RED — real text is a hard boundary, not underfitting.**

The real corpus was also given an epoch-aligned budget (16,000 updates), two folds (the first, plus the fold with
the largest training set):

| Fold | train pairs | top-1 after 16k | bigram | delta | CE tail/head |
| :--- | ---: | ---: | ---: | ---: | ---: |
| fold 1 | **118** | 0.0604 | 0.1754 | **-11.49pp** | 0.13 (converged) |
| largest | **252** | 0.0652 | 0.1669 | **-10.17pp** | 0.39 (converged) |

The largest fold's full curve (every 2,000 steps): test CE **rises monotonically 6.35 -> 7.93**, while top-1 peaks at
**0.0691 at 4,000 steps** and then declines.

**Both statements hold at once**: (1) **it is not underfitting** (CE ratios 0.13 / 0.39, fully trained); (2) **nor is
it "train longer and catch up"** -- even at the curve's best point it is still **9.8pp** behind bigram.

**This is the branch's RED boundary**: on 34k characters of real conversation with a 1,246-character vocabulary, a
feedforward LIF with a locally-trained readout **cannot match a character bigram**, and the gap does not shrink with
more training. The dose response is real (118 -> 252 pairs drops CE at 16k from 11.26 to 7.93), but **top-1 stays
between 0.060 and 0.069** regardless of data size.

**Deviation**: the pre-registration said six folds, two were run; the stopping rule is recorded in the results file
(fold 1 showed decisive overfitting, and one fold takes 1.75 h in practice, so six would take 8-10 hours). This does
not affect the directional conclusion.

**Only three fixes remain**: more data (synthetic expansion into the hundreds of thousands of pairs, to bring the
parameter/data ratio down), a smaller readout (or stronger regularisation), and early stopping (which limits damage
but cannot lift top-1). **More training is not among them.**

# Retrieval line: cross-modal R@K

[中文](README.md) · [日本語](README.ja.md)
**⚠️ Breaking-change grace period: until 2027-09** (plan section 12.5, ADR-0005).

## What it tests

The one phase-two success criterion that is still ❌ in plan section 7: **cross-modal
retrieval R@1 50%+**.

The pre-registration is at
[`sweep_results/retrieval_procedural_prereg.md`](../../sweep_results/retrieval_procedural_prereg.md),
written **before the data**. Read it before reading the result.

## Round one: outer-product Hebbian (2026-10-17, five seeds)

**The bar is not met. R@1 = 0.141** (0.121–0.172 across seeds), chance 0.034;
**R@5 = 0.528**.

| Column | Result | Meaning |
| :--- | :--- | :--- |
| Within-modality (reading a) | text 1.000; image 0.759–0.931 | the encoders preserve identity |
| Bus compared directly (no association) | 0.034 / 0.000–0.069 | **at chance** — the two paths are different projections, as the pre-registration predicted |
| Outer-product Hebbian association (the bar) | **R@1 0.141 / R@5 0.528** | four times chance, far from 50% |

R@5 reaching 0.528 while R@1 sits at 0.141 says the association **does** carry cross-modal
information, but the top of the ranking is diffuse: the correct item is often nearby, rarely first.

## Round two: the delta rule (2026-10-17)

Pre-registration: [`retrieval_delta_prereg.md`](../../sweep_results/retrieval_delta_prereg.md).
Same data, split, seeds and encoders; only the association rule changes.

| Association rule | R@1 (mean of five seeds) | R@5 |
| :--- | ---: | ---: |
| Outer-product Hebbian (round one) | 0.141 | 0.528 |
| **delta (Widrow–Hoff, this round)** | **0.359** | **0.845** |

**Still short of 50%.** But R@5 rose from 0.528 to 0.845 and R@1 from 0.141 to 0.359, which
confirms the diagnosis from round one: the outer-product sum only averages training targets
weighted by similarity and has nothing to suppress near neighbours; error correction does tighten
the top of the ranking.

⚠️ **delta is not unsupervised Hebbian**: the target `v` in the error term is a teacher signal
supplied during paired training, so this is a **supervised local rule** — each synapse uses only
its own pre- and post-activity plus an error, with no chain rule. That boundary is not negotiable.

⚠️ **Large step sizes diverge**: some grid points at η=0.1 overflow to inf/NaN. Divergence is now
flagged explicitly and scored zero in cross-validation, so it cannot poison the average into NaN.
The first version lacked that step and the log carried
`RuntimeWarning: invalid value encountered` — **that warning is not noise; it is a selection
being contaminated by NaN**.

## Round three: a linear upper bound — the bottleneck is the representation, not the rule (2026-10-17)

Pre-registration: [`retrieval_ridge_prereg.md`](../../sweep_results/retrieval_ridge_prereg.md).
A closed-form ridge regression (`W = (TᵀT + λI)⁻¹ Tᵀ V`) measures the **capacity ceiling of a
linear map** between the two bus representations.

| Association rule | R@1 (mean of five seeds) | Local? |
| :--- | ---: | :--- |
| Outer-product Hebbian | 0.141 | yes |
| delta (Widrow–Hoff) | 0.359 | yes (supervised) |
| **ridge (linear upper bound)** | **0.466** | **no — sees the whole training set, inverts a matrix** |

**The linear upper bound is also short of 50%.** That locates the problem: **the bottleneck is the
representation, not the association rule.** delta already captures most of the linear ceiling
(0.359 of 0.466), so further work on the rule has little room; crossing 50% means changing the
representation — the pooling, the bus dimension, or sharing one projection across both channels.

⚠️ **This is a *linear* upper bound only**: a nonlinear readout could do better, so it must not be
read as "50% is impossible" — only as "linearly, this representation is not enough". Ridge
regression is not a local rule and **enters no acceptance column**.

## Round four: a representation grid — more dimensions do not help (2026-10-17)

Pre-registration: [`retrieval_representation_prereg.md`](../../sweep_results/retrieval_representation_prereg.md).

| Association | bus@128 | bus@256 | bus@512 | plugin (no bus projection) |
| :--- | ---: | ---: | ---: | ---: |
| ridge (linear upper bound) | 0.466 | 0.476 | 0.469 | **0.479** |
| delta (local) | 0.359 | **0.393** | 0.407 | 0.403 |

**Bus dimension is not the bottleneck.** Raising `bus_dim` from 128 to 512, and even bypassing the
bus projection entirely (raw plugin outputs, 1152 image / 256 text), leaves the linear upper bound
at 0.47–0.48.

Under the pre-registered decision rule this triggers the **change the encoding** branch: the image
side's event encoding has only three frames (blank → image → image) and keeps a single onset edge;
the text side embeds each token as a standalone sentence and then applies z-score top-k. What each
encoding loses is not something a linear map can put back.

⚠️ An a-priori constraint: **cross-modal cannot exceed within-modal**. The image side's
within-modal R@1 is 0.759–0.931, so a cross-modal 0.48 is already close to the ceiling set by
images that barely separate from each other.

## Rounds five to seven: a nonlinear ceiling and two "more local" rules (2026-10-17)

| Readout | R@1 (mean of five seeds) | Is it a learning rule? |
| :--- | ---: | :--- |
| delta (linear, local) | 0.393 | yes |
| fixed random features + delta | 0.383 | yes (features are fixed) |
| competitive associative memory (M ≤ 64) | 0.114 (five seeds) | yes |
| **k nearest neighbours (nonlinear ceiling)** | **0.555 / 0.562** | **no — nothing is learned** |

Three conclusions, in order of importance:

1. **Only kNN crosses 50%, and kNN does not learn.** It performs no mapping; it finds the most
   similar training examples and averages their image codes. That is not meeting the bar, it is
   carrying the training set around.
2. **Fixed random features did not help** (0.383 < 0.393). So kNN's advantage is not
   nonlinearity — it is instance-level voting.
3. **Competitive memory underfits when M ≪ n** (0.103). With too few units each unit averages
   several unrelated image codes and emits a blurred one. Approaching kNN would require
   M ≈ the number of training items — **which is memorisation, not learning.**

⚠️ **The question behind this criterion therefore has to be restated**: on the procedural paired
set, "R@1 ≥ 50%" is satisfiable by **memorisation**, and satisfying it says nothing about whether
local learning works. This is the trap this repository keeps recording — **a number can pass a bar
trivially**.

## Rounds eight and nine: real data (Flickr8k) — the current front ends sit at chance (2026-10-17)

Pre-registrations:
[`retrieval_flickr8k_prereg.md`](../../sweep_results/retrieval_flickr8k_prereg.md),
[`retrieval_frozen_frontend_prereg.md`](../../sweep_results/retrieval_frozen_frontend_prereg.md).
Official split (train 6000 / test 1000); chance is **0.1%**.

| Front end | Readout | R@1 | Note |
| :--- | :--- | ---: | :--- |
| current plugins (24×24 DVS + per-token) | kNN | **0.002** | essentially chance |
| current plugins | — | — | the ceiling is already at chance; local rules need not be run |
| **frozen ResNet18 + whole-sentence nomic** (never co-trained) | ridge (linear ceiling) | **0.076** | 76× chance, far from 50% |
| frozen ResNet18 + whole-sentence nomic | kNN | 0.052 | nonlinearity does not help |

Two conclusions:

1. **The current encoders carry no cross-modal signal on real data.** A three-frame 24×24
   greyscale difference code paired with per-token sentence embeddings is at chance on 1000-way
   retrieval.
2. **Two individually strong frozen front ends still give a linear ceiling of 7.6%.** They were
   **never trained together**, so stitching the two modalities remains entirely the association
   layer's job — and that is all the association layer can recover.

⚠️ **The 50% bar has lost its discriminating power here.** The only practical way to reach 50%
on Flickr8k is a **jointly trained cross-modal encoder** (CLIP-like) — but then the alignment is
already done by pretraining, the local association rule has nothing to do, and passing the bar
says nothing about local learning. **This is a problem with the criterion, not with the
implementation.**

If this line continues, the next step is to **restate the criterion** — for example, measure how
much of an aligned-encoder oracle's alignment a local rule recovers — rather than tuning against
a measure that cannot distinguish the two.

## Round ten: restating the criterion — recovery rate (2026-10-17)

Pre-registration: [`retrieval_recovery_prereg.md`](../../sweep_results/retrieval_recovery_prereg.md).

The original bar (absolute R@1 ≥ 50%) is decided by three things at once: how strong the encoders
are, how well aligned they are, and how good the association layer is. Separating them means
switching to a **recovery rate**:

```text
recovery = (R_local − R_zero) / (R_upper − R_zero)
```

| Dataset | local (delta) | linear ceiling (ridge) | zero association | **recovery** |
| :--- | ---: | ---: | ---: | ---: |
| procedural paired set (bus@128) | 0.359 | 0.466 | 0.034 | **0.752** |
| Flickr8k (frozen ResNet + nomic) | 0.052 | 0.076 | 0.001 | **0.673** |

**Both datasets clear 50%.** This is the first cross-dataset-stable conclusion in nine rounds: a
purely local, teacher-signalled rule recovers two thirds to three quarters of what a linear
upper bound achieves.

⚠️ Three caveats that must be read together:
1. **This is not a quiet loosening of the old bar.** The absolute R@1 stays marked ❌ and is
   reported alongside; a high recovery rate does not change that verdict.
2. **delta on Flickr8k is under-trained** (`epochs ∈ {1,3}`, 3 folds, one seed — the compute cap
   declared in the pre-registration), so 0.673 is a **lower bound**.
3. **The ceiling itself is low** (7.6%). Recovery answers "how far the local rule is from the
   linear ceiling"; it does **not** answer "is this front end good enough" — that remains open.

## One finding that has to be recorded: this rule has no tunable hyperparameter

The pre-registration named two hyperparameters, `η` (learning rate) and `λ` (forgetting).
Neither can actually be selected:

* the transform is `η·(t ⊤ Σ v t⊤) / (1+λ)`, so **η and λ are both global scalings**;
* cosine is invariant to global scaling → every grid point scores **identically** under
  leave-one-out, and the "selected" point is just the first of a tie.

So the selection step selects nothing. That is not an implementation defect; it is the
algebra of this rule under cosine ranking. Tuning it properly needs a pseudo-inverse
association, which is no longer pure Hebbian — that is a new pre-registration.

## How to reproduce

```bash
uv run pytest research/retrieval -q                  # 36 tests
uv run python -m research.retrieval.run_retrieval --seeds 0,1,2,3,4
```

The text side uses the local Ollama `nomic-embed-text` model (frozen front end); token-level
embeddings are cached in `data/embeddings/nomic-embed-text.json`. **When Ollama is
unavailable the experiment does not run**; there is no fall back to a fake backend.

## Known boundaries

1. **The dataset is procedurally generated** (4 shapes × 3 shades × 6 positions × 2 sizes = 144)
   and cannot be compared against the literature. The **Flickr8k loader exists** (`flickr8k.py`:
   load only, no download; the layout and the `Flicker8k_Dataset` spelling with one r are in the
   module docstring), but **no run on real data has been made** — the download source and its
   per-file checksums are not registered yet.
2. **The two bus paths do not share a space**: `semantic` and `temporal` each go through an
   independently seeded random projection, so comparing them directly stays at chance. The
   control measured 0.034 = 1/29, exactly what that sentence predicts.
3. **The association is directional**: each direction fits its own `W`, with no assumption
   that the two are transposes of each other.
4. **Only R@1 and R@5 are reported**; the pre-registration fixed those two and nothing else.

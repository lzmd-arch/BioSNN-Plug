# Retrieval line: cross-modal R@K

[中文](README.md) · [日本語](README.ja.md)
**⚠️ Breaking-change grace period: until 2027-09** (plan section 12.5, ADR-0005).

## What it tests

The one phase-two success criterion that is still ❌ in plan section 7: **cross-modal
retrieval R@1 50%+**.

The pre-registration is at
[`sweep_results/retrieval_procedural_prereg.md`](../../sweep_results/retrieval_procedural_prereg.md),
written **before the data**. Read it before reading the result.

## Current result (2026-10-17, five seeds)

**The bar is not met. R@1 = 0.141** (0.121–0.172 across seeds), chance 0.034;
**R@5 = 0.528**.

| Column | Result | Meaning |
| :--- | :--- | :--- |
| Within-modality (reading a) | text 1.000; image 0.759–0.931 | the encoders preserve identity |
| Bus compared directly (no association) | 0.034 / 0.000–0.069 | **at chance** — the two paths are different projections, as the pre-registration predicted |
| Outer-product Hebbian association (the bar) | **R@1 0.141 / R@5 0.528** | four times chance, far from 50% |

R@5 reaching 0.528 while R@1 sits at 0.141 says the association **does** carry cross-modal
information, but the top of the ranking is diffuse: the correct item is often nearby, rarely first.

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
   and cannot be compared against the literature; neither the Flickr8k loader nor a run on
   real data exists yet.
2. **The two bus paths do not share a space**: `semantic` and `temporal` each go through an
   independently seeded random projection, so comparing them directly stays at chance. The
   control measured 0.034 = 1/29, exactly what that sentence predicts.
3. **The association is directional**: each direction fits its own `W`, with no assumption
   that the two are transposes of each other.
4. **Only R@1 and R@5 are reported**; the pre-registration fixed those two and nothing else.

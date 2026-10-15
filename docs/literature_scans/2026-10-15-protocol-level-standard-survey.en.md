# Protocol-level standard survey, 2026-10-15

[中文](2026-10-15-protocol-level-standard-survey.md) · [日本語](2026-10-15-protocol-level-standard-survey.ja.md)

> This is the **re-run** demanded by
> [`2026-10-14-local-learning-landscape.md`](2026-10-14-local-learning-landscape.en.md): a search with a
> complete log, reproducible by a third party. It is stronger than the 10-14 record in exactly one
> way — **the procedure is auditable**. It is weaker in exactly one way, and section 4 says which:
> **this search found near neighbours, so the "no equivalent exists" claim has to be narrowed.**

## 1. What it sets out to answer, and the criteria

The target claim (`paper/main.tex` §2.4):

> To our knowledge no equivalent protocol-level standard exists for local-rule SNN training.

The criteria are those of the 10-14 record; all three must hold at once for a hit:

1. it is a **standard** (a checklist, a policy, a mandatory submission item), not one paper's own practice;
2. it targets **local-rule SNN training** (relaxed: SNN training);
3. it prescribes **what a report must contain** (budget, seed count, optimizer, evaluation protocol, ...).

## 2. The search log (verbatim)

**Date:** 2026-10-15 (every entry was retrieved on this day).
**Sources:** the `WebSearch` index available to this agent (**US index** — see the limits in section 5)
and direct fetches of specific URLs.
**Counting:** "returned" is the number of link entries (the same item repeats across blocks within one
query, so these are approximations); "included" is how many reached the assessment in section 3.

| # | Source | Query string / URL (verbatim) | Returned | Included | Disposition |
| :--- | :--- | :--- | :--- | :--- | :--- |
| S1 | WebSearch | `reproducibility checklist spiking neural network training standard` | ~10 | 2 | In: a TechRxiv practical-guidelines document for SNNs, and Pauli et al. 2018. The rest were mostly **NeurIPS checklist pages embedded in individual papers** (artefacts of a submission template, not the standard itself) |
| S2 | WebSearch | `neuromorphic benchmarking standard checklist NeuroBench reporting protocol` | ~10 | 1 | In: NeuroBench. The rest were mirror copies of it |
| S3 | WebSearch | `local learning rules spiking neural networks benchmark comparison protocol budget seeds reporting` | ~10 | 2 | In: NeuroSeqBench, STEP. The rest were single papers |
| S4 | WebSearch | `"spiking neural network" benchmark reporting requirement multiple seeds training budget local learning rule standard` | ~10 | 2 | In: arXiv:2402.01782 (a locality-varying benchmark), NeuroEval. The rest were single papers' own checklist pages |
| F1 | WebFetch | `https://www.techrxiv.org/doi/full/10.36227/techrxiv.175322635.58911591/v1` | — | 0 | **403 Forbidden**, not verified. S1 hit it, but the content was never read, so **no conclusion may be drawn from its title** |
| F2 | WebFetch | `https://arxiv.org/abs/2304.04640` (NeuroBench) | — | 1 | Abstract-level information obtained |
| F3 | WebFetch | `https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2018.00046/full` | — | 1 | Full-text-level points obtained |
| F4 | WebFetch | `https://github.com/liyc5929/neuroseqbench` | — | 1 | README-level information obtained |

**Inclusion/exclusion:** anything unrelated to "how a report should be written" was excluded — the
NeurIPS checklist pages inside individual papers, incidental citations in code repositories, and SNN
papers with no bearing on reproducibility all stay out of section 3. **Excluded does not mean absent**:
they simply are not a "standard" in the sense of criterion 1.

## 3. The hits, assessed one by one

| Hit | What it is | Crit. 1 standard? | Crit. 2 local-rule SNN training? | Crit. 3 prescribes report contents? |
| :--- | :--- | :--- | :--- | :--- |
| NeuroBench (Yik et al., 100 authors, arXiv:2304.04640 / Nature collection, 2023–2025) | A benchmarking framework for neuromorphic computing, split into an algorithm track (hardware-independent) and a system track; it supplies tasks, metrics and baselines | Half: a framework plus guidelines, but **not** a mandatory submission item | No: organised by system and task, and the abstract **does not touch** learning rules such as e-prop, STDP or Hebbian | Half: it fixes the metrics to report for its own tasks (accuracy, latency, energy), but **not** training budget, seed count or optimizer |
| NeuroSeqBench (liyc5929, GPL-3.0, 24 stars) | A benchmark suite for SNN temporal tasks that **places `E-prop` alongside STBP/OTTT/SLTT**, and claims guidelines "to guarantee fair and consistent evaluations" | No: a code suite plus guidelines | Yes: the **closest thing to a criterion-2 hit** in this search | Half: it reports accuracy, PPL, parameter count, latency, training speed, memory and energy, and plugs into NeuroBench metrics; the **visible content fixes neither seed counts nor a compute budget** |
| arXiv:2402.01782, "Benchmarking Spiking Neural Network Learning Methods with Varying Locality" | A benchmark paper comparing SNN learning methods **stratified by locality** | No: it is a paper | Yes: the topic is almost exactly ours | No: one paper's own practice |
| STEP (OpenReview) | An evaluation platform for "fair and reproducible benchmarking" of spiking transformers | No: a platform | No: it targets transformer training, not local rules | Half: built into the platform, not an external standard |
| NeuroEval (Zenodo report) | A **meta-evaluation** framework using an LLM to score data transparency, reproducibility and neural-coding rigour in papers | No: an assessment tool and method | Half: it takes computational-neuroscience papers as its object | Half: it supplies scoring items, not submission requirements |
| Pauli, Weidel, Kunkel & Morrison 2018 (*Front. Neuroinform.* 12:46) | A best-practices guide to the reproducibility of spiking network **models**, built around reproducing polychronization | No: a guide, not a standard | No: its object is the reproducibility of **simulation models**, not training benchmarks | Half: share code, use version control, record software versions, run multiple times; note that it reports **seed sensitivity** itself (87% vs 13% land in different dynamical states) — a point that **supports** our claim rather than refuting it |
| Pineau et al. 2021 (*JMLR* 22(164), ledger [27]) | The protocol-level practice in ML: code submission policy, reproducibility challenge, mandatory submission checklist | Yes | No: it targets ML as a whole | Half: it prescribes report elements, but not the ones specific to local-rule training |

## 4. Conclusion: what was not found, and how the sentence must now read

**Entries satisfying all three criteria: 0.** The closest are **NeuroSeqBench** (criterion 2 yes,
criterion 3 half) and **NeuroBench** (criterion 1 half, criterion 3 half) — both are "a benchmark
suite plus its own guidelines". None is a **mandatory or community-adopted standard**, and none
prescribes the budget, seed count and optimizer sensitivity that local-rule training should report.

So **the sentence can be upgraded, but the upgraded sentence must name the near neighbours** — it
cannot just say "0 hits":

> A search of the sources and query strings recorded in Appendix X on 2026-10-15 returned no
> protocol-level standard for local-rule SNN training. The nearest work is benchmark-oriented:
> suites that fix tasks and metrics for comparing SNN learning algorithms, including e-prop,
> and reproducibility guides aimed at spiking model *simulation*; none of them prescribes the
> budget, seed and optimizer reporting this paper controls.

Warning: **the upgrade has not been applied, for two reasons, neither of them technical:**

1. That sentence names the near neighbours, and naming them means **citing** them (at least
   NeuroBench, NeuroSeqBench and Pauli 2018). The rule at the top of `paper/refs.bib` is that the
   file **adds no citation that this repository does not already have** — they would first have to be
   registered and metadata-verified in the [`docs/references.md`](../references.en.md) ledger, and
   that is a step needing its own ruling. Recording them in this file **is not** registering them:
   **none of the hits above is a ledger entry**.
2. The phrasing "a search of [sources] on [date] with [queries]" requires the search specification
   to enter the paper (an appendix section), or a reviewer has nothing to check against. That is a
   structural change to §2.4 and the appendix, not a one-line swap.

**Until both are done, §2.4 keeps its defensive wording.** The conclusion of the 10-14 record stands.

## 5. Limits, and how to re-run this

**Coverage is not exhaustive.** This search used the `WebSearch` index reachable by this agent
(**US region**) and per-URL fetches. It had **no** in-database access to ACM DL, IEEE Xplore, Scopus
or Web of Science, and no complete Google Scholar pagination. So what this file supports is "**with
the queries below, on this day, on this index, nothing was found**" — **not** "no such thing exists
in the field". Any claim of an "exhaustive" or "systematic review" is unsupported.

**To re-run:** copy each row's query string and URL from section 2, re-run it on any search engine or
database, and apply the inclusion/exclusion criteria from section 1. The criteria are written down
precisely so that a re-run may reach a **different** conclusion from mine.

**Known unfinished business:** F1 (the TechRxiv "Practical Guidelines for Spiking Neural Networks")
returned 403 and was not read. It is the entry whose *title* looks most like a standard, so **it
deserves priority next time** — if it turns out to prescribe report elements, the conclusion in
section 3 changes.

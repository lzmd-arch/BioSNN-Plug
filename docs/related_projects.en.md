# Related projects at a glance

[中文](related_projects.md) · [日本語](related_projects.ja.md)

> This table is a ledger of **licence boundaries** and **evidence strength**, not a literature review.
> Every entry answers three questions: what its learning rule is (and whether it is purely local),
> **whether its code may enter this repository**, and whether its self-reported numbers have been
> reproduced by anyone else.

## Why this table exists

Three things forced us to write it down:

1. **The technology choices in §8 of the project plan come straight from these projects** (EMBER →
   §4 LLM collaboration; MEMBRAIN → §4.3 encoding alternative; eprop-PyTorch → the e-prop
   implementation reference in §1.4/§8). Once a choice is written into the plan, "what state is that
   thing actually in" has to be answerable.
2. **This repository is Apache-2.0 and has a licence audit in CI** (`scripts/check_licenses.py`,
   plan §12.1). The field includes MIT projects (which may be vendored) and **non-commercial**
   projects (not one line may be copied). This column cannot be filled from memory.
3. **This repository's citation discipline** separates self-reported numbers from third-party
   reproductions. Some projects here turned **targets into capabilities**, and one was **refuted by
   its own benchmark** — that kind of information is worth more than a flattering accuracy figure.

## The table

| Project | Type | One-line positioning | Learning rule (purely local?) | Stack | Licence (can it be a dependency?) | Activity and evidence strength | Relation to this project |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BioSNN-Plug** (this project) | Research prototype | A purely-local-learning SNN cognitive core with modality plugins; the SNN decides when to call an LLM | e-prop (ALIF) + R-STDP + TD-LTP Critic + kernelized IB-Hebbian; **no surrogate gradients** | numpy skeleton library + PyTorch/SpikingJelly research lines | This repository is Apache-2.0; the `biosnn-bus` skeleton library follows its own semver | Phase 1 accepted on all three lines (W1 97.79% / W2 77.48% — a 30-epoch budget, which adds **+7.55** points at 100 epochs and still has not converged / W3 median 237.8 steps); numbers in the [reproducibility record](reproducibility.en.md) | — |
| **EMBER** (Savage, 2026, arXiv:2604.12167) | Paper (**no public repository**) | A hybrid architecture with an SNN as cognitive substrate and an LLM as a replaceable reasoning engine; the SNN decides "when to act / which associations to surface", the LLM decides "which action and what content" | Depression-dominant STDP ($A_-=1.05A_+$, $\tau_+=20$ ms / $\tau_-=30$ ms) + eligibility traces gated by a global dopamine signal + cascade-scaled decay; **no backpropagation, no e-prop** | **Self-written LIF, implemented in PyTorch**; no SpikingJelly / snnTorch / Nengo (zero hits, word by word) | **Not applicable** -- there is no repository to audit. ⚠️ Do **not** fill this column with arXiv's distribution licence (CC BY-NC-ND 4.0) or with any same-named repository's licence | Submitted 2026-04; code "released at publication" (NeurIPS checklist item 5 answers `[No]`; only v1 exists as of 2026-09). **N=1, no error bars** (checklist item 7 `[No]`); the author himself flags LLM confabulation risk | The entire evidence base of §4. **Worth borrowing**: the architectural division of labour, z-score top-k encoding, lateral-propagation triggering. **Not borrowable**: any code (there is none); treat its N=1 numbers as external evidence |
| **MEMBRAIN** (tfatykhov, 2026, GitHub) | Personal repository (a 4-day PoC) | A "neuromorphic memory bridge" for LLM agents: FlyHash sparse encoding + Nengo populations + attractor cleanup | Voja (local, unsupervised, shapes encoders) + PES (error-driven, shapes decoders, on by default); **no e-prop, no backpropagation** | Nengo (**CPU `nengo.Simulator` only**; Loihi/Lava exist **only in planning docs**) + numpy; **no torch / SpikingJelly / snnTorch** | **MIT** (standard text, no extra terms, vendorable with the copyright line retained). ⚠️ Two audit notes: **(a) MIT grants no patent licence**, and the author's `AGENTS.md` calls attractor dynamics a "Key for patent claim"; **(b) not on PyPI** (the `membrain` package on PyPI is an unrelated cryo-ET project) | 72 commits between 2026-01-31 and 02-03, then dormant; 0 stars / 0 forks / one author plus AI agents, no external contributors. ⚠️ **The README overstates**: it claims "100% completion at 20% noise", while its own benchmark records `MembrainStore` at **0.35** at 0.2 noise (cosine/FAISS baselines 0.55) and admits "~5000x slower"; at HEAD, `recall()` **skips the SNN simulation** (PES divergence). No third-party reproduction | §4.3's encoding alternative. **Worth borrowing**: the ~224-line numpy FlyHash encoder (int8 projection + top-k WTA) is a good vendoring candidate. **Not worth depending on** as a whole (dormant, claims do not match the code). **Do not cite** its numbers |
| **eprop-PyTorch** (ChFrenkel, 2022, GitHub) | Third-party reimplementation (**not official**) | A PyTorch reimplementation of the e-prop paper by Charlotte Frenkel (UZH/ETH, **not among the authors of Bellec et al.**) | Hard-coded e-prop (Eqs. (4)/(25)), **no autograd**; **LIF only, ALIF explicitly removed** (`main.py`: "Support for the ALIF neuron model has been removed.", `models.py`: `assert self.model == "LIF"`) | PyTorch (**no dependency manifest, no version constraints, no CI, no tests**); the only task is evidence accumulation | **Apache-2.0** (standard text, copyright University of Zurich; no commercial restriction, safe as a dependency) -- but engineering-wise **not advisable**: not on PyPI, no tags/releases, so it can only be anchored at commit `0f32a8f2` | A **single** initial commit (2022-02-18); 68 stars / 9 forks; one issue still unanswered. Note `updated_at` in 2026-08 is metadata churn, **not code activity** | The "e-prop implementation reference" named in §1.4/§8. **What it can actually be checked against is narrow**: the LIF + evidence-accumulation chain only. ⚠️ **For ALIF use the official implementation [15] and the original paper** (this repository removed ALIF). Two sentences in ADR-0008 decision 3 do not match its facts and were corrected in that ADR's later note |
| **ESPP** (Graf, Su & Indiveri, 2024, arXiv:2405.13976) | Paper + official repository | **EchoSpike Predictive Plasticity**: the previous sample's entire spike train is the prediction target (the "echo"); same-label pulled together, different-label pushed apart; self-supervised, no global error backpropagation | A predictive-plus-contrastive **inter-layer local rule** (not an eligibility-trace line) | PyTorch + **snnTorch + Tonic** (different from this project's stack, adding two dependencies to licence-check) | The repository [Zhe-Su/ESPP](https://github.com/Zhe-Su/ESPP) is **Apache-2.0** (⚠️ its LICENSE still has the copyright-holder placeholder unfilled; ⚠️ the paper's footnote link `largraf/EchoSpike` is **404** -- use the Zhe-Su one) | Preprint from 2024-05 (v2, self-described "submitted to IEEE"; **no journal version found** as of 2026-09); repository created 2025-08, no commits since 2025-09, 0 stars. 84.32% on SHD (self-reported; the authors claim it beats every local rule they know of) | **The original source of the phrase "event priority"**: it uses input-activity and loss thresholds to decide **which time steps get a weight update** (measured 18%-27%, decreasing with training); verbatim "ESPP intrinsically has the ability to selectively choose those time steps that matter the most" -- the same motivation as this project's §4.2 trigger. **But it cannot replace e-prop**: it is a parallel rule (the same treatment ADR-0010 gave TP). Also: "ESPP" is a high-frequency acronym (employee stock purchase plan, European Strategy for Particle Physics, ...); **in an SNN context it is this one** |
| **Javis** (BEKO2210, 2026, GitHub) | Personal repository (no paper) | An **associative SNN memory co-processor** for LLM agents: knowledge lives in cell assemblies, a query acts as a partial cue and pattern completion reactivates them, and only a few decoded concepts are handed to the LLM | All STDP-family (pair / iSTDP / triplet / reward-modulated STDP / BCM / SFA / structural plasticity, 12 options in total); **no e-prop, no backpropagation** | Pure Rust, **no SNN framework at all** (no Nengo / SpikingJelly / snnTorch) | ⚠️ **PolyForm Noncommercial 1.0.0** -- **not OSI-approved, not an SPDX standard**, with a research-use addendum banning production deployment and third-party services: **not one line of business code may be merged into this repository**; treat it exactly like `eligibility_propagation` (read-only, conclusions only) | No paper, no DOI; 0 stars; dormant since 2026-05. Every README number (100% self-recall, 35-45% token reduction, ≈2% associative recall, ≈50-concept capacity) comes from the author's own small corpus with **no third-party reproduction** | Same "SNN as substrate, LLM as mouth" family as EMBER, but with the **opposite division of labour**: EMBER's SNN decides **when** to act, Javis's decides **what goes into the context**. A useful second point of comparison for §4 |

## Name collisions (read before filling in a licence)

This repository has already mis-attributed a conclusion once (Khacef → Hajizada), so same-named
projects get their own section:

- **Javis**: GitHub also hosts **`JuliaAnimators/Javis.jl` (a Julia animation/visualisation library, MIT, 800+ stars)**.
  A licence audit that records "Javis = MIT" gets it exactly wrong -- the Javis in this table is
  `BEKO2210/Javis` (PolyForm non-commercial). There are further unrelated ones
  (`JavisVerse/JavisGPT`, `JavisVerse/JavisDiT`) and a completely different "JARVIS" lineage.
- **EMBER**: `Starlight-Unit-Studio/coreui` ("Ember CoreUI ... E.M.B.E.R. cognitive architecture") is
  also a local-LLM web UI, with **no SNN and no STDP**, and GitHub's API reports its licence as
  `NOASSERTION` -- the textbook case of "`NOASSERTION` misleads". Same for `pandeyAnush/ember`
  (a RAG assistant).
- **MEMBRAIN**: the `membrain` package on PyPI belongs to **another** project (membrane-protein
  localisation for cryo-ET, BSD-3-Clause); it has nothing to do with the MEMBRAIN discussed here,
  which is not on PyPI at all.

## How to use this table

- **Citing a conclusion** → check the "activity and evidence strength" column first. If it says "no
  third-party reproduction", say so when citing, and register the verification status in
  [`references.md`](references.en.md) as that file requires.
- **Using a piece of code** → check the "licence" column first. **Non-commercial / non-OSI licences
  mean read-only, never copy**; MIT may be vendored with the copyright line retained (and remember
  MIT grants no patent licence).
- **Judging whether a direction is viable** → look at whether its learning rule is purely local and
  what hardware it runs on. This project's proposition is **purely local + no surrogate gradients**,
  so a flattering number obtained with BPTT is not a precedent here.

## Maintenance

When adding a project, the four key fields must come from **files or pages actually opened** (LICENSE,
`pyproject.toml`, CI config, the paper PDF) -- never from memory or second-hand summaries; write
"not found" when a page cannot be reached. Record the verification itself in the commit message.

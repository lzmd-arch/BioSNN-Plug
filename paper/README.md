# paper/ — P2 benchmark paper (LaTeX)

Source for the **P2 paper**: an honest-measurement benchmark of local learning rules
(e-prop / Hebbian / R-STDP) on consumer hardware. Every number in it must trace back to a
reproduction record under `research/`, per the project plan's §12.4 rule.

Build: `latexmk -pdf main.tex` (or `pdflatex main && bibtex main && pdflatex main && pdflatex main`).
The resulting `main.pdf` is a **local build artifact and is not committed** — see below.

Status: **skeleton v0.2 + Related Work drafted** — all four subsections of §2 are written
(see the draft log at the end); §1 and §3–§7 are still structure and section plan.
Outstanding work is marked in-file with `\todo{}` (rendered red), so
`grep -c '\todo{' main.tex` counts the open items — **7**. Six are the original skeleton
placeholders (untouched by the drafting); the seventh was added by the search-specification
appendix, and it is deliberate: it marks the TechRxiv document that returned HTTP 403 and was
never read, which is the one hit whose title most resembles a standard.

## Compile status

**`main.pdf` is a local build artifact, not committed.** Run `latexmk -pdf main.tex` and it
appears in this directory; both it and the intermediates (`*.aux`, `*.log`, `*.out`, `*.blg`,
`*.bbl`) are gitignored. The reason it is not committed: a PDF **embeds a creation
timestamp**, so its hash changes on every rebuild even when the source is untouched —
committing it would add a binary diff to every documentation-only change.

**Compilability is guaranteed by CI instead**: the `latex-build` job runs `tectonic main.tex`
on every push. Tectonic pulls packages on demand and runs BibTeX itself, so the job also
checks the syntax of all 24 `refs.bib` entries. Watch that job, not a committed PDF.

**And the PDF is now downloadable from CI**: that same job uploads the compiled file as an
Actions artifact named **`paper-pdf`**, so after a push the current build can be fetched from the
run's Summary page on any device, phones included. Two things to keep straight about it: the
artifact is a **build product, not a reproducible one** — it embeds a creation timestamp, so its
hash changes on every rebuild and it is never an identity; and it lives as long as the run's
retention window (Actions' default), not forever. If you need a build from a specific commit and
the artifact has expired, re-run that run — or build locally.

Last verified locally 2026-10-15 (TinyTeX + pdflatex), after the second drafting batch:
`pdflatex → bibtex → pdflatex ×2`, no errors and no warnings, **7 pages**. `bibtex` now
resolves **14 of the 19** entries — it no longer prints `I found no \citation commands`,
which it did while the body was still citation-free.

⚠️ **The batch after that (the five near-neighbour citations, the §2.4 rewrite and the new
search-specification appendix) was NOT built locally.** The machine was running the LIF-250ep
training job under a standing rule that it may do nothing else until 23:00, so no pdflatex was
run. It is verified by CI's `tectonic` job instead — which is this repo's stated policy anyway
("watch that job, not a committed PDF"). By `grep` rather than by bibtex, the file now cites
**19 distinct keys out of 24 entries**; those counts are a proxy, not a build. (The five uncited entries —
`fremaux2010`, `pogodin2020kernelized`, `shen2025taaf`, `hu2026pssnn`, `hajizada2026clp` —
belong to §3 and later.)

Two things in the skeleton **looked** like they would fail to compile. **Both were wrong** —
asserted from memory before any LaTeX was installed, and the real build disproved both.
Recorded here as a correction rather than quietly deleted:

- line 6, `\title{...\\A Reproducibility-First ...}` — `\\` followed by `A` is fine:
  `\\` is a control symbol and does not need a space before the following text.
- line 98, `\bibliographystyle{plainnat}` — `natbib` ships with TinyTeX-1, and
  `plainnat.bst` resolves without an explicit `\usepackage{natbib}`.

## Known gaps

1. **`refs.bib` has grown from the task list's 15 to 24, in three moves.** First, three keys had
   zero hits in this repo's ledgers and were resolved by verification, recorded in
   `docs/references.md` [24][25][26]:
   - `miconi2018differentiable` — ICML 2018, PMLR 80, pp. 3559–3568;
   - `zanardi2024metaplasticity` — Phys. Rev. E 110(5), 054304;
   - `turcu2025switching` — NeurIPS 2025 **UniReps workshop**.
   Then four more on 2026-10-15, by explicit ruling, so that §2.3 and §2.4 could be written
   and so the D-RTRL/pp-prop slot had a home:
   - `membrain2026`, `javis2026` — **not new references**: these are the `@misc` forms of
     ledger entries [10] and [16], which already existed;
   - `pineau2021` — JMLR 22(164):1–20, the anchor for §2.4, registered as [27];
   - `wang2026braintrace` — Nature Communications, registered as [28]. ⚠️ **The identifier
     given in the ruling was wrong** (`arXiv:2609.08070` is a different paper by different
     authors); see the draft log below before touching this entry.
   And five more the same day, from the protocol-standard search, registered as [29]–[33]:
   `pauli2018reproducing`, `benchmarks2024locality`, `neurobench2023`, `neuroseqbench2025`,
   `step2025`. These are the **near neighbours** — the reason §2.4 can name what it is *not*
   rather than merely assert absence. Each one's per-criterion assessment is in its ledger
   entry, and §2.4 cross-references the search log in Appendix A.4.

   ⚠️ **The third carries a venue correction worth reading.** The task list's key was
   `neurips2025switching`, which assumed a **main-conference** paper. It is a **workshop**
   paper (also a DBM 2025 Findings Poster, and Cosyne 2026), from the Allen Institute, with
   **no arXiv preprint**. The key was renamed to match. Earlier attempts to fetch the author
   list all failed (NeurIPS virtual page renders authors client-side; bioRxiv refused;
   OpenReview API returned `403 ChallengeRequiredError`) — it was resolved by hand from the
   UniReps page and OpenReview, cross-checked.
   ⚠️ **`main.tex` line 44's comment carried the same imprecision** ("NeurIPS 2025 switching
   rules", with no workshop qualifier). It has since been **corrected in place** — see gap 3.
2. **`hajizada2026clp` is dated 2026** here, matching the task list and `docs/references.md`
   [8]. Note the project plan's prose says 2025; the two disagree.
3. **Three edits against the delivered file came from the reference-ledger work.** Two are
   comment-only; the third is **body text** and is called out as such below. ⚠️ This list is
   scoped to *that* work — it is **not** a running total of how far the file has diverged: the
   §2 Related Work draft changed it much further and logs its own changes in the **Draft log**
   section at the end of this file. Read the two together before drawing conclusions about what
   the file used to say.
   - (a) *comment* — line 44's comment said "NeurIPS 2025 switching rules" and now says "Turcu et
     al. 2025, UniReps workshop (switching rules; closest neighbor -- cite AND delimit)" — the
     venue correction;
   - (b) *comment* — the two comment lines under it carry the **delimiting** itself: "they switch
     discretely between plasticity rules per synapse; we mix continuously under a retention
     constraint". The `cite AND delimit` marker (a) was already there; the distinction it
     points at is what (b) writes down, so the related-work paragraph does not have to
     re-derive it.
   - (c) *body text* — the Contributions bullet "Interaction finding" read "at default
     hyperparameters the adaptation term can hurt (same-budget LIF leads ALIF by 1.04pp)". It now
     reads "...swept as a controlled grid; the adaptation term is a net drag at every learning rate
     tested (same-budget LIF leads ALIF by 0.93-1.23pp over lr 1e-3 to 1e-2; 5 seeds each)". **Two
     things were stale, not one**: "at default hyperparameters" stopped being true once the LIF
     learning-rate sweep landed (LIF leads at *every* lr tested), and 1.04pp was a single-point
     number. ⚠️ Unlike (a) and (b) this is **a claim-strength change**, not a factual touch-up; it
     was explicitly authorized on 2026-10-15 at positioning level. **The standing rule from that
     decision**: factual updates to Contributions go in without asking; strength escalations
     (adding "first", "contradicts", and the like) still need sign-off.

   **The evidence for (a) and (b) is the diff itself**: one line changed plus two comment lines
   added in `git diff --stat`, and the `\todo{}` count is unchanged at 6 (it stays 6 after (c) too
   — that bullet never carried a `\todo{}`). ⚠️ Do **not** use "the PDF
   bytes are the same" as evidence — `main.pdf` embeds a creation timestamp, so its **hash
   changes on every rebuild** even when the source is untouched (observed on the venue-correction
   commit: same 103,970 bytes, different blob hash).

## Draft log — §2 Related Work (2026-10-15)

**What was written:** §2.1 (five paragraphs) and §2.2 (four paragraphs) in full. §2.3 and §2.4
are still empty; §2.1/§2.2 keep the paragraph template the drafting plan prescribes — what the
work is, what it establishes, the gap, then a connective to our own setting.

**Two preamble/comment edits this required, both called out because neither is prose:**

- **`\usepackage{natbib}` added before `hyperref`.** The drafted text uses `\citet`, which is a
  natbib command — without the package the build fails with `Undefined control sequence`. The
  existing note above (line ~36) is still true as far as it goes (`plainnat.bst` resolves on its
  own), but it does **not** cover `\citet`, so the package is now loaded explicitly.
- **The `docs/related_work.md` pointer in the §2 comment was stale** — that file has never
  existed in this repo (checked with `git log --all`). The comment now points at
  `docs/references.md` and `docs/related_projects.md`, which are the actual grouping sources.

**One number in the delivered §2.1 opening was changed: "four" → "five" lines of work.** The
drafting plan enumerated four paragraphs (Bellec / TP / ESPP / event-driven); Millidge 2025 is
also listed for this subsection in the skeleton comment and is in `refs.bib`, so it is drafted
as a fifth paragraph and the count in the opening sentence was brought in line. Flagged because
the opening sentence was delivered as finished text.

**Open items inside §2, recorded as source comments rather than prose:**

- **D-RTRL / pp-prop** — named in the original skeleton comment, but there is no `refs.bib` entry
  for either, and that file's own rule forbids adding citations with no ledger entry. Left
  unwritten.
- **§2.3 (SNN–LLM hybrids)** — `savage2026ember` is in `refs.bib`; MEMBRAIN ([10]) and Javis
  ([16]) are `docs/references.md` entries but have no `refs.bib` keys. The subsection needs all
  three keys to be added together before it can be drafted.
- **§2.4 (reproducibility protocols)** — unwritten. The claim it rests on ("the literature is
  sparse") needs its own search record before it goes into prose.

**Two literature facts verified on 2026-10-15 while drafting, neither previously in the ledger:**

- **Turcu et al.: the "per synapse" clause in the old §2.2 comment is now confirmed.** The
  NeurIPS virtual page returns the abstract server-side: synapses "can independently change
  their governing plasticity rule during training", switching between a weak Hebbian-like
  pre-post rule and a strong credit-assignment rule (*backpropagation*) — so the strong rule is
  not local, which §2.2 now states. ⚠️ **The ledger's "[26] has no arXiv preprint" is still true
  but incomplete**: a **bioRxiv preprint exists** (10.64898/2026.06.10.731456, v2 titled
  *State-dependent plasticity routes learning signals in recurrent circuits*). The bioRxiv
  fetch itself returns **403**, the same block the ledger already records. Syncing that into
  `docs/references.md` means syncing three languages, so it is left to a deliberate edit rather
  than done here. *(Superseded the same day by the "three rulings" log below, which does sync it.)*
- **Zanardi et al.: the paragraph in §2.2 rests on the abstract** (arXiv:2403.13967) — a biased
  random walk on a cyclic feed-forward network with several levels, a Hebbian term plus a
  metaplastic reinforcement term on grouped edges, three regimes, and retrieval of a stored path
  after weights are reset to zero. The ledger's [25] entry previously recorded metadata only,
  which is not enough to write a sentence about what the paper shows.

## Draft log — §2.3, §2.4, and the three rulings (2026-10-15)

**What was written:** §2.3 (SNN–LLM hybrids) and §2.4 (reproducibility protocols), which
completes §2. Plus a half-paragraph appended to the §2.1 TP paragraph for BrainTrace, the
D-RTRL/pp-prop slot.

**All three rulings were carried out. Two of their premises did not hold**, and in both cases
it is the premise — not the goal — that was dropped:

1. **Turcu wording: implemented the goal, not the letter.** The ruling asked for
   "backpropagation-like", on the stated ground that the abstract writes "strong
   credit-assignment (BP-like)". The abstract does not say that. Re-fetched 2026-10-15, it
   reads "a strong, credit-assignment rule governing strong synapses (**backpropagation, BP**)".
   Writing "backpropagation-like" would therefore have been *less* faithful than the text it
   replaced, and "BP-like" appears nowhere in the source. The prose now says the abstract
   **labels** it backpropagation (BP) — which is what the ruling was actually after (leaving
   room, by attributing the label instead of asserting it).
2. **BrainTrace: the identifier in the ruling was wrong.** `arXiv:2609.08070` is *A
   Gradient-based yet Spike-Timing-Dependent Solution to the Feedback Learning Problem in
   Neural Microcircuits* (Zhang et al.) — not BrainTrace, and no Wang among its authors. The
   real work is **Nature Communications**, "Model-agnostic linear-memory online learning in
   spiking neural networks" (Chaoming Wang et al., online 2026-01-19), with **no arXiv
   preprint** (title search: 0 hits). It is registered as [28] and cited as
   `wang2026braintrace`; the key name the ruling proposed happens to be right for the real
   paper. Metadata came from the Janelia BibTeX export, since nature.com redirects to an IdP
   login; the algorithm list (D-RTRL, pp-prop) is from the project's own docs site.
3. **The Turcu bioRxiv sync was carried out as a deliberate trilingual edit**, into all three
   ledgers. The DOI in the ruling (`10.64898/2026.06.06.10.731456`) has an extra segment; the
   verified DOI is `10.64898/2026.06.10.731456`. And it is not just v2: bioRxiv is at **v3**
   (2026-08-30), retitled *Synaptic-state-dependent plasticity routes learning signals in
   recurrent circuits*; the licence is `cc_by_nc_nd`, so **no figure may be reused**.
   `refs.bib` keeps OpenReview as the canonical link and carries the bioRxiv provenance in
   `note`.

**The provenance gap behind §2.4's core sentence — partially closed, and the residual is
written into the file itself.** §2.4's central sentence is the defensive "To our knowledge no
equivalent protocol-level standard exists for local-rule SNN training", and the ruling says
its evidence is a **2026-10-14 literature scan**. That scan's own queries and sources are
**in no file in this repository** (searched the repo and the desktop on 2026-10-15; the only
10-14 commits are about the `biosnn-bus` 0.2.0 release). By a further ruling the scan record
now exists at `docs/literature_scans/2026-10-14-local-learning-landscape.md` (three
languages): what question the scan was meant to settle, the three criteria a counterexample
would have to meet, the ledger entries it swept with a per-entry "why this is not a standard"
column, and what it did not hit. ⚠️ **That file states plainly that it is a narrative compiled
from ledger entries on 2026-10-15, not a replay of the scan's procedure** — so what it gives
the sentence is a floor of "here is what we looked at and what each thing is", not "we swept
the field reproducibly". `pineau2021` anchors how another field does this; it is not evidence
for the absence claim. Ledger entry [27] carries the same warning. §2.4's wording therefore
stays defensive; escalating it needs the scan's actual query log first.

⚠️ **`urldate` in the two `@misc` entries does nothing.** `plainnat.bst` has no `urldate`
field; bibtex ignores it silently and the access date reaches the PDF only through the `note`
field ("Accessed 2026-10-15"), which is why both are present. Verified by reading the rendered
bibliography, not by assuming.

**Verification this batch:** local TinyTeX `pdflatex → bibtex → pdflatex ×2`, no errors, no
warnings, **7 pages**; 14 of 19 `refs.bib` entries cited. `check_references.py` (3 languages),
`check_md_tables.py` (87 files) and `check_translations.py` (78 files) all pass.

## Draft log — the search re-run, and why §2.4 did not get upgraded (2026-10-15)

A logged re-run of the literature search was authorised so that §2.4 could move from "to our
knowledge" to "a search of [sources] on [date] with [queries] found none". It landed at
`docs/literature_scans/2026-10-15-protocol-level-standard-survey.md` (three languages) with the
sources, query strings, date, return counts and inclusion/exclusion criteria recorded verbatim.

**The upgrade was not applied, because the search did not produce the clean result the upgrade
assumed.** Zero entries meet all three criteria — but there are **near neighbours**, and naming
them is unavoidable once the sentence claims to be search-backed:

- **NeuroSeqBench** — a benchmark suite that places `E-prop` alongside STBP/OTTT/SLTT and claims
  fairness guidelines; closest on "targets local-rule SNN training", but it is a code suite, and
  its visible content fixes neither seed counts nor a compute budget.
- **NeuroBench** — a neuromorphic benchmarking framework (algorithm + system tracks); its abstract
  does not touch learning rules at all.
- **arXiv:2402.01782** — a benchmark comparing SNN learning methods *stratified by locality*.
- **Pauli et al. 2018** — a reproducibility best-practices guide for spiking model *simulation*;
  it reports seed sensitivity itself (87% vs 13% land in different dynamical states), which
  **supports** our claim rather than refuting it.

Two things block the upgrade, neither technical: (1) naming them means **citing** them, and
`refs.bib`'s own rule forbids adding citations the repository does not have — they would need
ledger entries first, and **none of them is a ledger entry today** (recording them in the survey
file is not registration); (2) the "a search of ..." phrasing requires the search specification to
appear in the paper, i.e. an appendix section, or a reviewer has nothing to check against. Ledger
entry [27] carries the same warning in all three languages. **§2.4 therefore still reads "To our
knowledge ..."** — the surveyed sentence is written out in the survey file, ready to apply the day
the two blockers clear.

Also recorded in the survey, because it bounds what the file may be used for: coverage is **not
exhaustive**. It used a US-region web search index and per-URL fetches, with no in-database access
to ACM DL / IEEE Xplore / Scopus / Web of Science. It supports "with these queries, on this day, on
this index, nothing was found" — not "no such thing exists". One hit (a TechRxiv practical-guidelines
document, the title that looks most like a standard) returned **403** and was never read; it is
flagged in the file as the priority for next time.

## Draft log — the upgrade landed (2026-10-15)

The ruling came back with all three approvals at once, and with a reframing worth recording
because it changes how the section should be read: **the near neighbours are an asset, not a
threat.** "We found no equivalent" is a weak claim; "we searched, here are the four nearest
things, and here is why each fails a criterion" is a strong one — and it is *harder* to attack,
because the reviewer's obvious question has been answered in advance.

**Landed, in three parts:**

1. **Five ledger entries [29]–[33], each carrying its per-criterion assessment** (which of the
   three criteria it meets and which it misses) — that column is what Appendix A.4 and §2.4
   cross-reference. Written into all three language versions of `docs/references.md`.
   - `pauli2018reproducing` — fails all three (a guide; simulation, not training; nothing
     mandatory) yet is the batch's most useful **supporting** citation, because it reports seed
     sensitivity itself: 87% vs 13% of runs land in different dynamical states. Cited in the
     §2.4 body, not listed as a near neighbour.
   - `benchmarks2024locality` — the ruling required one more fetch before registering, to settle
     whether it fixes budget/seeds. **It does not**: the full text's only control statement is
     "consistent neural network architectures and training datasets", and its two repetition
     counts (CKA over 10 initialisations, a backdoor experiment over 5 random draws) are not
     training runs. That is the factual basis for the delimiting clause the ruling asked to give
     weight to.
   - `neurobench2023` — half on criteria 1 and 3, fails 2. ⚠️ Volume and article number were not
     obtained (nature.com → IdP login, PMC → browser check), so the arXiv version is registered
     and the journal version plus DOI live in `note`.
   - `neuroseqbench2025` — the closest hit found: it benchmarks `E-prop` beside STBP/OTTT/SLTT.
     Recorded as the entry needing the most care, since what it lacks is exactly what this paper
     controls.
   - `step2025` — fails all three, registered anyway on the ruling's instruction, so its
     "fair and reproducible benchmarking" wording is cited and bounded before a reviewer asks.
2. **§2.4 rewritten**, using the sentence prepared in the survey file, with the search log
   cross-referenced. **This replaces the defensive "To our knowledge ..." wording** — that is a
   claim-shape change, made under the ruling's explicit approval.
3. **A new appendix section, "Search specification"** (`\label{app:search}`): the three criteria,
   the ten search rows verbatim (source, query string, returned, included), the
   inclusion/exclusion rule, and the coverage boundary — a US-region index and per-URL fetches,
   no in-database ACM DL / IEEE Xplore / Scopus / Web of Science, so it is a **survey, not a
   systematic review**, and supports "with these queries, on this date, on this index, nothing
   was found", not "no such thing exists". The TechRxiv 403 item is carried as a `\todo{}` in the
   appendix — hence the open-item count moving 6 → 7. Keeping it red is the point: it is the one
   hit whose title most resembles a standard and it is still unread.

⚠️ **Not built locally.** The machine was under the ruling's training-only discipline, so no
pdflatex ran on this batch; CI's `tectonic` job is the verification channel. If it goes red, the
likely culprits are the appendix table's column widths or the `\ref{app:search}` label.

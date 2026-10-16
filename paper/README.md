# paper/ — P2 benchmark paper (LaTeX)

Source for the **P2 paper**: an honest-measurement benchmark of local learning rules
(e-prop / Hebbian / R-STDP) on consumer hardware. Every number in it must trace back to a
reproduction record under `research/`, per the project plan's §12.4 rule.

Build: `latexmk -pdf main.tex` (or `pdflatex main && bibtex main && pdflatex main && pdflatex main`).
The resulting `main.pdf` is a **local build artifact and is not committed** — see below.

Status: **full draft** — §1–§7 and all four appendices are written. What is left is not
prose but three measured items, marked in-file with `\todo{}` (rendered red). `grep -c '\todo{'
main.tex` returns **5** lines, but two of them are comments — the macro's own documentation at
the top of the file, and a note on the bibliography block — so the count of **open items in the
body is 3**: (i) the BPTT control arm's matched-budget number (its recipe pre-sweep is running),
(ii) memory/throughput as a function of network size, which has not been measured at all, and
(iii) the TechRxiv document that returned HTTP 403 in the search build, which is the one hit
whose title most resembles a standard.

**Figures.** Two of the repository's generated figures are in the paper —
`f2-w2-epoch-accuracy.png` (§5.2.3) and `f3-w3-greedy-curves.png` (§5.3). Both are regenerated
from ingested records by `scripts/make_figures.py`, which now writes each one **twice**: to
`figures/` (the canonical copy the READMEs cite) and to `paper/figures/` (the copy the paper
includes). The duplication is deliberate — see the script's docstring: including the canonical
copy via `../figures/` would make the paper read outside `paper/`, and that path's behaviour
under the CI engine (tectonic) has never been verified in this repository. Two generated copies
cannot drift; a hand-maintained second copy could.

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

⚠️ **Green does not mean warning-free.** The job fails on errors, not on box warnings, and
tectonic does emit some (verified by reading the run log of 2026-10-15, run 38689701424):
**Overfull** at `main.tex:33-34` and `:36-37` — the two long Contributions bullets, which predate
the drafting batches and are not caused by them; and **Underfull** (badness 10000) at
`main.tex:287-290` — the appendix A.4 search-log table, where narrow `p{}` columns stretch short
cell content. Underfull there is cosmetic; what would matter is *Overfull* in the table, and there
is none, so the table fits inside the text width. The 8-page artifact was downloaded and its text
extracted to confirm §2.4 and the appendix table are actually in the PDF, not merely in the
source.

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

## Pre-approval on record — the "W2 official number" batch (2026-10-15)

**Trigger: the 250-epoch LIF run's plateau verdict. EXECUTED 2026-10-16, 05:47.** Recorded here
because it was a standing instruction, not a one-off: the six edits went in **without further
sign-off**.

**Execution record (2026-10-16):**

- **Official number = `0.8889`** — LIF, lr = 1e-2, 250 epochs, seeds 0.8962 / 0.8854 / 0.8851,
  range 1.11 pp, wall clock 5.86 / 6.05 / 5.91 h.
- **Plateau verdict: passed.** Last-20-epoch window-mean increments **+0.0019 / +0.0007 /
  +0.0025** — the same order as ALIF's +0.0016 / −0.0037 / +0.0025. **But all three are positive**
  where ALIF had both signs, so the judgment rests on **magnitude, not sign**: epoch-to-epoch val
  spread in this arm is ≈ ±0.009, so the standard error of a 20-epoch mean is ≈ 0.002 and the three
  increments sit at 0.35–1.25 SE. Because the ALIF record's own phrasing was "both signs", this
  difference is written out in `research/eprop/README.*` rather than smoothed over — the next reader
  should see the criterion, not only the verdict.
- All six locations updated; **`grep -c 0.8831 paper/main.tex` → 0**.
  ⚠️ **One clarification of location 5's criterion, found on execution**: the grep was specified
  over `paper/`, but this README now *documents* the change and therefore quotes 0.8831 by
  necessity. The criterion that carries the meaning is the **paper source**, which is clean.
  Recorded rather than left looking like a miss.
- ⚠️ **A seventh change, not on the list: the official run's artefacts were moved into the
  repository.** They had been written to `data/w2_lif_long/`, and `data/` is gitignored — so the
  "only source of the official number" sat *outside* version control while its control sat inside.
  Copied to `sweep_results/w2_lif_long/` plus the log, following the existing practice for accepted
  runs. ⚠️ **The "byte-identical" claim needs its scope stated**: the two sides were byte-identical
  at the moment of copying (md5 equal), but `.gitattributes` mandates `eol=lf`, so the committed
  copies are LF while `data/`'s working copies are CRLF — **same content, different line endings**
  (equal again once CR is stripped), which is how this repo stores every text record. A reader who
  compares `git show … | md5sum` against `data/` will see a mismatch; it is the line-ending
  normalisation, not the data. Without the move, §5.2 and the 2026-10-22 deposit would both cite a
  path that git does not carry.

The official headline number for W2 becomes **LIF @ lr=1e-2 @ 250 epochs** (the arm the
pre-registered decision rule selected), and ALIF's **0.8831** is demoted to a **control** value.
The six locations, inventoried before the trigger so the execution is mechanical:

| # | Location | Action |
| :--- | :--- | :--- |
| 1 | `main.tex:35` — Contributions, "Convergence evidence" | `0.8831` → the official LIF value, four decimals, same convention as the ALIF number |
| 2 | `main.tex:24` — the abstract's `\todo{}` (its number preview says "converges at 0.8831 (250ep)") | use the official value when the abstract is written; `0.8831` must not appear in the abstract |
| 3 | `README.md:58` / `README.en.md:66` / `README.ja.md:64` — the "current status" table's W2 row | LIF leads across the sweep; official = LIF@1e-2@250ep; ALIF 0.8831 (control). **All three languages** |
| 4 | `research/eprop/README.{md,en.md,ja.md}` (the extended-budget row and the convergence discussion) + `sweep_results/README.md:86` | **Add a pointer, do not rewrite history.** The ALIF run really did produce 0.8831 — that entry is a true record. What moves is the *role*: mark 0.8831 as the control and say the official number's only source is the LIF run |
| 5 | `grep -rn "0.8831" paper/` | **Must come back empty.** Today it returns three hits: the two above plus `main.tex:240`, a comment on the Budget-extension subsubsection. A result-table control row is the one allowed exception — and §5.2 is not written yet, so it is out of this batch's grep either way |
| 6 | `BioSNN-Plug_项目计划书_v6.3.md:860` — approved as a sixth location after the fact | Append a note in the same format the `research/` docs use: **「官方收官数字 = LIF@1e-2@250ep = 0.XXXX（见 `sweep_results/w2_lif_long`）；本行 0.8831 为 ALIF 对照收敛值」**. The plan is a **living document**: it may grow more complete, it may not become false. ⚠️ **The value slot is filled from the same run as locations 1–3, in one edit** — writing the note early would leave a hole in the plan, which is neither "more complete" nor "false", and the plan currently carries no placeholders at all |

**Two edge cases, decided in advance:**

- **If the plateau criterion fails** (the last-20-epoch window increments have not settled to
  noise), the number still goes in but the word "converges" does **not**. That is a claim-strength
  issue and therefore outside the pre-approval — it comes back for a ruling.
- **`BioSNN-Plug_项目计划书_v6.3.md:860` also carries 0.8831** ("已实测 ... 且已收敛"). It was
  **approved as a sixth location on 2026-10-15**, on the ruling that the plan is a living document:
  it may grow more complete, it may not become false. As it stands the line is a record of what was
  measured rather than a claim about which arm is official, so it does not become *false* — it
  becomes *incomplete*, and row 6 above completes it in a single edit alongside the others.

**Not to be touched:** `sweep_results/w2_long*` (the log and the seed JSONs). Those are
reproduction records; the batch changes the *documentation's* role assignment, never the evidence.

## Publication batch — pre-approval on record (split decided 2026-10-15, for 2026-10-22)

**Two records, not two versions of one.** The earlier "v0.1 today / v0.2 on 10-22" framing was
dropped: versioning a skeleton invites reading today's deposit as a defective predecessor of the
paper, and it forced one licence onto files that have two different native ones.

- **Record A — today (2026-10-15): a findings report, not the paper.** Attachments:
  `sweep_results/README.md` plus the English edition of the W2 README
  (`research/eprop/README.en.md`), `publication_type = report`, licence **Apache-2.0** — both files
  are repository documents, so this is their native licence and a single one. **No PDF.** Dropping
  the PDF also retires the `\todo{email}` blocker, because the paper is not part of A.
  Package built at `Desktop/zenodo_v0.1/` (metadata JSON + upload steps + the pre-deposit scan).
  **Status 2026-10-15: PUBLISHED.** Version DOI **`10.5281/zenodo.23388709`**, concept DOI
  **`10.5281/zenodo.23388708`**; record created 2026-10-15T08:17:13Z.
  - **Verified from Zenodo's public API**, not from the upload response and not from this
    transcript: DOI, concept DOI, title, `publication` / `report`, `apache2.0`, `v0.1`, creator
    `lzmd-arch`, and both files at 18,154 and 35,937 bytes — md5 and size match the local copies
    byte-for-byte.
  - **The publish step itself was blocked by this environment's permission classifier** ("Create
    Public Surface"): minting a public DOI is reserved for a human action here. The ruling's
    sandbox route was also unavailable — `sandbox.zenodo.org` is a separate service needing its
    own token, and only production `ZENODO_TOKEN` is set. So a **private production draft** stood
    in for the sandbox run (an unsubmitted deposition is invisible to everyone but its owner), the
    pipeline was exercised through it end to end, and the human clicked Publish.
  - **What this DOI is, and is not.** It identifies the *findings and protocol record* (a report).
    It is **not** the software's DOI and **not** the paper's, so it does not belong in
    `CITATION.cff`'s `doi:` field — that field describes the repository itself.
  - **The attachments are frozen, and had gone stale — the record now says so.** A published
    Zenodo record's files **cannot be replaced**; only a new version can change them. One day
    after the deposit, the W2 official number moved arms (LIF 0.8889 replacing ALIF 0.8831 as the
    headline, with 0.8831 demoted to control), so the attached record index — the 2026-10-15
    snapshot — was stating a superseded number **with nothing to mark it as superseded**. Fixed
    the surgical way, on 2026-10-16: a **metadata-only** edit appending an update note to the
    record's description (`Numbers updated after this deposit ... 0.8889 ... the attachments are
    the 2026-10-15 snapshot`). Verified from the public API afterwards: DOI and concept DOI
    unchanged, status `published`, description 1,988 characters carrying the note, and **both
    attachments byte-identical to what was deposited** (18,154 and 35,937 bytes).
    ⚠️ Why not a new version: A is a *timestamp*, and version-churning it would turn a dated
    snapshot into a moving target. Why not leave it to B: B is a separate record and will not
    retro-fix A — a reader who lands on A alone needs the correction *on A*.
    ⚠️ **Mechanics worth remembering**: editing a published Zenodo record goes
    unlock (`actions/edit`) → `PUT` metadata → re-publish. A failure between the first two steps
    leaves the record as a **draft, invisible to the public** — so the script that did this
    re-publishes on every failure path. Editing metadata does **not** mint a new DOI; replacing
    files would.
  - ⚠️ **A private URL in a tracked file fails CI, and local checks will not tell you.**
    The intermediate commit `3d5c3b0` recorded the *draft* page (`https://zenodo.org/deposit/23388709`)
    in this file; an unsubmitted deposition is visible only to its owner, so lychee got 403/404 and the
    `引用链接存活检查（lychee）` job went red. `check_readme_links.py` — the local check — validates
    relative paths and self-referential URLs only, by design; external URLs are lychee's job. So
    "local link check green" says **nothing** about a newly added external URL, private ones
    especially. Fixed by replacing the draft URL with the DOI text in `3af9771`; the lesson is the
    one the repo keeps re-learning: **watch the run, not the local check**.
- **Record B — 2026-10-22: the full paper**, Zenodo + arXiv, `publication_type = preprint`,
  licence CC-BY-4.0.

**Standing items for B:**

1. **Contact method = the GitHub repository URL, not an email.** It replaces the author block's
   `\todo{email}`: zero new exposure, functionally equivalent.
2. **Cross-reference A's DOI — the value is now known: `10.5281/zenodo.23388709` (version) /
   `10.5281/zenodo.23388708` (concept).** Add it to B's related identifiers (relation
   `isSupplementedBy` or `references`, decide when B's metadata is written), and edit A's record
   to point back at B once B exists — the two must read as siblings, never as a draft and its
   sequel.
3. ⚠️ **arXiv requires an endorsement for a first submission.** Verified against arXiv's own pages
   on 2026-10-15: a new submitter needs one "before submitting their first paper to arXiv or a new
   category", obtained either automatically (needs an institutional email address, which this
   project does not have) or personally, by requesting it from an established arXiv author in the
   category. **That has lead time — it cannot be arranged on 10-22.** If B is going to arXiv,
   finding an endorser starts now.
4. ⚠️ **arXiv's identity policy is the sharper question.** The same pages state: "It is a violation
   of our policies to misrepresent your identity or organizational affiliation", with misrepresentation
   grounds for "immediate and permanent suspension". Pseudonyms are **not** explicitly banned, so
   this is not a hard no — but it is a grey zone, and item 1's repository URL solves the contact half
   of it, not the identity half. **This one needs a human decision before 10-22**, not a mechanical
   edit.

## Draft log — the full draft (2026-10-16)

**What was written:** §1 (motivation, the measurement gap, contributions, positioning), §3
(methods: systems under test, the evaluation protocol written out in full, the triviality
analysis, the boundary statement), §4 (setup), §5 (all five result subsections), §6
(discussion), §7 (conclusion), and appendices A.1–A.3. The abstract is written too. §2 was
already complete.

**Every number traces to a record**, per the file header's rule: W1's two readouts and the
five-seed range from `w1_seeds/`, the four learning rates and their LIF counterparts from the
`w2_lr*`/`w2_lif_lr*` series, the extended arms from `w2_long/` and `w2_lif_long/`, W3's
acceptance and tuning batches from `step14_confirm`/`step13_lr001`, SHD from `w2_shd/`, and the
memory peaks from the same runs. Where a number does not exist, the text says so and marks it
rather than reaching for a neighbouring one — the BPTT subsection, the convergence figure, and
the size-scaling curve are the three cases.

**Two claims deliberately narrowed while writing:**

- **W1's accuracy is presented as a joint result** of local features and a supervised readout,
  at the point where the number appears, not in a footnote. The source paper's tables invite the
  opposite reading.
- **The SHD arm is attributed to the configuration, not to the rule.** The text states that no
  hyperparameter search was run for SHD, so the 5.34% measures this configuration's
  non-transfer, and explicitly refuses the stronger reading.

**One author-block change, under a standing ruling:** `\todo{email}` is replaced by the
repository URL (the 2026-10-15 ruling: contact method = repo URL, zero new exposure, functionally
equivalent). That is also why the open-item count moved by one.

⚠️ **Not built locally.** The machine is running the BPTT pre-sweep, so the training-only
discipline applies and no pdflatex ran. CI's `tectonic` job is the verification channel for this
batch, as the repo's own policy states; the run is being watched, and a compile error will be
fixed as soon as it appears rather than left for the next reader.

## Draft log — figures (2026-10-16)

**The figure problem was not "a figure is missing" but "the figure that exists is stale".** The
repository's per-epoch W2 figure stopped at 100 epochs, because it was drawn before the
250-epoch arms existed; including it beside text that runs to 250 would have been exactly the
kind of mismatch this paper argues against. So the figure was **extended rather than swapped**:
`make_w2_curve` now reads four batches — 30, 100, and the two 250-epoch arms — and shades the
last-twenty-epoch window, because that is where the convergence criterion is computed and a
criterion whose sampling region is invisible is not checkable by a reader. The 30/100 prefix
overlap still verifies exactly (maximum pointwise difference 0.000000).

**What the figure shows is deliberately unflattering to the paper's own story**: the two
250-epoch arms are 0.8889 and 0.8831, half a percentage point apart, and the two curves sit on
top of each other. They were not separated for legibility.

**Two caption corrections made by looking at the render, not at the script.** The W3 caption
first claimed the ensemble "wanders above and below the threshold for the whole run" — the
figure says otherwise: the individual runs swing from ~10 to ~500 steps, but the *median* stays
below the threshold until roughly episode 700. The caption now describes what the figure shows.
This is the second time in this repository that a claim was corrected by reading the artifact
rather than the code that produced it.

**`f3-w3-greedy-curves.png` regenerates byte-identically** from the acceptance records, which is
how we know the committed copy was current — checked by running the generator and finding no
diff, not by trusting its date.

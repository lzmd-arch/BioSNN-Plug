# paper/ — P2 benchmark paper (LaTeX)

Source for the **P2 paper**: an honest-measurement benchmark of local learning rules
(e-prop / Hebbian / R-STDP) on consumer hardware. Every number in it must trace back to a
reproduction record under `research/`, per the project plan's §12.4 rule.

Build: `latexmk -pdf main.tex` (or `pdflatex main && bibtex main && pdflatex main && pdflatex main`).
The resulting `main.pdf` is a **local build artifact and is not committed** — see below.

Status: **complete draft — zero open items in the body.** §1–§7, all four appendices, three
figures, the headline control arm and the resource scaling curve have all landed. The last open
item, the TechRxiv document that returned HTTP 403 in the search build, is now written up in the
appendix as a **stated limitation** rather than a red marker: it exists, every route to it was
blocked from this environment, and it is the single most likely candidate to falsify §2.4's claim
--- which is exactly why it is recorded with its DOI for a reader who can reach it. `grep -c
'\todo{' main.tex` returns **1**, and that one is a comment documenting the macro itself.

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

First verified locally 2026-10-15 (TinyTeX + pdflatex), after the second drafting batch:
`pdflatex → bibtex → pdflatex ×2`, no errors and no warnings, **7 pages**. `bibtex` now
resolves **14 of the 19** entries — it no longer prints `I found no \citation commands`,
which it did while the body was still citation-free.

**Re-verified locally 2026-10-17 (revision 8)**, in a scratch copy outside the repository:
`pdflatex → bibtex → pdflatex ×2`, 20 pages, **zero overfull boxes**. The local channel came
back when `\usepackage{CJKutf8}` was dropped from the preamble --- see the author-name log
below for why removing it was a net gain rather than a loss.

⚠️ **The batch after that (the five near-neighbour citations, the §2.4 rewrite and the new
search-specification appendix) was NOT built locally.** The machine was running the LIF-250ep
training job under a standing rule that it may do nothing else until 23:00, so no pdflatex was
run. It is verified by CI's `tectonic` job instead — which is this repo's stated policy anyway
("watch that job, not a committed PDF"). By `grep` rather than by bibtex, the file now cites
**19 distinct keys out of 24 entries**; those counts are a proxy, not a build.

*(Superseded 2026-10-16: after the two entries added that day, `refs.bib` holds **26** and the body
cites **22**; the four uncited ones — `shen2025taaf`, `hu2026pssnn`, `hajizada2026clp`,
`bellec2018lsnn` — are either plan-phase topics this paper does not discuss or, in the last case, an
entry added for a literature comparison that §1.3 no longer makes. `bellec2018lsnn` is kept: it is
a verified ledger entry, and a later version of this paper may want it.)*

## Release plan: two versions, one source

**Venue facts, checked against the source on 2026-10-17** (IEEE WCCI 2026 *Information for
authors*, <https://attend.ieee.org/wcci-2026/information-for-authors/> — WCCI is the congress that
hosts IJCNN, so these govern an IJCNN submission):

| Requirement | What the venue says | What we have |
| :--- | :--- | :--- |
| **Page limit** | **up to 6 pages, *including references and appendices***; up to 2 extra pages for an added charge (8 max) | **20 pages** |
| **Template** | IEEE conference proceedings template, 10 pt body; LaTeX: `\documentclass[10pt,conference]{IEEEtran}` | `article`, 11 pt, single column |
| **Deviations** | "may result in desk rejection without further review" | — |
| **Review** | **double blind**; the submission "should be anonymized" | author block + repository URL + two DOIs, all identifying |
| **AI text** | "must be disclosed in **acknowledgements**" | disclosed in a standalone section |

⚠️ **Correction to an earlier note in this file.** An earlier version of this plan said the limit was
"≈10 pages" and proposed cutting appendices A–D plus merging §4 into §3.2. That was wrong on the
number and, because of it, wrong on the plan: **6 pages including references is not a trim, it is a
different paper.** Cutting all four appendices and §4 removes roughly four pages of a twenty-page
manuscript. The honest statement is that **no compressed version exists yet**, and producing one is
a rewrite task with its own budget, not a post-processing step.

**What the compressed version has to decide**, in order:

1. **Which of the three rule lines survives.** Three lines, a neuron-model × learning-rate grid, a
   matched-budget BPTT control, a negative SHD result, a resource sweep, and a protocol — that is
   five papers' worth of claims in six pages. The protocol and the control are the paper's
   contribution; W1 or the resource sweep is the candidate to drop to a sentence.
2. **Anonymisation.** Author block, repository URL, the Zenodo DOIs, and the "sole author" phrasing
   all identify. Everything that points at the repository has to go or be neutralised for the
   submission, and restored for the arXiv version.
3. **The AI disclosure moves into Acknowledgements**, per the venue rule — carrying the substance,
   with the full version's section cited as the detail.

**What must survive *in substance* in the compressed version**, however short it gets: §5.5's
self-refutation (the control wins by 7.41 points, and the resource curve does not support the
quadratic claim), §1.4 and §6.3's stated boundaries, §2.2's framing of Miconi as a contrast rather
than support, W3's disclosure of the tuning-to-acceptance selection effect, and the AI disclosure.
⚠️ A compression pass may shorten these; it may not silently turn any of them into a more
flattering reading, and it may not drop one to make room for a result.

⚠️ **The two versions must not diverge in numbers.** The conference version is a subset of the
arXiv version at the same commit; if a number changes, both are rebuilt. And the arXiv version
stays the record of what was measured — the compressed one is what fits. (The five uncited entries —
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
  - **v0.2 published 2026-10-17 — this time the fix *was* a new version.** Version DOI
    **`10.5281/zenodo.23428426`**; the concept DOI is unchanged (`10.5281/zenodo.23388708`), and
    v0.1 keeps its own DOI and its frozen attachments. What changed is the point: last time only
    the *description* had gone stale, so metadata sufficed; this time the **attachments** were what
    needed updating, and a published record's files cannot be replaced. New attachments are the
    current `sweep_results/README.md` (23,645 B) and `research/eprop/README.en.md` (37,245 B),
    **verified byte-identical from Zenodo's side** after upload.
    Three measurements landed between the versions, and the attached index now carries all three:
    the W2 official arm moving to LIF (0.8889), the matched-budget BPTT control (0.9630, a 7.41-point
    gap), and the six-size resource sweep. **The paper is deliberately not in this deposit** — it is
    record B, planned for 2026-10-22.
    ⚠️ **Two version numbers, one distinction worth keeping straight**: v0.1's metadata note and
    v0.2's contents say different things about the W2 headline, and that is correct — v0.1 is a
    timestamp of 2026-10-15, v0.2 of 2026-10-17. A reader who lands on v0.1 should follow the note;
    a reader on v0.2 has the current numbers.
  - ⚠️ **A private URL in a tracked file fails CI, and local checks will not tell you.**
    The intermediate commit `3d5c3b0` recorded the *draft* page (`https://zenodo.org/deposit/23388709`)
    in this file; an unsubmitted deposition is visible only to its owner, so lychee got 403/404 and the
    `引用链接存活检查（lychee）` job went red. `check_readme_links.py` — the local check — validates
    relative paths and self-referential URLs only, by design; external URLs are lychee's job. So
    "local link check green" says **nothing** about a newly added external URL, private ones
    especially. Fixed by replacing the draft URL with the DOI text in `3af9771`; the lesson is the
    one the repo keeps re-learning: **watch the run, not the local check**.
- **Record B — the full paper. Zenodo: PUBLISHED 2026-10-17 as v1.0.** DOI
  **`10.5281/zenodo.23429028`**, concept DOI `10.5281/zenodo.23429027`, `publication_type = preprint`,
  licence CC-BY-4.0, one attachment `BioSNN-Plug_P2_paper_v1.0.pdf` (20 pages, 658,248 bytes,
  `md5:7a36024d…`), built from commit `508c6b3`. The 2026-10-22 date in the earlier framing was a
  plan, not a constraint; the deposit was moved up on the author's instruction. Verified from
  Zenodo's **public** API after publishing, not from the upload response.
  ⚠️ **arXiv is still pending**: the endorsement question (standing item 3 below) is unresolved and
  is a human action, not a script.

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

## Draft log — correctness pass (2026-10-16)

A read-through looking for **claims that were wrong or unsourced**, rather than for prose to
improve. It found one of each plus a missing standard section.

**① A misattribution the repository had already corrected elsewhere.** The paper said the W3
critic is trained by "the TD-LTP rule of Frémaux et al. (2010)". TD-LTP's **name and form** come
from the **2013** paper; the 2010 paper is where the *bias* of reward-modulated STDP is analysed
— which is precisely the phenomenon §5.3 reproduces structurally. `docs/references.md` [12]
records this correction for the project plan; the paper never got it. Both papers are now cited,
each for what it actually says, and the sentence states the distinction rather than hiding it.
⚠️ Worth noting how this happened: the error was in the **delivered** text and survived several
passes because it reads correctly. It was caught by checking the citation against the ledger,
not by re-reading the sentence.

**② An unsourced literature claim.** §5.4 said "the e-prop literature reports SHD in the 63–81%
range" with no citation. The range comes from the e-prop rows in the traces-propagation paper's
comparison table, which those rows footnote as transcribed from a third party — a fact the
sentence already asserts, so it was odd not to cite. Now cited.

**③ A vague comparison replaced by a real one.** The Contributions bullet said 0.8889 sits at the
"lower edge of the e-prop literature band". There is no such band in the ledger: the comparable
sMNIST numbers in this repository's records belong to the **LSNN**, trained by **BPTT + DEEP R**,
not by e-prop, and they average 93.8–94.2 across twelve runs. Rather than drop the comparison,
the bullet now states it with its source and its size: the local-rule gap at this scale is four
to five points. Also note what the old phrasing would have implied — that 0.8889 is merely
"literature-typical", when the honest reading is that it is *below* a non-local baseline on the
same task.

**④ A standard section that was simply missing**: "Data and code availability", carrying the
deposit's version and concept DOIs. It states that the archived record is a **snapshot** whose
attachments are frozen and whose metadata records what was superseded — which is only true
because that note was added to it earlier today.

`refs.bib` went 24 → 26 entries, and **neither is a new reference**: `fremaux2013` and
`bellec2018lsnn` are ledger entries [12] and [22], verified there with verbatim quotes from the
sources. The paper was missing them, not the repository.

## Draft log — the control arm lands (2026-10-17)

**BPTT 0.9630 against e-prop 0.8889: a 7.41-point gap in the control's favour**, at matched
learning rate, seeds, budget and neuron model, with each side tuned at its own best recipe — the
control's recipe chosen by a pre-registered pre-sweep, so the comparison does not rest on the
optimizer asymmetry that the pre-sweep was built to expose. In fact it does not rest on it at all:
at the learning rate finally chosen, the two optimizer variants differ by 0.88 pp, inside the
pre-registered tie band.

**The plateau criterion was applied here too, and it passes more cleanly than e-prop's.** The
increments are +0.0003 / −0.0001 / −0.0002 — mixed signs, and roughly a tenth of the within-window
standard deviation, against e-prop's +0.0019 / +0.0007 / +0.0025. This matters because the two
claims share a sentence shape: "converges at X" now has the same evidential basis on both sides.

**What the number does to the paper.** It is the largest single effect measured anywhere in this
work, and it points away from the family the paper set out to benchmark. Two things were changed
rather than left standing:

- **§5.2.4** was written to report it as measured, including the budget dynamics: the gap
  *narrows* from about 14 points at 30 epochs to 7.41 at 250, so budget closes part of the
  distance and none of the story is "the local rule simply loses".
- ⚠️ **The abstract now carries it** ("Four findings, two of them uncomfortable"). That is a change
  to the paper's headline framing, not a factual touch-up to a number — flagged here for review.
  The alternative, an abstract that omits the paper's largest measured effect, did not seem
  defensible.

**One item in §6.3 had become false and was rewritten**: "the control arm is incomplete" was true
yesterday and is not true now. It is replaced by what the completed control does and does not
license — one control, one scale, one task family, and the observation that a local-rule paper
reporting accuracy without a matched same-recipe backprop control is reporting a number with no
denominator.

**Recorded with its caveat:** the arm's wall clocks (2.59 / 3.77 / 3.05 h) grow across three
identical consecutive runs, and the fastest rate observed anywhere in this work (0.59 min/epoch,
baseline 0.92) came from this arm. Neither is attributed to a cause, and neither enters a
performance baseline — see the pre-registration and `docs/reproducibility.md`.

## Draft log — the scaling curve, and a claim it does not support (2026-10-17)

**The sweep ran clean and the result is a negative one.** Six layer sizes, two epochs each, on a
quiet machine (pre-run check: 2,108 MiB, no Ollama, 1-epoch probe 0.98 min/epoch against a 0.92
baseline). Log-log slopes: **0.86 for peak memory**, 1.15 for time per epoch.

**What that does to the paper's own framing.** The curve was built to corroborate the per-synapse
trace accounting — e-prop's traces scale as $N^2$, which is the storage argument §2.1 carries from
the source paper. It does not corroborate it as the *dominant* term, and the magnitudes matter:
by the project's own formula for the four per-synapse tensors, the traces are about 2.0 GB at
$n_{\text{rec}}=2{,}048$ against an observed peak of 11.7 GB — **17.7%**, rising from 0.34% at the
bottom of the sweep. So they are present and their share climbs fastest, but what fills the card
at these sizes is a sub-linear term. The paper now cites the quadratic accounting as **the source
paper's**, explicitly not as our measurement, and §5.5 says why. The pre-registration anticipated
exactly this branch ("the slope is a result, not a premise").

⚠️ **A self-caught error, recorded because of how it was made.** The first version of that
paragraph said the trace term was "about 17 MiB, three orders of magnitude below the peak". That
came from computing **one** $N^2$ tensor and forgetting the batch dimension — the same mistake the
project plan had already corrected once for the same formula. The number is now ~2.0 GB and the
conclusion survives the correction, but the error was in the *flattering* direction: it made the
measurement look like stronger evidence than it is. Found by checking the paper's number against
the plan's own arithmetic rather than by re-reading the sentence.

**The largest size did not fit, in a way the pre-registration had not enumerated.** It did not
OOM: all six points returned `rc=0`. But the 2,048 point *allocated 11,744.8 MiB on an 8,151 MiB
card* — Windows fell back to shared memory — so its throughput (2,646.8 s/epoch, six times the
next point) is a paging measurement. The record keeps the number and labels it rather than
replacing it with an estimate, and the prereg's execution record names the third outcome it
hadn't listed.

**Two sentences elsewhere had become false and were rewritten**: §4 said resource behaviour by
size "is not measured here", and §6.3's limits said "we do not have a scaling curve". Both are
updated to what the curve does and does not license — including the part that is genuinely useful:
up to 1,024 neurons this pipeline fits a consumer card, and the binding constraint is time, not
capacity.

This is the third time in this paper's drafting that a measurement has contradicted the claim it
was commissioned for (the LIF/ALIF interaction, the BPTT control, and now the scaling curve). All
three are reported; two of them are now in the abstract.

## Draft log — the revision round (2026-10-17)

Four issues were raised against the draft, and the useful part of this entry is **which of them
were real** — because two were not, and the reporting of the two that were is where the lesson is.

**① A self-contradiction in §1.3 — real, and the most serious.** The Contributions bullet carried
a *second* gap claim: 0.8889 against an LSNN baseline from the literature, "four to five points".
Written before the control arm existed, it now contradicted §5.2.4's measured 7.41 points — two
different gaps in one paper — and it compared across configurations, which is exactly what the
protocol forbids. The bullet now states the converged number and hands the comparison to §5.2.4
alone: **a number gets one home.** `bellec2018lsnn` becomes the fourth uncited entry (kept, not
deleted).

**② Broken tables — real, but not where I first looked, and my first answer was wrong.** The
report was that tables render with their rows crammed. Text extraction of the built PDF looked
clean, and I said so. It was clean because **the text is still there**: an overfull table pushes
its right-hand column past the margin rather than overlapping anything, so `pdftotext -layout`
cannot see the defect. The compiler can: three `Overfull \hbox` warnings, two of them ~2 cm and
~1.7 cm too wide. Cause, once located: **unbreakable content** — long `\texttt{}` strings (a
package list, a DOI) and two table cells that were `l` columns with long prose in them. Fixed by
making the text columns `p{}` and the strings breakable; the build now reports **zero** overfull
boxes. ⚠️ **My earlier claim that the tables were fine came from reading a build log from before
those tables existed** — the same mistake as trusting a stale artifact, one level up.

**③ The abstract's last sentence is not truncated, and ④ the correspondence footnote is not
empty.** Both render complete in the current PDF (`...a benchmark of this kind invites.`;
`Correspondence: https://github.com/lzmd-arch/BioSNN-Plug`). Recorded rather than silently fixed,
because the likely explanation is a **stale PDF in a viewer** — the same file name has now held
three different builds, which is its own defect, and is why the Desktop copy now carries the
commit hash in its name.

**Consistency items carried out:** Table 1's caption now says the two accuracies are from the
acceptance run (seed 0) and gives the five-seed mean beside them, so 0.9779 and 0.9774 no longer
read as a typo. The NeuroBench entry's dangling "100 authors" fragment is reworded. The ESPP
author name was checked against ledger [20] and **is correct** — no change.

**And a process failure worth its own line:** fixing the last overfull box, I inserted
`\end{sloppypar}` with `sed`, which **ate the backslash**, leaving `end{sloppypar}` in the source.
Caught by printing the bytes. That is the fourth instance of the same shell-quoting family in one
session, and it happened *after* I had written the rule and the memory entry about it. The rule
works; using `sed` instead of the editor is what failed.

## Draft log — the AI-use disclosure (2026-10-17)

**Added: "Use of AI assistance", a disclosure section beside "Data and code availability".** It
states four things, and the reason it is this long is that the alternative — "AI was used" — cannot
be checked by anyone.

- **What the agent did**, split by kind: the literature search and the ledger's metadata
  verification; drafting §2 against a template the author set; the sweep drivers, tooling and figure
  generator, under the author's direction; the remaining prose. Plus the part that matters most for a
  benchmark: **the agent executed the experiments.**
- **That the decision rules preceded the data**, in three cases, and that the pre-registrations are
  in the public record — so a reader can check the ordering rather than take it on trust.
- **Who is responsible**: the author, sole author, for every claim, every number, and the
  verification of each against its record. The agent is not a co-author and cannot be one — it
  cannot consent to authorship or bear responsibility for content.
- **That the verification caught things**, with three named examples: the inherited misattribution,
  the arithmetic error that flattered a negative result, and the stale figure. A disclosure that only
  asserts diligence is worth less than one that shows where diligence bit.

⚠️ **The section is in the never-cut list** for the compressed conference version, alongside the
other four. A venue that asks for an AI statement and receives a trimmed one is worse served than
one that receives none.

⚠️ **One more overfull box came with it** — `\texttt{paper/README.md}`, unbreakable at the slash —
fixed with `\allowbreak`. Worth noting because it is now the *second* time a new section has
introduced a margin violation that the compiler flagged and text extraction did not.

## Draft log — the verification round (2026-10-17)

Verification was done **mechanically where possible**: a script recomputed every reported number
from `sweep_results/` and checked for the resulting string in the source, rather than re-reading
the paper for typos. 35 numbers, 34 clean, **1 real discrepancy**, plus a second one found by
following the first.

**① The learning-rate table's difference column mixed two arithmetic routes.** The LIF−ALIF
column read +0.93 / +1.04 / +1.05 / +1.23. Recomputing: the 1e-3 value came from subtracting the
*rounded* means (0.7657 − 0.7564) while the 2e-3 value came from the *unrounded* ones (0.7867 −
0.7764 = 1.038). So a reader subtracting the table's own cells matched three rows and missed one.
Fixed by choosing a convention and stating it: the differences are now of the accuracies **as
printed**, and Table 2's caption says so and says what the alternative would move. Nine places
across the paper, the W2 README (three languages) and the record index. ⚠️ **Two files were
deliberately not touched**: `paper/README.md`'s quotation of the old delivered text (that is a
historical record of an edit) and `w2_lif_lr020_prereg.md` (a pre-registration is a snapshot of
what was written before the data — editing it would be the one thing this project cannot do).

**② W3's "27% selection effect" had no arithmetic behind it.** The claim is that the tuning batch's
median (319.8) falls to the acceptance batch's (237.8), "about 27%". It is 25.6% — and neither
alternative reading (relative to the acceptance batch: 34.5%; as a ratio: 74.4%) gives 27 either.
So it was a derived percentage that no computation produced, inherited from the record into the
paper. Corrected to 26% in eight places: the paper, the R-STDP README (three languages), the
reproducibility checklist (three languages), and the record index. ⚠️ The *other* "18–27%" in this
repository is ESPP's self-reported fraction of updated time steps and is unrelated — left alone.

**③ Citations: 22 of 22 verified against the ledger**, mechanically, on (first author, year).

⚠️ **Both of my checking scripts lied first.** The number checker reported four false misses because
the LaTeX writes thousands as `1{,}688.6`; the citation checker reported four false misses because
`Fr{\'e}maux`, `Korcs{\'a}k-Gorzo` and `{BEKO2210}` are not the strings the ledger uses. A
verification tool that has not been verified is not evidence — and this is the second time in this
project that the checker, not the checked, was what needed fixing.

**④ The venue check changed the plan.** See the release plan above: the target's limit is **6 pages
including references**, not the ~10 assumed earlier, which makes the compressed version a rewrite
rather than a trim, and the submission **double-blind**, which the current front matter and both
DOIs are not.

## Draft log — citation verification and the author name (2026-10-17)

**Citations: all 22 verified against the ledger by title *and* year**, mechanically — a step past
the earlier author-and-year check. One metadata action came out of it: `pes2025tp` is **published**
(*Neuromorphic Computing and Engineering* 6(1):014002, 2026), which ledger entry [2] had flagged as
"出处不完整". Its note now carries the journal version and DOI, while the arXiv version stays the
cited one — that is the version whose table this paper's numbers were checked against.

**The ledger's status column, read against what this paper actually cites**: `pes2025tp` is
`metadata-error` (the dataset name — the paper already says N-MNIST, which is the corrected
reading); `pogodin2020kernelized` is `partial` (an over-strong phrase in the *plan*, not in this
paper); `membrain2026` is `metadata-error` about the author field (the entry already uses the GitHub
handle). The remaining flagged entries are not cited here.

⚠️ **The checker lied twice more before it worked** — and both times it was the checker, not the
bib. First it reported 11 misses because normalising punctuation to spaces produces *double* spaces
while the needle was single-spaced; then, fixing that line with `perl -pe`, the `\s` in the
replacement lost its backslash and the regex became "delete every letter s", which made the misses
worse. **That is the sixth shell-quoting failure of this session**, in the same family as ㉜, and it
happened *while* fixing a checker. The rule is in the plan and in memory; what keeps failing is
reaching for `sed`/`perl` instead of the editor.

**Author name: `lzmd-arch` → `Jinyao Shen`.** The paper now carries a real name rather than the
handle, which also resolves the arXiv pseudonym question from the release plan in the direction of
the real-name policy — and introduces a new requirement: **the double-blind submission must strip
it**, along with the repository URL and both DOIs.

⚠️ **A first attempt to typeset the Chinese form 沈锦垚 did not survive verification.**
Adding `\usepackage{CJKutf8}` and the characters around the name fails on this machine ---
**TinyTeX-1 does not ship `cjkutf8.sty`** --- so the local build stopped at a missing package. CI's
`tectonic` fetches the package on demand and went green, so the characters *were* in one build.
Whether they *rendered* could not be checked from here: CJK bitmap glyphs carry no ToUnicode
mapping, so extraction returns nothing for them, and this machine has no PDF renderer
(`pdftoppm`, `pdffonts`, `mutool`, `qpdf` all absent). The page-1 check that would have settled it
never happened.

**Final decision (2026-10-17): the byline carries `Jinyao Shen` alone.** The Chinese form is not
needed for the English submission; it belongs to a Chinese-language edition of the PDF, when one is
made. `\usepackage{CJKutf8}` is gone with it, and so is the dependency it introduced: **the local
build channel works again** --- verified this round with `pdflatex → bibtex → pdflatex ×2` in a
scratch copy outside the repository, 20 pages, zero overfull boxes. The paper is not CI-only any
more, and the unanswered glyph question goes away with the glyphs.

⚠️ **A third name lives in this repository**: `CITATION.cff` lists `name: "离中梦蝶"` with
`alias: lzmd-arch`. The paper and the citation metadata now disagree. Aligning them is a decision
about the repository's public identity, not a paper edit --- flagged, not done.

**Resolution on the identity question:** the real name goes in the paper only --- `CITATION.cff`,
the Zenodo creator field, and every other identity field stay as they are. The byline is
`Jinyao Shen`, and the double-blind submission will strip it along with the repository URL and both
DOIs. The lesson from the glyph episode is the same one the overfull boxes taught, and it is kept
even though the glyphs are gone: **compiling is not rendering.**

## Draft log — reading a stale copy, and the one defect that was real (2026-10-17)

A report came back with four "must-fix" items (Table 3 crammed together with an `offcial` header;
the abstract's last sentence cut off; an empty correspondence footnote) and three more (garbled
characters near Figure 2; a family of sentences truncated at page breaks; a truncated last sentence
in Data Availability). Every one was checked against the source before anything was edited.

**None of the seven reproduces in the committed source, and none reproduces in the PDF built from
it.** Concretely, in `main.tex`:

| report | what the source says |
| --- | --- |
| `offcial` in the Table 3 header | the string occurs **zero** times; all four occurrences are `official` |
| abstract's last sentence cut off | ends `...solutions a benchmark of this kind invites.` |
| empty correspondence footnote | reads `Correspondence: github.com/lzmd-arch/BioSNN-Plug.` |
| "V3(R-S1DP+ID-LIP" near Figure 2 | no such string; it is `W3 (R-STDP + TD-LTP)` degraded character by character |
| `...and this paper is about the second` stopping there | the sentence continues; see below |
| `...and for nothing measured` stopping there | the sentence ends there by design; nothing follows it |
| Data Availability's last sentence | ends `...Cite that record for the protocol, and this paper for the result.` |

The degradation pattern is the tell: `W3`→`V3`, `ST`→`S1`, `L`→`I` is what a bad rasteriser or a
cached OCR layer does, not what a LaTeX build does. Combined with a PDF that was locked by another
program earlier in the day, the reading is that the report was made against an older copy still open
in a viewer.

**Two changes make that harder to repeat.** First, `\date` now prints `draft revision 7
(2026-10-17)` on page 1, so a screenshot or a quote identifies its own build. Second, the only real
defect found by the check — and it came from the CI log, not from the report — was fixed: tectonic
reported `Overfull \hbox (1.54933pt too wide)` in the Data Availability paragraph, invisible to the
local pdflatex run because the two engines use different font metrics. The paragraph is now inside
`sloppypar`. **The current build reports zero overfull boxes**; the local build being clean was
never evidence about the CI build, and that distinction is now written down.

**One sentence did need changing, and it was not truncated.** §1.1 closes with `What the promise
does not come with is a measurement convention, and this paper is about the second thing.` The
nearest preceding `the second` enumerates the second cost of BPTT — the surrogate gradient — so the
sentence read as "this paper is about the surrogate gradient", the opposite of its subject. There
was no two-item list for "the second thing" to point at. It now reads `...is a measurement
convention. This paper is about that gap.` — and `gap` hands directly to the next subsection,
*The measurement gap*.

The general lesson, recorded because it has now cost two rounds: **text extracted from a PDF
truncates at page breaks and degrades at glyph level, so a reported truncation is a hypothesis about
the source, not evidence about it.** `grep` the source first; fix only what the source shows.

**Build of record:** commit `508c6b3` (the last commit to touch `main.tex`), 20 pages, tectonic
reports **zero** overfull boxes, sha256 `0b152e8987…`. The hash identifies *this* build; it is not
a reproducibility claim --- the PDF embeds a creation date, so a rebuild will differ byte for byte.

## Draft log — record B published, and where the "broken PDF" reports come from (2026-10-17)

**The paper is deposited.** Record B is live at `10.5281/zenodo.23429028` (concept
`10.5281/zenodo.23429027`): `publication / preprint`, CC-BY-4.0, version `v1.0`, one attachment,
20 pages, 658,248 bytes, `md5:7a36024d…`, built from `508c6b3`. The record was created as a private
draft, the file was uploaded and read back **from Zenodo's side** (size and checksum matched the
local PDF before the publish button was pressed), and the published record was then re-read from
the **public** API — title, type, licence, version, creator, related identifiers and file checksum
all as intended. The deposit package lives outside the repository, at
`Desktop/zenodo_v1.0/` (`zenodo_metadata.json`, `zenodo_deposit.ps1`, `README-上传.md`), following
the same draft-first pattern as record A.

**The three "must-fix" items were not defects, and this time the cause is identified.** A third
report said Table 3 was still crammed with an `offcial` header, the abstract still ended at
"…a benchmark of this kind", and a garbled line `V3(R-S1DP+ID-LIP` still sat before Figure 2. Every
PDF copy on the machine was scanned — the Desktop file and the four older copies in the WeChat file
folder — and **`offcial` occurs zero times in all five, as does `V3(`**; the abstract's closing
phrase `kind invites` occurs once in each. What produces exactly those three symptoms is a viewer
that **rasterises the page and re-recognises the text**: `official` loses its `i`, `W3 (R-STDP +
TD-LTP)` comes back as `V3(R-S1DP+ID-LIP`, two table rows collapse into one cell, and a sentence at
the page foot is cut where the raster ends. That is the WeChat PDF preview, and the files reached
it because they sit in WeChat's received-file folder. **Read the paper in a real PDF reader**; the
`draft revision N` stamp on page 1 now identifies the build in any screenshot that is quoted back.

**One thing is unfinished, and it is blocked by this environment.** A design check against the
A/B rules turned up three statements on record A that are now false or missing, all on the A side:
its **description** and its **notes** both say the preprint "is planned for 2026-10-22", and it has
**no related-identifier pointing at B** — the design calls for the two records to point at each
other, not one way. A fourth item is cosmetic but visible: A's **title still says "records v0.1"**
while its `version` field says `v0.2` (the title was not revisited when the new version was cut).
The fix — a metadata-only edit (unlock → PUT → re-publish, with recovery on every failure path so
the record cannot be left invisible) — is written and staged at
`Desktop/zenodo_v0.1/zenodo_link_to_B.ps1`; it corrects the description, the notes and the related
identifier, and deliberately leaves the title alone, because retitling a published record is a
different decision. **The edit was refused by this environment's permission classifier** as an
external-system write. It needs one manual run:

```powershell
powershell -ExecutionPolicy Bypass -File "$env:USERPROFILE\Desktop\zenodo_v0.1\zenodo_link_to_B.ps1"
```

Until that runs, the two records do not yet read as siblings: B points at A, A does not point at B.

**And one item the check raised that is not an edit but a decision.** The paper's Data and code
availability section cites A's **v0.1 version DOI** (`10.5281/zenodo.23388709`), which was current
when the paragraph was written; A is now at v0.2 (`10.5281/zenodo.23428426`), and the numbers this
paper reports live in the v0.2 attachments. The sentence is not false — it names v0.1 as a dated
snapshot whose metadata records the supersession, which is exactly what v0.1 is — but a reader who
follows it lands on the older index first. Whether to point the paper at the concept DOI alone (or
at the latest version) is a judgement about what a citation should resolve to, and changing it means
a v1.1 of the deposit. Flagged, not changed.

# paper/ — P2 benchmark paper (LaTeX)

Source for the **P2 paper**: an honest-measurement benchmark of local learning rules
(e-prop / Hebbian / R-STDP) on consumer hardware. Every number in it must trace back to a
reproduction record under `research/`, per the project plan's §12.4 rule.

Build: `latexmk -pdf main.tex` (or `pdflatex main && bibtex main && pdflatex main && pdflatex main`).
The resulting `main.pdf` is a **local build artifact and is not committed** — see below.

Status: **skeleton v0.2 + first body text** — §2.1 and §2.2 of Related Work are drafted
(see the draft log at the end); the rest is still structure and section plan. Outstanding
work is marked in-file with `\todo{}` (rendered red), so `grep -c '\todo{' main.tex` counts
the open items — **6**, unchanged by the Related Work draft.

## Compile status

**`main.pdf` is a local build artifact, not committed.** Run `latexmk -pdf main.tex` and it
appears in this directory; both it and the intermediates (`*.aux`, `*.log`, `*.out`, `*.blg`,
`*.bbl`) are gitignored. The reason it is not committed: a PDF **embeds a creation
timestamp**, so its hash changes on every rebuild even when the source is untouched —
committing it would add a binary diff to every documentation-only change.

**Compilability is guaranteed by CI instead**: the `latex-build` job runs `tectonic main.tex`
on every push. Tectonic pulls packages on demand and runs BibTeX itself, so the job also
checks the syntax of all 15 `refs.bib` entries. Watch that job, not a committed PDF.

Last verified locally 2026-10-15 (TinyTeX + pdflatex), after the Related Work draft:
`pdflatex → bibtex → pdflatex ×2`, no errors and no warnings, **5 pages**. `bibtex` now
resolves **9 of the 15** entries — it no longer prints `I found no \citation commands`,
which it did while the body was still citation-free.

Two things in the skeleton **looked** like they would fail to compile. **Both were wrong** —
asserted from memory before any LaTeX was installed, and the real build disproved both.
Recorded here as a correction rather than quietly deleted:

- line 6, `\title{...\\A Reproducibility-First ...}` — `\\` followed by `A` is fine:
  `\\` is a control symbol and does not need a space before the following text.
- line 98, `\bibliographystyle{plainnat}` — `natbib` ships with TinyTeX-1, and
  `plainnat.bst` resolves without an explicit `\usepackage{natbib}`.

## Known gaps

1. **`refs.bib` now has all 15 entries.** Three keys had zero hits in this repo's ledgers and
   were resolved by verification, recorded in `docs/references.md` [24][25][26]:
   - `miconi2018differentiable` — ICML 2018, PMLR 80, pp. 3559–3568;
   - `zanardi2024metaplasticity` — Phys. Rev. E 110(5), 054304;
   - `turcu2025switching` — NeurIPS 2025 **UniReps workshop**.

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
  than done here.
- **Zanardi et al.: the paragraph in §2.2 rests on the abstract** (arXiv:2403.13967) — a biased
  random walk on a cyclic feed-forward network with several levels, a Hebbian term plus a
  metaplastic reinforcement term on grouped edges, three regimes, and retrieval of a stored path
  after weights are reset to zero. The ledger's [25] entry previously recorded metadata only,
  which is not enough to write a sentence about what the paper shows.

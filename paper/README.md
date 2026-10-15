# paper/ — P2 benchmark paper (LaTeX)

Source for the **P2 paper**: an honest-measurement benchmark of local learning rules
(e-prop / Hebbian / R-STDP) on consumer hardware. Every number in it must trace back to a
reproduction record under `research/`, per the project plan's §12.4 rule.

Build: `latexmk -pdf main.tex` (or `pdflatex main && bibtex main && pdflatex main && pdflatex main`).
The resulting `main.pdf` is a **local build artifact and is not committed** — see below.

Status: **skeleton v0.2** — structure and section plan only. Outstanding work is marked
in-file with `\todo{}` (rendered red), so `grep -c '\todo{' main.tex` counts the open items.

## Compile status

**`main.pdf` is a local build artifact, not committed.** Run `latexmk -pdf main.tex` and it
appears in this directory; both it and the intermediates (`*.aux`, `*.log`, `*.out`, `*.blg`,
`*.bbl`) are gitignored. The reason it is not committed: a PDF **embeds a creation
timestamp**, so its hash changes on every rebuild even when the source is untouched —
committing it would add a binary diff to every documentation-only change.

**Compilability is guaranteed by CI instead**: the `latex-build` job runs `tectonic main.tex`
on every push. Tectonic pulls packages on demand and runs BibTeX itself, so the job also
checks the syntax of all 15 `refs.bib` entries. Watch that job, not a committed PDF.

Last verified locally 2026-10-15 (TinyTeX + pdflatex): clean, **3 pages**. `bibtex` prints
`I found no \citation commands` — **expected at skeleton stage**, since the body still has no
`\cite{}` (the single grep hit sits inside a comment).

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
3. **`main.tex` has three edits against the delivered file.** Two are comment-only; the third is
   **body text** and is called out as such below.
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

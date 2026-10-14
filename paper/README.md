# paper/ — P2 benchmark paper (LaTeX)

Source for the **P2 paper**: an honest-measurement benchmark of local learning rules
(e-prop / Hebbian / R-STDP) on consumer hardware. Every number in it must trace back to a
reproduction record under `research/`, per the project plan's §12.4 rule.

Build: `latexmk -pdf main.tex` (or `pdflatex main && bibtex main && pdflatex main && pdflatex main`).

Status: **skeleton v0.2** — structure and section plan only. Outstanding work is marked
in-file with `\todo{}` (rendered red), so `grep -c '\todo{' main.tex` counts the open items.

## Compile status (verified 2026-10-15)

The full pipeline runs clean and produces `main.pdf` (**3 pages**):

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`main.pdf` is committed; the intermediates (`*.aux`, `*.log`, `*.out`, `*.blg`, `*.bbl`)
are gitignored. `bibtex` prints `I found no \citation commands` — **expected at skeleton
stage**, since the body still has no `\cite{}` (the single grep hit sits inside a comment).
It still parses `refs.bib`, so the run doubles as a bibliography syntax check.

Two things in the skeleton **looked** like they would fail to compile. **Both were wrong** —
asserted from memory before any LaTeX was installed, and the real build disproved both.
Recorded here as a correction rather than quietly deleted:

- line 6, `\title{...\\A Reproducibility-First ...}` — `\\` followed by `A` is fine:
  `\\` is a control symbol and does not need a space before the following text.
- line 98, `\bibliographystyle{plainnat}` — `natbib` ships with TinyTeX-1, and
  `plainnat.bst` resolves without an explicit `\usepackage{natbib}`.

## Known gaps

1. **`refs.bib` has 14 entries; the task list named 15.** Three keys had zero hits in this
   repo's ledgers. Two were verified online and added, with the verification recorded in
   `docs/references.md` [24][25]:
   - `miconi2018differentiable` — ICML 2018, PMLR 80, pp. 3559–3568;
   - `zanardi2024metaplasticity` — Phys. Rev. E 110(5), 054304.

   The third, `neurips2025switching`, is **still missing its author list**: title and venue
   are confirmed ("Learning using switching synaptic plasticity rules", NeurIPS 2025), but
   the page renders authors with JS and the bioRxiv preprint was unreachable. **The authors
   are not guessed** — an incomplete entry stays out rather than carry a plausible-looking
   author list.
2. **`hajizada2026clp` is dated 2026** here, matching the task list and `docs/references.md`
   [8]. Note the project plan's prose says 2025; the two disagree.
3. **`main.tex` is committed byte-for-byte as delivered** and has not been edited — the two
   items above were checked, not changed.

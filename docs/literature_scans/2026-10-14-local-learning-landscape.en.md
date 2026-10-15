# Literature scan, 2026-10-14: the local-learning landscape

[中文](2026-10-14-local-learning-landscape.md) · [日本語](2026-10-14-local-learning-landscape.ja.md)

> This directory holds literature scans, one file per scan, named by date. Each record answers
> three things: **what question this scan was meant to settle**, **which entries it hit**, and
> **what it did not hit, plus the boundary of the record itself**. It is not a review, and it
> does not repeat bibliographic data already in the ledgers — those live in
> [`docs/references.md`](../references.en.md) and
> [`docs/related_projects.md`](../related_projects.en.md). What is here is the narrative and
> the pointers.

## 1. What this scan was meant to settle

It supports the sentence in `paper/main.tex` §2.4:

> To our knowledge no equivalent protocol-level standard exists for local-rule SNN training.

The claim is **defensive** ("to our knowledge"). What it asserts is not that nobody has
discussed reproducibility, but that **no protocol-level reporting standard exists** — that is,
nothing satisfies all three of the following at once:

1. it is a **standard** (a checklist, a policy, a mandatory submission item), not one paper's
   own practice;
2. it targets **local-rule SNN training** (relaxed: SNN training);
3. it prescribes **what a report must contain** (budget, seed count, optimizer, evaluation
   protocol, ...).

Missing any one of the three means it is not a counterexample to that sentence. The criteria
are written down here so that a later reader can **refute** the claim, rather than only seeing
a conclusion.

## 2. Entries the scan hit, and what each one covers

The "No." column points at the ledger numbering in
[`docs/references.md`](../references.en.md); the last column is the three criteria applied.

| No. | Entry | What it covers | Why it is not a protocol-level standard |
| :--- | :--- | :--- | :--- |
| [1] | Bellec et al. 2020 (e-prop) | The rule itself, and its derivation for ALIF | Self-reported results; it does not tell anyone else how to report |
| [2] | Pes et al. 2025 (TP) | Per-neuron traces, memory order | A whole block of its SHD comparison is **transcribed from a third party** (the paper footnotes it) — an instance of inconsistent protocol, not a standard |
| [17] | Korcsák-Gorzo et al. 2025 | Event-driven updates, scale on NEST | Engineering and scale; the axis is efficiency, not reporting practice |
| [18] | Millidge 2025 | Trace recursion in the depth direction | A pure math note with no experiments; it produces no numbers of its own |
| [20] | Graf et al. 2024 (ESPP) | Selective time-step updates, 84.32% on SHD | Single configuration, self-reported, no independent reproduction |
| [21] | eprop-PyTorch | A third-party reimplementation | No dependency list, no CI, no tests — a cautionary example of reproducibility tooling, not a standard |
| [22] | Bellec et al. 2018 (LSNN) | The same-family baseline on sMNIST | Its numbers were already distorted in third-party transcription (mean 93.8 rendered as 93.7, max placed beside the mean) |
| [27] | Pineau et al. 2021 | **A protocol-level practice in ML** (code submission policy, reproducibility challenge, mandatory submission checklist) | It is the **positive** case showing such a thing can exist — but it targets ML as a whole; SNN training is not covered, local rules less so |
| [28] | Wang et al. 2026 (BrainTrace) | System-level linear-memory online learning | A toolchain contribution; the axes are memory, throughput and scale |

The same sweep covered the three sibling projects in
[`docs/related_projects.md`](../related_projects.en.md) (EMBER, MEMBRAIN, Javis): all three
rest on self-reported numbers with no third-party reproduction, and none of them is a standard.

## 3. What the scan did not hit, and the boundary of this record

**What it did not hit:** none of the entries above is a mandatory submission checklist or a
reporting policy, whether for SNN training or for local-rule training specifically. That
absence is precisely the gap §2.4 points at.

Warning: **the boundary of this record itself — read before citing it.** **The 2026-10-14
scan's own queries, sources, and hit/miss counts are in no file in this repository** (checked
across the whole repo and the desktop on 2026-10-15). The two sections above are a narrative
and a set of pointers **compiled** from ledger entries on 2026-10-15; they are **not a replay
of that scan's procedure**. So this record supports "here is what we looked at and what each
thing is"; it does **not** support "we exhaustively swept the field with a reproducible
query".

Two hard constraints follow: the wording in §2.4 must stay defensive and must **not** be
escalated into an assertion, and the note on ledger entry [27] says the same thing — do not
change one without the other.

**Two ways to close it:** add that scan's query record to this file, or run a **logged** scan
again and record it as its own file under this file's date convention (one file per scan).

**Minimum spec for the next scan** (so the second one is not just another conclusion): record
verbatim the sources and query strings, the date, the hit count per source, the inclusion and
exclusion criteria, and how a "miss" was decided — what was searched, what came back, and why
that did not count as a hit.

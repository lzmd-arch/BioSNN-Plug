# ADR-0010: e-prop's quadratic storage holds, but §3.1's remedy (adopting Trace Propagation) is not adopted

[中文](ADR-0010-eprop-quadratic-storage-and-trace-propagation.md) · [日本語](ADR-0010-eprop-quadratic-storage-and-trace-propagation.ja.md)

- **Status**: accepted
- **Date**: 2026-09-23
- **Related**: project plan §3.1 key correction 1, §3, §8 selection table, §11 risk table;
  `research/eprop/`; [ADR-0007](ADR-0007-td-ltp-critic-provenance.en.md) (also a provenance
  investigation); [docs/references.md](../references.en.md) item 2

## Context

The project plan's §3.1 "key correction 1" (`BioSNN-Plug_项目计划书_v6.2.md:161-163`) says two things:

> e-prop 的资格痕迹按突触存储，空间复杂度随神经元数量**二次增长** [Pes et al., 2025].
> Traces Propagation（TP）是一种前向、内存高效、可扩展的完全局部学习规则，结合资格痕迹与
> 逐层对比损失，**无需辅助逐层矩阵**……

and §8's selection table turns that into a direct replacement: "e-prop 存储优化 | Trace Propagation |
**资格痕迹存储从 O(N²) 降至 O(N)**" — the same assertion also appears at §3 lines 104/159, the
architecture diagram (line 86), line 363, line 604, and the risk table lines 496/516.

`docs/references.md` item 2 previously recorded three findings, the third of which said: **"the paper
attributes that complexity specifically to OSTTP/OSTL, and does NOT give e-prop's space complexity
separately — the plan attributing it to e-prop specifically is an extrapolation (引申)."**

This ADR settles two things: **whether that finding is itself correct**, and **whether §3.1 should be
implemented**.

## Decision

### One: the third finding is wrong and is retracted

Table 3 was parsed **cell by cell** from the LaTeXML HTML of arXiv:2509.13053v2 (not from the PDF's
text layer, which is misaligned for tables), cross-checked against v1 and the PDF:

| Model | Update Locking | Weight Transport | Time Local | Space Local | **Space** | Time | Aux |
| :--- | :-: | :-: | :-: | :-: | :--- | :--- | :--- |
| BPTT | ✗ | ✗ | ✗ | ✗ | TLH | TLH² | − |
| **E-prop [2]** | ✗ | ✗ | ✓ | ✗ | **LH²** | **LH²** | **−** |
| E-prop(rnd) [2] | ✗ | ✓ | ✓ | ✗ | **LH²** | LOH | LOH |
| OSTL [8] | ✗ | ✗ | ✓ | ✗ | LH² | LH² | − |
| OSTTP [11] | ✓ | ✓ | ✓ | ✓ | LH² | LOH | LOH |
| TESS [13] | ✓ | ✓ | ✓ | ✓ | LH | LOH | LOH |
| TP (ours) | ✓ | ✓ | ✓ | ✓ | LH | LH | OH |

§1.3.1 goes further and **names** e-prop: it first gives the generic trace
`ϵ_l^t[i,j] = βϵ_l^{t-1}[i,j] + g(s) f(s)` (Eq. 9, indexed by the synaptic pair `[i,j]`), then writes

> For instance, the eligibility trace of **E-prop [2]** defines the presynaptic factor as a
> low-pass filtered version of the spiking activity … and the postsynaptic factor as the surrogate
> derivative of the spike function …

and then the family property: "eligibility traces are stored per synapses, leading to a space
complexity that scales quadratically with the number of neurons, i.e., 𝒪(H²L)". **e-prop is the first
instance named in that passage, so "these solutions" covers it literally.**

**So §3.1's premise holds and is a direct statement of the paper's** ("quadratic in neurons, linear in
layers" matches `LH²` exactly). The repo's earlier finding was **true of the prose and false of the
table** — v2's §2.3 names only OSTTP (v1's sentence is more explicit: it says ETLP and OSTTP are "based
on **E-prop [2]** and OSTL [8]"), so reading only the prose produces the wrong verdict. The retraction
is described under Consequences.

### Two: the "adopt TP" remedy is **not** adopted — three hard pieces of evidence

1. **TP is not a memory-saving version of e-prop; it is a different rule that replaces its spatial
   credit assignment.** The paper lists the two as **separate rows on separate axes** (e-prop: time-local
   ✓ / space-local ✗ / LH² / no auxiliary matrix; TP: all ✓ / LH / LH / OH) and never says TP optimises
   e-prop. TP uses **two per-neuron** activity traces (Eq. 11/12, indexed `(batch, neuron)`; the
   recursions contain no ψ_j, no ε_v/ε_a, no e_ij) plus a one-hot target pathway, a batch×batch
   contrastive loss (Eq. 13-15), and an update (Eq. 18) whose surrogate derivative is evaluated at the
   **current** instant. **Adopting it means replacing W2's learner**, not optimising its memory.
2. **It does not cover W2's neuron model.** Every TP result in Table 1/2 is **LIF** (the two SHD rows at
   400/450 are LIF too; the ALIF row belongs to ETLP). **W2's acceptance configuration is ALIF (β=0.07).**
3. **On the paper's own headline dataset e-prop beats TP.** N-MNIST: `eProp [2] … 97.90` versus
   `TP (ours) … 97.33 ± 0.06`. And the abstract's "outperforms other fully local learning rules"
   **does not include e-prop** — e-prop is marked `Partial (time)` local in Table 3, so it is outside the
   "fully local" comparison set. **A reader taking that sentence to mean "the rule this project chose has
   been shown to be worse than TP" has it backwards.**

(One aside, **not** used as a reason: TP needs batch ≥ 2 for its contrastive loss and cannot update online
per sample, and the paper's own Eq. 25 shows its memory advantage over TESS only holds when `O > B`,
whereas sMNIST is O=10, B=64. That argument uses a third party as the target, and TP's per-neuron traces
are only about 0.125 MiB at this operating point. **The three reasons above are the real ones.**)

## Consequences

- **`docs/references.md` item 2's third finding must be rewritten** (tri-lingual): from "an
  extrapolation" to "the premise holds; the paper gives it both in Table 3's `E-prop [2]` row and by
  naming e-prop in §1.3.1; the prose (v2 §2.3) names only OSTTP, which is easy to misread — that is
  probably how the repo got it wrong". One wording elsewhere also needs tightening: the paper does
  **not** never use the token `MNIST` (it appears 3 times: the abstract's `NMNIST` and §3.1.1 explaining
  where N-MNIST comes from); the accurate statement is that **it never uses MNIST as an evaluation
  dataset**.
- **`research/eprop/README.md`'s boundary item 1 must be rewritten** (tri-lingual): from "Trace
  Propagation has not been implemented" (which reads as a debt) to "on investigation, §3.1's remedy
  **does not apply to this line**, for three reasons", plus a note that **the one thing this
  implementation actually should do has been done** (below).
- **What this implementation actually did**: in `traces.py`, `epsilon_v` was allocated as
  `(batch, n_pre, n_post)` although its recursion **contains no postsynaptic index j** — it is the same
  value for all j at a given instant. Allocating `(batch, n_pre, 1)` instead is **bit-identical**
  (`torch.equal=True`, `max|diff|=0`) and gives **−18.2%** wall clock at the acceptance shape.
  `epsilon_a`'s coefficient contains `(ρ − β·ψ_j)` and **does not factorise**, so the bulk of this
  section's memory is set by the ALIF adaptation term; for LIF (β=0) both degenerate to outer products
  and the whole trace state drops to O(batch·N). That boundary is written into the code comment.
- **`research/rstdp/README.md`'s boundaries** (tri-lingual) also lists TP as "not implemented this
  phase" — it must be changed at the same time, or the repo will describe the same thing two ways in two
  lines.

## Alternatives

- **Implement TP as §3.1 asks**: **rejected**, for the reasons in Decision Two. If it is ever built it
  should be its own line — **another learning rule** (replacing e-prop's learner), not "a memory
  optimisation of e-prop" — and it would first require moving W2's neuron model from ALIF to LIF, or
  deriving an ALIF version of TP, which the paper does not provide.
- **Only fix the prose-versus-table wording and leave the implementation alone**: **rejected** — because
  this implementation genuinely has one wasted O(N²) buffer (`epsilon_v`), and fixing it is risk-free
  (bit-identical, −18%). Fixing only the docs would leave a free improvement on the table.
- **Fall back to "semi-local" to buy smaller storage**: **not applicable** — that is §11's degradation
  path and has nothing to do with this storage question.

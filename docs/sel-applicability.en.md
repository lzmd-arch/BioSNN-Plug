# Does SEL apply to recall and gating?

[中文](sel-applicability.md) · [English](sel-applicability.en.md) · [日本語](sel-applicability.ja.md)

**Date**: 2026-11-03
**Answer**: **No.** SEL (spiking expression layer) solves "style in sequence generation";
recall is ranking and gating is binary classification. The only shared part is the spiking
neuron, and that part already has three consistently negative results on these two lines.
SEL is worth doing, but it must not be used to justify recall or gating.

## One-paragraph version

- **Recall**: queries and memories are static sets; the time steps are artificial. A firing
  rate is a quantised dot product — it can only lose information. 512 ALIF neurons reach
  hit@1 0.024 (lexical 0.259); with TI-LIF it is **0.012**.
- **Gating**: only three static features, and both kNN and a single threshold reach AUROC
  **1.000**; any deterministic map of those three features adds no mutual information (data
  processing inequality). The 512 population reaches 0.41 on gold; the lexical single
  threshold reaches 0.825.
- **What transfers is method**: preregistration, baselines, held-out sets, negative-result
  registration, two-sided criteria. Not the architecture.

## Recall

**Why the mechanism does not apply.** Recall scores and ranks a candidate set: given a query
and a memory, the answer is in the top few. SEL's temporal structure, cross-token state and
spiking attention all assume "the output is a sequence"; recall has no sequence to generate.
Encoding a query into a spiking current, running T steps and reading a firing rate is a
thresholded, refractory, quantised nonlinearity applied to the same dot product — information
can only decrease.

**Evidence.**

| Experiment | Result |
| :--- | :--- |
| `research/spiking_512` (512 ALIF, gold) | hit@1 **0.024** vs lexical **0.259**; hit@5 0.094 vs 0.424 |
| Same, TI-LIF D=1/D=2 (2026-11-03, 10 seeds) | hit@1 **0.012**; hit@5 0.047 |
| Same, ALIF + K-window averaging (5 independent projections) | hit@1 **0.035** (+2.3pp; positive control holds) |
| Same, TI-LIF D=2 + K-window | hit@1 **0.012** (no gain from K-window) |
| Axo substrate completion (512 ALIF) | hit@1 0.000 vs linear 0.025; latency +40.7 ms |

The K-window row matters: it is the only readout improvement ever shown to work (∝K^0.6),
and it reproduces on ALIF (0.012 → 0.035), but TI-LIF gets no such gain. The problem is not
"binary vs signed spikes"; it is that a firing rate throws away the resolution ranking needs.

**Reopen when**: there is a genuinely streaming recall task (queries arrive over time, one
forward pass only, no candidate list to re-rank) and the linear/lexical baseline is not at
the floor (above 0.10).

## Gating

**Why the mechanism does not apply.** Gating asks "does this sentence have support in the
library?" over three scalars (`top_score` / `mean_topk` / `density`), and answers a binary
label. The kNN(5) density gate and a single `top_score` threshold both reach AUROC **1.000**
on this data. AUROC depends only on ranking; any deterministic map of these three features
(including spiking population coding) cannot add information — at best it moves the decision
boundary on finite samples, which is more likely overfitting.

**Evidence.**

| Experiment | Result |
| :--- | :--- |
| Axo metacognitive gate (217 real memories) | kNN **1.000** vs single threshold **1.000**; hard positives 0.996 vs 0.996; criterion 2 fails, not wired |
| `research/spiking_512` gating (gold) | 512 ALIF **0.535**; TI-LIF D=2 **0.410**; lexical top-1 cosine **0.825** |
| Product path | one threshold 0.20 (ADR-0028), calibrated on the real distribution |

**Reopen when**: there is a human-labelled set of near-domain hard OOD queries (single-threshold
AUROC < 0.95) and a rate-based baseline has first shown headroom. Without both, any "spiking
gate" is just another way to write the same threshold under a 1.000 ceiling.

## Relation to SEL

- SEL and recall/gating **share the neuron implementation** (`biosnn_bus.AdaptiveMembrane` /
  `TernaryMembrane`), not the task structure.
- These negatives **cannot** be used to dismiss SEL: SEL generates text and is evaluated on
  style and fact retention; its risks are data, scale, and teacher logits — not the word
  "spiking".
- Conversely, SEL's narrative **must not** be used to endorse recall/gating. Their
  preregistrations, baselines and reports stay separate; that is why this file exists.

## What transfers

1. **Write the criteria before the run** (S1–S3, A1/A2/B1/B2, T1–T5 were all frozen first);
2. **Report the positive control alongside the trivial solution** (the K-window +2.3pp here);
3. **Register negative results** (`research/spiking_512`, `benchmarks/substrate_spiking.md`);
4. **Two-sided criteria** (B1/B2: check whether spiking leads *and* whether it lags).

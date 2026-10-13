# ADR-0012: TAAF's locality red line, the projection-layer training rule (open), and the SpikeBus output-shape interface contract

[中文](ADR-0012-taaf-locality-and-bus-output-contract.md) · [日本語](ADR-0012-taaf-locality-and-bus-output-contract.ja.md)

- **Status**: partially accepted (locality and the output contract are accepted; the projection-layer training rule is **left open for phase two**)
- **Date**: 2026-10-13
- **Related**: project plan §2.2 (spike bus: TAAF + dual-channel fusion), §3.1, §7 phase two; [ADR-0002](ADR-0002-numpy-core-torch-optional.en.md) (skeleton library does not depend on torch), [ADR-0006](ADR-0006-plugin-interface-fidelity.en.md) (plugin interface fidelity)

## Context

Project plan §2.2 says the spike bus integrates "TAAF temporal-attention-guided adaptive fusion", and §7 phase two lists it as the task "TAAF quick fine-tuning". But the repository says otherwise: `FusionStrategy` has only one implementation, `ConcatenateFusion`, and `bus.py` says the `FusionStrategy` is where TAAF will **be inserted in future** -- **TAAF was never implemented**. The plan's "quick fine-tuning" phrasing assumes it already exists, so the wording and reality disagree.

Three constraints were made explicit at the start:

1. **Locality of the attention weights**: TAAF's time-attention weights must be driven by **locally computable signals** (e.g. each path's firing rate), and **must not be trained by backpropagation** -- otherwise the "purely local learning" claim is punctured right at the fusion layer;
2. **The projection layer's training rule must be nailed down in an ADR**: membrane layer with IB-Hebbian, or simple local normalization -- **even if undecided now, list the options**; this is a core phase-two question, not something to answer in passing;
3. **Write the `bus` output shape into the interface docs**: fused output = **unified core dimension** + **modality labels preserved**.

## Decision

### 1. The locality red line (accepted, hard constraint)

The fusion layer's attention weights are computed on the fly from each path's own firing rate (rate-based gating), implemented as `TemporalAttentionFusion.attention_weights`:

```text
rate_i = mean(|data_i|)          # each path's mean firing rate -- locally readable per neuron
w      = softmax(rate / tau)
out    = sum_i w_i * projected_i
```

**No backward, no trainable parameters, no torch.** `test_taaf.py::TestLocalityRedLine` pins this with an **AST structural assertion** (it inspects imports and calls, not source text by eye), because once this red line is bypassed, "purely local learning" stops holding.

**Why rate-based rather than Hebbian**: both satisfy the red line (both are locally computable), but rate-based is simplest and the weights can be recomputed from data immediately and asserted in tests. A Hebbian variant accumulates traces across time and is harder to check. **This is not an exclusive ruling** -- if later experiments show Hebbian weights do better, swap the implementation; the red line stays.

Warning: **this red line matters even more once the skeleton library becomes a product.** The product side will be tempted to "just learn the fusion weights". That trades performance for the loss of the "purely local learning" claim.

### 2. The output-shape interface contract (accepted, hard constraint)

What `SpikeBus.step()` returns:

| Field | Value |
| :--- | :--- |
| Shape | `(T, bus_dim)` -- the **second dimension is always** `bus_dim`, **regardless** of how many modalities are registered or what each one's `spike_dim` is |
| Channel | The sum across channels; the `channel` field is meaningless afterwards; per-modality channel information lives only in `describe()` (grouped by `modalities_by_channel`) |
| dtype | `float64`: input current from a sparse random projection with ±1 weights |

**The second dimension being `bus_dim` is a promise, not a coincidence**: a downstream cognitive core can size itself to `bus_dim` without knowing how many modalities feed it. This stays true when the fusion strategy changes -- the two work-points differ only in **how one path is produced**, not in the output dimension.

**Modality labels are preserved**: after fusion a single sequence has no bit that says "which modality is which", so labels do not live on `SpikeTrain.channel` (that field means "which fusion channel", not "which modality"). The right place is `describe()`'s `modalities_by_channel` -- **already implemented**; this ADR adds it to the interface docs.

### 3. The projection layer's training rule (**left open for phase two**)

This is a core phase-two question, and this ADR **does not decide it**; it registers the options. Three candidates:

| Option | Content | Cost | Note |
| :--- | :--- | :--- | :--- |
| **A. Fixed-seed random projection** (status quo) | `SparseRandomProjection`, ±1 weights, seed derived via `zlib.crc32` | Not learnable; distance preservation is Johnson-Lindenstrauss, but it is **not optimized for any task** | Current implementation. **No training, no backpropagation** -- consistent with the red line |
| **B. IB-Hebbian local learning** | Projection matrix updated online with kernelised IB-Hebbian (as in `research/ib_hebbian`) | Introduces learning rates, HSIC kernels, divisive normalization and more hyperparameters; **but still local** | Same lineage as the existing perception-layer choice |
| **C. Simple local normalization** | Fixed projection + per-neuron local normalization (divisive norm / membrane) | Lightest; but normalization is not learning, just stabilisation | The `Membrane` protocol and `PassThroughMembrane` placeholder already exist |

Warning: **all three candidates satisfy the locality red line** (none uses backpropagation), so this is not a red-line question but a **capacity** question: the fixed projection is least expressive, IB-Hebbian most expressive but with the most hyperparameters. Which one to pick depends on whether the cross-modal retrieval R@1 success criterion (§7 phase two) can be met.

Warning: **do not change `SparseRandomProjection` semantics before this is decided.** Its "±1 projection preserves distances" property underpins all three research lines in `research/`; re-verify it before changing it. This round's C3 change only **added** `_projection_for` for per-path-dimension projections -- it did **not** touch the projection matrix itself or its seed derivation.

## Consequences

**The two accepted constraints** (locality red line, output contract) bind every future fusion strategy: any new strategy must pass `test_taaf.py::TestLocalityRedLine`'s AST assertion, and its output's second dimension must be `bus_dim`.

**The open third constraint** is a real phase-two gap: the current projection matrix is random and **never optimised for a task**. If cross-modal retrieval R@1 50%+ (the §7 success criterion) cannot be reached, look here first.

**The project plan's wording needs to change**: §7 phase two's "TAAF quick fine-tuning" should become "**implement TAAF temporal-attention fusion** (rate-based gating, no backpropagation)" -- "quick fine-tuning" assumes TAAF already exists.

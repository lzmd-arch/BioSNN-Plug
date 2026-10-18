# Arbiter: cross-rule conflict detection and weight adjustment

[中文](README.md) · [日本語](README.ja.md)
**Warning: destructive-change grace period through 2027-09** (project plan section 12.5,
ADR-0005).

## What claim is being tested

Class **1** of the arbitration dimensions in plan section 3.4: Hebbian <-> e-prop
**conflict detection and weight adjustment**. (Classes 2-3 are the other rule pairs and
class 4 is the "handle it myself vs call the LLM" dispatch decision -- all later phases.)

Conflict judgement follows section 3.4 verbatim:

> Conflict judgement: compute the cosine similarity between the two rules' update
> directions on the same synaptic weight; an angle greater than 90 degrees is a conflict.

The acceptance criterion is the phase-two bar "the arbiter reduces dual-rule conflict
events by 50%+", and section 9's "three-rule coordination / conflict-event fraction < 20%".

## Current conclusion

**Conflict detection is done and the ES arbiter is done, but "conflict events drop 50%"
is trivially satisfied on synthetic data -- it tests nothing.**

1. **Per-synapse cosine is degenerate; only per-row cosine carries a real angle.** The
   "direction" of a single scalar is just its sign, so per-synapse cosine is **always
   +-1**, and "angle > 90 degrees" collapses to "opposite sign", making the threshold
   meaningless. For the angle to mean anything, the cosine has to be taken over **a
   group of synapses** (e.g. one presynaptic neuron's whole row of outgoing weights).
   Both readings are implemented (`axis=None` per synapse / `axis=1` per row; per row is
   the default) and **must not be mixed**.
   Warning: **`axis` follows numpy's convention**: on an `(n_pre, n_post)` update tensor
   `axis=1` reduces over `n_post` giving one value per **row** (a presynaptic neuron's
   outputs); `axis=0` reduces over `n_pre` giving one value per **column** (a
   postsynaptic neuron's inputs). Two different questions -- choose deliberately.

2. **Warning: the trivial solution is `alpha -> 0`, i.e. switching rule B off.** If the
   objective is only "minimise conflict with A", the optimum is **always not to use B**
   -- that is not arbitration, it is degeneration, and it **passes the 50% bar
   trivially**. On a synthetic trace (`B = -A + noise`, outright opposition) measured
   across 5 seeds: **100% reduction every time**, with `alpha` drifting from 0.5 to
   0.2-0.3. That is a **symptom, not a score**.
   So **"conflict events dropped 100%" is uninformative on its own**; report it together
   with `retention` (how much of B survives). `fit(..., min_retention=...)` closes the
   degenerate answer.
   Warning: this is the same class of problem as "a green test that verifies a trivial
   property" -- the `reduction >= 50%` assertion holds under the degenerate solution,
   so it tests nothing.

3. **ES parameterisation stays at one scalar.** Confavreux et al. found meta-learning
   **starts failing as co-activation rules grow more complex**, so the arbiter tunes only
   `alpha`, on purpose -- this is not an unfinished feature.

## How to reproduce it

```bash
uv run pytest research/arbiter -q                  # 32 tests
```

Two modules:

| Module | What it does |
| :--- | :--- |
| `conflict.py` | The conflict criterion itself: `cosine_similarity` / `is_conflict` / `measure_conflict`. **Knows nothing about Hebbian or e-prop** -- it takes two update tensors and answers one question |
| `es_arbiter.py` | The ES meta-learned arbiter: `blend` (mix the two rules' updates by `alpha`) + `fit` (tune `alpha` with ES). **Depends on no research line**, so it is testable in isolation |

## Ruling (2026-10-18): the synergy benchmark is P0; the shared interface is a counterfactual arm only

* **The shared-interface approach is rejected.** Wiring three rules into one interface, or
  projecting R-STDP's update onto `(784, 256)`, is a **semantic distortion** -- those updates
  describe different learning problems. Such constructions are allowed **only as a clearly
  labelled counterfactual arm**, never in a main conclusion.
* **Priority inverted.** P0 is a **synergy benchmark**: at the weight the two lines already share
  (the `784 -> 256` input mapping), **H alone must fail, E alone must fail, and the mix must
  succeed**. Specification:
  [`sweep_results/arbiter_synergy_task_spec.md`](../../sweep_results/arbiter_synergy_task_spec.md).
* **The e-prop <-> R-STDP counterfactual arm comes after** the synergy benchmark, and is designed
  once the benchmark has results.
* **Three-way blending and the scheduling dimension stay in phase five**; untouched here.
* Infrastructure kept: the 43 tests, per-row cosine, and joint reporting of `retention` and
  conflict **remain as they are**.

## Known boundaries

1. **Only class 1 (Hebbian <-> e-prop) is done.** Classes 2-3 (e-prop <-> R-STDP,
   Hebbian <-> R-STDP) and class 4 (dispatch decisions) are not -- the plan puts them in
   phase three.
2. **Wired into both research lines** (`shared_synapse.py`). This boundary used to read
   "nowhere do their updates reach the same arbiter"; now there is such a place: the two
   rules **share one input synapse matrix** (shared numbers, re-synced each step) and
   their aligned updates go to the arbiter. Warning: **only the input->hidden mapping is
   shared**; the two downstreams stay independent with their accepted widths (1024 and
   256). Warning: **storage conventions differ** -- `ib_hebbian` uses PyTorch's
   `(out, in)`, `eprop` uses `(in, out)` -- so sharing is **numeric**, not by pointer,
   maintained by copying each step. `align_update` rejects the wrong convention (comparing
   a matrix with its own transpose yields a meaningless cosine).
3. **`alpha` blends update tensors, not learning rates.** The alternative is adjusting
   each rule's learning rate, which is another reading of "weight adjustment"; this
   implementation takes the former (closer to section 3.4's wording) and **has not been
   argued to be better**.
4. **A conflict event counts per training step**, not per synapse: one conflicting
   synapse in a step is one event. That is the unit behind section 9's "conflict-event
   fraction"; for per-synapse counting use `ConflictReport.conflict_fraction`.

# Roadmap v7.0 candidates

[中文](roadmap_v7_candidates.md) · [日本語](roadmap_v7_candidates.ja.md)

**What this is**: a candidate list, not a schedule. Entry means "thought through, deliberately not
started". Every item states its **activation condition** and a **review date**; until the condition
holds it does not start. The plan proper (v6.2 authoritative, v6.3 with citation corrections) is
unaffected by this file.

---

## Candidate 1: PINN x spiking networks (grounding axis)

**Form ranking**: **B (PINN-regularised world model) > A (physical modality plugin) > C
(physics-gated predictive core)**. Why: B reuses the world model that
[ADR-0015](adr/ADR-0015-world-model-as-phase-four-component.en.md) already registered and adds one
constraint term; A touches only the perceptual layer; C would push physics into the learning rule
itself, which is the highest-risk option.

### Calibration (the half that does not hold)

- PINNs buy **data efficiency, extrapolation stability, and a usable memory checker** — not a
  better representation;
- the 2023--24 literature repeatedly reports training instability (loss weighting, stiffness,
  multi-scale behaviour);
- **the family has split**: neural operators (FNO / DeepONet), Hamiltonian/Lagrangian networks,
  differentiable physics simulators. **A PINN is not necessarily the answer**; it is the most
  famous branch, not the only one;
- the value of this entry is therefore not "use a PINN" but that **neuromorphic x physical priors
  is currently empty** — the known neuromorphic world models (the Hajizada line) carry neither
  physical constraints nor an interface between spiking dynamics and continuous physics.

### Activation condition

Phase-three acceptance passes **and** the continual-learning probe lands. Both are needed: without
the first the dynamics axis is not standing yet; without the second the world model has no
measurable across-time object.

### Three pre-registered questions (answer them first, then look at data)

1. **Locality**: can a physical constraint be injected under **purely local rules** (no
   backpropagation unfrozen) and still be measurable? If only backpropagation works, the answer is
   "do not do it", because it contradicts the project's central claim.
2. **Constraint form**: hard constraint, soft penalty, or conservation embedding — which
   extrapolates better **at matched budget**? "Structural fidelity beats photorealism" is a bet
   already placed on the grounding axis; this question tests it.
3. **Third factor**: can prediction error serve as a **local learning signal** without introducing
   a global gradient?

### Alternative-route disclaimer

If any of the three answers is "no", **the whole line may be taken over by alternatives**: neural
operators for extrapolation, Hamiltonian networks for conservation, differentiable physics for the
checker. **Not doing PINN does not mean abandoning the grounding axis** — it means switching branch.

### Review date

**Summer 2027.** The review answers exactly one question: has the activation condition been met?
If not, it slips; whether to start early is not a question for the review.

---

## Candidate 2: mixed memory (consolidation store x long context, two recall paths)

**Where it sits**: the crossing of the dynamics and grounding axes -- one set of memories, two
recall paths.

**The question**: on the same conversational task, which is stronger -- the model's own
long-context "memory" (paste the history into a 1M-token window) or this project's consolidation
store (cross-session, structured, auditable)? Under what conditions does each win?

**Three arms**: (1) context only; (2) consolidation store only (retrieval-injected); (3) both,
merged under a conflict rule.

**Prerequisite**: the consolidation daemon runs in the terminal but has **zero research
validation** (ADR-0016's inversion). This experiment shares a metric language with the
consolidation comparison and does not reuse its conclusion.

**Activation condition**: the consolidation comparison yields its first AF/recall data **and** the
terminal has at least one month of consolidation logs.

**Three pre-registered questions**: (1) where does the capacity-versus-latency trade-off flip?
(2) when the two paths disagree (one fact, two versions), how is the conflict adjudicated?
(3) is the gain **just "a bit more context"** -- an equal-token control is mandatory.

**Alternative-route disclaimer**: if context cost becomes negligible, the value of mixed memory
lies in **cross-session persistence and auditability**, not recall rate; the experiment should then
be rewritten rather than forced.

**Review date**: spring 2027.

---

## Candidate 3: DeepInfer (an observation item, not an experiment)

**What it is**: an inference framework open-sourced on 2026-10-20 (Apache-2.0), source grade C,
currently missing KV-cache and load balancing.

**Relation to this project**: it lives in the **terminal's** engine layer, not in the research
line -- this project does not run LLM inference, so it changes no experimental design.

**Observation condition**: KV-cache support appears, or there are signs of sustained maintenance
(two consecutive releases with fixes).

**Review date**: spring 2027, together with Candidate 2.

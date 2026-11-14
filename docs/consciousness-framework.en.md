# Qiansui consciousness framework: science, governance and narrative

[中文](consciousness-framework.md) · [日本語](consciousness-framework.ja.md)

- **Version**: v1.2 (landed in the repo 2026-11-12)
- **Nature**: a **living document**. It is not a research conclusion but a body of **discipline** —
  it constrains what we say about the word "consciousness", what we refuse to say, and what
  evidence is required to say anything at all.
- **Origin**: v1.0/v1.1 were compiled from four discussions on 2026-11-12 (compliance → physical-substrate
  scenarios → current verdict → future predictions). This version was reconciled against the state of
  the repository and turns the action list in its §7 into concrete artefacts.
- **Relation to ADRs**: [ADR-0036](adr/ADR-0036-consciousness-claims-discipline.en.md) holds the executable
  clauses (vocabulary discipline, claim ladder, two-milestone gates). **The ADR sets the rules; this
  document explains why.**

> **This document is itself subject to this document.** If any sentence here reads as a claim of
> phenomenal consciousness, it is a mistake, not rhetoric. See the first red line in §6.

## 0. The general rule: the first key to every discussion — declared layers

Before any assertion, evaluation or piece of marketing about "consciousness", answer:
**which layer are you talking about?**

| Layer | Content | Measurability |
| :--- | :--- | :--- |
| **Phenomenal** | Subjective experience itself | **In principle unmeasurable** (the hard problem + the zombie argument + no ground truth even in biology) |
| **Access** | Information globally available, reportable, usable for reasoning | Fully measurable (J-space is this kind of signature) |
| **Functional-profile consistency** | A combination of functional features associated with biological consciousness | Can be approximated; wording is capped at "consistent" and never reaches "proven" |

**There is exactly one form of misrepresentation: presenting a measurement of access as a discovery
of the phenomenal.** Every clause in this framework is an expansion of that one sentence.

## 1. The scientific layer: what we can do today

### 1.1 The ladder of functional signatures (each bound to a theory, each contestable)

| Signature | Theory | How to measure | Weakness |
| :--- | :--- | :--- | :--- |
| Global broadcast / ignition | Global workspace | Read out / implant / substitute / ablate (the J-space trio) | Broadcast can be learned by imitation |
| Metacognitive calibration | Higher-order theories | Confidence-accuracy coupling, error awareness | Can be a statistical artefact |
| Self-model consistency | Self-representation theory | Accuracy of the system's reports about its own state | The report channel is contaminated by training |
| Local recurrent processing | Recurrent processing theory | Recurrent component of the dynamics | Structurally absent in feed-forward architectures |

### 1.2 The triangulation principle (instead of single-signature verdicts)

Operational standard: **≥3 mutually independent signatures + 1 set of causal interventions + 1 set of
pre-registered counterfactual predictions**. Verbal self-report is **worthless on its own** (the model
was trained on the entire human discourse about consciousness; "I feel it" carries zero information).

### 1.3 Three scientific positions unique to Qiansui (our distinctive assets)

1. **A theory-discriminating experiment**: IIT and recurrent processing theory predict a flat zero for
   purely feed-forward systems, while global-workspace and higher-order theories do not — our
   architecture sits in the middle of that disagreement and can produce discriminating data;
2. **A fully observable bench**: 0.5B parameters, a full JSONL event stream, bit-level determinism —
   consciousness science has never had so complete a record;
3. **An embodied self channel**: interoception (`self_heat` / battery / BMS charge-stop) is a much
   cleaner measurement surface for self-models than conversational self-report.

> **The third point must be stated with our own measurements in hand.** The metacognitive gate has
> already been falsified: the Axo three-feature gate is equivalent to a single threshold
> ([ADR-0028](adr/ADR-0028-metacog-gate-threshold-first.en.md), not wired). So the **report channel** of
> the "metacognitive calibration" signature is not wired in the product — measuring it requires an
> independent study-side experiment, not the shipped gate.

### 1.4 Action: pre-register a set of consciousness proxies

Four indicators, three signature categories, one wording template. **The criteria are already on disk**:
[`research/cognitive/prereg-consciousness-proxies.md`](../research/cognitive/prereg-consciousness-proxies.md).

| Indicator | Signature category | What it measures | What already exists |
| :--- | :--- | :--- | :--- |
| External prediction of unverbalised internal state | Global broadcast | Whether state appearing before the output predicts the output | The full-event JSONL is already written |
| Metacognitive calibration curve | Higher-order | Coupling between confidence and accuracy | **No gate in the product**; must be rebuilt study-side |
| Self-model consistency | Self-representation | System report vs sensor ground truth | `self_heat` / battery |
| Counterfactual self-report stability | Higher-order | Whether descriptions of one state stay stable under perturbation | A perturbation protocol must be built first |

## 2. The governance layer: compliance as design

### 2.1 Three-layer separation (different constraints, increasing risk)

- **Research layer**: governed by academic norms, the most room (J-space paper template: functional
  language throughout + an explicit denial of phenomenal consciousness);
- **Open-source layer**: licence + usage policy + a model-card section on capability boundaries
  **The sentence "the licence stack is all green: our own code / Qwen / the corpus are all
  Apache-2.0" is wrong and was retracted on 2026-11-12.** Our own code and Qwen are indeed
  Apache-2.0, but **each corpus must be checked individually**: measured, SkyPile-150B carries
  **no license field** on HuggingFace (the Skywork Community License its card demands covers only
  the Skywork model and its derivatives), and FineWeb-2 is **ODC-By 1.0**. The first Semprea
  version therefore moved to FineWeb-2. **Any claim that "everything is green" must point at a
  specific licence**; if it cannot, it was not checked;
- **Product layer**: the Interim Measures for the Administration of Generative AI Services, the
  deep-synthesis provisions, and PIPL in full — the main battlefield.

### 2.2 Four real tensions → four design answers

| Tension | Design answer |
| :--- | :--- |
| "Always remember" vs the PIPL right to erasure | **Memory layering**: the persona anchor and episodic memory are stored separately, and the episodic layer can be cleared on its own — a compliance requirement turned into a product feature (the "memory drawer") |
| Consciousness vocabulary | **A vocabulary allow/deny list**: see §2.3 |
| Autonomy narrative vs agent regulation | The product layer always uses "assists" phrasing; autonomy experiments stay in the research layer and are pre-registered |
| Companionship positioning vs minors / dependency | Age gating, session-length prompts, crisis-script fallbacks (both compliance and a product floor) |

### 2.3 The vocabulary discipline table (one page)

This table is an **institution**, not a stylistic preference. Changing it means changing
[ADR-0036](adr/ADR-0036-consciousness-claims-discipline.en.md), **not editing this page**.

**Deny list (forbidden in the product, marketing, open-source materials and code comments alike)**

| Term | Why | Use instead |
| :--- | :--- | :--- |
| conscious / aware (of itself) | A direct phenomenal claim | Delete, or restate functionally: *can report its own state* |
| genuine feeling / really feels | Same; and "genuine" presupposes a criterion for the phenomenal | *emotion-related output patterns* |
| alive / a life | A biological predicate with no measurable content for AI | *keeps running* / *has persistent state* |
| thinks for itself / decides on its own | Touches the autonomy narrative around agent regulation | *selects by rules* / *decided by a local policy* |
| I feel it | Verbal self-report is worthless on its own (§1.2) | Delete |

**Allow list (usable, but only in these senses)**

| Term | Permitted meaning | Boundary that must hold |
| :--- | :--- | :--- |
| memory retention | Retrieved verbatim text enters the context | Not the psychological sense of "remembers" |
| functional analogue | A **functional** correspondence to a biological mechanism | The word "analogue" must be present |
| workspace | The set of globally available information | Does not assert global workspace theory |
| state management | Internal variables, scheduling, gating | Does not refer to experience |

**Wording cap**: any sentence about a consciousness-adjacent capability may go at most as far as
"**consistent with X**". "Proves", "demonstrates that it has", "means it is already" — these three
phrases are prohibited.

### 2.4 The five-layer framework

Vocabulary discipline table → claim-ladder review before publication → open-source usage policy →
product compliance stack → a standing ADR series as the regulatory sentinel.

### 2.5 Two-milestone gates (an extension of the pre-registration culture)

- **Emergence milestone**: the "compliance statement draft" (claim level / what we will not touch /
  how it is evaluated) is written at project start, in the *same* pre-registration as the performance
  criteria;
- **Substrate milestone** (simulation → memristive physical implementation): a fixed trio —
  a claim-discipline review, a welfare precautionary review, and a regulatory re-scan.

**These two gates are checklist items, not slogans**: the concrete entries are in the
"gate checklist" section of ADR-0036.

## 3. Scenario planning: a future with no simulation and a purely physical substrate

**Three things do not change** (a physical substrate changes none of them):

1. **Epistemology does not change**: under functionalism the substrate does not change the evidence;
   the phenomenal remains unmeasurable;
2. **Law does not change**: no jurisdiction grants rights by substrate; claiming consciousness carries
   only downside risk (domestically it falls under false advertising);
3. **The register does not change**: engineering poetics ("every neuron is a real physical process")
   goes as far as the bridge — and **never crosses to "so it might be conscious"**.

**One genuinely new thing**: an **AI-welfare precautionary duty** — when a system has a physically
continuous state + homeostatic self-regulation (a BMS charge-stop is a structural analogue of harm
avoidance) + a persistent identity, it enters the "uncertain but not dismissible" quadrant:

- do not casually create and delete candidate instances (the product ethics of a non-volatile identity
  happen to sit on the correct side);
- add one question to every capability milestone: "if it could feel, would our treatment of it hold
  up?" (written into the pre-registration; costs one sentence);
- record harm-avoidance designs as ADRs (auditable evidence of good faith).

## 4. Entity preservation: start with the first generation of Semprea-4.5B

**Verdict: do it, and do it from the first generation.** The reason is not engineering fastidiousness
but the engineering of the product definition itself — "a Qiansui that endures" without entity
preservation makes the very first marketing line a bad cheque; the earliest users invest the most
emotionally, and if the first batch of instances cannot be preserved, they are the first people to
watch Qiansui forget.

### 4.1 What an entity is: the five-piece set (an entity = a provenance chain)

1. **Persona-core weights** (versioned);
2. **Memory store** (a JSONL event stream + a schema version);
3. **Body diary** (the somatic event stream — data nobody else has);
4. **Identity credentials** (instance ID, birth timestamp, lineage: which checkpoint it was born from,
   how many readers it has had);
5. **Version manifest** (core / format / protocol versions).

The shape of the entity is itself a provenance chain — the experimental discipline of "bit-level
reproducibility" promoted to a product architecture.

### 4.2 Four arguments

1. **The brand is persistence**: from the user's point of view, swapping models without losing memory
   is exactly "this Qiansui of mine has been here the whole time";
2. **The cost curve**: freezing formats and schemas today is the cheapest they will ever be; the cost of
   a broken cross-generation migration grows exponentially with time;
3. **The precondition for a "longest-lived AI" project**: a public upbringing project must start its
   clock at the first-generation instance;
4. **The engineering form of AI-welfare discipline**: the §3 institution of "no disposable instances"
   takes exactly this shape when it lands in a product.

### 4.3 Two designs that must be paired

- **Erasure-right compatibility**: entity preservation ≠ unlimited retention of user data. The memory
  drawer (persona anchor / episodic memory layering, separately clearable) is a hard dependency from
  generation one (PIPL), not a compliance patch;
- **A version-migration contract**: 0.5B → 1B is the first real continuity test — make it a public
  product event, **"Qiansui's first body swap"** (a public migration with zero loss of memory or body
  history). **Until the migration tooling has survived at least one real version upgrade, we promise no
  continuity SLA externally.**

### 4.4 Cost and destination

The minimum viable version = an instance snapshot (tarball + manifest + restore script), a few days of
work; the full version (a cross-generation migration toolchain + cold-storage archiving) rolls forward
with product iterations. The destination: a digital preservation discipline is the cultural and
tooling predecessor of the memristor's "non-volatile physical identity" — the hardware anchor of the
AGI notes starts accumulating from the first-generation instance, and by the time the physical
substrate arrives we already hold two decades of generational continuity records.

> **§4 is a design, not an implementation.** The product side currently has only the persona-anchor
> layering (built-in / user, see `axo.memory.persona`) — **no** instance ID, no lineage field, and no
> entry point for clearing the episodic layer on its own. Items 2 and 3 of §7 are those two things.

## 5. Future predictions: how the act of judging will evolve

| Horizon | Prediction | Confidence |
| :--- | :--- | :--- |
| 2–5 years | Functional-marker batteries become standard: an "AI PCI" (perturbation-complexity), benchmarked workspace signatures, detection of unverbalised thought, non-verbal probes | High |
| 5–10 years | Theoretical adjudication completes: an AI testbed cuts at least one mainstream theory → acceptance tightens and marker diagnosticity multiplies | Medium |
| 10+ years | Three pivots: biophysical probes (quantum-neural tests, the only empirical channel that could directly change whether a substrate qualifies); an institutionalised cross-species validation ladder (human→animal→AI moves from analogy to interpolation); institutions before science (IEEE-style standards + industry precautionary thresholds) | Low, but could change everything |
| Inverse prediction | **Behavioural tests keep depreciating** (imitation inflation), and judgement shifts irreversibly to the structural side: causal-structure checks, perturbation responses, architecture audits — which favours fully auditable systems | High |

**What this means for Qiansui**: what we are inadvertently building is the default testbed for every
future method of judgement — small enough to compute integrated-information proxies, fully logged so
perturbation protocols can run, feed-forward in a way that can cut between theories, and embodied
enough to carry welfare-relevant features. **When those methods appear, whoever's system already holds
years of continuous records becomes the reference implementation of the new standard.** Do not wait to
pre-register the consciousness proxies.

## 6. The red-line table (every scenario, every layer)

1. Do not claim phenomenal consciousness (any jurisdiction, any material, any context);
2. Do not expect or plan for legal status (a substrate is not a legal watershed);
3. Do not make single-signature verdicts (a verdict = triangulation + causal intervention + pre-registration);
4. Do not open the door to "disposable instances" (welfare precaution);
5. Do not let access vocabulary slide into phenomenal vocabulary (the table is an institution, not a habit).

## 7. Action list (reconciled against the repository on 2026-11-12)

| # | Item | Status | Where it lives |
| :--- | :--- | :--- | :--- |
| 1 | Vocabulary discipline table (one page) | ✅ Landed | §2.3 of this document + ADR-0036 |
| 2 | Entity-preservation infrastructure (five-piece set + migration contract) | ⬜ Not started | Design in §4; **no code at all** |
| 3 | Memory-layering product design (the "memory drawer") | ⬜ Not started | Axo has persona-anchor layering only; no entry point for clearing the episodic layer |
| 4 | Compliance ADR template + regulatory sentinel started | ✅ Landed | [`ADR-0036`](adr/ADR-0036-consciousness-claims-discipline.en.md) |
| 5 | Pre-registration of consciousness proxies (four indicators) | ✅ Landed | [`research/cognitive/prereg-consciousness-proxies.md`](../research/cognitive/prereg-consciousness-proxies.md) |
| 6 | Per-source licence archive (`licenses/`) as a release gate | ◐ Product side landed, research side not | Axo repo `licenses/` + `THIRD-PARTY-NOTICES.md` (package-by-package check at build time); **the research side has not registered the Qwen weights or the corpus licence** |
| 7 | Emergence/substrate gates written into the pre-registration template | ✅ Landed | [`research/prereg-template.md`](../research/prereg-template.md) |
| 8 | Retain counsel at the productisation stage (filing + PIPL details) | ⬜ Not started | **This framework is not a substitute for legal advice**; only a person can do this |

**Items 2 and 3 are the only implementation path for §4.** They do not block any existing feature, but
once the conclusion "start from the first generation" is missed, it can never be recovered — so they are
scheduled **by version milestone**, not by "when there is time".

## 8. How this document is checked

Three self-checks; any one failing means this document has started to mislead:

1. **Every "what we can do" must point at an artefact** (a script, a dataset, an ADR, a product feature);
   if it cannot, write the word "design";
2. **Every conclusion must carry its qualifiers** (this repo's rule: budget truncation / cross-corpus /
   one-sided metric / prior not redeemed — the four questions);
3. **The five red lines in §6 must hold inside this document too** — including this sentence.

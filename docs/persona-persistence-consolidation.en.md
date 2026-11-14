# Qiansui · Semprea session digest v1.0 (evening of 2026-11-12 — 11-14)

[中文](persona-persistence-consolidation.md) · [日本語](persona-persistence-consolidation.ja.md)

> Main theme: **persona persistence** — Qiansui's moat itself.
> "Remembering you" is not a feature, it is the product definition; this file gathers two
> evenings of sessions into five blocks.

## 1. Persona persistence (main line)

Persona persistence = memory persistence + stylistic consistency + identity continuity. Five clusters:

| Cluster | Content | Source |
| :--- | :--- | :--- |
| **Capability** | Memory narration ("tell the story of you two") / affect sensing (arousal channel ADR-0016, dynamical modulation rather than prompt engineering) / proactivity (A1-A2 ladder) / persona consistency with opinions (can refuse, can close — anti-dependency is a capability, not a compliance patch) / singing | Companion-capability discussion |
| **Measurement** | Character metrics (memory persistence, style consistency); delayed-recall probes (N≈16→90%, N≈64→50%, N≥128 chance); **identity distance** (behavioural consistency + representational similarity + memory probes); continuity tests | Experiment line |
| **Mechanism** | The five-part entity kit (weights / memory store / body journal / identity credential / version manifest); memory drawer (deletion-right compatible); migration contract; four public migration events (model → body → device → substrate) | Entity-preservation framework |
| **Research frontier** | **The state-modality dispute** (text state vs spiking state — the mechanism-level argument for persona persistence, §5); the **improvement–drift Pareto front** (a measurable Ship-of-Theseus) | Literature / self-iteration |
| **Ethical boundary** | Privacy tiers (biological data > camera); anti-dependency design; death and inheritance (digital legacy); consciousness vocabulary discipline | Consciousness framework |

Headline anchors: "**swap the brain once, it still remembers you**" (general audience) + "0.038%" (technical audience).

## 2. Companion capabilities and modalities (evening of 11-12)

- **Five strong capabilities**: memory narration / affect sensing / proactivity / persona consistency with opinions / singing; plus two mandatory safety capabilities: crisis recognition and referral, and minor protection.
- **Modality-gap diagnosis**: the output side is complete (voice / face screen / light band); the **input side is missing the two core companion channels** — **touch** (brow-rim capacitive touch, BOM ¥3) and **presence sensing** (24 GHz mmWave, BOM ¥30–60; it does not look at you, it only knows you are there).
- **Voice details** (zero hardware): interruption and barge-in, night whisper mode, dialect layer (a hard requirement for elderly users).
- **Camera**: deepening from event-level to expression-level stays inside the licensing framework (it would need a "fourth-tier licence"); storing images is never done.
- **Horizon**: smell (the Proust modality, already anchored by the gas-sensing collaboration).
- **Three-question filter** (every new modality or capability must answer): how is it licence-tiered / which character metric does it feed / is it a new sense of the same entity, or another application.

## 3. Senses and the hardware product line (evening of 11-12)

- **Gas sensing** (collaboration with a doctoral team at Northeast Normal University): **two lines, SNN + PINN** — the SNN line is edge classification and event generation (exit: a scent event plugin into SpikeBus, whose modality slots already exist); the PINN line is sensing-physics modelling, digital twin, drift compensation; the coupling point is PINN supplying physically consistent synthetic data and online-corrected features to the SNN. **The two-line template (PINN digital twin + SNN edge plugin) is the general integration recipe for any new sense**; the PINN capability also transfers to memristor device modelling (memristor layout is out of scope here).
- **Biosensing**: technology lineage table (PPG / bioelectric / enzyme electrode / mmWave / rPPG / sweat / breath VOC / vocal biomarkers); Axo's three realistic paths are wearable-data ingestion (pure software) → mmWave vital signs (the same path as presence sensing, not a new modality) → rPPG (reusing the camera, last); the **zero-hardware hidden entrance is vocal biomarkers** (load-bearing for psychological companionship and cognitive decline); red lines are wellness wording (the NMPA boundary) plus PIPL sensitive-tier biological data, with biological events not retained by default.
- **Watch category**: **build a window, not a watch; build a token, not a computer** — watch app and faces (Gen 3 window standard, first) → Qiansui band or token (no screen: light ring + haptic pulse + voice notes, BOM ¥50–80, mid-term) → smartwatch (never: engineering, ecosystem and stage all veto it). The band has three entry conditions: the box is stable / the watch-face retention data proves itself / certification is scoped.
- **Six home bodies plus one bridge**: goodnight lamp (first pick: mmWave sleep staging + breathing light + sung lullaby + morning dream flashback) / ambient lamp (room-level light language) / Axo mini (multi-room window) / TV big screen (pure-software big face) / doorway terminal / sensor nerve endings (¥30–80 each, room-scale body) plus a Matter bridge (existing appliances become organs). Not doing: standalone camera products, child-specific hardware, mobile robots.
- **Projection**: **build a shadow, not a projector** — buy the pico optical engine, integrate in-house; avoid the word "projector" (it enters lighting-gift channels); the strongest scene is the bedroom-ceiling dream mode.

## 4. Release engineering (11-14)

- **Gen1 suffix settled**: `Semprea-4.5B-Gen1` — the rule "parameters do not occupy the version name" made operational; three-part filename `Semprea-4.5B-Gen1-qwen4b-Q4_K_M.gguf`; a two-repo structure (fused body + a `Semprea-Core-0.5B-Gen1` Delta); the cloud form carries neither suffix nor parameter count.
- **ModelScope setup**: the organisation name "semprea" is registered (one more platform in the naming campaign); licence Apache-2.0 ✓; **build the repo first, fill the card first, open the gate later** — README and model card first, weight upload bounded by the four release gates (criteria / LoRA ablation / multiple seeds / licences).
- **Three headline pillars**: (1) a measurable persona-memory win ("4.5B beats a cloud flagship at 'remembering you'", attributed causally to architecture) (2) a significant ablation gap plus an equal-parameter LoRA-0.5B that is not worse (3) engineering spectacle (0.038% / bit-level determinism / edge readiness).
- **Five minimum-magic lines** (in priority order): flawless swap-the-reader demo → win at least one task family against a cloud flagship → significant ablation → general dialogue ≥ 95% of the bare reader → edge readiness.
- **Not chasing**: general leaderboard rank, multimodal metrics, broad SOTA claims.

## 5. The state-axis literature front (checked 11-14)

**Lineage (three species on the state axis plus a frontier signal)**:
**Reflexion** (2023, textual reflection memory injected into the prompt, NeurIPS) → **JitRL** (2026, textual memory → **logit modulation**, ICML 2026 Spotlight, $98 cost beating WebRL on-the-fly) → Qiansui (spiking memory → logit modulation). Corroboration: **NVIDIA's "You don't need long context"** (arXiv 2611.05306: frontier models lose almost nothing with short context plus compression; proposes the SSSM concept without implementing it) — a **second frontier-convergence signal** alongside J-space (July).

**JitRL's three absorption points** (the closest relative so far):

1. **Same interface**: logit modulation is the persona core's style-control channel; its theorem "an additive logit update is the closed-form solution of KL-constrained policy optimisation" is the channel's mathematical backing;
2. **A door left open**: JitRL's limitations section states plainly that textual-state retrieval is limited for non-text domains and that non-text state representations are future work — exactly the persona core's strength;
3. **A baseline column**: the state-modality experiment uses **the same backbone, the same logit interface, only the state modality differs** (textual memory vs spiking plus external memory), a textbook variable isolation at $98 cost.

**Methodological literature**: WebArena (the ancestor of self-hostable, programmatically verifiable evaluation environments — the harness release follows that tradition) / WebRL (its task-synthesis machinery is reusable: an LLM generates memory and style test tasks plus verifiers).

**Honest boundaries**: SSSM has no implementation; JitRL is a Spotlight, not an Oral (cite the repo); the evidence is scoped to agentic tasks; note the chip vendors' incentives.

## 6. Landing checklist

1. The technical report's introduction cites three papers: Reflexion / JitRL / NVIDIA (the state axis is no longer alone; each gets one explicit point of difference);
2. The tier-2 claims table **proactively includes JitRL** (same interface, different state) so reviewers do not add it for us;
3. Pre-register the "state-modality dispute": baseline columns are JitRL and Reflexion, same backbone and same interface, tasks grown by a WebRL-style synthesiser;
4. The model card text is ready (ModelScope); weights upload after the four gates;
5. Companion line: touch and presence sensing enter Axo's next BOM (the three questions are answered; cost is negligible);
6. Sense line: advance the Northeast Normal University two lines, with authorship split agreed early (PINN → sensing journals / SNN → neuromorphic conferences);
7. Hardware line: the goodnight lamp is second in the home queue (after the box); the watch line does the faces first and the band later.

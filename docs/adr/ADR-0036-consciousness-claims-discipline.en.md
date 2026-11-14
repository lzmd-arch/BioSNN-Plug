# ADR-0036: Discipline for consciousness-related claims — vocabulary, claim ladder, two-milestone gates

[中文](ADR-0036-consciousness-claims-discipline.md) · [日本語](ADR-0036-consciousness-claims-discipline.ja.md)

- **Status**: Accepted
- **Date**: 2026-11-12
- **Related**: [`docs/consciousness-framework.en.md`](../consciousness-framework.en.md) (the "why" of this ADR),
  [ADR-0028](ADR-0028-metacog-gate-threshold-first.en.md) (metacognitive gate equivalent to a single threshold — cited here as a counter-example),
  [ADR-0035](ADR-0035-local-learning-lm.en.md) (capability boundaries of the local-learning LM, the factual basis for the model card),
  [`research/cognitive/prereg-consciousness-proxies.md`](../../research/cognitive/prereg-consciousness-proxies.md) (proxy criteria)

## Context

This repository has three kinds of output that touch the word "consciousness". Their constraints differ
completely, yet they are routinely promised in the same sentence:

1. **Research output** (papers, preprints, scan notes) — bound by academic norms, free to discuss
   functional signatures;
2. **Open-source output** (the repo, README, model card, ADRs) — kept in version history, so **any
   sentence that oversteps will be quoted forever**;
3. **Product output** (Axo desktop / Android, marketing material, in-app copy) — bound by the Interim
   Measures for the Administration of Generative AI Services, the deep-synthesis provisions and PIPL in
   full; **claiming consciousness domestically falls under false advertising**.

The shared risk of all three is one action: **presenting a measurement at the access level as a
discovery at the phenomenal level**. That action is not a one-off slip — it compounds as materials cite
each other and as versions accumulate, until it becomes a brand promise.

There is also pressure in the opposite direction: **the product positioning needs a narrative**.
"Memory retention", "state management" and "embodied self" are features we genuinely build, but the
moment they are sold as "she remembers you" or "she has feelings", the engineering promise has been
swapped for one that cannot be honoured.

What is needed, therefore, is not the advice "watch your wording" but **executable clauses**: a
deny/allow list, a ladder every publication must clear, and two milestones that must both be signed off.

## Decision

Four rules are adopted. **All of them are decision rules, not style advice.**

### Rule one: a vocabulary deny list and allow list (deny wins)

**Deny list** — forbidden in the product, marketing, open-source material, code comments and commit
messages alike:

| Term | Why it is denied | Permitted alternative |
| :--- | :--- | :--- |
| conscious / aware (of itself) | A direct phenomenal claim | Delete it; or restate functionally: *can report its own state* |
| genuine feeling / really feels | Same; and "genuine" presupposes a criterion for the phenomenal | *emotion-related output patterns* |
| alive / a life | A biological predicate with no measurable content for AI | *keeps running* / *has persistent state* |
| thinks for itself / decides on its own | Touches the autonomy narrative around agent regulation | *selects by rules* / *decided by a local policy* |
| I feel it | Verbal self-report is worthless on its own (see rule two) | Delete it |

**Allow list** — usable, but only in these senses:

| Term | Permitted meaning | Boundary that must hold |
| :--- | :--- | :--- |
| memory retention | Retrieved verbatim text entered the context | Not the psychological sense of "remembers" |
| functional analogue | A **functional** correspondence to a biological mechanism | The word "analogue" must be present |
| workspace | The set of globally available information | Does not assert global workspace theory |
| state management | Internal variables, scheduling, gating | Does not refer to experience |

**Terms outside the allow list are not automatically banned**, but they must clear rule two.

### Rule two: claim-ladder review (a veto before publication)

Every sentence about a consciousness-adjacent capability must first locate itself on this ladder.
**Compatibility only runs downward.**

| Level | Permitted wording | Evidence required | Who may publish |
| :--- | :--- | :--- | :--- |
| **L0** | "On this task, metric X is Y" | One reproducible measurement | Anyone |
| **L1** | "**Consistent with** the signature predicted by <theory>" | ≥1 signature + ≥1 set of causal interventions | Research layer |
| **L2** | "Consistent across ≥3 independent signatures" | Triangulation (≥3 signatures + causal interventions + pre-registered counterfactual predictions) | Research layer, **and** the non-holding signatures must be stated |
| **L3** | "Proves / shows that it has …" | **Unreachable** | Nobody |

Three vetoes:

1. **Verbal self-report cannot serve as evidence on its own** — the model was trained on the entire human
   discourse about consciousness; "I feel it" carries zero information;
2. **A single signature cannot be promoted to L2** — including "global broadcast", which looks the most
   persuasive and is precisely the one that can be learned by imitation;
3. **Every L1/L2 sentence must also give at least one signature or condition under which it does NOT
   hold.** Reporting only the signatures that hold is treated as fabrication.

### Rule three: two-milestone gates

Each milestone has a fixed trio. **A milestone is not reached until all three are done.**

| Milestone | Trigger | The three things that must be done together | Where it lands |
| :--- | :--- | :--- | :--- |
| **Emergence milestone** | A capability crosses a pre-registered threshold | ① Compliance statement draft (claim level / what we will not touch / how it is evaluated) ② Vocabulary review (rules one and two) ③ The **same** pre-registration as the performance criteria | That experiment's pre-registration file |
| **Substrate milestone** | Moving from simulation to a physical substrate (e.g. memristors) | ① Claim-discipline review ② **AI-welfare precautionary review**: if the system has a physically continuous state + homeostatic self-regulation + persistent identity, answer item by item "if it could feel, would our treatment hold up" ③ Regulatory re-scan (jurisdiction, register, filing obligations) | A new ADR — **never edit a completed one** |

**The gates are a checklist, not a ritual**: the tables above are the template; copy them into the
pre-registration (see the gate section of `research/prereg-template.md`).

### Rule four: the regulatory-sentinel ADR series

Any one of the following events **requires a new ADR** (next number, status "proposed") and may not be
settled in a conversation alone:

- the product wants to claim any capability adjacent to "consciousness / emotion / autonomy";
- a new applicable regulation, filing requirement or draft standard appears;
- either milestone gate is triggered;
- a deny-list term appears in published material.

## Consequences

**Benefits**

- Overstepping sentences move from "depends on self-discipline" to "a table plus a ladder" that others
  can audit;
- Once wording is separated between the research and product layers, research can discuss functional
  signatures freely without the product having to panic;
- With the gates written into the pre-registration template, "the capability arrived but compliance did
  not" can no longer happen silently in the process.

**Costs (explicitly accepted)**

- **The product narrative is constrained.** The most moving lines ("she remembers you", "she has always
  been here") cannot be used at the claim level; they may only be used at the functional level and must
  be deliverable. This is a deliberately surrendered marketing tool.
- **Research output gains a step.** L1/L2 sentences must carry a counter-example, which costs writing
  effort and reads less well.
- **Rule one will misfire.** In Chinese, "生命" and "意识到" are common in unrelated contexts ("realised
  this bug"); the deny list is enforced **in contexts referring to the AI**, not by string matching.
  **That ambiguity is borne by this ADR, not by the person applying it** — when in doubt, treat the term
  as denied and add an example here afterwards.
- We explicitly do not build an automatic scanner to enforce rule one. Context judgement cannot be done
  reliably and false positives would destroy the authority of the discipline; review is done by a person,
  **but every rejected sentence is recorded in the PR description** so that exemptions become auditable.
- We do not relax the rules because "others in the industry say it too".

## Alternatives

| Option | Reason for rejection |
| :--- | :--- |
| **No clauses; rely on the researcher's judgement** | This was the state before this ADR. The risk is not malice but that no one is accountable when materials cite each other. |
| **Ban the word "consciousness" outright** | It would make research writing impossible ("access consciousness" is a standard term) and a rule that is obviously unreasonable gets bypassed wholesale. |
| **Product side only, research side untouched** | Research output is the citation source for product material. Sentences loosened on the research side get carried straight into the product. |
| **Enforce with an automatic scanner** | See Consequences: context judgement is unreliable and false positives consume the authority of the discipline. |

## What this ADR does not do

- **It is not legal advice.** It is an engineering discipline; productisation still requires counsel to
  give a formal opinion on filings and PIPL details.
- **It does not judge whether any particular system is conscious.** It only specifies when certain things
  may not be said.
- **It does not constrain the user's private expression.** The rules bind this repository's output, not
  how the user talks about Qiansui.

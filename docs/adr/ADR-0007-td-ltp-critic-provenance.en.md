# ADR-0007: Critic uses TD-LTP, with provenance in Frémaux et al. (2013)

[中文](ADR-0007-td-ltp-critic-provenance.md) · [日本語](ADR-0007-td-ltp-critic-provenance.ja.md)

- **Status**: Accepted
- **Date**: 2026-09-21
- **Related**: project plan §3.2, §11 P0 to-do; [docs/references.md](../references.en.md)

## Context

§3.2 of the project plan writes the execution layer's Critic training rule as **"TD-LTP"**, and
marks it "⚠️ provenance to be added", while §11 lists it as a **P0 to-do** and stipulates:

> TD-LTP provenance verification | The Critic's training rule is the cornerstone of the claim
> that the whole R-STDP loop is local; **if there is no provenance, phase 5 does not get
> approved as a project**.

At the same time, §3.2 of the project plan binds "TD-LTP" into a single citation with
**[3] Frémaux, Sprekeler & Gerstner (2010)** — but the 2010 paper is about the unsupervised
bias of R-STDP and the requirement that "there must be a stimulus-specific reward prediction
system"; it **neither proposes nor names** TD-LTP. Whether that pairing holds needs
verification.

## Decision

**Adopt TD-LTP, with provenance in Frémaux, Sprekeler & Gerstner (2013), *PLoS Computational
Biology* 9(4):e1003024. The P0 to-do is closed, and phase 5 clears its approval gate.**

Verbatim from the original (the Critic learning section, corresponding to Eq. 17):

> Because it has, roughly, the form of "TD error signal × Hebbian LTP", we call this
> learning rule **TD-LTP**.

The Figure 2A caption: "TD-LTP is the learning rule given in Eq. 17." "TD-LTP" appears 40 times
in the full text. The paper is open access, so you can re-verify it yourself.

**The rule itself** (three-factor, counting pre-before-post only):

Δw ∝ δ(t) · κ ∗ [ x_i(t) · y_j(t) ]

- Factors 1 and 2: the pre and post spike trains pass through a coincidence window that
  **counts pre-before-post only** (post-before-pre pairings are ignored — this is the
  substantive difference between TD-LTP and TD-STDP);
- The filtering kernel κ acts as the eligibility trace (the original says it "serves a role
  similar to the eligibility trace");
- Factor 3: δ(t), the **global scalar** TD error, carried by a dopamine-style broadcast.

**The only non-local quantity is this one scalar**, and there is no back-propagation of an
error vector, which is compatible with the project plan's "purely local" claim.

**The citation pairing in the project plan has to be split into two**:

| Item | Should cite |
| :--- | :--- |
| The name TD-LTP and the rule itself | Frémaux et al. (2013), PLoS Comput Biol 9(4):e1003024 |
| R-STDP's unsupervised bias, and the need to introduce stimulus-specific reward prediction | Frémaux et al. (2010), J Neurosci 30(40):13326-13337 (the subject of Figure 3 in that paper) |

## Consequences

**Benefits**

- The P0 question in §11 has a definite answer, and phase 5 is no longer left hanging;
- The architectural choice has external precedent behind it: [13] Tihomirov et al. (2025) uses
  TD-LTP as the Critic of a spiking actor-critic, isomorphic to the design of §3.2 — independent
  evidence that "this route is viable";
- The rule satisfies the project's hardest constraint — the non-local signal is a single scalar.

**Costs (accepted)**

- **The form of the rule can only be implemented from a second-hand description.** The 2013
  paper is open access and can be checked; but the applied follow-up (the journal version by
  Tihomirov/Rybka) sits behind a paywall. So what this ADR records is "the naming and the
  skeleton of the rule", not a verbatim transcription of the equations. If the details do not
  line up when it is implemented in phase 1, the 2013 original is authoritative.
- The body of the project plan needs a v6.3 that corrects the citation pairing. Until that
  correction, wherever §3.2 of the project plan is cited, this ADR is authoritative.

**Explicitly not claimed (avoiding over-citation)**

1. The sentence "no back-propagation signal has been observed in experiments" in Frémaux 2013
   **cannot** be used to argue that "biological networks do not back-propagate an error vector".
   The back-propagation in that sentence refers to the credit assignment property of the TD
   error along **time** (the original also calls it "this back-propagation phenomenon is a
   signature of TD learning algorithms"), and it is unrelated to this project's locality claim.
   To argue for locality, cite the three-factor form itself together with the original's
   wording "the global signal".
2. For the relationship between TD-STDP and TD-LTP, the original's wording is "behaves
   similarly" and "only slightly worse". **Do not write it as "functionally equivalent".**
3. This ADR does not verify, word by word, the notation system of the paywalled journal
   version. Wherever the concrete implementation is concerned, Eq. 17 and Figure 2A of the
   2013 original are authoritative.

## Alternatives

**A. Rule that there is no provenance, and trigger the fallback path of §11 (a moving-average
baseline Critic).**
Rejected. The verification conclusion is that a provenance exists, and that it is a formal
naming by the authors (three pieces of evidence: a naming statement, an equation number and a
figure caption). The premise for the fallback does not hold.

**B. Adopt Tihomirov/Rybka et al. (2025) as the primary provenance of TD-LTP.**
Rejected. Both papers really exist, but the 2025 one **uses** TD-LTP, while the 2013 one
**names and defines** it. Cite the naming provenance, not the using provenance.
(Two search angles during verification attributed it to 2025, and an adversarial
re-verification overturned them — the reviewer admitted that their search did not go back as
far as 2013.)

**C. Do not implement TD-LTP; use TD-STDP directly (the smallest change, differing from the
existing R-STDP code in one place only).**
Deferred, not rejected. The original verifies that the two behave similarly; TD-STDP only
requires replacing the reward modulation term S=R−⟨R⟩ with δ(t). If implementing TD-LTP runs
into obstacles in phase 1, this is the lowest-cost fallback. Open a separate ADR at that point.

**D. Switch wholesale to the RL form of e-prop (Bellec et al. 2020 already contains an
actor-critic).**
Deferred. It can replace "R-STDP + a separate Critic" wholesale, and is theoretically more
unified, but it would bind the execution layer and the cognitive layer to the same set of
rules, weakening the main line of "a combination of three local rules". Leave it until phase 2,
when rule conflicts are evaluated.

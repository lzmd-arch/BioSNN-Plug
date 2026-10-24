# ADR-0014: adaptive arbitration is not measurable, `ESArbiter.fit` stays on standby, and P4 changes its headline

[中文](ADR-0014-adaptive-arbitration-unmeasurable-and-fit-retained.md) · [日本語](ADR-0014-adaptive-arbitration-unmeasurable-and-fit-retained.ja.md)

- **Status**: Accepted
- **Date**: 2026-10-24
- **Related**: [ADR-0013](ADR-0013-synergy-benchmark-priority-and-arbiter-scope.en.md) (the synergy benchmark becomes P0); [`sweep_results/arbiter_synergy_task_spec.md`](../../sweep_results/arbiter_synergy_task_spec.md) (preregistration and the two objective checks); [`sweep_results/arbiter_synergy/`](../../sweep_results/arbiter_synergy/README.md) (results); [research/arbiter](../../research/arbiter/README.en.md)

## Context

The five-seed v3 run showed that a fixed 50/50 mix clearly beats either arm alone (Fisher 0.651
against 0.472 / 0.331; synergy margin +0.179), but the **alpha the ES learned was bit-identical to
a frozen 0.5** — the objective was an integer conflict-event count that is identically zero near
alpha = 0.5, so there was no gradient and alpha never moved.

The two candidate objectives were then checked, preregistration first:

1. **Fisher as the objective**: on 8 / 50 / 200 / 800-step probes, Fisher's range across alpha was
   0.026–0.039, neither growing with probe length nor monotone in alpha — noise. Full training
   needs 37,480 steps to reach 0.65; a probe long enough to see alpha costs about 7 hours.
2. **A continuous proxy** (mean cosine of the blend against both arms): its peak sits at alpha = 0.9
   (Fisher's peak is at 0.5), and it gives **the same value at alpha = 0 and alpha = 1** (0.501)
   although the two endpoints' Fisher differ by 0.14.

Each candidate failed its own check, so **the adaptive arm was not run**.

## Decision

**① Wording red line: "not measurable" is not "disproved".** No document or manuscript may say
"the arbiter has no value". The accurate statement is: **adaptive arbitration is not measurable in
this benchmark** — within the budget (a single evaluation ≤ 5% of one main-run seed, total ≤ 2×)
no objective passed its pre-registered check; the only known-valid objective (a full-course probe)
is priced at about **7 hours** and is left as future work. The static 0.5 arm winning means only
that "on this benchmark nobody can beat it cheaply".

**② `ESArbiter.fit` is kept, labelled standby rather than in service.** It is a library improvement
backed by 72 tests and reverting it would waste that; but three conditions apply:

1. its docstring states it is **used by no reported result** — the synergy arm that was meant to
   call it never ran because no objective passed;
2. its public status: **no sanctioned objective is mounted on this `fit()`**;
3. this ADR is its registration.

**③ P4's headline question changes.** It is no longer "can arbitration buy back the cost of
locality" but **"When Can Arbitration Be Measured?"** — turning "we could not measure it" into
"the cost structure of measuring it".

**④ The two negative diagnostics go into the Discussion**, because their diagnostic value exceeds
the result itself:

* **timescale separation**: a mixture weight's effect on representation quality only appears after
  convergence, so a short-horizon metric cannot in principle serve as an objective for long
  training — a concrete warning for ES meta-learning work that optimises an outer loop on short
  metrics;
* **alignment is not separability**: the proxy measures a different construct from the acceptance
  metric (equal at alpha = 0 and 1 on the proxy, 0.14 apart on Fisher).

## Consequences

- "Does adaptive alpha beat static 0.5" stays **open**, now with an explicit price tag: an upgrade
  from "infeasible" to "priced", not a negative finding.
- The arbiter's three-language READMEs, `DESIGN.md`, the results record and the task spec are
  brought to the new wording; the preregistration text stays untouched and the wording change is
  marked with a **post-data date**, changing no criterion.
- Reopening this line requires **first** an objective that passes its discrimination check at an
  affordable cost; until then no claim of the form "adaptive alpha is better" may be made.
- P4's §3.3 triviality analysis now has four cases: the alpha -> 0 degeneracy, the per-synapse
  cosine degeneracy, the inert integer-objective ES, and the proxy/construct mismatch.

# ADR-0009: W3's behaviour policy becomes Boltzmann sampling, with the trace centred by the sampling probability

[中文](ADR-0009-w3-behaviour-policy-and-trace-centring.md) · [日本語](ADR-0009-w3-behaviour-policy-and-trace-centring.ja.md)

- **Status**: accepted
- **Date**: 2026-09-23
- **Related**: project plan §3.2 and §7 (phase 1); [ADR-0007](ADR-0007-td-ltp-critic-provenance.en.md);
  `research/rstdp/rstdp.py`, `research/rstdp/cartpole.py`, `research/rstdp/README.en.md`

## Context

W3 (R-STDP + TD-LTP Critic on CartPole) had long failed acceptance: median over >=10 seeds
**100.0 steps** against a criterion of 200. Once the failure mode was measured properly, **two
problems stacked on top of each other** came into view.

### One: under a deterministic policy there is no learnable target

`learn_step` updates the Actor only on **non-exploring** steps, and on those `action = argmax` is a
**deterministic** function of the state. So the update lands on the chosen column only — measured
`U touches one column in 100% of states` (a discrete fact, no threshold involved).

More fundamentally, `A(s, pi(s)) ≡ 0` is an **identity** (forcing the action the policy itself would
take and then following the policy *is* the definition of `V^pi`). So the "advantage-weighted
direction" target is **identically zero at the visited states** — measured
`per-state |T| median = 0.0000`. The reason is that at the states a good policy visits, the pole is
near upright and **the two actions are nearly equivalent**.

Together: the rule does not face a wrong direction, it faces **no target at all**. The chosen
column receives drift from an action-blind residual delta. That also explains two long-standing
observations: the policy wanders with large amplitude throughout (all twenty curves swing 5–30x
between adjacent checkpoints), and `|U|/|T|` is large (a **small numerator** — a near-zero target —
not a large denominator).

### Two: the evaluation path itself was defective

`select_action` **ignores `exploration`** in the `boltzmann` branch and samples from the softmax
unconditionally, while `_greedy_score` (used by both the mid-training curve and the final
acceptance run) passes `exploration=0.0`. So that arm's "greedy" evaluation measured a **stochastic
policy**, and the ceiling of that scale is brutally low: a **perfect** policy (the heuristic, 500
steps) is worth only 97.4 steps at 70% action accuracy and 36.3 at 60%. This is the **fifth defect
of its class**, alongside truncation, RNG pollution, the sampling-pool cap and the wrong sign
argument — and its cost was concrete: it made this phase's only effective lever read as "no effect",
and that reading went into the tri-lingual README.

## Decision

1. **`action_sampling="boltzmann"`**: sample from `softmax(logit_scale * score)` and **update the
   Actor on every step** (there is no such thing as an "exploration step"). `logit_scale` is
   required — after L1 normalisation the two columns' score gap is only 0.05–0.5, so an unscaled
   softmax is nearly uniform.
2. **`trace_center="sampling"`**: the trace's second factor becomes `a_j − pi_j`, where pi is the
   **sampling** distribution. Its expectation is zero, so the bias term in
   `E[dw] = eta·E[x·(a−pi)·S]` — the one that demands a per-state-unbiased Critic — cancels exactly;
   and **the unchosen column is weakened**, which is precisely the **action comparison** that was
   structurally missing.
   **Assertion**: `trace_center="sampling"` is meaningful only under `boltzmann`. Under
   epsilon-greedy `a_j − pi_j ≡ 0` freezes the Actor completely — the code raises `ValueError`
   rather than let that happen silently.
3. **Inverse-temperature annealing**: `logit_scale` interpolates in **log space** from **2** to
   20 (or 40). A constant value loses at both ends: near-uniform (5) learns but stays blunt (median
   141.4); sharper (20) reaches higher (held-out 177.3, max 434) but **roughly one trajectory in
   ten never starts at all** (peak only 10–12 steps). Annealing goes blunt-then-sharp and lifts the
   minimum from 9 to 90–123. Log-space interpolation because `logit_scale` is the inverse
   temperature: interpolating it linearly is a hyperbolic temperature schedule (too slow early, too
   fast late).
4. **`select_action(greedy=True)`**: return `argmax` **regardless of the sampling scheme**. The
   evaluation path must use it. Under epsilon-greedy it is bit-identical to the old
   `exploration=0.0` path, so every recorded number stands.
5. The acceptance configuration keeps this repository's convention of being **specified
   explicitly** (the same layer as `actor_lr_final_fraction=0.1`), but the four items above become
   defaults, so that a default run *is* the passing configuration.

## Consequences

- **Both criteria of §7 are met**: on held-out seeds (25–44, twenty of them, none involved in any
  selection) the median is **260.0** (criterion 200) and the maximum offset ratio is **0.0359**
  (criterion 0.10, all seeds pass). The selection set (0–9) gives median 250.4 — close, so this is
  not a seed-selection effect.
- **The additional "stable" condition is not met**: the minimum is **74** (our own pre-registered
  reading requires >= 100), with 2 of 20 seeds below 100. Diagnosis: those two have **peaks of
  90–124**, i.e. they are "not learning well enough", **not collapsing** (contrast the genuine
  collapses under a constant temperature, whose peaks were only 10–12). The failure mode has
  changed kind.
- **What W3 tests has changed**: the training-time behaviour policy goes from "epsilon-greedy
  decaying to 0.02" to "Boltzmann sampling with inverse-temperature annealing (2 -> 20)".
  **Deployment and acceptance remain `argmax`**, so the object §7 measures is unchanged.
- **Locality is unchanged**: the only non-local quantity is still the scalar delta; what is added is
  each action cell's **own** pi_j — a scalar computed from that cell's own input, which
  `trace_center="sampling"` already used.
- Default behaviour changes, so historical numbers such as 114.6 for `--seed 0` correspond to the
  **old default**; they remain in the README's ledger as a control but are no longer "what the
  default configuration produces".

- **Later update (2026-09-23, numeric retuning under the same decision)**: the figures recorded
  above belong to the configuration as of the decision — annealing 2 -> 20, learning-rate final
  fraction 0.1, and 260.0 / minimum 74 measured on seeds **25–44**. Retuning continued on **held-out
  seeds**, and the final configuration is **annealing 1 -> 40, learning-rate final fraction 0.01**,
  giving median 237.8 and minimum 121 on the fresh seeds 45–64 (both seed batches pass). The
  **decision** in this ADR — Boltzmann sampling, trace centred by the sampling probability, inverse
  temperature annealing — is unchanged; only two of its numbers were retuned. By repository
  convention (`docs/adr/README.md`: an ADR's body is not edited once recorded) the body stays as
  written and this note carries the update; current numbers are the README's conclusion and the
  code's defaults.

  **Verified 2026-09-23: the defaults now match, and that was checked.** This note says "current
  numbers are the code's defaults", but the default annealing start was **2** while the acceptance
  batch ran with **1** -- inconsistent. The default is now 1, and re-running
  `cartpole.py --seed 45 --device cpu` gives **232.1 steps**, byte-identical to seed 45's 232.1 in the
  acceptance batch (`sweep_results/step14_confirm`). So "the default run is the acceptance
  configuration" is now verifiable rather than asserted.

## Alternatives

- **Action mixture `pi = (1−kappa)·onehot(argmax) + kappa/n`** (a fixed kappa decoupling policy
  sharpness from the minority-action sampling rate): **rejected**. The adversarial review showed
  that at kappa=0.2 the distribution `(0.9, 0.1)` is the **same distribution** as epsilon-greedy
  with epsilon=0.2, and both deploy argmax — while the boltzmann branch **already has no gate**, so
  the only thing it adds over the status quo is the evaluation fix, which does not need it.
  Moreover its own pass line (per-state cosine > 0.95) is **mathematically unreachable** under its
  own rule (identically `1/sqrt(2)`).
- **Split the Critic by action (two Q populations driving the Actor with `A = Q − V`)**:
  **rejected**. Frémaux's Actor rule already states that only those directly responsible for the
  action choice are reinforced, so this is not a departure; and replacing delta with A **changes
  only the modulator, not the trace**, so the Actor's credit structure is bit-identical. Its family
  has also already been measured (Boltzmann without centring: paired −26.0, 2/10 improved).
- **The paper's Eq. 37 performance-driven learning-rate schedule**: **deferred**. The review showed
  its existing form is missing a factor of 7, giving `eta/peak ∈ [0.871, 1.000]`, so the mechanism
  ("when performance collapses, the step grows again") **cannot occur** in the implementation. Even
  with the factor restored, the late-time eta stays above the 3e-4 endpoint of the only effective
  lever. Separately, the paper's `eta_act = 0.5·eta_crit` (Critic at twice the Actor) directly
  conflicts with this repository's "speeding up the Critic is a 0/10 catastrophe", and needs its own
  inquiry and ADR.
- **A fixed signed readout (`readout="random"`)**: **rejected**. Frémaux's Eq. 13 defines the value
  as the population's **average firing rate** — `readout="uniform"` *is* the paper, and the signed
  readout is the departure. It did move explained variance from −1.00 to −0.21 and halve the mean
  absolute error, but **0/10 seeds improved** (paired −50.7; still −40.6 after compensating sigma_R).

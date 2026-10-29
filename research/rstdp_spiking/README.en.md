# Spiking execution layer: spiking R-STDP on CartPole

[中文](README.md) · [日本語](README.ja.md)
**⚠️ Breaking-change grace period: until 2027-09** (plan section 12.5, ADR-0005).

## What it tests

Phase three's **spiking execution layer** (plan section 7): replace the rate-based actor-critic in
`research/rstdp/` with a spiking implementation (LIF + **eligibility traces with a real Δt window**
+ a reward-predicting critic), so that the structural difference between the `STDP(Δt_ij)` formula
in section 3.2 and "pre-before-post only" becomes **testable for the first time**.

The rate version stays as the **control baseline** (same task, same rule, spiking vs rate). It
answers "how much performance does spiking cost".

## ⚠️ 2026-10-29 measurement re-check: `best_steps` is not "current level"

The "best evaluation" figures on this page take the **maximum over 32 evaluation points (10 episodes
each)** — the best moment seen in training, not the level at the end of training. Recomputed on the **last evaluation**, the 26 historical
arms have medians of only **9.2–18.3 steps** (the highest, sign-consistent, is 18.3), while D1
rate-features goes 62.15 → 13.05 and the D2 linear policy 179.8 → 9.6. **No arm holds its level at the end.** The preregistered verdicts (all failures) are unchanged, but
do not use the `best` figures on this page to write "just short of the line". ⚠️ The best peak is
not noise: diagnosis ③ re-evaluated the best snapshot on 20 fresh eval seeds and got medians
73.8 / 76.5 (lr=1e-4/TD) — the problem is end-of-training degradation, not measurement. See
[`sweep_results/stage3_sustained_audit.md`](../../sweep_results/stage3_sustained_audit.md).

## ⚠️ Every number on this page dated before 2026-10-18 is void

Training episodes did not receive an environment seed (`env.reset(seed=None)`, whose RNG is seeded
from system entropy), so **the same seed produced a different training trajectory on two runs** —
the evaluation itself was seeded, but the weights being evaluated came from unreproducible
training. Discovery, fix and the full impact list:
[`sweep_results/rstdp_spiking_seeding_fix.md`](../../sweep_results/rstdp_spiking_seeding_fix.md).

Re-running with five seeds after the fix **reversed the ranking wholesale** (linear TD 109.8 → 51.0;
the original default configuration came out on top). The sections below are kept as history; their
numbers and the conclusions drawn from them **no longer hold**. New conclusions rest on the
10-seed comparison at the end of this page.

### 2026-10-29 diagnostic conclusion: entropy collapse, not the readout

- **D3 (rate readout)**: best median 65.2, last 9.55 — changing the readout did not move the
  end-of-training level;
- **D4 (state dependence)**: after training the action is identical at pole angles ±0.2
  (slope 0 for all 10 seeds);
- **D5 (collapse vs silence)**: final entropy <0.10 nats in **10/10** seeds, silent probe windows
  **0/10** — this is entropy collapse; by episode 100 entropy is already 0.0001–0.026;
- **Mechanism**: policy logits are 25-step spike counts and sampling is
  `softmax(logit_scale×counts)`, while the spiking actor's `logit_scale` is fixed at 1.0
  (the temperature schedule previously applied only to the linear policy);
- **D6**: three-arm temperature experiment (1.0 control / 0.1→0.02 / 0.03→0.01), criteria in
  [`sweep_results/rstdp_spiking_temperature_prereg.md`](../../sweep_results/rstdp_spiking_temperature_prereg.md).

### Records after the fix (reproducible, 2026-10-18)

| Configuration | Mean | Per seed |
| :--- | ---: | :--- |
| population, `gradient` + uniform + norm (original default `b=0.4, g=8`) | **82.2** | 65.6 / 81.3 / 55.1 / 61.3 / 147.5 |
| population, `gradient` + uniform + no norm | 60.5 | 44.6 / 69.0 / 73.9 / 58.1 / 56.9 |
| population + unit homeostasis | 53.7 | 36.0 / 101.8 / 69.8 / 40.1 / 20.6 |
| linear TD (1200 episodes) | 51.0 | 61.5 / 156.9 / 17.3 / 9.3 / 9.9 |
| population, operating point `b=0, g=16` | 40.0 | 64.6 / 54.7 / 35.2 / 15.8 / 29.7 |
| population, `rate` + uniform + norm | 37.2 | 9.2 / 80.9 / 64.0 / 20.4 / 11.7 |
| population, fixed input subspace `p=0.5` | 37.0 | 79.1 / 51.5 / 30.9 / 9.2 / 14.1 |
| population, signed readout | 27.7 | 71.8 / 9.2 / 22.9 / 25.4 / 9.3 |
| population, operating point + signed readout | 21.2 | 9.2 / 9.2 / 26.5 / 51.5 / 9.4 |
| population, signed + no norm | 17.2 | 9.5 / 9.2 / 49.0 / 9.2 / 9.3 |

⚠️ **The ranking of these means is not trustworthy**: individual seeds range from 17 to 157, and the
standard error at five seeds is the same order as the gaps between configurations. The table exists
to show which old conclusions reversed, not to rank configurations; a meaningful comparison needs ten
seeds and overlapping intervals.

**Three old conclusions refuted**: (1) "linear TD is the only critic that learns" (51.0, 0/5);
(2) "removing normalisation is better" (60.5 without vs 82.2 with); (3) "the `b=0, g=16` operating
point is better" (40.0, below the original default).

### Ten-seed comparison: the three main arms are indistinguishable (2026-10-18)

| Configuration | Median | IQR | Min–Max | Mean | Over 200 |
| :--- | ---: | :--- | :--- | ---: | ---: |
| linear TD (1200 episodes) | 50.4 | 12.9 – 128.9 | 9.3 – 156.9 | 64.8 | 0/10 |
| population, `gradient` + uniform + norm (default) | **55.0** | 36.0 – 65.6 | 23.3 – 147.5 | 59.8 | 0/10 |
| population, operating point `b=0, g=16` | 37.1 | 29.7 – 54.7 | 9.5 – 67.8 | 40.1 | 0/10 |

**Conclusion: all three intervals overlap, no configuration is distinguishable, and none reaches
200.**

* The linear TD median (50.4) cannot be told apart from the population default (55.0); its interval
  is in fact the **widest** (12.9–128.9), meaning this critic behaves very unstably on this harness.
* The `b=0, g=16` operating point (37.1) sits below the default (55.0) with overlapping intervals —
  the earlier "the operating point is better" claim fails under both readings.
* The **maxima** (156.9 / 147.5 / 67.8) show how unstable the task is with this hyperparameter set:
  occasionally ~150, most of the time 10–60.

⚠️ **Every number and mechanism explanation in the earlier (2026-10-17) sections below came from
unseeded training**; they are kept only to record what was thought at the time, not as evidence.

## Historical result (2026-10-17, five seeds, 1200 episodes each, **unseeded, void**)

**It learns, but unstably: only 1 of 5 seeds crosses the 200-step bar.**

| Seed | Best evaluation (steps) |
| :--- | ---: |
| 0 | 102.8 |
| 1 | 39.0 |
| 2 | 19.6 |
| 3 | 138.5 |
| 4 | **249.0** |
| **Mean** | **109.8** |

Controls:

| Implementation | Result |
| :--- | :--- |
| rate version (`research/rstdp/`) | median **237.8 steps**, acceptance passed |
| spiking + EMA baseline (no critic) | 65.3 steps (one seed, 300 episodes) |
| spiking + linear TD critic | **109.8 steps mean**, 1/5 over the bar |

Two claims the data supports:

1. **The critic is structural, not a tuning detail.** Replacing reward prediction with an
   exponential moving average of returns leaves the same actor at 65 steps; adding the linear TD
   critic lifts the mean to 109.8 — consistent with what section 3.2 quotes from Frémaux et al.
   (R-STDP needs stimulus-specific reward prediction).
2. **Spiking costs stability.** The rate version's median is 237.8 and it passes; the spiking
   version peaks at 249 but bottoms out at 19.6. Much larger variance, a direct consequence of
   "the same weights give different actions on two runs" (see the evaluation protocol below).

## How to reproduce

```bash
uv run pytest research/rstdp_spiking -q
uv run python -m research.rstdp_spiking.cartpole --smoke
uv run python -m research.rstdp_spiking.cartpole
uv run python -m research.rstdp_spiking.cartpole --seeds 0,1,2,3,4 --episodes 1200 --learning-rate 1e-4
uv run python -m research.rstdp_spiking.diagnose_critic
```

**Evaluation protocol: the random stream is frozen.** A spiking network's output is itself random
(the input is a Bernoulli spike code), so the same weights produce different actions on two runs.
Every evaluation episode therefore resets the environment with a fixed seed and re-draws the input
spikes with a fixed seed — same weights + same seed → same action sequence. Without this the
"greedy" evaluation measures a random policy (the rate version hit the same trap in its
`boltzmann` branch).

## The TD-LTP population critic: implemented, but it does not learn (added the same day)

Following the structure in plan section 3.2, the critic was replaced with a **population +
fixed readout** design (`pop_critic.py`): unit activity `y_j = σ(g(w_j·x − b))`, value
`V = Σ u_j y_j + out_bias`, eligibility `e_ji ← λ e_ji + x_i y_j`, update
`w_j ← w_j + η δ e_j` followed by row normalisation. **The output bias is learnable** — the rate
version's README records that disease: near death the true value is ≈0 while a population readout
cannot reach that low, so the terminal step gets a reliably large negative δ.

**The result is negative.** Same actor, different critic:

| Configuration | Best evaluation (steps) |
| :--- | ---: |
| linear TD (five seeds × 1200) | **109.8 mean, 1/5 over the bar** |
| population, actor lr 1e-4, critic 1e-2 | 9.2 |
| population, actor 1e-3, critic 1e-2 (no output bias) | 28.9 |
| population, actor 1e-3, critic 1e-2 (with output bias) | 20.8 |
| population, actor 1e-3, critic **1e-1**, 800 episodes | 78.4 |
| population, **same, 1200 episodes** | **9.3** |
| population, critic 5e-2 / 2e-1 / 3e-1 / 1.0 | 27.2 / 18.0 / 9.2 / 9.2 |
| **population, five seeds at critic 1e-1, 800 episodes** | **36.2 / 12.9 / 9.4 / 40.8 / 15.7 — mean 23.0, 0/5** |

### Ablation: porting the omitted mechanisms shows the problem is elsewhere (same day)

The earlier version's hypothesis was "the mechanisms this port omitted are load-bearing". After
porting them, **the hypothesis refuted itself** — each one toggled separately, five seeds × 800
episodes, everything else fixed:

| Variant | Mean best | Per seed |
| :--- | ---: | :--- |
| `rate` post-factor + uniform readout + row norm (original) | 23.0 | 36.2 / 12.9 / 9.4 / 40.8 / 15.7 |
| `gradient` post-factor + uniform + norm | 32.9 | 61.7 / 14.3 / 9.2 / 69.8 / 9.7 |
| `gradient` + **signed readout** + norm | 14.3 | 9.2 / 9.2 / 34.4 / 9.2 / 9.7 |
| `gradient` + uniform + **no row norm** | **56.8** | 69.2 / 37.3 / 104.9 / 63.1 / 9.3 |
| `gradient` + signed + no norm | 41.3 | 49.2 / 9.2 / 105.1 / 33.9 / 9.3 |
| reference: linear TD (not a population) | 109.8 | 102.8 / 39.0 / 19.6 / 138.5 / 249.0 |

Readings:

1. **The `gradient` post-factor helps a little** (23.0 → 32.9); **the signed readout hurts**
   (32.9 → 14.3, and 56.8 → 41.3 without normalisation).
2. **The biggest single factor is row normalisation**: removing it takes 32.9 → **56.8**. That
   normalisation was copied from the rate version, where it prevents divergence; on this line it
   flattens the dynamic range of the value.
3. **Porting everything still misses 200, and does not even reach the linear TD critic's 109.8**
   (0/5 against 1/5).

### Operating point: from "almost all dead" to "tracks the return" (same day)

The previous section measured 76% of units below 0.05 activity — the bias `b = 0.4` is an order of
magnitude larger than the scale of the pre-activation it receives (0.16). Pre-registration:
[`rstdp_spiking_operating_point_prereg.md`](../../sweep_results/rstdp_spiking_operating_point_prereg.md).

Screening (one seed, 300 episodes, `gradient` and no normalisation fixed):

| bias | gain | units | best evaluation |
| ---: | ---: | ---: | ---: |
| 0.4 (current) | 8 | 64 | 73.5 |
| 0.0 | 8 | 64 | 42.2 |
| −0.2 | 8 | 64 | 33.0 |
| **0.0** | **16** | **64** | **123.4** |
| 0.0 | 8 | **256** | 9.2 |

**What works is the gain (stretching `z` to span ±2), not the unit count** — H1 (operating point)
holds, H2 (capacity) does not.

Confirmation (five seeds × 800 episodes, `b = 0, g = 16`): **mean 64.8** (previous best 56.8),
per seed 44.4 / 89.4 / 33.2 / 72.4 / 84.6. ⚠️ **Still 0/5**, but **the worst seed rises from 9.3
to 33.2** — markedly more stable.

The direct criterion, `corr(V,G)`: **−0.496 → +0.697** (linear TD is +0.965). The critic now
**tracks the return**.

### Homogenisation: the hypothesis holds, and worse than expected (same day)

Pre-registration:
[`rstdp_spiking_homogenisation_prereg.md`](../../sweep_results/rstdp_spiking_homogenisation_prereg.md).
Measurement script: `diagnose_homogenisation.py` (800 episodes, `b=0, g=16`, `gradient`, no norm).

| Quantity | Before | After | Criterion |
| :--- | ---: | ---: | :--- |
| mean unit-to-unit cosine | −0.002 | **+0.823** | chance sd is 0.125 ⇒ about 6.6σ, past the pre-registered 5σ threshold |
| effective rank `exp(H)/n` | 0.613 | **0.025** | homogenisation ⇒ falls |
| pairwise activity correlation | — | **+0.928** | homogenisation ⇒ rises |
| mean activity | — | **0.003** | the units are almost silent |

**After training the 64 weight vectors point in nearly the same direction** and the activities
collapse to zero together. Hence `V = Σ u_j y_j + out_bias ≈ 200 × 0.003 + out_bias`: the value is
carried almost entirely by one scalar bias (inferred ≈ 15–16, consistent with the measured
`V ∈ [14.4, 17.6]`). That is the mechanism behind "`V` only 3.2 wide, bootstrapping dead".

The inference holds: with a `gradient` post-factor the trace is always non-negative (`x ≥ 0`,
`σ' > 0`) and the readout is all positive, so their product pushes every unit's weights toward the
**same non-negative direction**, and the units converge.

### Unit-level homeostasis: units come alive, homogenisation stays (same day)

Pre-registration:
[`rstdp_spiking_homeostasis_prereg.md`](../../sweep_results/rstdp_spiking_homeostasis_prereg.md).

Rule: `b_j ← b_j + β(y_j − y*)`, using **only that unit's own activity and a constant target** —
it reads no other unit and does not pass through `δ` (homeostasis is a second local rule running
in parallel with TD-LTP).

Screening (one seed, 300 episodes): `y*=0.05, β=0.01` was best (69.0); the others gave 30.6 / 26.9 /
25.7. ⚠️ **Screening pointed the wrong way**: at the same 300 episodes the non-homeostatic
configuration scored **123.4**, so homeostasis looked worse; by 800 episodes the non-homeostatic one
falls to 64.8 while the homeostatic one holds at 69.4 — **the one without homeostasis degrades over
time, the one with it does not**.

Confirmation (five seeds × 800, `y*=0.05, β=0.01`): **mean 69.4**, per seed 72.9 / 87.1 / 57.8 /
52.4 / 76.6, against 64.8 (33.2–89.4) without. **The worst seed rises from 33.2 to 52.4 and all five
are ≥ 52** — markedly more stable. **Still 0/5.**

### The mechanism is not broken: activity level is not the cause of homogenisation

Same measurement (800 episodes):

| Quantity | No homeostasis | Homeostasis |
| :--- | ---: | ---: |
| unit-to-unit cosine | +0.823 | **+0.778** |
| effective rank | 0.025 | **0.024** |
| mean activity | 0.003 | **0.089** |
| `V` range | 14.4 – 17.6 | **11.7 – 202.4** |
| mean unit threshold | 0 | −5.15 |

**The units are pulled back into a working range (0.003 → 0.089) and the weights converge all the
same (cosine +0.78, effective rank 0.024).** Silence is therefore not the cause of homogenisation —
the common-mode push comes from the sign structure of the trace and is independent of activity
level.

⚠️ **`V` hit the `value_scale=200` ceiling** (202.4 measured). The pre-registration says: record
that configuration as failed and **do not secretly change the scale**. This also explains why the
score did not follow: δ is contaminated by a clipped signal.

### Signed readout: retested at the fixed operating point, still harmful

The earlier 14.3 came from the **saturated** operating point, where the units barely moved and the
sign had no room to act. The pre-registration required a retest. Result:

| Readout | Mean | Per seed |
| :--- | ---: | :--- |
| uniform (`b=0, g=16`) | **64.8** | 44.4 / 89.4 / 33.2 / 72.4 / 84.6 |
| signed (same operating point) | **29.2** | 43.9 / 9.2 / 74.2 / 9.3 / 9.3 |

**Still worse, and less stable** (three seeds fall back to chance). So "flip the readout sign to fix
homogenisation" is refuted — consistent with the reading that the sign changes how the readout
combines units, not the fact that **the weights themselves converge**.

### Still unresolved: training collapses the value's dynamic range

The same diagnostic reports `V ∈ [14.4, 17.6]`, **only 3.2 wide**, while returns span 1–10.5. A
short run (2 episodes) still shows a 3.9–36.8 spread; 800 episodes collapse it into a narrow band.

Consequence: in `δ = r + γV(s′) − V(s)` the differences in `V` are about a third of the reward
scale, so `δ` is dominated by the reward term and bootstrapping barely contributes. This is
consistent with "with a `gradient` post-factor the trace is always non-negative and the readout is
all positive, so weights get pushed the same way and units homogenise" — **the most explanatory
next hypothesis**, but it needs its own pre-registration and is not part of this round's
conclusion.

### Diagnosis: the population critic never learns the value function

`diagnose_critic.py` freezes the policy after training and compares `V(s_t)` with the Monte-Carlo
return `G_t`:

| Critic | corr(V, G) | V range | G range |
| :--- | ---: | :--- | :--- |
| linear TD(0) | **+0.965** | 0.8 – 17.2 | 1.0 – 10.5 |
| population (gradient, row norm) | **−0.496** | **18.3 – 19.9** | 1.0 – 10.5 |
| population (gradient, no norm) | +0.217 | 12.0 – 18.2 | 1.0 – 27.5 |

The population critic's `V` wanders inside a range 1.6 wide and is anti-correlated with returns —
**the actor is being fed noise**, which is why changing the readout or the step size does not
rescue it. Removing normalisation moves it from "anti-correlated" to "weakly correlated": the
right direction, still far from +0.965.

⚠️ **Do not read this as "the TD-LTP population critic is infeasible"**: it says that **this
implementation**, on **this harness**, does not learn. With the three known suspects
(normalisation, gradient scale, readout sign) ruled out, what remains is "the population's capacity
and initialisation do not match this 64-dimensional population code" — which needs its own
pre-registered round.

## Actor trace rule: new `sign-consistent` option (2026-10-26)

The default rule is **unchanged** (`pair-stdp`, bit-for-bit identical to every recorded number).
A new explicit switch, `--trace-rule sign-consistent`, keeps the real-Δt LTP/LTD but accumulates
two non-negative traces only for the selected action's column, then picks the phase by the sign
of the success signal at episode end (pre-registration:
[`sweep_results/rstdp_spiking_actor_fix_prereg.md`](../../sweep_results/rstdp_spiking_actor_fix_prereg.md);
amendment:
[`..._amend1.md`](../../sweep_results/rstdp_spiking_actor_fix_prereg_amend1.md)).

The problem A1 measured (the selected action's eligibility column sum was larger in only 7 of
20 windows, and negative in 3) becomes **18/18** in A1′ (two silent windows excluded), negative
20/20, gating 20/20. But the 10-seed paired Stage 2:

- paired median **+17.95 steps** (bar +20, not met; bootstrap 95% CI −0.30 … +29.20);
- **8/10** seeds improve (this one passes);
- median **75.65** steps, 0/10 at 200 (acceptance fails).

Conclusion: **sign consistency fixed the trace, but it is not the (only) bottleneck**; the next
suspect is the critic / credit horizon. Full table and provenance:
[`sweep_results/rstdp_spiking_actor_fix/README.md`](../../sweep_results/rstdp_spiking_actor_fix/README.md).

A2 (oracle EMA) also failed; D1 (`rate-features`, replacing the eligibility with the rate-style
`x·a`) gives a 10-seed median of **62.15**, below sign-consistent's 75.65 (3/10 better), so
**eligibility shape is not the blocker**; the next suspect is **spike encoding / policy
parameterisation** (the action is still chosen by spike counts). See
[`sweep_results/rstdp_spiking_credit_diagnosis/rate_features/README.md`](../../sweep_results/rstdp_spiking_credit_diagnosis/rate_features/README.md).

D2 (the linear-policy recipe, a sanity control) reaches a median of **179.80** in the same
harness (paired **+124.40** vs D1, better in 10/10, 4/10 over 200): **the harness can learn**;
the bottleneck is the **spike-count action path**. The next step is a local eligibility aligned
with the count-based policy, not switching to the linear policy. See
[`sweep_results/rstdp_spiking_policy_control/linear_rate/README.md`](../../sweep_results/rstdp_spiking_policy_control/linear_rate/README.md).

## Known boundaries

1. **The default critic is a linear TD(0) one.** The TD-LTP population critic is implemented
   (`pop_critic.py`, `--critic pop`) but does not currently learn (see the section above), so it is
   **an experimental entry point, not a usable default**.
2. **Single LIF layer only.** No multi-layer stacking, and the encoding layer (population coding +
   Bernoulli sampling) does not learn.
3. **No systematic hyperparameter sweep.** Four variants tried were all worse: window 10 steps →
   9.2, features 128 → 10.3, critic rate 1e-1 → 66.9, γ = 0.95 → 60.2 (one seed, 800 episodes
   each). The current configuration is a local optimum, **not** a proven best.
4. **Training length matters a lot**: the same configuration gives 83.9 at 400 episodes and 102.8
   at 1200 (seed 0).
5. **This is not a controlled comparison with the rate version — two protocols differ.**
   (a) *Evaluation*: the rate version is deterministic, while this one must freeze the random
   stream and average over episodes. (b) *Encoding*: the rate version draws population centres
   uniformly in `[-1.5, 1.5]` on all four axes with `--encoding-sigma` defaulting to 0.5, while
   this directory uses per-axis `CARTPOLE_HALF_WIDTHS` (±2.4 / ±3.0 / ±0.21 / ±3.0) and
   `--sigma` defaulting to 1.0. **The implementation is shared**
   (`research/components/cartpole_encoding.py`); only the parameters differ. The reported gap
   therefore mixes the cost of spiking, the evaluation protocol and the encoding parameters —
   **it is not attributable to a single variable**.

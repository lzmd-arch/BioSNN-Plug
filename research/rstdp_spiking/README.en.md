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

## Current result (2026-10-17, five seeds, 1200 episodes each)

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
5. **This is not a like-for-like comparison with the rate version**: its evaluation is
   deterministic, while the spiking one must freeze the random stream and average over episodes.
   The reported gap mixes the cost of spiking with the difference in evaluation protocol.

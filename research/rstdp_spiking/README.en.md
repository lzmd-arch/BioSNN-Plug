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
```

**Evaluation protocol: the random stream is frozen.** A spiking network's output is itself random
(the input is a Bernoulli spike code), so the same weights produce different actions on two runs.
Every evaluation episode therefore resets the environment with a fixed seed and re-draws the input
spikes with a fixed seed — same weights + same seed → same action sequence. Without this the
"greedy" evaluation measures a random policy (the rate version hit the same trap in its
`boltzmann` branch).

## Known boundaries

1. **The critic is a linear TD(0) one, not the TD-LTP population critic of section 3.2.** Both are
   stimulus-specific reward prediction, but the former is a single weight vector. Porting the
   population critic is remaining phase-three work.
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

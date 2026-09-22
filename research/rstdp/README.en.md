# W3: R-STDP execution layer + TD-LTP Critic

[中文](README.md) · [日本語](README.ja.md)

**⚠️ Breaking-change grace period: through 2027-09** (project plan §12.5, ADR-0005).

## Current conclusion

**All three criteria are met.**

| Criterion | Seeds 45–64 (**fresh, never involved in any selection**) | Seeds 25–44 (used for tuning) | Threshold |
| :--- | ---: | ---: | :--- |
| CartPole, 200 steps or more | median **237.8** | 327.9 | ≥ 200 ✓ |
| Success offset | max **0.0403** | 0.0359 | < 10% sigma_R ✓ |
| (our own added "stable" condition) minimum | **121** | 109 | ≥ 100 ✓ |

**Both seed batches pass**, and **no seed falls below 100**. To be honest about it, a gap of about
27% remains between the two (327.9 → 237.8), so **a selection effect is real** — but both batches sit
above 200 at the median and above 100 at the minimum, and that is the part that matters.

**Acceptance configuration** (now the defaults, see [ADR-0009](../../docs/adr/ADR-0009-w3-behaviour-policy-and-trace-centring.en.md)): Boltzmann behaviour policy +
trace centred by the sampling probability + inverse temperature **annealed in log space from 2 to
40** + Actor learning rate decayed to **0.01x**. So `cartpole.py --seed 0` runs **exactly that**.

Getting here, **most of the hypotheses along the way were refuted** — and those negative results are
just as much a product of this phase: they pinned down "not this cause" one at a time, and what was
left was the real one (the ledger is the table below and the "Known boundaries" section). The failure
mode also changed twice: first **oscillation**, then, after the decay was added, **a mediocre fixed
point**, and finally what remained was a "slow to learn" tail.


### Acceptance rule (fixed before measuring)

Per-seed paired difference: **median > 0 and ≥7/10 seeds improve**. Baseline is
`actor_lr_final_fraction=0.1` (10-seed median 100.0). Multiple seeds are mandatory: within one
configuration, seeds differ by 5–30×, so single-seed conclusions are unusable here.

### What was tried, and what happened

| Lever | Paired median | Improved | Verdict |
| :--- | ---: | ---: | :--- |
| **Actor learning rate 3e-3 → 1e-3** | — | — | Works, but only reaches 93.7 |
| Actor learning rate 3e-4 / 1e-4 / 3e-5 / 1e-5 / 3e-6 | −26 … −65 | ≤4/10 | No effect |
| Encoding width σ ≥ 1.0 | — | — | Catastrophic (median 9.4) |
| Training budget 2400 / 400 / 200 / 100 | −23 … −55 | ≤3/10 | No effect (less is worse) |
| Actor L1 normalisation off (scale-matched) | −1.6 | 5/10 | No effect |
| Trace second factor → softmax | −88% | 0/5 | Catastrophic |
| Trace second factor → `a_j − 1/n` | — | — | No effect by construction (row-common vector, argmax-invariant) |
| Critic learning rate 2e-3 / 8e-3 / 3e-2 | −69 … −66 | **0/10** | Catastrophic |
| Critic learning rate 1e-4 | −2.4 | 5/10 | No effect |
| Critic units 256 / 1024 | −3.9 / +2.2 | 3/10 / 5/10 | No effect |
| Critic gain 4; threshold 0.6 / 0.8 / 1.0 | −37 … +6 | ≤5/10 | No effect |
| Critic output bias (learnable / fixed −20) | −6.3 / +0.3 | 4/10 / 5/10 | No effect |
| Critic semi-gradient post-factor `y(1−y)` | −7.2 | 5/10 | No effect |
| Actor signal clip | — | — | **Wrong grid**: all three values sat below σR≈4.8, so it swept a smaller effective learning rate again |
| Actor Polyak averaging τ=0.99 | −25.9 | 2/10 | No effect (flattens the wandering and the good excursions alike) |
| Behaviour policy → Boltzmann + trace centring | +204.0 | 9/10 | **adopted** (the earlier "no effect" was polluted by the evaluation defect, see defect 5) |
| Critic fixed **signed** readout + semi-gradient | −50.7 | **0/10** | No effect — but it works at the mechanism level, see below |

### Two decisive measurements

**One: the policy wanders throughout; the final value is one phase sample of that wander.**
Greedy evaluation every 50 episodes:

```text
frac=1.0 seed 3:  78 374 176 38 39 12 9 12 96 83 18 39 94 47 22 9
frac=0.1 seed 0:  24 30 29 32 11 10 24 21 32 94 92 428 226 130 111 61
frac=0.1 seed 9:  12 9 9 56 10 34 19 40 54 37 31 28 31 21 244 88
```

**All twenty** curves look like this, with adjacent checkpoints differing by 5–30×. So the
acceptance statistic — the mean of 10 greedy episodes on the final weights — largely measures
**which phase the wander stopped in**, not the level learned. The median peak is 158.0, **also
short of 200**.

**Two: making the Critic genuinely more accurate makes the policy worse.** The pre-registered
fork rule said "EV < 0.5 ⇒ the Critic is the bottleneck". That inference **was refuted by its own
intervention** — see below.

### Why the Critic cannot fit, localised to the readout

Setting the actor learning rate to 0 gives a **completely stationary** target, which rules out
non-stationarity: EV(V, V\*) is still −1.00 / −0.51 / −0.81. Nor is it step size — lowering the
critic learning rate makes EV monotonically worse (5e-4 → 1e-4 → 2e-5 gives −0.75 → −1.30 → −6.45).

Replacing the fixed readout with a **least-squares solution**, changing nothing else, moves EV on
the **same unit activities** from −0.9 to **+0.6**: the features carry the information, and the
"fixed, all-positive, uniform" readout throws it away. (Control: **untrained** random directions
already score +0.53, so training the unit directions buys very little.)

So I added `critic_readout="random"` (fixed but **signed**) together with the complete
semi-gradient post-factor `u_j·g·y_j(1−y_j)`. **It works at the mechanism level**: EV −1.00 →
−0.21, mean absolute error 10.8 → 5.4.

**But 0/10 seeds improved** (paired median −50.7). The reason is a measurable confound: a signed
readout turns `V = Σu_j y_j` into a cancelling difference, dropping **σR from 4.30 to 1.40** —
which shrinks the calibrated Actor step by 3×.

**The step-compensated control has now finished, and the conclusion holds.** Scaling the Actor
learning rate by the σR ratio to 9e-3 (`3e-3 × 4.30 ≈ 9e-3 × 1.40`, whose product is the
effective step) still gives a paired median of **−40.6, 4/10 improved**, median 45.9. **So
"making δ more accurate does not make the policy better" survives the compensation** — the
pre-registered fork inference (EV < 0.5 ⇒ the Critic is the bottleneck) was refuted by its own
intervention.

### What is now established

- **The Actor's rule never compares two actions, and the target it faces is identically zero.**
  `learn_step` only updates the Actor on non-exploring steps, and on those `action = argmax` is a
  deterministic function of the state — so the update lands on **the chosen column only** (measured
  `U touches one column in 100% of states`; a discrete fact, no threshold involved). Deeper still,
  `A(s, pi(s)) ≡ 0` is an **identity** under a deterministic policy, so the
  "advantage-weighted direction" target is **identically zero at the states the policy visits**
  (measured `per-state |T| median = 0.0000`) — because at the states a good policy visits the pole
  is near upright and **the two actions are nearly equivalent**. So the rule does not face a wrong
  direction; it faces **no target at all**: the chosen column receives drift from an action-blind
  residual delta.
- **The rule never compares actions.** `learn_step` only updates the Actor on non-exploring
  steps, and on those `action = argmax` is a deterministic function of the state — so it only ever
  reinforces the action it just happened to take. That is also why centring by `a_j − π_j`
  **on its own freezes the Actor completely** (`a_j − π_j ≡ 0`); an assertion in the code blocks it.
- **Boltzmann + centring is the only lever this phase to pass the adoption rule** — once the
  evaluation defect is fixed. The earlier "no effect" record was polluted (see item 5 of "Five real
  defects fixed this phase"): re-measured with a truly greedy evaluation, constant logit 20 gives
  median 283.8 (selection seeds) / **177.3** (held-out 10–24), paired **+204.0**, 9/10 improved.
  **It still falls short of 200, and roughly one seed in ten collapses** (peak only 10–12 steps: it
  never starts, rather than learning and losing it).
- **Capacity is not the bottleneck**: a supervised linear policy on the same encoding reaches 497 steps.
- **The Critic's value range is not the bottleneck**: V's width is 0.73–0.84 of V*'s. The shape is wrong.

### Five real defects fixed this phase

1. **Truncation was treated as termination.** An episode reaching the 500-step cap took
   `δ ≈ 1 − 99.3 = −98` on its last step, about 56× a typical weight, and **the better the policy
   the harder it hit** — the most direct explanation of "peak 500 → collapse to 14". After the fix,
   the first 5 seeds of the undecayed arm are **bit-identical** to before, so the change is scoped
   exactly to the truncation path.
2. **The observation perturbed the observed.** Mid-training greedy evaluation advanced the
   training environment's own RNG stream, changing the initial state of every later episode
   (same config and seed: seed 6 went from 107.3 to 335.4). Fixed by using a separate env instance;
   now bit-identical.
3. **The Critic's direction-sampling pool silently capped `critic_units`** (~850–960): with
   `critic_units=1024` all ten seeds failed.
4. **A sign argument of mine was wrong.** "The readout must be all-positive" only holds for the
   `rate` post-factor; with `gradient`, the sign is carried by `u_j` itself.
5. **The `boltzmann` branch ignores `exploration`, so its "greedy" evaluation measured a stochastic
   policy.** `select_action` samples from the softmax unconditionally under
   `action_sampling="boltzmann"`, while `_greedy_score` (used by both the mid-training curve and the
   final acceptance run) passes `exploration=0.0`. The ceiling of that scale is brutally low: a
   **perfect** policy (the heuristic, 500 steps) is worth only 97.4 steps at 70% action accuracy and
   36.3 at 60%. The cost was concrete — it made this phase's only effective lever read as "no effect"
   and that reading went into the docs. The fix is `select_action(greedy=True)`, which returns
   argmax **regardless of the sampling scheme**; under epsilon-greedy it is bit-identical to the old
   path, so every recorded number stands.

### Before the change (single-unit Critic, kept as a control)

`--critic-kind single` reproduces the earlier behaviour:

| seed | steps | \|offset\|/σR |
| ---: | ---: | ---: |
| 0 | 9.3 | 0.9676 ✗ |
| 1 | 23.1 | 0.4364 ✗ |
| 2 | 37.3 | **0.0530** ✓ |
| 3 | 9.4 | 0.7571 ✗ |
| 4 | 9.5 | 1.3705 ✗ |

No seed reached 200 steps; a random policy gets about 9–10 steps on CartPole. **The seeds with the
largest offset were exactly the ones that failed to learn** (0.97 / 0.76 / 1.37 → 9 steps) —
directionally consistent with §3.2, though the quantitative threshold boundary was not reproduced.

Provenance record (`--seed 0`, generated by `research/common/provenance.py`):

```text
实验名称：rstdp/cartpole
日期：2026-09-23 05:21:05 中国标准时间
Git commit：75bc258e2e6e1745ac1ee706faef1377bacebf9f
Git 状态：干净
Python：3.12.14
操作系统 / 架构：Windows 11 / AMD64
硬件：CPU（Intel64 Family 6 Model 183 Stepping 1, GenuineIntel，无 GPU 参与）
随机种子：base=0；Actor 与 Critic 初始权重=3589114572；Critic 感受野采样=1786091376；动作探索=3138835151；感受野采样环境=2269638270；环境动作空间种子=4106697854；环境随机种子=2726622797；群体编码中心=4160090208
依赖快照：uv.lock sha256=cbafb161e71421f9f228e23e2ab6f4f742f899e9958d290f214dac1dd948e37f
运行命令：cartpole.py --seed 0 --device cpu
耗时：23.4 s
显存峰值：0.0 MiB（§6.2 硬约束 8GB）
§6.2 降级路径：未触发
备注：N=64, sigma=0.5, eta_actor=0.003, eta_critic=0.0005, trace_decay=0.9, gamma=0.99, success_signal=td_error
备注：Actor：normalize=True, clip=None, polyak_tau=None, sampling=boltzmann, logit_scale=20.0, trace_center=sampling, lr_final_fraction=0.1
备注：Critic：kind=population, units=64, value_scale=200.0, gain=8.0, threshold=0.4, output_bias=0.0, bias_lr=None, post_factor=rate
备注：探索：start=0.3, end=0.02, 回合数=800
备注：状态编码：4 维连续状态的高斯群体编码（本项目自己的选择）
备注：速率型单元：STDP 窗口形状与 TD-LTP/TD-STDP 的差别在此化简下无从体现
```

## How to reproduce it

```bash
uv run python -m research.rstdp.cartpole              # acceptance run (~4 seconds)
uv run python -m research.rstdp.cartpole --seed 2     # a different seed
uv run python -m research.rstdp.cartpole --smoke      # CI smoke
```

## Equations mapped to modules

| Paper | Content | Module |
| :--- | :--- | :--- |
| Frémaux 2013 Eq. 17 | TD-LTP: `Δw = η·δ·e`, `e ← λe + x·y` | `td_ltp.TDLCritic` |
| §3.2 | R-STDP: `Δw = η·S·e`, `e ← λe + x·a` | `rstdp.RSTDPActor` |
| §3.2 | Weight normalization | implemented in both classes |
| §3.2 / §七 | Success-signal offset / σR | `measure_bias` |

## Three structural bugs fixed while implementing this

Each has a regression test, and none of them could be worked around by tuning.

**1. The Critic's zero-lock.** The eligibility trace is `e ← λe + x·V`, gated by the
Critic's **own output**. With zero initialisation, `V ≡ 0` → the trace stays 0 → the update
`η·δ·e` stays 0 → the Critic can never learn. A lock-in that comes from the three-factor
structure itself, where the second factor is the postsynaptic activity.

**2. Positive feedback runs away.** With small random initialisation it moved, but raising
the learning rate made the weights diverge to NaN: `V` grows → `e = x·V` grows → `Δw` grows
→ `V` grows further.

**3. The Critic's weight normalization was missing.** §3.2 says plainly: "weight
normalization is introduced at the same time: keep the weight sum constant per neuron, to
prevent the synaptic dynamics from running away" — I had applied it only to the Actor.
Adding L2 normalization broke the loop and took CartPole from 9.5 to 84 steps. A
`value_scale` restores the overall gain that normalisation removes.

Two **measurement/reproducibility** defects were also fixed: the progress log's trace column
was read after `end_episode()` (so it was always 0); and **the CartPole environment was
never seeded** — the same seed run twice gave 20.4 and 9.3 steps. The latter matters for the
conclusion: before the fix, there was no way to tell how much of the "seed-to-seed
variation" came from the algorithm and how much from the environment's RNG.

## Capacity probe: a negative result

`research/rstdp/capacity_probe.py` uses CartPole's classic heuristic controller as a target to
measure **the capacity ceiling for a linear policy on this encoding** — the linear classifier's
form is identical to `RSTDPActor`'s action selection.

| Features N | σ=0.25 | σ=0.5 | σ=1.0 |
| ---: | ---: | ---: | ---: |
| 16 | 24.6 | 209.8 | 494.2 |
| 64 | 294.2 | **497.2** | 500.0 |
| 256 | 449.4 | 500.0 | 500.0 |
| 1024 | 498.4 | 500.0 | 500.0 |

(The table gives the fitted linear policy's mean survival steps; the expert heuristic itself
scores 500.0.)

**The current configuration is N=64, σ=0.5, which corresponds to 497.2 steps.** In other
words: a linear policy that nearly solves CartPole **exists** on this encoding, and R-STDP did
not find it.

So the "the Actor lacks capacity" hypothesis is **refuted**, and adding capacity is not the
answer. That also rules out what looked like the cheapest explanation.

## Before and after: why the population structure

The single-unit Critic treated "the unit's output" and "the value" as the same quantity, so its
eligibility trace was `e = x·V`. That produced three chained problems, all observed:

| Problem | Symptom | Why the population structure fixes it |
| :--- | :--- | :--- |
| Zero-lock | With zero init, `V ≡ 0` → trace stays 0 → never learns | `y_j = σ(…)` is 0.5 at 0, not 0 |
| Positive feedback | `V`↑ → `e=x·V`↑ → `Δw`↑ → `V`↑, weights diverge to NaN | the second factor is `y_j ∈ (0,1)`, **bounded** |
| Scale freedom removed | To plug the previous one I added `‖w‖=1`, so `V` pinned to its ceiling and `δ` became constant | the scale is carried by the **fixed readout**; `w_j` only sets the shape |

The numbers after the change are in the previous section: offset 1/5 → 5/5, steps 9–37 → 60–120.
**Those two numbers are all it takes to decide whether this change was worth making** — no need
to guess which hyperparameter is more sensitive.

## Known boundaries

1. **Acceptance is met** — both of §7's criteria and our own added stability condition; see
   Current conclusion. But the boundaries below still hold, and the most valuable part of this
   entry is **not the number** — it is the directions that were refuted one at a time: they ruled
   out what was not the cause, and what was left was the real one.
2. **The success signal's default was changed.** §3.2 writes `S = R − ⟨R⟩`, but CartPole pays a
   **dense, constant** reward of +1 per step, so `⟨R⟩ ≡ 1` and `S ≈ 0` — no contrast at all. That
   expression was written for **sparse terminal** rewards (Frémaux 2010's original setting). The
   default is now the TD error; both remain selectable. So this entry does **not** test the success
   signal written literally in the plan.
3. **Rate-based, not spiking.** With rates there is no "spike ordering", so the `STDP(Δt)` window
   shape degenerates into a same-instant product, and **the one substantive difference between
   TD-LTP and TD-STDP — whether there is a post-before-pre component — cannot be expressed here**.
   ADR-0007's alternative C (falling back to TD-STDP) is therefore not a switch in this
   implementation; falling back would first require a spiking implementation. **That fallback has
   not been tried.**
4. **The policy does not converge, and this is the most important boundary.** Greedy evaluation
   every 50 episodes shows **all twenty** trajectories oscillating over the whole run (adjacent
   checkpoints differ by 5–30×). So the acceptance statistic — the mean of 10 greedy episodes on
   the final weights — largely measures **a phase**, not a level; and the **median peak is 158.0,
   also short of 200**. Any conclusion here that reports a single final number should be discounted.
5. **"Actor capacity is insufficient" is refuted.** On the **same encoding**, a supervised linear
   policy reaches 497 steps.
6. **Making the Critic more accurate does not improve the step count — this refutes this phase's
   own pre-registered fork.** The rule said "EV < 0.5 ⇒ the Critic is the bottleneck"; EV was in
   fact negative (−1.0 … −2.0), so I fixed the Critic: a fixed signed readout plus the complete
   semi-gradient post-factor took **EV from −1.00 to −0.21 and the mean absolute error from 10.8 to
   5.4** — a mechanism-level success. **The result was 0/10 seeds improved** (paired median −50.7).
   **The confound is measured and compensated**: the signed readout drops `σR` from 4.30 to 1.40
   (shrinking the effective Actor step by 3×), and after scaling the Actor learning rate by the
   same ratio to 9e-3 the paired median is still **−40.6, 4/10 improved**. **So "fix the Critic
   first" is ruled out, not merely untried.**
7. **The Actor's rule never compares actions, and under a deterministic policy it has no target.**
   `learn_step` only updates the Actor on non-exploring steps, and there `action = argmax` is a
   deterministic function of the state, so the rule only reinforces the action it happened to take
   (`U touches one column in 100% of states`). More fundamentally, `A(s, pi(s)) ≡ 0` is an identity,
   so the advantage-weighted direction is **identically zero at the visited states**
   (`per-state |T| median = 0.0000`) — at the states a good policy visits the two actions are nearly
   equivalent. **Switching to Boltzmann + `a_j − pi_j` centring is the only lever this phase to pass
   the adoption rule** (held-out 177.3, paired +204.0), but it does not yet meet the criterion and
   has roughly a one-in-ten collapse rate. **Two claims formerly written here are retracted**: "the
   rule's direction is fine, cos = 0.74" (that cosine is noise when the target is near zero — seed 1
   gives 0.078) and "a step ratio of 20–36× means the step is too large" (it is a small denominator,
   not a large numerator).
8. **Retuning must be redone after environment seeding.** The earlier "hyperparameter-sensitive"
   conclusions (e.g. "critic learning rate 1e-3 versus 5e-4 is 44 steps versus 223 steps") **were
   measured before the environment was seeded and are therefore untrustworthy** — back then the
   same configuration run twice could differ by a factor of two. Most of this entry's earlier
   tuning judgements have been retracted.
9. **One grid-design mistake, recorded here.** The Actor signal-clip values were {0.3, 1.0, 3.0}
   while the measured median `σR` is 4.78 — so all three sat **below the signal's own standard
   deviation**, and that round actually swept a smaller effective learning rate rather than
   trimming the tail. To test the latter, the threshold must sit above `σR`.
10. **The single-unit Critic's two pathologies are located and fixed, but they only explain part of
    the earlier behaviour.** Its eligibility trace was gated by its own output (zero lock), its
    update was proportional to `V` (positive feedback → divergence), and the normalisation I added
    to stop that removed its scale freedom (`V` pinned to its ceiling → `δ` became a constant).
    The population structure removes all three. But replacing the single-unit version with standard
    TD(λ) also only reached 21–40 steps — so **other factors were already at work then, and those
    factors are most likely still present**.
11. **§3.2's quantitative boundary was not reproduced.** "An offset of ~25% σR prevents learning"
    requires a task designed to keep the action distribution mixed in order to measure cleanly;
    `examples/paper_fremaux2013.py` deliberately does **not** demonstrate it, because my first
    version was actually measuring "which action the actor happened to pick", not the bias structure.
12. **Seed variance is enormous, and it is dynamical rather than statistical.** CartPole's dynamics
    are deterministic; the randomness comes only from the initial state, exploration, and the
    initial weights. A 10-seed median still has a wide confidence interval, so every
    pass/fail here is reported over ≥10 seeds with **min and max always given**.
13. **CPU and CUDA numbers are not comparable** (different floating point and random streams), so
    every sweep is pinned to CPU.

### Structural directions not yet tried

Ordered by strength of evidence; all remain within "no surrogate gradients, purely local":

1. **Split the Critic by action** (two populations giving `Q(s,0)` and `Q(s,1)`, driving the Actor
   with `A = Q − V`). A single **state-value** Critic can only ever give a state-level advantage,
   while CartPole's failures are action-specific. This is the classical fix and the one structural
   piece this entry has not touched. **But note boundary 6**: making the Critic more accurate has
   already been refuted once, so the prior on this should be lowered accordingly.
2. **Let local plasticity shape the encoding itself** (currently a fixed random encoding plus a
   linear readout). The capacity probe shows the *ceiling* of that combination is high (497 steps),
   but the learning rule may not reach it.
3. **Stimulus-specific reward prediction** — the structural fix §3.2 itself names: replace "one
   global δ" with per-stimulus-channel predictions.
4. **Trace Propagation** (Pes 2025) — not implemented this phase, **and this is the same question as
   the W2 line's**: on investigation, the plan's §3.1 treatment of it as "a memory optimisation of
   e-prop" **does not hold** (it is a different rule, covers LIF only, and is worse than e-prop on
   the paper's own N-MNIST). Verdict: [ADR-0010](../../docs/adr/ADR-0010-eprop-quadratic-storage-and-trace-propagation.en.md).

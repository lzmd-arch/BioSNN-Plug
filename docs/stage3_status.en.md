# Stage-three status and route (2026-11-02)

[中文](stage3_status.md) · [日本語](stage3_status.ja.md)

**Nature: closing the books, not passing acceptance.** This file gathers the stage-three record that
was scattered across `sweep_results/` and the main README, for scheduling and for the "should we
re-open it" decision. **It replaces no pre-registration**, and it does not turn "we understand why"
into "we built it".

## 0. One-page conclusion

| Sub-item | Status | One line |
| :--- | :--- | :--- |
| Execution layer, rate version | ✅ **accepted** | CartPole greedy-evaluation median **237.8** (criterion 200) |
| Execution layer, spiking version | ✗ **unmet, archived** | 10-seed medians **50.4 / 55.0 / 37.1**, 0/10; sixteen rounds of diagnosis (D3-D18) converge on the **learning signal** |
| Continual learning, task-aware CIL | ✅ passes (constructive) | oracle per-task **0.8195 / AF 0.0** |
| Continual learning, pure CIL | ✗ unreachable, archived | **0.0876 / AF 0.290**; best task inference **0.5855 / 0.1285** |
| Relearn 3x faster | ✗ unmet | paired median **0.957** (target <=1/3), calibration about 1.06 |

**Archive basis:** [ADR-0029](adr/ADR-0029-research-line-archival.en.md) (pure CIL and the W3 spiking actor).

## 1. Criteria (acceptance wording)

- **Execution layer**: CartPole greedy-evaluation mean steps, **median over 10 seeds >= 200**;
  `best_steps` is only a training-time peak and **does not count**.
- **Continual learning**: ACC / AF for task-aware CIL; the same wording for pure CIL; relearn 3x is
  judged on the paired median.

## 2. Execution layer: why the spiking version did not land

### 2.1 The main comparison after the fix (post 2026-10-18)

> Warning: **every number before 2026-10-18 is void** -- training episodes dropped the environment
> seed, so two runs with the same seed followed different trajectories.

| Arm | 10-seed median | Passes 200 |
| :--- | ---: | ---: |
| Linear TD critic | 50.4 | 0/10 |
| TD-LTP population critic (default) | 55.0 | 0/10 |
| Same (operating point) | 37.1 | 0/10 |
| **Rate version (control)** | **237.8** | ✅ |

The three spiking arms have **fully overlapping intervals** and cannot be told apart.

### 2.2 The diagnostic chain (D3-D18)

| Round | Question | Answer |
| :--- | :--- | :--- |
| D3 | Does swapping the readout (rate vs count) save it | No (65.2 / 9.55) |
| D4 | Does the policy depend on state | After training, slope is all zero |
| D5-D6 | Output collapse / can temperature fix it | Entropy collapses; temperature fixes entropy, the final median stays 9.5 |
| D7 | Can the degradation be attributed | 5/5 `unattributed` |
| D8 | How much state information is left in the final readout | 5/5 `no_state_information` |
| D10 | Does a difference direction exist | **Yes** (4/5), but its relative size is only **0.5%-7.3%** |
| D11-D13 | Threshold/budget/other readouts (low-pass rate, membrane potential) | Not enough (best 0.7-3.45 counts against a threshold of 2) |
| D14 | K-window averaging | **The only effective lever**: 0.57 -> 2.12 (K=1 -> 8), but 5/5 only reach 3/5; longer windows gain nothing |
| D15 | First-spike latency | Signal on the seeds where counts are zero, but still <2 |
| D16 | Orthogonalise the difference direction against the state mean | Cannot make the sign cross zero (0/5) |
| **D17** | **Upstream check: is the encoder sensitive in the pole angle** | **Only 2.9%** (cos 0.9988), 1.1-1.4 on the other three axes -> D4/D8/D9 measured slope=0 along a nearly constant dimension |
| **D18** | **Re-measure after per-axis normalisation of the encoding** | Median delta range 0.2 -> **6.2 (30x)**: the readout now sees state; but **slope is still 0/10 positive and the final median is still 9.85** (seed 4 = 269.1) |

**In one line**: once the readout can see state, **the policy still does not vary with state** -- the
gap is the **learning signal**, not the readout, not the magnitude, not the noise.

The full chain is in [`sweep_results/stage3_readout_audit.md`](../sweep_results/stage3_readout_audit.md).

## 3. Continual learning

| Item | Number | Note |
| :--- | :--- | :--- |
| Task-aware CIL | ACC **0.8195** / AF **0.0** | needs an oracle for the task ID, **constructive pass** ([ADR-0018](adr/ADR-0018-task-aware-cl-gate.en.md)) |
| Pure CIL | ACC **0.0876** / AF **0.290** | unreachable |
| Task inference | **0.5855 / 0.1285** | the best setting, still far from feeding CIL |
| Relearn 3x faster | paired median **0.957**, calibration about **1.06** | v1-v3 all unmet ([ADR-0019](adr/ADR-0019-stage-four-entry-decoupled-from-relearn.en.md)) |

## 4. Boundaries (what to know when reading the numbers above)

- The network has **256 spiking neurons** against the plan's target of about 50,000 -- a factor of about **195**;
  the measured ceiling on 8 GB is about **1.35k** recurrent neurons;
- The operating point is **L1 normalised (each column sums to 1) + threshold 1.0**; other operating
  points were not measured;
- Before D17 several diagnostics were run **along the pole angle**, a dimension that barely changes at
  the input -- read the old conclusions with that note in mind;
- The 0.957 for "relearn 3x" is a **paired median**, not a single best run.

## 5. Archive and re-opening

**Archived** (ADR-0029): pure CIL, the W3 spiking actor.

**Re-opening conditions** (from ADR-0029, stated so they can be tested):

| Line | What evidence would void the archive |
| :--- | :--- |
| Spiking actor | A scale experiment (5k-8k neurons) showing the readout limit comes from **scale**, not from rate-based readout as such |
| Pure CIL | A long sequence with **real task boundaries** where task inference reaches **0.8** |

**Third condition added on 2026-11-02 (from D18)**: if what changes is the **learning signal** (rather
than readout/magnitude/noise), that is no longer "re-opening stage three" -- it is spiking SoS, which
is now the main line ([ADR-0030](adr/ADR-0030-mainline-shift-to-spiking-sos.en.md)).
In other words: **the execution-layer item will not be completed by "tweaking it again"; it will only
be inherited by "changing the learning signal".**

### Verdict of this review (2026-11-02)

**Do not re-open the stage-three experiments.** D18 has already localised the gap to the learning
signal, and changing the learning signal is exactly what spiking SoS is; until the SoS
pre-registration lands, this stage-three item stays archived.

## 6. Where the records live

| Content | Location |
| :--- | :--- |
| Readout diagnosis summary (D3-D18) | [`sweep_results/stage3_readout_audit.md`](../sweep_results/stage3_readout_audit.md) |
| Spiking actor main line | [`research/rstdp_spiking/README.md`](../research/rstdp_spiking/README.en.md) |
| Rate version (the only accepted implementation) | [`research/rstdp/README.md`](../research/rstdp/README.en.md) |
| Continual learning | [`research/continual/`](../research/continual/) |
| Archive decision | [ADR-0029](adr/ADR-0029-research-line-archival.en.md) |

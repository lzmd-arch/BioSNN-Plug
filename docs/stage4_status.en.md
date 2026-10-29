# Phase-4 component status and route (2026-10-27)

[中文](stage4_status.md) · [日本語](stage4_status.ja.md)

**Nature: a status close-out, not a passed acceptance.** The phase-4 entry was opened under
[ADR-0018](adr/ADR-0018-task-aware-cl-gate.en.md) (task-aware CIL) and
[ADR-0019](adr/ADR-0019-stage-four-entry-decoupled-from-relearn.en.md)
(decoupled from relearn 3×). This document records component status, boundaries, and open
research items as of 2026-10-27 for phase-4 scheduling; it replaces no preregistration and does
not turn "component ready" into "phase-4 end-to-end holds".

## 1. Scorecard

| Component | Status | Key numbers | Boundary/caveat |
| :--- | :--- | :--- | :--- |
| Multi-timescale working memory | **v4 + v9 ready** | v4 three cues R² **0.9994 / 0.9982 / 0.9994**; v9 spiking trace A/B **0.9257 / 0.9281** (LIF), **0.9261 / 0.9292** (ALIF) | v4 local RLS; v9 independent afferent population + spike trace (arm-state median diff 0.130, non-trivial); v5/v6/v7 negative, v8 trivial (no evidence); v9 is a spike-count/low-pass code, timing not required |
| Metacognitive gate | **v3 + v8 + v11 ready** | v3 CIFAR: selective 80% **0.5423**, AUROC **0.7530**, ECE **0.0296**; v8 (seeds 5–9): coverage **0.810**, selective MSE **0.033**, quantized abstain **0.898**, AUROC **0.925**, scale abstain **0.851**, validity **6.43×** | v4 scale only; v5 rotation not OOD; v6/v7 quantization gates fail; v8 reconstruction-consistency gate (errors + state_norm), input-transform OOD only; **v11 (max/mean/norm) passes all four**: time_shift 0.998, time_reverse 0.990, dim_dropout 0.858, correlated 0.987, scale 0.855; v9 (0.596/0.539) and v10 (dim 0.762) negatives preserved. Still not general abstention |
| World model P3 | **state/open-loop/rolling closed-loop termination ready; pure-imagination loop open** | closed-loop H=10 NMSE ~2e-8; drift 115.5; open-loop v5 AUROC **0.999996**, FP **0.000504**, hit **0.998**; rolling closed-loop v6 AUROC **1.0**, FP **0.000912**, hit **1.0**, terminal error **1.14e-5**; pure imagination **0.634–0.665** | open-loop and rolling closed-loop ready; **500-step pure-imagination loop still fails** (v2 0.6469, v6 control 0.645), due to action feedback + float32 bit sensitivity; `p3_closed_loop_imagination_ready=false` |
| Episodic memory | **DPAR non-local + local delta usable** | v4: literal NN **0.938**, pseudo **0.828**, Hebbian **0.311**; v5 local delta K=8 **0.805**, K=4 **0.922**, state control **0.123** | local delta approaches pseudo-inverse (0.805 vs 0.816); K8 mean passes but 3/5 seeds <0.80; K4 stable; capacity K=8–16; one-shot Hebbian still unusable |
| Predictive E-prop | **1/3 seeds pass; `ready=false`** | Reconstruction 0.011–0.022× (all seeds); free-running amplitude **0.584 / 0.536 / 0.119**, DTW **0.088 / 0.137 / 0.231** (baseline 0.228–0.268) | Self-sustained oscillation unstable across seeds; **an error-input curriculum (gain 1 -> 0 over the last 30% of epochs) breaks reconstruction (0.089-0.169), so the error channel is a load-bearing drive, not a shortcut**; Lorenz/noise/interpolation and TBPTT comparison untested; adaptation, not bit-exact replication |
| Scale / neurogenesis / energy | **boundary data for scale/event priority/neurogenesis; energy proxy only** | ~**1.35k** under 8 GB; 500k forward **5.02 s**, one update **5.27 s**; 10k 6000-sample 1 epoch **205.6 s / acc 0.520**, full 60k 1 epoch **2,397 s / acc 0.697** (single epoch, not acceptance); event priority select **0.339** but accuracy **0.6213 vs 0.7748**; neurogenesis linear 0.432/0.310, NCM 0.766/0.277, **v3 replay/competition guard: replay 0.634 with old tasks within 2pp on 14/15 cells, guard 0.697 but old tasks drop up to 34pp, none of the four arms reaches 0.90**; measured sparsity **0.8758 → proxy 41.2×** | 500k training untested; no power meter; event priority/NCM not ready |
| Phase-4 integration demo | **runnable** | WM/world model bit-identical; metacognitive regression AUROC **0.562**; episodic recall MSE 1.194 | Composition/equivalence holds; end-to-end metacognition not inherited, not a capability conclusion |
| End-to-end task (delayed QA + OOD abstention) | **4/5 criteria** | R² **0.9994**, selective MSE **0.00053**, OOD AUROC **0.927**, abstain **0.887**; episodic MSE **0.822** | WM + metacognition + world-model surprise cooperate; **episodic recall is the only miss**, `e2e_ready=false` |
| End-to-end v2 (four cues + discrete episodic) | **5/5 ready** | R² **0.9983**, selective MSE **0.00172**, OOD AUROC **0.8895**, abstain **0.824**, E recall **1.000** | All five pass; but E recall is "pipeline works", not hard memory; seed 2 abstains only 0.746 |
| End-to-end v3 (capacity/noise/delay stress) | **4/5 criteria** | Primary K=8/σ=0.3/t=100: R² **0.9092**, selective MSE **0.0675**, OOD AUROC **0.8350**, abstain **0.7056**, E recall **0.9040** | Abstention misses (all five seeds <0.80); capacity boundary K=8–16 (K=16 only 0.7736); v2's abstention margin was already thin. Metacognitive v4's density gate reaches 0.8456 on the same data, but relies on a scale feature and does not change this row's verdict |
| End-to-end v4 (DPAR + WM coexistence) | **4/6 criteria** | WM R² **0.6558**, DPAR **0.9628**, state control **0.2652**, OOD AUROC **0.8612**; selective MSE **0.3507** and abstain **0.7648** miss | DPAR and WM interfere; the prereg's R²≥0.50 and MSE≤0.10 are mutually inconsistent, so this round's criteria are unreachable; `e2e_v4_ready=false` |
| End-to-end v5 (two-stage WM → DPAR) | **6/6 ready** | WM R² **0.99995**, selective MSE **6.16e-05**, DPAR **0.9584**, state control **0.2472**, OOD AUROC **0.9572**, abstain **0.9492** | Sequential composition: WM is probed at t=35, DPAR at t=85; they do not share the final state. v4's coexistence interference is preserved; not an overall phase-4 acceptance |
| End-to-end v6 (single vs dual bank coexistence) | **neither ready** | single: R² **0.527**, MSE **0.339**, OOD abstain **0.793**; dual: R² **0.994**, MSE **0.0046**, OOD abstain **0.622**; DPAR 0.962, state control 0.235 | Routing fixes WM but breaks OOD (AUROC 0.874→0.782); coexistence and abstention are not satisfied together |
| End-to-end v7 (dual bank + bank-W confidence) | **5/6** | WM R² **0.9938**, MSE **0.0045**, DPAR **0.962**, state control **0.235**, OOD AUROC **0.882**, abstain **0.786** misses | Removing bank D from confidence raises abstain 0.622→0.786 and AUROC 0.782→0.882; still 1.4 pp short; t=85 bank-W decay is the remaining boundary |
| End-to-end v8 (shared state + bank-W scale gate) | **6/6 ready** | WM R² **0.9938**, MSE **0.0045**, DPAR **0.962**, state control **0.235**, OOD AUROC **0.9562**, abstain **0.9368** | Shared final state; the gain is closed by `state_norm` on bank W (gauss2 0.823→gauss3 0.937); scale-OOD only; not general abstention or overall phase-4 acceptance |
| End-to-end v9 (v8 robustness) | **main ready, robust false** | scale 1.5/2/3/5 abstain **0.456 / 0.720 / 0.937 / 0.996**; K2/4/8 WM R² **0.998 / 0.994 / 0.89975** (K8 misses by 0.00025); σ0.1/0.3/0.5 abstain **0.945 / 0.937 / 0.919** | `v9_main_ready=true`, `v9_robust=false`, `v9_ready=false`; v8 covers only **s≥3, K≤4, σ≤0.5**; s=2 already drops to 0.720 |
| End-to-end v10 (shrunk covariance gate) | **s2/s3/s5 ready** | s2 coverage **0.7900**, abstain **0.8276**, AUROC **0.8937**; s3/s5 abstain **0.9764 / 0.9984**; K8 WM R² **0.89975**, coverage **0.7532** | `v10_ready=true`: s2 rises 0.720→0.828; K8 remains a capacity/coverage boundary; s=1.5 untested |
| End-to-end v11 (K=8 three arms) | **no arm ready** | baseline R² **0.8926**/coverage **0.7740**; snapshot85 R² **0.9936**/coverage **0.7552**; longtau R² **0.90145**/coverage **0.7776**; DPAR 0.934, state 0.12 | The only miss is K=8 test coverage (<0.78); snapshot85/longtau fix WM capacity, leaving gate calibration/coverage |
| End-to-end v12 (K=8 diag gate) | **two arms ready** | snapshot85+diag: R² **0.9936**, coverage **0.7820**, abstain **0.9292**; longtau+diag: R² **0.90145**, coverage **0.8052**, abstain **0.9244**; shrunk still **0.7552 / 0.7776** | `v12_ready=true`: K8's boundary is covariance estimation, not WM; both arms pass, but snapshot coverage is only 0.002 above 0.78 |
| End-to-end v14 (18-cell matrix) | **14/18 ready** | s3/s5 all pass; K4-s2 σ0.1/0.3 pass; failures **K4-s2-σ0.5 abstain 0.7756**, **K8-s2 three noise levels 0.7656/0.7448/0.7024** | `v14_ready=false`; the only failing criterion is s=2 OOD abstention; WM/DPAR/state/AUROC pass |
| End-to-end v15 (fill s=2 four cells) | **0/4** | K4-σ0.5 longtau **0.7868**; K8-s2 snapshot **0.7700/0.7416/0.7044** | Swapping arms (longtau/snapshot85) still misses 0.80; s=2 boundary not closed |
| End-to-end v16 (input-scale gate) | **18/18 ready** | K4/K8 × s2/3/5 × σ.1/.3/.5: coverage **0.796–0.810**, abstain **0.882–1.000**, AUROC **0.928–1.000**; WM/DPAR/state pass | **Input-level** input_norm gate, not state-level metacognition; v14 state-level remains 14/18; scale OOD only |
| End-to-end v17 (state-level reconstruction gate) | **12/18 ready** | seeds 10–14: s3/s5 all pass; six s2 cells abstain **0.620–0.736**, all fail; other metrics R² 0.906–0.994, DPAR 0.930–0.964, state 0.125–0.243 | state-level kNN[errors,state_norm] does not close s=2 and is two cells below v14’s 14/18; v16’s input-level 18/18 is not mixed in |
| End-to-end v18 (state-level prediction-norm gate) | **18/18 ready** | seeds 15–19: s2 abstain **0.837–0.907**, coverage **0.795–0.814**, AUROC **0.900–0.942**; WM R² 0.906–0.994, DPAR 0.937–0.958 | state-level gate without input_norm; v14 14/18 and v17 12/18 preserved |

## 2. What "ready" means

- It only means **passing that component's preregistered criteria** (see each result README); it
  does not include:
  - the phase-4 end-to-end task;
  - integration with the bus/plugins;
  - full locality (metacognitive calibration head); v9 spiking WM is a count/low-pass code, timing untested;
- All negative results/boundaries are kept as-is; no criteria were changed after the fact. Ledger
  entries ㊽–110 in the project plan are the complete index.

## 3. Open research items (in suggested order)

1. **End-to-end v3 (stress test) is done, 4/5**: the primary K=8/σ=0.3/t=100 misses only OOD
   abstention (0.7056 < 0.80), with all five seeds below the line; OOD ranking itself still works
   (AUROC 0.835). The capacity boundary lies between K=8 and K=16 (K=16 E top-1 0.7736). v2's
   0.824 was only 2.4 pp above the threshold and already thin. The next step should switch to a
   task that **truly requires memory**, or replace the fixed-coverage abstention threshold with
   calibration/risk control; do not keep adding the same family of stress, and do not treat K=8's
   0.904 as general memory ability. **v4 has now integrated DPAR**: DPAR 0.9628 passes, but WM
   is interfered with (R² 0.6558, MSE 0.3507), and the prereg's R²≥0.50 vs MSE≤0.10 are
   inconsistent; fix the criteria or split the targets before calling v4 ready. **v5 now reaches
   6/6 with the two-stage design** (WM at t=35, DPAR at t=85, no shared final state), but that is
   only a sequential composition bound and does not overturn v4's coexistence interference.
   **v6's dual-bank routing** raises WM R² from 0.527 to 0.994 but drops OOD abstention from
   0.793 to 0.622; neither arm is ready, so coexistence and abstention are still not satisfied
   together. **v7 computes confidence on bank W only**: abstain 0.622→0.786 and AUROC
   0.782→0.882, still 1.4 pp short of 0.80; the gap comes from A/B/Q decay at t=85, not the
   threshold position. **v8 adds bank-W `state_norm`** and reaches abstain 0.9368 / AUROC
   0.9562 with all six criteria passing; this is **scale-OOD only**, not general abstention,
   and does not change the v6/v7 information-loss OOD boundary. **v9 robustness**: s=2 abstains
   0.720 and K=8 WM R² is 0.89975, so `v9_robust=false`; v8's ready must be written as
   s≥3, K≤4. **v10 uses a bank-W shrunk Mahalanobis** and lifts s=2 abstention to 0.8276
   (coverage 0.7900, AUROC 0.8937); s3/s5 also pass. K=8 remains a capacity/coverage boundary
   (0.89975 / 0.7532). **v11 splits K=8 into three arms**: snapshot85 R² 0.9936 and longtau
   0.90145, but coverage 0.7552/0.7776 both below 0.78; the remaining K8 boundary is gate
   calibration, not WM capacity. **v12 switches to a diag gate**: snapshot85 0.7820 and
   longtau 0.8052, both arms pass all seven; shrunk still fails. K8's boundary is the 28x28
   covariance estimate, not WM or DPAR. **v14's 18-cell matrix is 14/18**: s3/s5 all pass;
   the remaining failures are s=2 abstention (K4-σ0.5 0.7756; K8 0.766/0.745/0.702).
   **v15's arm swap is still 0/4** (longtau 0.7868; snapshot 0.770/0.742/0.704); s=2 needs
   a stronger scale feature. **v16 switches to an input-scale gate `max(||A||,||B||)` and passes
   18/18**; but it is an input-level gate, not state-level metacognition, and v14's 14/18 remains
   the state-level result. **v17 applies the metacognitive v8 kNN[errors,state_norm] gate to the 18 cells and reaches only 12/18**: s=3/5 pass, but all six s=2 cells abstain 0.620–0.736 and fail; this is two cells below v14, and state-level s=2 remains open. **v18 switches to the prediction-norm z gate and reaches 18/18**: six s=2 cells abstain 0.837–0.907, coverage 0.795–0.814, AUROC 0.900–0.942; state-level s=2 is closed.
2. **Spiking working memory v5–v9 (closed)**: v5 counts 0.217/0.187, v6 membrane 0.309/0.121,
   v7 population membrane 0.788; the bottleneck moved from "5 Bernoulli samples" to "spike reset
   erasure". v8's `v+thr·trace` cancels reset algebraically (LIF/ALIF identical to ≤2e-7) and is a
   trivial solution, not evidence. **v9 uses an independent afferent population + pure spike trace
   `s=αs+z`**: LIF/ALIF 0.9269/0.9276, arm-state median diff 0.130, all four criteria pass.
   Boundary: v9 is a spike-count/low-pass code; timing is not required, and the readout is a
   measurement head.
3. **Metacognition v4 is done (with a boundary)**: an independent calibration split plus a
   three-dimensional density gate (disagreement / surprise / state_norm) raises e2e v3 abstention
   from 0.7056 to **0.8456** and AUROC from 0.8350 to **0.9003**, with selective MSE essentially
   unchanged. But **gauss2 is actually worse** (0.6984), so the gain comes from `state_norm`
   detecting the A/B×3 scale shift; it must be re-tested on **scale-independent OOD**
   (covariance/frequency/phase shifts), and cannot be treated as general abstention ability. v3's
   classification calibration head is still not transferred to regression. **v5's norm-preserving
   orthogonal rotation is not OOD** (error 0.0679 vs in-dist 0.0681), so v5 gives no generalization
   conclusion; the next attempt must first pass an "OOD error ≥2× in-dist" validity gate before
   interpreting abstention. **v6 quantization OOD passes that validity gate** (6.38×), but gauss3
   abstains 0.1716 with AUROC 0.4876 and all four gates fail; **v7 replaces the features with
   higher moments** (mean z²/|z|/z⁴/tail ratio) and still gets 0.2048/0.5024, with scale sanity
   dropping to 0.7544. So v4 is limited to scale shifts and cannot be written up as general
   abstention ability. **v8 switches to a reconstruction-consistency gate** (kNN(5) on
   [errors, state_norm]): seeds 5–9 coverage 0.810, quantized abstain 0.898/AUROC 0.925, scale
   0.851, validity 6.43×, `v8_ready=true`; the boundary is input-transform OOD only, and coverage
   leaves the band on individual seeds.
4. **World model P3 (split)**: closed-loop state/stability passes (H=10 NMSE ~2e-8, drift 115.5),
   but **closed-loop termination** still fails: v1 0.616, dedicated 0.665, one-step imagined 0.632,
   v2 multi-step 0.6469; v3/v4/v5 closed-loop controls are only 0.634–0.640. The cause is action
   feedback: the balancing policy flips actions near `θ+0.5θ̇`, tiny model errors change the action
   sequence, and a 115-step imagination diverges; DAgger cannot fix it (one-step error already 1e-5).
   **Open-loop fixed-action** termination risk is usable: v3 AUROC≈1 but all-or-nothing separation
   misses by 1.4e-4, v4 rate FP 0.00244 fails, **v5 calibrated threshold reaches AUROC 0.999996,
   FP 0.000504, hit 0.998, terminal error 0.00305**. `p3_v5_ready=true` is open-loop only; **v6 rolling-horizon closed loop** (real observations for actions, one-step model risk) reaches AUROC 1.0, FP 0.000912, hit 1.0, terminal error 1.14e-5, `p3_v6_ready=true`; **the 500-step pure-imagination loop remains open**, `p3_closed_loop_imagination_ready=false`.
   `p3_closed_loop_ready=false`.
5. **Episodic memory**: DPAR (v4) shows the task really requires binding (state control 0.121
   ≈ chance); **non-local pseudo-inverse** passes at K=8/σ=0.1 (0.828), literal NN is higher at
   0.938, and **local Hebbian still only reaches 0.311**; capacity boundary K=8–16. The original
   CartPole task/noise/temporal-order boundary is preserved. The next step should try local or
   low-dimensional keys, or accept the lookup baseline; do not keep tuning k/β/encoding
   dimension. **v5's local delta rule** raises K=8 to **0.805** and K=4 to **0.922**, close to
   pseudo-inverse 0.816; boundaries: K8 passes on the mean but 3/5 seeds <0.80, K=16 drops to 0.531,
   σ=0.3 to 0.553.
6. **Working-memory close-out**: both v4 (rate) and v9 (spike trace) are ready; v9's boundary is a
   spike-count/low-pass code, not a timing code. Claiming temporal coding needs a new prereg and a
   time-shuffle guard.

## 4. Relation to phase 3

- The phase-3 CL gate passed as task-aware CIL under ADR-0018 (constructive AF=0, not an
  anti-forgetting algorithm);
- **relearn 3× is unmet**, decoupled from the phase-4 entry under ADR-0019, and remains open;
- This document **does not change** either statement, nor does it turn phase-4 component ready
  into full phase-3 acceptance.

## 5. Index

- Working memory: [v4 result](../sweep_results/working_memory_v4/README.md),
  [v5 result](../sweep_results/working_memory_v5/README.md),
  [v6 membrane](../sweep_results/working_memory_v6/README.md),
  [v7 population membrane](../sweep_results/working_memory_v7/README.md),
  [v8 trivial](../sweep_results/working_memory_v8/README.md),
  [v9 spike trace](../sweep_results/working_memory_v9/README.md)
- Metacognition: [v3 result](../sweep_results/metacognitive_gate_v3/README.md),
  [v4 risk control](../sweep_results/metacognitive_gate_v4/README.md),
  [v5 rotation validity check](../sweep_results/metacognitive_gate_v5/README.md),
  [v6 quantization OOD](../sweep_results/metacognitive_gate_v6/README.md),
  [v7 higher-moment gate](../sweep_results/metacognitive_gate_v7/README.md),
  [v8 reconstruction gate](../sweep_results/metacognitive_gate_v8/README.md)
- World model: [P3 closed loop](../sweep_results/world_model_closed_loop_h500/README.md),
  [termination-head integration](../sweep_results/world_model_closed_loop_head/README.md),
  [imagined-state head](../sweep_results/world_model_closed_loop_imagined/README.md),
  [termination head v2](../sweep_results/world_model_termination_v2/README.md),
  [v3 open-loop](../sweep_results/world_model_termination_v3/README.md),
  [v4 rate](../sweep_results/world_model_termination_v4/README.md),
  [v5 calibrated](../sweep_results/world_model_termination_v5/README.md),
  [v6 rolling closed loop](../sweep_results/world_model_termination_v6/README.md)
- Episodic memory: [v3 result](../sweep_results/episodic_memory_v3/README.md),
  [DPAR result](../sweep_results/episodic_paired_associate/README.md),
  [local delta v5](../sweep_results/episodic_paired_associate_v5/README.md)
- Integration demo: [result](../sweep_results/stage4_integration_demo/README.md)
- End-to-end: [v2 result](../sweep_results/stage4_e2e_v2/README.md),
  [v3 stress test](../sweep_results/stage4_e2e_v3/README.md),
  [v4 DPAR coexistence](../sweep_results/stage4_e2e_v4/README.md),
  [v5 two-stage](../sweep_results/stage4_e2e_v5/README.md),
  [v6 single/dual bank](../sweep_results/stage4_e2e_v6/README.md),
  [v7 bank-W confidence](../sweep_results/stage4_e2e_v7/README.md),
  [v8 bank-W scale gate](../sweep_results/stage4_e2e_v8/README.md),
  [v9 robustness sweep](../sweep_results/stage4_e2e_v9/README.md),
  [v10 shrunk covariance gate](../sweep_results/stage4_e2e_v10/README.md),
  [v11 K8 three arms](../sweep_results/stage4_e2e_v11/README.md),
  [v12 K8 diag gate](../sweep_results/stage4_e2e_v12/README.md),
  [v14 18-cell matrix](../sweep_results/stage4_e2e_v14/README.md),
  [v15 s=2 four cells](../sweep_results/stage4_e2e_v15/README.md),
  [v16 input-scale gate](../sweep_results/stage4_e2e_v16/README.md),
  [v17 state-level reconstruction gate](../sweep_results/stage4_e2e_v17/README.md),
  [v18 prediction-norm gate](../sweep_results/stage4_e2e_v18/README.md)
- Technical report: [paper/stage4/README.md](../paper/stage4/README.md) · Zenodo
  [10.5281/zenodo.23696829](https://doi.org/10.5281/zenodo.23696829) (v1.0, 2026-10-28)

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
| Multi-timescale working memory | **v4 ready** | Three-cue R² **0.9994 / 0.9982 / 0.9994**, mean 0.9990 > ridge 0.9931 | Local RLS readout (rank-one, no backprop); spiking v5 not ready |
| Metacognitive gate | **v3 + v4 ready (scale only)** | v3 CIFAR: selective 80% **0.5423**, AUROC **0.7530**, ECE **0.0296**; v4 e2e v3: abstain **0.8456**, AUROC **0.9003**, coverage **0.8028** | v4's gain comes from `state_norm` detecting the A/B×3 scale OOD. v5 rotation is not OOD; v6 quantization OOD is valid (error 6.38×) but abstain 0.1716/AUROC 0.4876, so the gate fails; v7 higher moments give 0.2048/0.5024 and still fail. Not general OOD |
| World model P3 | **partial** | Closed-loop H=10 NMSE ~2e-8; first-step drift **115.5**; closed-loop termination AUROC **0.632–0.665** | State/closed loop stable; termination suffers distribution shift in closed loop, `p3_ready=false` |
| Episodic memory | **DPAR non-local usable** | DPAR (K=8, σ=0.1): literal NN **0.938**, pseudo **0.828**, Hebbian **0.311**; state control **0.121** (chance 0.125) | Only non-local pseudo-inverse passes; local Hebbian still unusable; capacity boundary K=8–16; the original CartPole task boundary is preserved |
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

## 2. What "ready" means

- It only means **passing that component's preregistered criteria** (see each result README); it
  does not include:
  - the phase-4 end-to-end task;
  - integration with the bus/plugins;
  - spiking (working memory) or full locality (metacognitive calibration head);
- All negative results/boundaries are kept as-is; no criteria were changed after the fact. Ledger
  entries ㊽–102 in the project plan are the complete index.

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
   s≥3, K≤4.
2. **Working memory v6 (spiking)**: v5's failure is in the **representation** (60-step counts
   only single digits), not the readout; v6 needs membrane/subthreshold state, a longer
   integration window, or rate input. Note v5 took 2598.7 s; the next round should cut
   seeds/windows or parallelize.
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
   abstention ability.
4. **World model P3 follow-up**: the closed-loop termination distribution shift (offline 0.999 →
   closed loop 0.665) is unresolved; directions are a stronger state model (push the first-step
   drift well beyond 115) or a termination head matched to the imagined distribution;
   one-step imagination augmentation was tried, 0.632, not enough.
5. **Episodic memory**: DPAR (v4) shows the task really requires binding (state control 0.121
   ≈ chance); **non-local pseudo-inverse** passes at K=8/σ=0.1 (0.828), literal NN is higher at
   0.938, and **local Hebbian still only reaches 0.311**; capacity boundary K=8–16. The original
   CartPole task/noise/temporal-order boundary is preserved. The next step should try local or
   low-dimensional keys, or accept the lookup baseline; do not keep tuning k/β/encoding
   dimension.
6. **Working-memory v5 alternative**: if the project insists on "pure spike counts", first solve
   the low-count representation problem; otherwise accept v4's rate leaky integrator as the
   usable local representation for phase 4.

## 4. Relation to phase 3

- The phase-3 CL gate passed as task-aware CIL under ADR-0018 (constructive AF=0, not an
  anti-forgetting algorithm);
- **relearn 3× is unmet**, decoupled from the phase-4 entry under ADR-0019, and remains open;
- This document **does not change** either statement, nor does it turn phase-4 component ready
  into full phase-3 acceptance.

## 5. Index

- Working memory: [v4 result](../sweep_results/working_memory_v4/README.md),
  [v5 result](../sweep_results/working_memory_v5/README.md)
- Metacognition: [v3 result](../sweep_results/metacognitive_gate_v3/README.md),
  [v4 risk control](../sweep_results/metacognitive_gate_v4/README.md),
  [v5 rotation validity check](../sweep_results/metacognitive_gate_v5/README.md),
  [v6 quantization OOD](../sweep_results/metacognitive_gate_v6/README.md),
  [v7 higher-moment gate](../sweep_results/metacognitive_gate_v7/README.md)
- World model: [P3 closed loop](../sweep_results/world_model_closed_loop_h500/README.md),
  [termination-head integration](../sweep_results/world_model_closed_loop_head/README.md),
  [imagined-state head](../sweep_results/world_model_closed_loop_imagined/README.md)
- Episodic memory: [v3 result](../sweep_results/episodic_memory_v3/README.md),
  [DPAR result](../sweep_results/episodic_paired_associate/README.md)
- Integration demo: [result](../sweep_results/stage4_integration_demo/README.md)
- End-to-end: [v2 result](../sweep_results/stage4_e2e_v2/README.md),
  [v3 stress test](../sweep_results/stage4_e2e_v3/README.md),
  [v4 DPAR coexistence](../sweep_results/stage4_e2e_v4/README.md),
  [v5 two-stage](../sweep_results/stage4_e2e_v5/README.md),
  [v6 single/dual bank](../sweep_results/stage4_e2e_v6/README.md),
  [v7 bank-W confidence](../sweep_results/stage4_e2e_v7/README.md),
  [v8 bank-W scale gate](../sweep_results/stage4_e2e_v8/README.md),
  [v9 robustness sweep](../sweep_results/stage4_e2e_v9/README.md)

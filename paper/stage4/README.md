# Stage Four of BioSNN-Plug: What the Cognitive Components Actually Support

**Technical report - v1.0 - 2026-10-28**

**Author:** lzmd-arch (independent researcher) - <https://github.com/lzmd-arch/BioSNN-Plug>

**Scope:** working memory, episodic memory, metacognition, world models, and end-to-end
composition in the phase-four research programme. This report records what passed, what failed,
and--most importantly--the exact protocol under which each statement holds.

---

## Abstract

Phase four added five cognitive components on top of the local-learning substrate: multi-timescale
working memory, episodic memory, a metacognitive gate, a world model, and an end-to-end delayed
recall task. Two results survive their preregistered criteria. A **sequential composition** of
working memory and delayed paired-associate recall reaches 6/6 criteria (`e2e_v5`). A **shared-state
dual-bank** design with an explicit scale gate reaches 6/6 on its primary condition (`e2e_v8`), but a
follow-up sweep shows that this result covers only OOD scales of 3x or more and DPAR set sizes of
four or fewer; at 2x the abstention rate falls to 0.720, and at K=8 working-memory R^2 is 0.89975
against a 0.90 floor.

The negative results are equally specific. Pure class-incremental learning does not pass; the
phase-three entry was opened by a task-aware CIL protocol. Relearning 3x remains unmet and is
recorded as an open item. Spiking working memory loses information in its count representation.
Local Hebbian episodic memory remains unusable on a binding task that literal nearest-neighbour and
non-local pseudo-inverse solve. The world model's closed loop is stable but its termination head is
not. The metacognitive gate detects scale shifts but not information-loss shifts: a valid
quantization OOD is answered with AUROC 0.4876 by the gate that reaches 0.9003 on a 3x scale shift.

The report's main claim is therefore narrow and structural: **component readiness is not end-to-end
readiness, and a scale-specific gate is not a general uncertainty signal.** Every number below is
tied to a preregistration, a result README, and a five-seed run.

---

## 1. Evidence discipline

All phase-four results follow the same rules:

- the criterion is written before the run, in a file under `sweep_results/*_prereg.md`;
- the run starts from a clean worktree (including untracked files);
- five seeds are reported, not the best seed;
- test and OOD splits never participate in fitting the gate;
- a negative result stays in the record; criteria are not changed after seeing data;
- a result's README contains a machine-generated Repro block with commit, Python, hardware, seed
  list, dependency hash, command, and elapsed time.

The phrase **ready** below means *passing that component's preregistered criteria*. It never means
phase-four acceptance, bus integration, or a general capability claim.

---

## 2. Phase-three entry

The phase-four work began after a governance decision, not after a full phase-three pass.

| Item | Status | Evidence |
| :--- | :--- | :--- |
| Pure CIL (no task id) | **failed** | ACC 0.0876, AF 0.290 against 0.65 / 0.08 |
| Task-aware CIL (ADR-0018) | **passed, constructively** | oracle per-task 0.8195 / AF 0.0 |
| Task-inference CIL | **failed** | best local route 0.5855 / 0.1285; final routing 0.528 |
| Relearn 3x | **unmet, open** | formal paired median 0.957; calibration around 1.06 |

ADR-0019 decoupled the phase-four entry from relearn 3x. The correct public wording is
**partially accepted**: task-aware CIL passes under its own protocol; relearn 3x is unmet. The
constructive AF=0 (frozen heads plus task id) is not an anti-forgetting algorithm.

---

## 3. Components

### 3.1 Working memory

| Arm | Status | Result |
| :--- | :--- | :--- |
| Multi-timescale leaky integrators + local RLS (`v4`) | **ready** | three-cue R^2 0.9994 / 0.9982 / 0.9994; mean 0.9990 above ridge 0.9931 |
| LIF / ALIF spike-count representation (`v5`) | **not ready** | multi-cue R^2 about 0.21 / 0.19; 60-step counts remain single-digit |

The v5 failure is representational, not a readout failure: a ridge probe on the same spike counts
is no better than the local RLS readout. The practical phase-four representation remains the
rate-based leaky integrator with a local, rank-one readout.

### 3.2 World model P3

The state model is essentially exact in closed loop (H=10 NMSE around 2e-8, ten orders below
`copy_last`), and the closed-loop first-step drift is 115.5. The termination head does not survive
the closed-loop distribution shift: AUROC 0.632-0.665 against a 0.75 criterion. `p3_ready=false`.
A dedicated termination head and a one-step imagined head were tried (0.999 offline, 0.665 closed
loop, 0.632 imagined); the problem is the distribution, not the classifier alone.

### 3.3 Episodic memory

The original CartPole episodic task is a sealed boundary: literal nearest-neighbour reaches action
0.762, termination balance 0.962, and temporal order 0.595; Hebbian, pseudo-inverse, kNN and
attention all fail to beat it, and temporal order never reaches 0.75.

A new delayed paired-associate recall (DPAR) task asks the model to bind K key-value pairs, delay,
and recall the value paired with a query key. It is genuinely memory-demanding: a linear readout
from the final state reaches 0.121 top-1 against chance 0.125.

| DPAR arm (K=8, query noise 0.1) | Top-1 |
| :--- | :--- |
| Literal nearest neighbour | **0.938** |
| Non-local pseudo-inverse | **0.828** |
| Local Hebbian outer product | **0.311** |

The preregistered best associative arm passes (0.828), but the result is explicitly non-local and
still 11.0 points below literal lookup. Capacity falls between K=8 and K=16 (K=16: 0.774). The old
CartPole boundary remains.

### 3.4 Metacognition

| Version | Protocol | Result |
| :--- | :--- | :--- |
| v3 | CIFAR selective prediction | selective 80% 0.5423, AUROC 0.7530, ECE 0.0296 |
| v4 | e2e v3, independent calibration, 3D density | abstain 0.8456, AUROC 0.9003, coverage 0.8028 |
| v5 | norm-preserving rotation OOD | invalid manipulation: OOD error 0.0679 vs in-dist 0.0681 |
| v6 | sign-quantized OOD, validity 6.38x | gauss3 abstain 0.1716, AUROC 0.4876 |
| v7 | higher-moment gate on the same OOD | moment4 abstain 0.2048, AUROC 0.5024 |

The v4 gain comes from `state_norm` on an A/B x3 scale shift, not from local calibration in
general: the two-feature Gaussian is actually worse than the z-score sum (0.6984 vs 0.7208). The
v5 rotation is a symmetry of the isotropic Gaussian task, not an OOD; its failure to abstain is
correct behaviour. The v6/v7 family establishes the boundary: simple state statistics, including
higher moments, do not detect an information-loss shift that preserves the first two moments.

---

## 4. End-to-end composition

All tasks share the same discipline: five seeds, a calibration split separate from test, and no
OOD labels in fitting. The table is the complete sequence.

| Version | Structure | Key result | Verdict |
| :--- | :--- | :--- | :--- |
| v1 | A/B/Q + continuous episodic target | R^2 0.9994; selective MSE 0.00053; OOD 0.927 / 0.887; episodic MSE 0.822 | 4/5 |
| v2 | A/B/Q/E (2 classes) | R^2 0.9983; MSE 0.00172; OOD 0.8895 / 0.824; E top-1 1.000 | 5/5, but E is a pipe |
| v3 | K=8, sigma=0.3, t=100 stress | R^2 0.9092; MSE 0.0675; OOD 0.8350 / 0.7056; E 0.9040 | 4/5 |
| v4 | DPAR and WM share one final state | WM R^2 0.6558; MSE 0.3507; OOD 0.8612 / 0.7648; DPAR 0.9628 | 4/6; criteria inconsistent |
| v5 | WM at t=35, DPAR at t=85 | R^2 0.99995; MSE 6.16e-05; OOD 0.9572 / 0.9492; DPAR 0.9584 | **6/6** |
| v6 | single bank vs routed dual bank | single R^2 0.527; dual R^2 0.994; dual abstain 0.622 | neither ready |
| v7 | dual bank, bank-W-only confidence | R^2 0.9938; MSE 0.0045; OOD 0.8822 / 0.7860 | 5/6, 1.4 pp short |
| v8 | v7 + bank-W state_norm | R^2 0.9938; MSE 0.00448; OOD 0.9562 / 0.9368 | **6/6** |
| v9 | v8 robustness sweep | s=2 abstain 0.720; K=8 R^2 0.89975 | main ready, robust false |

### 4.1 What v4 exposed

v4 placed the DPAR pairs and the A/B/Q cues into the same 28-dimensional state. Working memory
collapsed (B cue R^2 0.513; selective MSE 0.351), while DPAR remained strong (0.9628). The run also
exposed a preregistration error: R^2 >=0.50 and selective MSE <=0.10 cannot both hold for
unit-variance targets; the criteria were impossible, not merely hard. The run is preserved as a
negative result and as a lesson about criterion consistency.

### 4.2 What v5 established

v5 moves the WM probe to t=35, before the DPAR pairs are written, and reads DPAR at t=85. All six
criteria pass. This is a **sequential composition** result: it does not share a final state, so it
does not overturn v4's interference. It does show that the WM readout, DPAR memory, state-control
check, and scale gate can coexist in one episode pipeline when their probes are separated in time.

### 4.3 What v6 and v7 established

v6 routed the two input families into two 28-dimensional banks and concatenated them. The routing
fixed WM (R^2 0.527 -> 0.994, MSE 0.339 -> 0.0046) but broke OOD confidence: the 56-dimensional
readout diluted the scale signal, and abstention fell from 0.793 to 0.622. v7 restricted readout,
members, world model, and confidence to bank W; abstention recovered to 0.786 and AUROC to 0.882,
but remained 1.4 points below 0.80. The remaining gap came from the t=85 decay of A/B/Q in bank W,
not from a threshold position.

### 4.4 What v8 and v9 established

v8 added `state_norm` to the bank-W confidence (gauss3). All six criteria pass: abstention 0.9368,
AUROC 0.9562, WM R^2 0.9938, DPAR 0.9620, state control 0.2352. The gain is a scale feature:
gauss2 alone is 0.823 and norm-only is 0.898.

v9 swept one axis at a time:

| Axis | Values | OOD abstention |
| :--- | :--- | :--- |
| scale s | 1.5 / 2 / 3 / 5 | 0.456 / 0.720 / 0.937 / 0.996 |
| K | 2 / 4 / 8 | 0.928 / 0.937 / 0.926 |
| noise sigma | 0.1 / 0.3 / 0.5 | 0.945 / 0.937 / 0.919 |

Working-memory R^2 at K=2/4/8 is 0.9983 / 0.9938 / **0.89975**; the K=8 value is 0.00025 below the
0.90 floor. The state-control criterion was corrected before the formal run from a fixed 0.35 to
`1/K + 0.10`, because K=2 has chance 0.5; this correction is recorded in the preregistration's
revision 1 and does not affect the K=4 primary result.

**The honest v8 claim is therefore: shared-state dual-bank composition with a bank-W scale gate,
for OOD scales at least 3x, DPAR K at most 4, and cue noise at most 0.5.** At 2x scale the gate
already fails (0.720), and K=8 is a capacity boundary.

---

## 5. What is supported, and what is not

### Supported

- Local RLS working memory with three cues (v4; 0.9990 mean, above ridge).
- Non-local associative recall on DPAR (pseudo-inverse 0.828; literal NN 0.938 at K=8).
- A sequential WM -> DPAR composition with all six e2e criteria (v5).
- A shared-state dual-bank composition with a bank-W scale gate under the v9 envelope (v8).
- Task-aware CIL as a phase-three gate (0.8195 / AF 0.0, constructive).
- The negative result that the current metacognitive gate is scale-specific, not general.

### Not supported

- Pure CIL (0.0876 / 0.290) and task-inference CIL (0.5855 / 0.1285).
- Relearn 3x (0.957 paired median; open under ADR-0019).
- Spiking working memory (v5; around 0.21 / 0.19).
- Local Hebbian episodic binding (0.311 at K=8).
- World-model termination in closed loop (0.632-0.665).
- General OOD detection: sign quantization passes the validity gate (6.38x error) but the gate
  reaches AUROC 0.4876; higher moments reach 0.5024.
- Single-bank shared-state coexistence (v4/v6 single) and DPAR K=8 capacity (R^2 0.89975).
- Scale OOD at 2x (abstention 0.720).

---

## 6. Reproduction

All preregistrations and result READMEs are in the repository:

- `sweep_results/stage4_e2e_v5_prereg.md` and `sweep_results/stage4_e2e_v5/README.md`
- `sweep_results/stage4_e2e_v6_prereg.md` and `sweep_results/stage4_e2e_v6/README.md`
- `sweep_results/stage4_e2e_v7_prereg.md` and `sweep_results/stage4_e2e_v7/README.md`
- `sweep_results/stage4_e2e_v8_prereg.md` and `sweep_results/stage4_e2e_v8/README.md`
- `sweep_results/stage4_e2e_v9_prereg.md` and `sweep_results/stage4_e2e_v9/README.md`
- `sweep_results/episodic_paired_associate_prereg.md` and its README
- `sweep_results/metacognitive_gate_v4_prereg.md` through `v7_prereg.md`
- `docs/stage4_status.md` for the current component scorecard

Canonical commands:

```bash
python -m research.cognitive.run_e2e_delayed_recall_v5 --seeds 0-4
python -m research.cognitive.run_e2e_delayed_recall_v6 --seeds 0-4
python -m research.cognitive.run_e2e_delayed_recall_v7 --seeds 0-4
python -m research.cognitive.run_e2e_delayed_recall_v8 --seeds 0-4
python -m research.cognitive.run_e2e_delayed_recall_v9 --seeds 0-4
python -m research.cognitive.run_paired_associate_recall --seeds 0-4
```

The report snapshot was assembled from commit `d683518` (phase-four status before this report).
Each result README carries its own Repro block with the exact commit used for that run; the
numbers in this report are copied from those blocks, not re-derived.

---

## 7. Limitations

1. **Synthetic tasks.** DPAR keys and values are random vectors; the e2e cues are Gaussian. The
   results do not establish natural episodic memory or natural OOD.
2. **Per-episode memory.** DPAR builds a memory per episode; this is not lifelong episodic memory
   and has no cross-episode interference.
3. **Scale-specific gate.** `state_norm` is a scale detector. The v8 result is for A/B x3 (and,
   by v9, x5); it is not a general uncertainty signal. The sign-quantization OOD that preserves
   first two moments is not detected.
4. **Linear readouts.** WM and state control use local RLS; the negative result about final-state
   linear readout does not rule out nonlinear readouts.
5. **No bus integration.** Phase-four components are research scripts; they are not wired into
   `biosnn-bus` or the plugin stack. Component readiness is not system integration.
6. **Capacity.** K=8 is a boundary; the 0.89975 R^2 is below the preregistered 0.90 floor.
7. **Hardware and budget.** Five seeds per condition; the v9 sweep took 617.1 s. Longer seeds,
   larger K, and larger state banks are untested.

---

## 8. Open items

1. **General information-loss OOD.** Replace low-dimensional state statistics with a generative
   state model, cycle consistency, or an input-level detector; first pass the "OOD error at least
   2x in-dist" validity gate, then interpret abstention.
2. **K=8 capacity.** Change the state/readout capacity or accept K<=4; do not tune the 0.90 floor.
3. **Scale 2x.** A stronger scale/covariance feature is needed; current gate fails at 0.720.
4. **Spiking working memory v6.** Fix the count representation (membrane/subthreshold state,
   longer window, or rate input).
5. **World-model termination.** A state model or termination head matched to the imagined
   distribution.
6. **Local episodic memory.** Try local or low-dimensional keys; current local Hebbian is 0.311.
7. **Phase-three 3x.** Reopen only with a new task/algorithm and a new preregistration; ADR-0019
   keeps it decoupled from the phase-four entry.

---

## 9. Conclusion

Phase four produced two defensible end-to-end results and a large set of explicit boundaries. The
sequential composition (v5) is the cleanest: it passes all six criteria and does not pretend that
the shared state is solved. The shared-state result (v8) passes all six on its primary condition,
but v9 shows that it is a **scale-gated, K<=4** result, not a general one. The components that
remain open--spiking working memory, local episodic binding, world-model termination, and general
OOD--are open for specific, measured reasons, not for lack of trying.

The value of the phase-four record is therefore not a single headline number. It is the mapping
between a claim and the protocol under which the claim is true. That mapping is what the
preregistrations, the clean-worktree discipline, and the negative results preserve.

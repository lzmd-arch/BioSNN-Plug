# W2: The e-prop cognitive layer

[中文](README.md) · [日本語](README.ja.md)

**⚠️ Breaking-change grace period: through 2027-09** (project plan §12.5, ADR-0005).

## What claim is being tested

The cognitive-layer learning rule of project plan §3.1: **can e-prop (ALIF + adaptive
threshold + eligibility traces) work on a sequential task, with an active-neuron fraction
above 60%** — the latter is the hard criterion of §七 phase 1 and the trigger line of the
"dead neuron protection" in §3.1.

## Current conclusion

**Met. Test accuracy 77.48% (the seed-0 acceptance run), active-neuron fraction 1.0000.**

The project plan gives no number for "works", so it is broken into two checkable things,
and both pass:

| Metric | Measured | Criterion |
| :--- | ---: | :--- |
| Test accuracy (sMNIST) | **0.7748** | well above chance (10-class baseline 0.10) |
| **Active-neuron fraction** | **1.0000** | §七: "> 60%" |
| Spike sparsity | 0.8527 | §9 sets a threshold only at phase 2 (> 90%); not a criterion here |

⚠️ **But 0.7748 is a 30-epoch budget-truncated figure, not the ceiling of this rule on sMNIST**:
on the same seeds, opening the budget to 100 epochs raises test accuracy to **85.19%** (mean over
5 seeds) and it has **still not converged by epoch 100** -- see "Two budgets" below.

**Added 2026-10-13: once the learning rate is opened up, this rule does converge.** Three batches
run serially, 18.7 h in total, artifacts in `sweep_results/w2_lr001` ... `w2_lr010`, `w2_lif`,
`w2_long`:

| Batch | Configuration | Test accuracy (mean) |
| :--- | :--- | ---: |
| Learning-rate sweep | 30 epochs x 5 seeds, `eta_in = eta_rec` | 1e-3 -> 0.7564; 2e-3 -> 0.7764; 5e-3 -> 0.7983; **1e-2 -> 0.8002** |
| LIF control | 30 epochs x 5 seeds, `--beta 0`, lr still 2e-3 | 0.7867 |
| Extended budget (ALIF, **control**) | 250 epochs x 3 seeds, lr = 1e-2 | **0.8831** |
| Extended budget (LIF, **official**) | 250 epochs x 3 seeds, lr = 1e-2, `--beta 0` | **0.8889** (0.8851-0.8962) |

Two points, both written into the known boundaries below: **the learning rate matters far more
than the neuron model** -- moving lr from 2e-3 to 1e-2 alone is **+2.38 percentage points** at
30 epochs; and **the 250-epoch setting finally converges** (last-20-epoch window-mean increments
+0.0016 / -0.0037 / +0.0025, both signs present), so "every W2 number is budget-truncated" **stops
being true here**. There is also a **counter**-finding on boundary 8: **LIF scores higher than ALIF
at every one of the four learning rates** (+0.93 / +1.03 / +1.05 / +1.23 points).

**2026-10-16: the official number changes arms.** The official headline figure becomes **LIF @
lr=1e-2 @ 250 epochs = 0.8889** (three seeds 0.8962 / 0.8854 / 0.8851, range 1.11 points; artefacts
in [`sweep_results/w2_lif_long/`](../../sweep_results/w2_lif_long/)). The **0.8831** in the table
above is a real result of the ALIF run and stays **as the control, with the number unchanged** --
same lr, same seeds 0-2, same 250 epochs, differing only in the neuron model, so this is a clean
comparison rather than two different experiments.
Warning: the official arm passes the same convergence criterion -- last-20-epoch window-mean
increments **+0.0019 / +0.0007 / +0.0025**, the same order as ALIF's +0.0016 / -0.0037 / +0.0025.
**But all three are positive** (ALIF had both signs), so what is judged here is **magnitude**, not
sign: epoch-to-epoch val fluctuation in that arm is about ±0.009, the standard error of a 20-epoch
mean is ~0.002, and the three increments sit within 0.35-1.25 standard errors -- noise level. That
is written out because "all positive" differs literally from ALIF's "both signs": a later reader
should see the criterion, not only the verdict.

### Two budgets: 30 epochs (acceptance) and 100 epochs (opened up)

Under the acceptance budget, validation accuracy rose from 0.4165 at epoch 1 to 0.7695 at epoch 30
and **had not converged** -- so 0.7748 is not a converged value, it is a **budget-truncated** one.
Opening the budget to 100 epochs (same seeds 0-4, same code, only the epoch count changed):

| | 30 epochs (acceptance budget) | 100 epochs (budget opened) |
| :--- | ---: | ---: |
| Test accuracy, mean over 5 seeds | 0.7764 | **0.8519** |
| Test accuracy, seed 0 (acceptance run / same seed re-run) | 0.7748 | 0.8534 |
| Spread over 5 seeds | 1.79 percentage points | **0.82 percentage points** |
| Validation accuracy, mean over 5 seeds (final epoch) | 0.7735 | 0.8491 |

**Paired per seed** (same code, same seed, only the budget changed), truncation costs a mean of
**+7.55 percentage points** (min +6.05, max +8.66, 5 pairs). **The first 30 epochs are point-for-point
identical to the 30-epoch batch** (5/5 seeds x 30/30 epochs, cross-checked across two independent
runs), so the 30-epoch figures are a **prefix** of this trajectory, not a separate result. The curve
is in [`figures/f2-w2-epoch-accuracy.png`](../../figures/f2-w2-epoch-accuracy.png): red stops at
epoch 30, blue runs to epoch 100, and the grey vertical line marks where the acceptance budget ends.

⚠️ **Conventions**: the two kinds of "test accuracy" above are **not the same quantity** -- the
acceptance run (seed 0, with a reproducibility record) and the 5-seed mean are two different things,
and this repository states both, each labelled. It never subtracts one convention from the other.
Likewise **+7.55 is the paired difference** (the mean of 5 same-seed differences); seed 0 alone reads
**+7.86**. Say which one you are quoting.

⚠️ **But 100 epochs has still not converged either.** The 20-epoch window means gain
+0.1428 → +0.0423 → +0.0206 → **+0.0114** -- halving, but not reaching zero; and **per seed, the
final 20-epoch window gain is positive in all 5** (+0.0084 … +0.0147), so the mean is not hiding an
outlier. **0.8519 is therefore also a budget-truncated figure**, just truncated later -- **not** the
ceiling of this rule on sMNIST.

⚠️ **The criterion is window means, not "the final epoch's increment" -- that step is noise.** On the
100-epoch batch the cross-seed mean curve reads **−0.0003** at epoch 100 (0.8493 at epoch 99 →
0.8491 at epoch 100), and some individual seeds are negative too. These numbers are computed, not
eyeballed, by
[`scripts/analyze_w2_multibudget.py`](../../scripts/analyze_w2_multibudget.py); the artefacts are in
[`sweep_results/w2_seeds_100ep/`](../../sweep_results/w2_seeds_100ep/) and
[`sweep_results/w2_seeds_100ep.log`](../../sweep_results/w2_seeds_100ep.log).

One side note pointing the same way: spike sparsity **falls** with the longer budget in **all 5
seeds** (5-seed mean 0.8465 → 0.8406) -- training longer makes the network less sparse, further from
§9's phase-2 target of > 90%.

Reproducibility record (generated by `research/common/provenance.py`):

```text
实验名称：eprop/smnist
日期：2026-09-23 06:59:54 中国标准时间
Git commit：75196c9094f01abe843c1530f5037c39f5e04d9a
Git 状态：干净
Python：3.12.14
操作系统 / 架构：Windows 11 / AMD64
硬件：NVIDIA GeForce RTX 5060，8,123 MB，sm_120
随机种子：base=0；全局种子=1741413617；划分验证集=1435468909；权重初始化=3486733650；每批打乱=3990329895；脉冲编码=1681094898
依赖快照：uv.lock sha256=cbafb161e71421f9f228e23e2ab6f4f742f899e9958d290f214dac1dd948e37f
运行命令：train_sequential.py --device cuda
耗时：1753.6 s
显存峰值：1,104.1 MiB（§6.2 硬约束 8GB）
§6.2 降级路径：未触发
备注：n_rec=256, beta=0.07, thr=0.62, eta_in=0.002, eta_rec=0.002, eta_out=0.05, decay_out=0.95
备注：输入编码：像素值作发放概率的伯努利采样（本项目自己的选择）
```

GPU memory 1.08 GB of 8 GB; none of the §6.2 fallback paths fired.

### SHD cross-check: **it ran, and it did not learn** (a negative result, recorded as such)

| Metric | Measured | Criterion |
| :--- | ---: | :--- |
| Test accuracy (SHD, 20 classes) | **0.0534** | chance is 0.05 — **it did not learn** |
| Active-neuron fraction | 1.0000 | §7: "> 60%" ✓ |
| Spike sparsity | 0.9385 | §9 sets a threshold only at phase 2; not a criterion here |

The same hyperparameters chosen for sMNIST (`n_rec=256`, 30 epochs, `eta_in=eta_rec=2e-3`,
`eta_out=5e-2`, `beta=0.07`, `thr=0.62`) **do not learn** SHD: the training cross-entropy
moves only from 2.9948 to 2.9529, while `ln 20 = 2.996` — it stays at chance throughout.

**Diagnosis** (reading the `w_in` initialisation in `research/eprop/neurons.py` plus a
forward probe):

* `w_in` is initialised as `1/√n_in`, so with 700 input channels each weight is only one
  fifth the size it has with 28 channels (measured **0.0302** against sMNIST's **0.1507**);
* the hidden layer's firing rate is therefore about half sMNIST's (**0.0526** against
  **0.0997**) — not dead, but weakly driven;
* SHD also brings 20 classes and 100 time steps, neither of which sMNIST (10 classes,
  784 steps) has to carry.

**The literature's e-prop SHD configuration is a different one**, and it comes from [2]
(Traces Propagation): its text reads verbatim "we used a single hidden layer of **450 neurons**,
with and without recurrent connections. We find the best hyperparameters to be **β = 0.96**,
**α = 0.97**, and **v_th = 1.0**, for the feed-forward architecture"; in Table 1 the SHD rows are
**450 neurons / 100 time steps / 100 epochs** for everything except TP's own row, which uses
400 neurons, and the batch size for all fully connected architectures is **128**.

⚠️ **Two corrections that have to be stated first:**

- **This configuration is not [1]'s (the original e-prop paper).** The word `SHD` **never appears**
  in [1]'s main text or its supplementary material (see item 3 below) -- the paper's three
  benchmarks are TIMIT, the temporal credit assignment task, and ATARI. This sentence used to read
  "the paper's SHD configuration", which filed [2]'s configuration under [1]'s name; **corrected**.
- **The `Adam 1e-4` that stood in the same sentence has been deleted.** It cannot be found anywhere
  in the full text of [1] or [2] (the single occurrence of `Adam` in [1] is the person Adam Santoro,
  in the acknowledgements), its source is undetermined, and it **must not be reproduced from**.

**e-prop's SHD performance is on record in the literature, but again it is not self-reported.**
Table 1 of [2] carries two e-prop SHD rows -- **feed-forward LIF 450: 63.04%**, **recurrent LIF
450: 80.79%** -- with a footnote reading verbatim "1 Results from [10]." ⚠️ **That footnote is not
confined to the two e-prop rows**: read cell by cell from the table markup, **five** SHD rows carry
the `1` (`eProp` feed-forward / `DECOLLE` feed-forward / `eProp` recurrent / `ETLP` recurrent /
`DECOLLE` recurrent; the N-MNIST block has two more). [2] lifted **an entire comparison block**
from [10] (its own reference, the ETLP paper), with the two e-prop rows inside it. So those numbers
are **neither self-reported by [1]** ([1] contains no SHD at all) **nor produced by [2] itself**.

Viewed across the literature: e-prop reaches 63.04%-80.79% on SHD, while this entry gets 0.0534
(chance) under the **sMNIST hyperparameters and a 30-epoch budget**. The two differ in **both
hyperparameters and budget**, so this is not evidence that "the implementation is wrong" --
**no SHD-specific tuning was done in this phase**, and what this says is "these hyperparameters are
not enough" -- **not** "the rule does not apply", and **not** "the implementation is wrong".
Turning this into a comparable cross-check requires giving SHD its own hyperparameter budget
(size, epochs, α/β/v_th) rather than pushing the sMNIST configuration harder.

Reproducibility record (`--dataset shd --device cuda`, local GPU, ~24 minutes):

```text
实验名称：eprop/shd
日期：2026-09-23 08:32:19 中国标准时间
Git commit：e2d754ffe18f3ace20a735bbd6668bf94298e859
Git 状态：干净
Python：3.12.14
操作系统 / 架构：Windows 11 / AMD64
硬件：NVIDIA GeForce RTX 5060，8,123 MB，sm_120
随机种子：base=0；全局种子=1741413617；划分验证集=1435468909；权重初始化=3486733650；每批打乱=3990329895；脉冲编码=1681094898
依赖快照：uv.lock sha256=cbafb161e71421f9f228e23e2ab6f4f742f899e9958d290f214dac1dd948e37f
运行命令：train_sequential.py --dataset shd --device cuda
耗时：1413.7 s
显存峰值：3,220.6 MiB（§6.2 硬约束 8GB）
§6.2 降级路径：未触发
备注：n_rec=256, beta=0.07, thr=0.62, eta_in=0.002, eta_rec=0.002, eta_out=0.05, decay_out=0.95
备注：输入编码：SHD 脉冲时刻铺到 100 个箱、箱内二值化（本项目自己的选择）
```

## How to reproduce it

```bash
uv run python scripts/download_data.py mnist              # first time: download and verify MNIST
uv run python -m research.eprop.train_sequential          # acceptance run (local GPU, ~25 minutes)
uv run python -m research.eprop.train_sequential --smoke  # CPU smoke, for CI only
```

Multi-seed, multi-configuration batches go through `sweep_seeds.py` -- it **does not change the
training logic**, it only swaps `args.seed` and forwards the remaining arguments to
`train_sequential`:

```bash
# The three batches of 2026-10-13. Artifacts land in data/ first (gitignored) and are moved into
# sweep_results/ afterwards -- landing in the repo mid-run makes provenance report a dirty
# worktree, and every later record block gets stamped "conclusion not trustworthy"
uv run python -m research.eprop.sweep_seeds --seeds 0-4 --epochs 30 --device cuda \
    --learning-rate-in 5e-3 --learning-rate-rec 5e-3 --out-dir data/w2_lr005
uv run python -m research.eprop.sweep_seeds --seeds 0-4 --epochs 30 --device cuda \
    --beta 0 --out-dir data/w2_lif
uv run python -m research.eprop.sweep_seeds --seeds 0-2 --epochs 250 --device cuda \
    --learning-rate-in 1e-2 --learning-rate-rec 1e-2 --out-dir data/w2_long
```

The BPTT arm (the other half of the P2 headline number) runs through `sweep_bptt.py` -- the BPTT
counterpart of `sweep_seeds.py`: it borrows `BPTTLearner` and **does not touch
`bptt_baseline.py`** (that one produces something else, the quantitative gap-report artifact the
CI `gap-report` job depends on). It derives its seeds from the same purpose names, so the split,
weight init, batch order and spike encoding are **bit-identical** to the e-prop run -- a real
paired design, not just "same seed". `--variant` governs how the readout is treated: A is a
single SGD over all parameters (the status quo), B is parameter groups where the readout gets
e-prop's separate `eta_out`. Both variants go into the paper; A doubles as a handicap control:

```bash
# The P2 headline batch: presweep to fix the lr and the variant, then 250ep at the winner.
# Also lands in data/ first, moved into sweep_results/ afterwards
uv run python -m research.eprop.sweep_bptt --seeds 0-2 --epochs 30 --device cuda \
    --variant B --learning-rate-out 5e-2 --learning-rate 3e-3 --out-dir data/w2_bptt_lr003_vB
```

The secondary dataset SHD needs a separate ~169 MB download:
`uv run python scripts/download_data.py shd`. **It ran — and it did not learn**, see the
"SHD cross-check" section below.

## Equations mapped to modules

| Paper | Content | Module |
| :--- | :--- | :--- |
| LIF/ALIF dynamics | `v ← αv + W_in·x + W_rec·z − ϑ·z`; `a ← ρa + z`; threshold `ϑ + βa` | `neurons.ALIFCell` |
| Pseudo-derivative | `ψ = κ·max(0, 1−\|v_scaled\|)` | `neurons.pseudo_derivative` |
| Eq. (25) | Eligibility traces `ε_v`, `ε_a`, `e = ψ(ε_v − βε_a)` | `traces.eligibility_traces` / `traces.eprop_gradient` |
| Eq. (28) | `ΔW_ij = −η Σ_t L_j^t e_ij^t` | `eprop.EPropLearner.update` |
| Learning signal | `L_j^t = (W_out·∂E/∂y)_j^t` | `eprop.EPropLearner.learning_signal` |

## Predictive E-prop (control route, `predictive.py`)

A minimal control following the **description** in [Noè et al., 2026]: the third factor no longer
comes from the task label but from `L_t = (−e_t)·B`, where `e_t = x_t − ŷ_t` is the network's own
prediction error and `B` is a fixed random matrix (broadcast alignment). The error is fed back
through a fixed `W_in`, the network's own output through a fixed feedback matrix; the readout is a
**double-exponential filter** of spikes times `W_out`.

⚠️ **The paper's equations are images on the bioRxiv page and were not obtained verbatim**, so this
is an "adaptation from the description", not a bit-exact replication: updates run at trial end (or
every `chunk` steps), no OU noise, no mixed LIF/ALIF, no TBPTT baseline. Criteria and results:
[`../../sweep_results/predictive_eprop_prereg.md`](../../sweep_results/predictive_eprop_prereg.md)
and `sweep_results/predictive_eprop/`.

## What "local" precisely means

§9's "no global backpropagation" is **checkable** in this implementation, not just a
documentation promise:

- the forward simulation runs under `torch.no_grad()`; there is no graph inside the
  spiking network to back-propagate through;
- the eligibility traces are computed by explicit recurrence, not by differentiation;
- the learning signal has a **closed form** (`W_out` times the softmax error), which even
  removes the "differentiate as far as the filtered spikes" step — the whole of `eprop.py`
  contains no `backward()` or `autograd.grad`;
- the readout layer is updated by ordinary gradient descent (it is simply a linear
  classifier; that is the paper's approach).

Three tests watch this: the weight change must **equal the hand-computed**
`−ηΣ_t L_j^t e_ij^t`; after an update `w_rec.grad is None`; and there is a **contrast** —
the same architecture switched to the BPTT baseline has `.grad` non-`None` and a forward
pass that must build a graph. Looking at only one side shows "no graph was built"; looking
at both shows where the difference actually lies.

## The factorization: exact and approximate are two different things

e-prop's `Σ_t L_j^t e_ij^t` is **not** the BPTT gradient. It drops two families of paths:
cross-neuron recurrent paths, and the self spike-mediated path from the **reset term**
`−ϑ·z^{t−1}` — the latter is independent of `W_rec`, so zeroing the recurrent weights does
not remove it either.

Two tests pin this down separately:

- **Exact**: in a configuration with "no path left to drop" (`W_rec` zeroed, no
  refractoriness, loss depending only on the final timestep), the factorization and BPTT
  agree to a **relative difference of 0.000e+00**. This nails Eq. (25)'s formula and its
  time alignment;
- **Approximate**: in the general case the two differ, measured at 20–40%. That is
  e-prop's design trade-off; the test **measures** the difference and asserts it falls in
  an explainable range, rather than asserting a precision it does not meet.

## Minimal reproduction script

`examples/paper_bellec2020.py` (project plan §12.3). It does not depend on downloading
MNIST, uses a synthetic sequential task that finishes on CPU in tens of seconds, and CI
runs it directly; it reproduces the "exactly equal" result above on the spot and asserts
that the learning path contains no autograd.

`examples/paper_pes2025.py` is a minimal reproduction of **Pes 2025's Trace Propagation** —
likewise a synthetic task, likewise tens of seconds on CPU. **It does not verify the
learning rule used here** (TP is not it, see [ADR-0010](../../docs/adr/ADR-0010-eprop-quadratic-storage-and-trace-propagation.en.md));
it verifies that the paper's own rule runs, and lists the places where the paper's equations
and its official implementation disagree (the off-by-one reset in Eq. (1), Eq. (15) using the
previous layer's trace, the necessity of detaching across layers, and the output layer's
update rule that the paper never states).

## Known boundaries

This section matters more than the conclusion.

1. **Trace Propagation is not implemented here — and that is an investigated conclusion, not a
   debt.** §3.1's "key correction 1" asks for eligibility-trace storage to be reduced to O(N).
   After checking the source (arXiv:2509.13053):
   - **The plan's premise holds.** Table 3 has its own `E-prop [2]` row with Space Complexity =
     `LH²`, and §1.3.1 names e-prop as an instance of the "stored per synapse" family property.
     So attributing that complexity to e-prop specifically is **not** an extrapolation (the
     earlier finding in `docs/references.md` is retracted).
   - **But the "adopt TP" remedy does not apply.** TP is not a memory-saving version of e-prop;
     it is **a different rule that replaces the spatial credit assignment** (the paper lists the
     two as separate rows on separate axes and never says TP optimises e-prop). Every TP result
     is **LIF**, while this line's acceptance configuration is **ALIF (β=0.07)**. And on the
     paper's own N-MNIST, **e-prop 97.90 > TP 97.33 ± 0.06** — e-prop is marked
     `Partial (time)` local and is not even in the "fully local" comparison set. Verdict:
     `docs/adr/ADR-0010`.
   - **The one thing this implementation actually should do has been done.** `epsilon_v` was
     allocated as `(batch, n_pre, n_post)`, but its recursion **contains no postsynaptic index
     j**; allocating `(batch, n_pre, 1)` is **bit-identical** (`torch.equal=True`,
     `max|diff|=0`) and gives **−18.2%** wall clock at the acceptance shape. `epsilon_a`'s
     coefficient contains `(ρ − β·ψ_j)` and **does not factorise**, so the bulk of the memory is
     set by the ALIF adaptation term; for LIF (β=0) both degenerate to outer products and the
     whole trace state drops to O(batch·N).
   - The online-accumulation step remains worthwhile: what it removes is the **time dimension**
     (`O(T·batch·N²)` → `O(batch·N²)`), measured going from "not even one epoch finishes at
     n_rec=256" to 50 s/epoch.
2. **The acceptance budget is 30 epochs and has not converged; opening it to 100 epochs still
   has not converged.** ⚠️ The criterion is **not** "the final epoch's increment" -- that step is
   noise (on the 100-epoch batch the cross-seed mean curve reads −0.0003 at epoch 100, and
   individual seeds go both ways). The criterion is **window means**: the 20-epoch window gain
   halves from +0.1428 all the way to **+0.0114** without reaching zero, and the **final 20-epoch
   window gain is positive in all 5 seeds in both batches** (30-epoch batch +0.1247 … +0.1338;
   100-epoch batch +0.0084 … +0.0147) -- see "Two budgets". Every test accuracy W2 reports so far
   is therefore a **budget-truncated** figure, not the ceiling of this rule on sMNIST.

   🔄 **Updated 2026-10-13: this limitation stops at the 250-epoch, lr=1e-2 setting.** The same
   criterion (last-20-epoch window-mean increment) on
   [`sweep_results/w2_long/`](../../sweep_results/w2_long/) gives **+0.0016 / -0.0037 / +0.0025**
   (seeds 0-2) -- **both signs present, down at the noise floor** -- in direct contrast with the
   100-epoch batch's "positive in all 5 seeds, +0.0084 … +0.0147". So **0.8831 is a converged
   value, not another truncated one**. ⚠️ But that setting moved **two** variables at once (budget
   30->250 and lr 2e-3->1e-2); the budget axis alone, at the same lr and the same seeds, is in
   [`w2_lr010/`](../../sweep_results/w2_lr010/): 0.8036 at 30 epochs, so 250 epochs is worth
   **+7.95 percentage points** net.
3. **Multi-seed statistics now exist (5 seeds); but "better or worse than the paper" has no
   referent -- the paper never evaluated this task.** Same convention as W1
   (`uv run python -m research.eprop.sweep_seeds --seeds 0-4 --device cuda`):

   | Seed | Test accuracy | Active-neuron fraction |
   | ---: | ---: | ---: |
   | 0 | 0.7748 | 1.0000 |
   | 1 | 0.7678 | 1.0000 |
   | 2 | 0.7789 | 1.0000 |
   | 3 | 0.7857 | 1.0000 |
   | 4 | 0.7746 | 1.0000 |
   | **Mean** | **0.7764** | **1.0000** |
   | Min / max | 0.7678 / 0.7857 | 1.0000 / 1.0000 |
   | **Spread** | **1.79 percentage points** | **0.00 percentage points** |

   ⚠️ **The active-fraction column has no seed variance at all** (all 5 seeds read
   1.0000) -- it sits far from §7's 60% threshold and is **not a discriminating metric**;
   it is reported only because the plan asks for it.
   ⚠️ **The key fact on the accuracy side is not the spread, it is that this has not converged**:
   the final 20-epoch window gain of the 30-epoch batch is positive in all 5 seeds
   (+0.1247 … +0.1338); the curve is in
   [`figures/f2-w2-epoch-accuracy.png`](../../figures/f2-w2-epoch-accuracy.png), so 0.7764
   is a **budget-truncated** figure, not the ceiling of this rule on sMNIST. It does keep rising
   once the budget is opened (see "Two budgets" above). But **"better or worse than the paper"
   has no referent at all** -- see below.

   ⚠️ **There is no sMNIST baseline in the paper: the original e-prop work never evaluated this
   task.** Verbatim re-check of [1]'s **main text and supplementary material** on 2026-09-25: in the
   main text (the Nature open-access page, 146,503 characters after tag stripping) `MNIST` and
   `mnist` occur **0 times each**, `permuted` 0, `sequential` 0, `SHD` 0; in the supplementary PDF
   (`41467_2020_17236_MOESM1_ESM.pdf`, 81,610 characters of extracted text) `shd` / `heidelberg` /
   `spike-hearing` are **all 0 as well**. The only tasks named in the main text are `TIMIT` (8),
   `ATARI` (1) and `temporal credit assignment` (9). The abstract page of the same work's preprint
   (arXiv:1901.09049) likewise has `MNIST` at 0. The paper's "Data availability" reads verbatim:

   > Data for the TIMIT and ATARI benchmark tasks were published in previous works with DOI
   > [https://doi.org/10.1613/jair.3912] and ref. 20 with DOI [https://doi.org/10.6028/nist.ir.4930].
   > Data for the temporal credit assignment task are generated by a custom code provided in the
   > abovementioned code repository.

   The paper's model **is** an LSNN (LIF + ALIF) -- `LSNN` occurs 32 times and `ALIF` 13 times --
   the same family as this entry's acceptance configuration, but not one of its three benchmarks is
   sMNIST. The official implementation [15] likewise contains **no MNIST code at all**: the default
   branch holds only `Figure_2_TIMIT/`, `Figure_3_and_S7_e_prop_tutorials/` and
   `Figure_4_and_5_ATARI/`, plus two numerical-verification scripts.

   **The paper from the same lab that actually did sMNIST is a different one ([22], LSNN 2018) --
   and its learning rule is BPTT + DEEP R, still not e-prop.** The body of [22] reads verbatim:

   > A performance comparison is given in Fig. 1B. LSNNs achieve **94.7%** and **96.4%**
   > classification accuracy on the test set when every pixel is presented for 1 and 2ms
   > respectively. An LSTM network achieves 98.5% and 98.0% accuracy on the same task setups.

   ⚠️ **94.7% and 96.4% are both the `max`, not the mean.** That row of [22]'s Table S1 (1 ms)
   reads verbatim `LSNN 100(A), 120(R) 12% 8185 (full 68210) 12 94.2% 0.3% 94.7%` -- 12 runs, mean
   **94.2 ± 0.3**; Table S2 (2 ms), same row, reads `12 93.8% 5.8% 96.4%` -- mean **93.8 ± 5.8**.
   Those tables also carry a single-run large network at `96.1%` (1 ms) / `97.1%` (2 ms), **# runs
   = 1, no standard deviation**, which cannot be listed alongside the two distributions above.
   ⚠️ The LSTM control in the same sentence is **likewise a max** (98.5% / 98.0%); its mean is only
   79.8 ± 26.6 (1 ms) and 48.2 ± 39.9 (2 ms) -- **that control is wildly unstable; do not cite it
   as "the paper-level number"**.

   The README of `IGITUGraz/LSNN-official` says the same thing, **not a second independent
   source**:

   > With `tutorial_sequential_mnist_with_LSNN.py`, you can reproduce Figure 1B from [1] and
   > achieve above __96%__ accuracy on the __sequential MNIST__ task with an LSNN (the code may
   > not reproduce exactlty the results published in [1]).

   ⚠️ **Three qualifications -- drop any one and the number gets misread:**
   - **(a) It is trained with BPTT + DEEP R rewiring, not e-prop.** That tutorial script uses
     `tf.train.AdamOptimizer` with `rewiring_optimizer_wrapper`, and has no e-prop switch anywhere
     in the file.
   - **(b) It is the repository's own claim, not a number from the paper's text.** In the paper the
     result is plotted in Fig. 1B and no value is given in the prose; the repository itself adds a
     "may not reproduce exactlty" caveat (spelling as in the original).
   - **(c) By this repository's own standard, a BPTT number is not a precedent here**
     (see "How to use this table" in [`docs/related_projects.en.md`](../../docs/related_projects.en.md)).
     The budget it costs is written in the same docstring (verbatim): `40% accuracy in 100
     iterations`, `60% in 200 iterations`, `above 90% with 20k ~ 30k iterations`,
     `up to 96% in 36k iterations (takes ~24h on GTX1080)`.

   So [22]'s 94.7% / 96.4% (mean 94.2 ± 0.3 / 93.8 ± 5.8) can only serve as a **same-family
   reference under a different learning rule, and under a different budget and hyperparameters**;
   it **cannot be used as the paper's e-prop baseline**. The accurate statement of this item is now
   two sentences:
   - **This project has no comparable published e-prop baseline on sMNIST** -- neither "far behind
     the paper" nor "ahead of the paper", because [1] never evaluated on this task at all.
   - **But the same family (LSNN) does have a paper-level number attached to it**: a **mean** of
     94.2 / 93.8, obtained by BPTT + DEEP R. ⚠️ **Align the basis**: their 94.2 is a **mean** over 12
     runs, so this project must be compared by **mean** too -- the 30-epoch batch's 5-seed mean is
     **0.7764** and the 100-epoch batch's is **0.8519** (the seed-0 figures 0.7748 / 0.8534 are
     single draws from the same thing). Whichever we take, that distance is real, **but it is the sum
     of two differences at once -- a different learning rule and a different budget/hyperparameter
     set** -- and must not be read as "the e-prop implementation is wrong".

   Quantifying the e-prop/BPTT gap **on its own** can only be done with this project's own
   **same-budget** comparison (item 6 below), never by borrowing the paper's TIMIT / ATARI numbers.

   Per-seed JSONs and the full stdout (5 reproducibility record blocks, each reading a clean
   `Git state`) are in [`sweep_results/w2_seeds/`](../../sweep_results/w2_seeds/) and
   [`sweep_results/w2_seeds.log`](../../sweep_results/w2_seeds.log).
   A second, **independent cross-check batch** (same configuration, run before per-epoch recording
   was added) is in [`sweep_results/w2_seeds_nocurve/`](../../sweep_results/w2_seeds_nocurve/): mean
   **0.7764** (min 0.7678, max 0.7857), which differs from the canonical batch's
   0.7764 by 0.00 percentage points -- two independent runs
   agreeing with each other.

   That cross-check only verified the **final accuracy**. A finer-grained one exists at the
   100-epoch budget: [`sweep_results/w2_converge/`](../../sweep_results/w2_converge/) is the
   **first exploratory run** of seed 0, and it and
   [`w2_seeds_100ep/seed0.json`](../../sweep_results/w2_seeds_100ep/) are two independent
   executions with the same `test_accuracy` of **0.85340** and **all 100 `curve` points identical
   point by point** (only `elapsed_s` differs: 4441 s vs 4372 s). Equivalently: **given the seeds,
   this training pipeline is deterministic**, verifiable down to every single epoch.

   ⚠️ This item used to read "**A single run, not an average.** One seed only (`base=0`)". That
   statement has been superseded by this batch.
4. **The SHD cross-check ran, and it did not learn** (see "SHD cross-check" above). This
   entry used to read "was not run" — actually running it exposed two defects in `load_shd`
   that only surface at runtime (variable-length data indexed after the file was closed;
   binning by milliseconds, which keeps only the first 100 ms of each clip), and after
   fixing them the result was **no learning under the same hyperparameters**. So this is a
   negative result, not a gap; and "tune for SHD" is something this phase did not do.
5. **The input encoding is this project's own choice.** MNIST's analog pixels become spikes
   by Bernoulli sampling. No paper prescribes this, and changing the encoding changes the
   result.
6. **The quantified gap report against BPTT: 23.3 percentage points at matched
   budget.** `bptt_baseline.py` (same architecture, same loss, same budget, same seed -- the only
   difference is the learning rule):

   | Arm | Learning rule | Test accuracy |
   | :--- | :--- | ---: |
   | e-prop | local eligibility traces, **no cross-layer gradient** | **0.5274** |
   | BPTT | surrogate gradient through spikes, graph not cut | **0.7602** |
   | | **Gap** | **+0.2328** |

   Configuration `n_rec=256, epochs=10, n_train=20000, batch=64, seed=0`. The raw stdout (including its reproducibility record block) is in
   [`sweep_results/w2_bptt.log`](../../sweep_results/w2_bptt.log).

   **Three caveats, stated up front so the number is not misread**:
   - **The two arms use different learning rates** (e-prop 2e-3, BPTT 5e-3 -- each script's own
     default). So this is the gap "under each method's default learning rate", **not** the gap
     after hyperparameter alignment -- part of it may be nothing but the learning rate.
     🔄 **2026-10-13: the magnitude is now measured.** The e-prop side alone was swept over four lr
     values (30 epochs x 5 seeds) -- 1e-3 -> 0.7564, 2e-3 -> 0.7764, 5e-3 -> 0.7983, 1e-2 -> 0.8002.
     **Moving lr from 2e-3 to 1e-2 alone is +2.38 percentage points**, so "part of it may be
     nothing but the learning rate" is no longer a guess.
     ⚠️ But you still **cannot** subtract that from 23.3: the gap report runs **10 epochs x 20k
     samples** while this sweep is **30 epochs x 54k** -- two different budgets. Turning "how much
     gap remains once the learning rate is aligned" into one number requires re-running the BPTT
     arm at the **same budget**; that has not been done.
   - **e-prop is far from converged at this budget** (10 epochs x 20k samples), so a large part of
     the 23.3 points is "slow to converge", not "lower ceiling".
   - **Across budgets, the striking number is compute efficiency** (this is **not** a paired
     comparison -- the budgets differ): BPTT reaches 0.7602 with **200k**
     sample presentations, whereas the acceptance configuration's e-prop needs about
     **1620k** (30 epochs x 54k, where 54k = the 60k training set minus a
     10% validation split) to reach 0.7748 -- about **8x the compute for one or two
     points**. That is the real information in this report: it turns "e-prop is slow" from an
     impression into a citable number, and it is the starting point for §9's phase-4 target
     ("the gap narrows or gets a clear attribution analysis").

   ⚠️ **Do not compare the 100-epoch figures from "Two budgets" against this report.** That batch
   is 100 epochs x 54k = 5.4M sample presentations, whereas the BPTT arm here is 10 epochs x 20k =
   200k -- **both budgets differ**, and BPTT was never run at the longer one. The only comparison
   available is the **matched-budget** pair of arms above.

   ⚠️ This item used to read "**the quantified gap report against BPTT has not been produced** ...
   does not yet exist". That statement has been superseded by this report.
7. **A single hidden layer.** `EPropLearner` is one ALIF layer plus a linear readout. §2.2's
   cognitive core is a multi-layer RSNN; stacking layers is not implemented.
8. **ALIF's gain over LIF is negative at every one of the four learning rates -- this is now
   settled.** (Rewritten 2026-10-15; it previously read "and this is not settled".)
   `--beta 0` degenerates `ALIFCell` into a LIF, which allows a same-architecture, same-budget,
   same-seed control. Group 4 filled in four learning rates for LIF (30 epochs x 5 seeds, at the
   **same lr values and the same seeds** as ALIF):

   | Same lr | LIF (beta=0) | ALIF (beta=0.07) | Delta |
   | :--- | ---: | ---: | ---: |
   | 1e-3 | **0.7657** (0.7616-0.7689) | 0.7564 | +0.93 pp |
   | 2e-3 | **0.7867** (0.7801-0.7899) | 0.7764 | +1.03 pp |
   | 5e-3 | **0.8088** (0.7989-0.8216) | 0.7983 | +1.05 pp |
   | 1e-2 | **0.8125** (0.8016-0.8230) | 0.8002 | +1.23 pp |

   **LIF is higher at all four, and no two of the eight per-seed ranges overlap**; moreover LIF is
   still climbing where ALIF has already flattened (5e-3 -> 1e-2 gains ALIF only +0.19 pp but LIF
   +0.37 pp). So the conclusion is not "the neuron type does not matter" -- it is that **at this
   task, this scale and this budget, the adaptation term is a net drag**.

   ⚠️ **This was unanswerable before only because LIF had run at 2e-3 alone** -- and ALIF's +2.38
   points came entirely from sweeping lr, so comparing a single point against a swept curve did
   not hold up. Group 4 removed that objection. **But a boundary remains**: LIF has **not
   flattened** across these four values (1e-3 -> 1e-2 rises monotonically), so "ALIF is worse than
   LIF under all conditions" **cannot** be inferred -- what these four support is "**at this task,
   this scale and this budget, LIF is always higher**". Whether to extend the grid further is
   governed by a **pre-registered** rule and stopping condition, in
   [`sweep_results/w2_lif_lr020_prereg.md`](../../sweep_results/w2_lif_lr020_prereg.md).
   **That probe has since run**: at 2e-2 the 5-seed mean is **0.8140**, only **+0.14 pp** above
   1e-2 -- inside the seed noise (spreads 1.22 and 2.14 pp), where 5e-3 -> 1e-2 was +0.37 pp. So
   the curve **flattens between 1e-2 and 2e-2**, the official operating point **stays at 1e-2**,
   and by the pre-registered stopping rule no 5e-2 point is added.

   This item and item 3 (the activity fraction has no discriminating power) were the same class of
   problem: **a choice treated as a premise of the acceptance configuration** (here beta=0.07)
   **was itself never controlled**. It has now been controlled, and the result is that it brings
   **no** gain.

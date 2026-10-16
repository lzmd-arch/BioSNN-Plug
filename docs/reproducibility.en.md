# Reproducibility checklist

[中文](reproducibility.md) · [日本語](reproducibility.ja.md)

> The project plan §12.4 requires "the exact environment for each phase (commit hash, dependency versions, random seeds)".

## Why this is needed

The most common reason a reproduction fails is not that the code is wrong, it is that the
**environment does not match**: a different numpy version, a different PyTorch version, an
unrecorded random seed, a run against local uncommitted changes.

So every time you report experimental conclusions externally, attach the block below.
**A conclusion reported without it counts as not reproduced.**

## Template

Copy this block, fill it in, and put it in an experiment log or an issue:

```text
Experiment name:
Date:
Git commit:           # git rev-parse HEAD
Git status:           # count of genuinely uncommitted entries (untracked files + tracked
#                     content changes); 0 means clean -- computed by count_dirty_entries() in
#                     research/common/provenance.py, which excludes CRLF stat-only noise
Python:               # python -V
OS / architecture:
Hardware:             # GPU model + GPU memory
Random seeds:         # every seed used in the experiment, noting what each one is for
Dependency snapshot:  # hash of uv.lock, or the output of `uv pip freeze`
Run command:
Duration:
```

## How to obtain this information

This repository manages dependencies with `uv`, and `uv.lock` is committed to version
control — that is the basis of reproduction: **the same lock file resolves to the same
dependency versions**.

```bash
git rev-parse HEAD                     # commit hash
git status --porcelain                 # any output = dirty tree, results not trustworthy
python -V
uv pip freeze                          # exact versions of installed dependencies
shasum -a 256 uv.lock                  # on Windows, use Get-FileHash uv.lock
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

One command that gets all of it (Windows / Git Bash):

```bash
echo "commit: $(git rev-parse HEAD)"; \
echo "dirty: $(git status --porcelain | wc -l) files"; \
python -V; uv pip freeze | shasum -a 256; \
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

## Random seeds

**The skeleton library has randomness in exactly one place**: the sparse random projection
at the end of the spike bus (`SparseRandomProjection`). Its seed is set by
`SpikeBus(seed=...)`.

Conventions:

- The skeleton library **does not use** any source of randomness that depends on process
  state. Concretely, the seed of the projection matrix is derived by `zlib.crc32` from
  `(seed, channel, modality composition)` — **not** Python's built-in `hash()`, which salts
  strings per process (`PYTHONHASHSEED`) and would give the same configuration a different
  projection matrix in different processes, so the sentence "a fixed seed is enough to
  reproduce" would hold only within a single process.

  A regression test watches this: `packages/biosnn-bus/tests/test_bus.py::test_projection_seed_survives_a_process_boundary`
  starts subprocesses with two different `PYTHONHASHSEED` values, and their output must
  match.

- A plugin's own randomness is the plugin's own responsibility. When you write a plugin,
  expose `seed` as a constructor parameter rather than calling `np.random.random()`
  directly.

**Randomness in the research code (`research/`)** goes through
[`research/common/seeding.py`](../research/common/seeding.py): register the purpose first,
then apply. Its `derive_seed` uses `zlib.crc32` rather than the built-in `hash()` for the
same reason the skeleton library does (the latter is salted per process, so it cannot be
reproduced across processes), and the same cross-process regression test watches it. The
validation lines should **not** call `np.random.seed(...)` directly — the seed would then
never appear in the reproducibility record.

## Pre-submit self-check

Besides the general checklist in [`CONTRIBUTING.md`](../CONTRIBUTING.en.md), confirm this
one item as well for anything reproduction-related:

```bash
uv sync --locked      # the lock file and pyproject must agree
```

The sentence above — "the same lock file resolves to the same dependency versions" — holds
only once this step passes.

## Environment record for the skeleton-library phase

Phase 0 (the skeleton library and open-source infrastructure) involves no GPU experiments,
and its conclusions are all **deterministic** — the test suite runs on CPU, and results are
independent of hardware. This phase therefore only needs dependency versions recorded.

| Item | Value |
| :--- | :--- |
| Skeleton library version | `biosnn-bus` 0.1.0 |
| Python | >= 3.10 (CI covers 3.10 / 3.11 / 3.12) |
| Runtime dependencies | `numpy>=1.24` only |
| Hardware requirements | None. CPU is enough |

**The three Phase 0 success criteria (project plan §7) have each been checked**:

| Criterion | Status |
| :--- | :--- |
| The skeleton library is `pip install`-able | ✅ `biosnn-bus 0.1.0` published to PyPI on 2026-09-23 (trusted publishing / OIDC; pipeline in [`release.yml`](../.github/workflows/release.yml)) |
| ...and passes CI | ✅ 12 jobs green (including the leg that installs the wheel into a clean venv and runs the tests) |
| The demo runs in a fresh, GPU-less environment | ✅ Ran [`examples/quickstart_register_plugin.py`](../examples/quickstart_register_plugin.py) in an environment with **only numpy + matplotlib**: it detected the missing package, installed it from PyPI, and completed the whole demo (decode error 0.0161, matching the level code's quantisation step of 1/32). All three language versions were run |

The first two are watched continuously by CI and the release pipeline; the third was **checked by
simulating Colab**. There is no Colab on this machine, so what was verified is the path "fresh
environment + install from PyPI + run to completion", not "someone clicked it once in a Colab tab".

## The Phase 1 (single-rule validation) environment record

Experiments in this phase **involve a GPU**, so they are no longer purely deterministic:
the same configuration can produce different floating-point results on CPU and on CUDA.
So besides dependency versions, the hardware and GPU memory must be recorded too.

| Item | Value |
| :--- | :--- |
| Python | **>= 3.11** (SpikingJelly 2.0.0rc1's floor, see `docs/adr/ADR-0008`) |
| Key dependencies | `torch`, `torchvision`, `torchaudio`, `spikingjelly==2.0.0rc1`, `gymnasium` (the `research` dependency group) |
| torch source | Per-platform: Windows -> `download.pytorch.org/whl/cu130` (covers sm_120); other platforms -> `whl/cpu`. The reasoning is in the `[tool.uv.sources]` comment in `pyproject.toml` |
| Hardware | NVIDIA GeForce RTX 5060, 8,151 MiB, sm_120 |
| Skeleton library | Unaffected: still depends on numpy only, still `requires-python >=3.10` (CI's `package` job covers it with a 3.10 leg) |

**Dependency-group isolation**: the `research` group is **not installed by default**.
`uv sync --locked` gives the Phase 0 environment with no torch; `uv sync --locked --group
research` is the Phase 1 experiment environment. The skeleton library's "does not depend on
torch" claim (ADR-0002) therefore remains verifiable locally as well.

Every module's docstring carries the 12-month breaking-change grace period and its expiry
date (to 2027-09).

### Experiment records completed so far

| Validation line | Where the record lives | Acceptance figure |
| :--- | :--- | :--- |
| W1 kernelized IB-Hebbian perception layer | [`research/ib_hebbian/README.en.md`](../research/ib_hebbian/README.en.md) | MNIST **98.01%** (supervised SGD readout) / **97.79%** (ridge closed-form readout, current default); criterion 70%. **Both readouts are supervised** — 98.01% is the joint result of locally learned features plus a supervised readout, not a purely local accuracy; see the README section on what local precisely means |
| W2 e-prop cognitive layer | [`research/eprop/README.en.md`](../research/eprop/README.en.md) | sMNIST 77.48%; active-neuron fraction 1.0000 (threshold > 60%). **The SHD cross-check is a negative result**: test accuracy 0.0534 ≈ chance (0.05) — the sMNIST hyperparameters do not learn SHD; diagnosis and next steps in the README. ⚠️ **77.48% is a 30-epoch-budget figure, not a converged value**: on the same seeds, opening the budget to 100 epochs reaches 85.19% (mean over 5 seeds; paired difference a mean of **+7.55** points) and has **still not converged by epoch 100** — see "Two budgets" in the README |
| W3 R-STDP + TD-LTP Critic | [`research/rstdp/README.en.md`](../research/rstdp/README.en.md) | ✅ **All three criteria met**: CartPole median **237.8 steps** (seeds 45–64, 20 **fresh** seeds never involved in any selection; criterion ≥ 200); max offset ratio **0.0403** (all seeds < 0.10 ✓); minimum **121** (our own added stability condition ≥ 100 ✓, no seed below 100). On the tuning seeds 25–44 the median is 319.8 with minimum 109 — **both batches pass**. Configuration and rationale: [ADR-0009](adr/ADR-0009-w3-behaviour-policy-and-trace-centring.en.md) |

## Raw experiment artefacts

**The identifier for the authoritative copy**: Zenodo record A (2026-10-15, a deposit of the
findings and the protocol): version DOI `10.5281/zenodo.23388709` (concept DOI
`10.5281/zenodo.23388708`). When you come to this checklist looking for the authoritative copy,
**read that DOI first instead of digging through git history**.
Warning: that record holds **document snapshots** (`sweep_results/README.md` and
`research/eprop/README.en.md`); **the raw json and log files are not in it** — those remain
authoritative in this repository. The DOI supplies an independent timestamp and a citable
identifier, not a substitute for the data: the two stand as index and corpus, not as backups of
each other.

Every number in the table above can be traced back to a **raw artefact**, and those live in
[`sweep_results/`](../sweep_results/) and are now **tracked in this repository** (since 2026-09-23 --
until then the directory was in `.gitignore`, which meant the data existed on exactly one disk; that
is not preservation). The directory listing, the meaning of each field in a json, and which directory
backs which number in the READMEs are in [`sweep_results/README.md`](../sweep_results/README.md).

Three retention rules: **never delete, never overwrite** -- new experiments get a new directory, and
an old one stays even if its conclusion was later refuted (for example the W1 run polluted by a
mid-run edit), with a note saying so; one run is one `.json`, one sweep is a directory plus a summary
`.log` (**which does not always share the directory's name** -- a tuning-phase batch log covers several branches whose results land in several directories; see the "Batch logs" section of `sweep_results/README.md`); and **datasets are not part of this** -- `data/` stays untracked and is
reproduced with `scripts/download_data.py` (per-file checksums), as §12.1 requires.

When writing a paper: draw curves from the `curve` field in the json; take medians and minima from the
summary logs or aggregate the json yourself; and **only report acceptance numbers from the fresh-seed
batch** (for W3 that is `step14_confirm`, seeds 45-64 -- the tuning batch is 25-44; both pass, but the
~27% gap between them is itself part of the conclusion).

## Where later phases record this

From Phase 1 (single-rule validation) onward, every experiment record must carry the full
template. This part is generated by
[`research/common/provenance.py`](../research/common/provenance.py):

```python
from research.common.provenance import DegradationLog, collect
from research.common.seeding import SeedBook

book = SeedBook(base=0)
book.derive("权重初始化")

record = collect(
    "ib_hebbian/mnist",
    seeds=book.render(),
    elapsed_s=123.4,
    peak_mb=None,  # 真实实验里传 measure_peak_memory() 给出的 stats["peak_mb"]
    degradation=DegradationLog(scale_reduction=True, notes=["5 万 → 1 万神经元"]),
)
print(record.render_block())  # 可直接粘进 Markdown 的 text 围栏块
```

It reads out four fields that are **impossible to fill in correctly by hand**:
`git rev-parse HEAD`, `git status --porcelain` (anything non-empty gets flagged "the
working tree is dirty, this conclusion is untrustworthy"), the SHA-256 of `uv.lock`, and
the hardware and GPU memory.

GPU experiments additionally need to record:

- Peak GPU memory (the project plan §6.2 lists 8GB as a hard constraint) — measured by
  `measure_peak_memory()` in `research/common/device.py`, using PyTorch's allocator
  accounting rather than `nvidia-smi`
- Whether the project plan §6.2 fallback paths were triggered (scale reduction / chunked
  training / INT8 trace quantization) — registered via `DegradationLog`; **not firing is
  the norm**, so only explicitly registered ones count
- Training duration and energy-related quantities (§9's energy-efficiency metrics are
  reported in the form of an "invocation-frequency vs. energy curve")

## Pre-run machine check

**This section exists because the same accident happened three times** — on one machine, a long
run's wall clock was quietly inflated by something else, and nobody noticed at the time:

1. **1.61×** (September 2026, already on record): the person measuring was running heavy work on
   the same machine;
2. **1.5×** (morning of 2026-10-16): the same 30-epoch configuration measured 56 s/epoch before
   09:00 and 84 s/epoch after 11:00;
3. **4×** (afternoon of 2026-10-16, mid-way through the sixth BPTT pre-sweep arm): *within one
   arm*, seed 1 measured 3.8 min/epoch while seed 0 and seed 2 measured 0.92 and 0.65.

The common thread: **accuracy is unaffected, the wall clock is polluted, and the pollution is
only ever discovered afterwards.** The rule, therefore: before any run expected to exceed **one
hour**, walk the five steps below, and write step ①'s snapshot into the driver log.

| # | Step | How |
| :--- | :--- | :--- |
| ① | **Snapshot memory and utilisation** | `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader`, into the driver log. Without that baseline there is no way afterwards to tell whether a run was slow |
| ② | **List compute processes** | `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`; if something unfamiliar is there, find out what it is before starting |
| ③ | **Stop Ollama** | `Get-Process \| Where-Object ProcessName -match 'ollama' \| Stop-Process -Force`. The memory terminal loads embedding models on demand and the **daemon keeps them resident for minutes after the last call** — a window that easily covers the middle of a long run. Stopping the daemon makes such a call fail fast instead of quietly taking the card |
| ④ | **Clear the desktop** | A wallpaper engine (`kwallpaperplayer`), browsers/WebView and chat apps (`Weixin`) all hold device memory; an observed snapshot had a screenful of them |
| ⑤ | **1-epoch probe** | Run one epoch at the **target configuration** and compare min/epoch against that configuration's historical baseline. **If it is > 1.5×, do not start the long run** — find the cause first, and start once the rate is back at baseline |

**Two rules about what slow timings do and do not invalidate** (they sit alongside the retention
rules in "Raw experiment artefacts"):

* **Accuracy is not polluted by a slowdown.** Slowness changes time, not arithmetic — runs caught
  inside a slow window keep their **accuracy**, and do not need re-running. `w2_bptt_lr010_vB` on
  2026-10-16 is the case in point: its three seeds went into the mean as normal.
* **Wall clocks from a polluted window must never enter a performance baseline.** Record them as
  an anomalous segment, usable for progress only. `w2_bptt_lr_driver.log` and the running
  `w2_bptt_final.log` both contain such a segment.

Warning: **the cause of the 4× episode was never confirmed.** The prime suspect is the Ollama
model that was loaded at the time, but by the time we looked it had already unloaded (`ollama ps`
was empty and `nvidia-smi` did not list it) — **the evidence had evaporated**. So what this
section records is a **procedure**, not a confirmed culprit; next time it happens, step ①'s
snapshot is the evidence that should have existed.

## Data

Per the project plan §12.1, this repository **ships no data mirror**.

The download and preprocessing script is
[`scripts/download_data.py`](../scripts/download_data.py):

```bash
uv run python scripts/download_data.py --list        # list the registered datasets
uv run python scripts/download_data.py mnist         # download + verify + convert to .npy
uv run python scripts/download_data.py --verify-only # verify existing files only, no network
```

Data lands under `data/`, which `.gitignore` ignores, and each file is **verified
individually against a hash** before it counts as usable; the SHA-256 that passed is
printed for the reproducibility record.

**The checksum is MD5, deliberately**: it is not a security mechanism but an integrity
check (against truncated downloads and mirror drift). MD5 is chosen because for MNIST it
is the **only checksum published by an independent third party** —
`torchvision.datasets.MNIST` hard-codes the four files' MD5s in its own source, and this
script pins those same values, so they can be cross-checked. Computing our own SHA-256 and
pinning that would just be vouching for ourselves.

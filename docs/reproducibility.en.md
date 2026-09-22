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
Git status:           # output of git status --porcelain; non-empty means uncommitted changes
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
| W1 kernelized IB-Hebbian perception layer | [`research/ib_hebbian/README.en.md`](../research/ib_hebbian/README.en.md) | MNIST 98.01% (threshold 70%) |
| W2 e-prop cognitive layer | [`research/eprop/README.en.md`](../research/eprop/README.en.md) | sMNIST 77.48%; active-neuron fraction 1.0000 (threshold > 60%) |
| W3 R-STDP + TD-LTP Critic | [`research/rstdp/README.en.md`](../research/rstdp/README.en.md) | ✅ **Both §7 criteria met**: CartPole median **260.0 steps** (held-out seeds 25–44, 20 of them, criterion ≥ 200); max offset ratio **0.0359** (all seeds < 0.10 ✓). **But the minimum is 74**, below our own added "stable" condition (≥ 100; 2 of 20 under 100). Configuration and rationale: [ADR-0009](adr/ADR-0009-w3-behaviour-policy-and-trace-centring.en.md) |

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

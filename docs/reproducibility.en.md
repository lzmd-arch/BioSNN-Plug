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

## Where later phases record this

From Phase 1 (single-rule validation) onward, every experiment record must carry the full
template. GPU experiments additionally need to record:

- Peak GPU memory (the project plan §6.2 lists 8GB as a hard constraint)
- Whether the project plan §6.2 fallback paths were triggered (scale reduction / chunked
  training / INT8 trace quantization)
- Training duration and energy-related quantities (§9's energy-efficiency metrics are
  reported in the form of an "invocation-frequency vs. energy curve")

## Data

Per the project plan §12.1, this repository **ships no data mirror**. Download and
preprocessing scripts will ship with Phase 1 — for now `scripts/` holds only the
repository's own check scripts, and no data scripts at all yet.

Convention: data scripts will place data under `data/`, which `.gitignore` ignores, and
verify a hash.

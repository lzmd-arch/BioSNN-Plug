# Research code (`research/`)

[中文](README.md) · [日本語](README.ja.md)

## ⚠️ API instability notice

**The code in this directory makes no compatibility promises for its first 12 months (until 2027-09).**

The project plan §12.5 gives research code a breaking-change grace period:

> Breaking-change grace period: research code is explicitly marked as API-unstable for its
> first 12 months; the skeleton library's separate semver is unaffected.

The reasoning behind that trade-off is in [`docs/adr/ADR-0005`](../docs/adr/ADR-0005-versioning-policy.en.md).

**If you need something to build on, use [`biosnn-bus`](../packages/biosnn-bus/README.en.md)** —
it follows semver and deliberately depends on none of the code in this directory
(see [`ADR-0001`](../docs/adr/ADR-0001-skeleton-as-separate-library.en.md)).

## Current scale

**This project currently runs on 256 spiking neurons.**

The networks of the three verification lines (numbers taken from each line's CLI defaults, i.e. the
acceptance configuration):

| Line | Network | Scale | Type |
| :--- | :--- | ---: | :--- |
| W1 perception | 3 layers x 1024 | **3,072** units | rate-based LReLU -- **not** spiking neurons |
| W2 cognition | single recurrent layer | **256** neurons | **ALIF spiking neurons** (beta=0.07) |
| W3 execution | population critic + actor | **64** units (+2 actions) | rate-based units |
| Skeleton-library demo | spike bus | 128-d (plugins 32 + 128 channels) | plugin encoding units, **not** neurons |

**The only genuine spiking neurons are W2's 256.**

⚠️ **That is about two orders of magnitude short of the plan's targets**, stated here so no reader
mistakes "phase 1 complete" for "the scale is nearly there":

| | Neuron count |
| :--- | ---: |
| **Now (measured)** | **256** |
| §2.2 cognitive core, "initially about 50,000 neurons" | 50,000 |
| §9 phase-2 target | 100,000 |
| §9 phase-4 target | 500,000 |

A gap of roughly **195x**. W1's 3,072 units are the largest count, but they are
rate-based and **do not count toward the "spiking neuron scale" metric**. The scaling path, the
memory constraint and the three degradation paths are in plan §6.2/§6.3; **the scaling experiments
themselves have not started** (§9 puts "100,000" in phase 2).

## What will be here

Following the phase breakdown in the project plan §7, this directory is filled in starting
from **phase 1**:

| Phase | Contents | Plan reference |
| :--- | :--- | :--- |
| Phase 1 (months 1-3) | e-prop cognitive core (ALIF + adaptive threshold; the Trace Propagation named by the plan was **ruled out** by provenance review, see [ADR-0010](../docs/adr/ADR-0010-eprop-quadratic-storage-and-trace-propagation.en.md)); kernelized IB-Hebbian perception layer (including divisive normalization); R-STDP execution layer (including the Critic) | §7 phase 1, §3 |
| Phase 2 (months 4-8) | Dual-rule coordination, TAAF fast fine-tuning, ES meta-learning arbiter | §7 phase 2 |
| Phase 3 (months 9-15) | Three-rule closed loop, continual-learning benchmark | §7 phase 3 |
| Phase 4 onward | Neurogenesis, scaling up, metacognitive gating | §7 phase 4 |

The docstring header of every module must state the grace period and its expiry date.

## Directory conventions

```text
research/
├─ common/                  # shared infrastructure (seeding / device / metrics / record)
├─ <phase-or-topic>/
│  ├─ README.md
│  ├─ <implementation>.py
│  └─ tests/
└─ ...
```

One README per topic, stating three things plainly: **what claim is being tested**,
**the current conclusion**, and **how to reproduce it**. The evaluation-metrics table in
the project plan §9 is the acceptance criterion for those conclusions.

`research/` is a **package**, not a pile of loose scripts — that is what lets the three
lines share the things in `common/` whose definitions must not drift. So entry-point
scripts are always run as **modules**:

```bash
uv run python -m research.<topic>.<script>
```
Running `python research/<topic>/<script>.py` directly also works, but then `sys.path`
holds the script's directory rather than the repository root, and `research.common`
cannot be imported.

Tests live under each topic's `tests/`. The `testpaths` in `pyproject.toml` already
includes `research`, so a local `uv run pytest` picks them up; in CI they belong to the
`research-smoke` job (which needs torch), while the skeleton library and the repository
toolchain belong to the `test` job — each job runs the half it can.

## Minimal reproduction scripts

The project plan §12.3 requires "one `examples/paper_<name>.py` per core paper", to catch
"the paper says it runs but the repository does not". Such scripts live in
[`examples/`](../examples/), not here — they have to be directly executable by CI.

## Additional requirements before submitting

Beyond the repository-wide pre-submit checklist (see
[`CONTRIBUTING.md`](../CONTRIBUTING.en.md)), research code additionally requires:

- experimental conclusions to come with a complete environment record
  (see [`docs/reproducibility.md`](../docs/reproducibility.en.md));
- external conclusions, when cited, to be registered in
  [`docs/references.md`](../docs/references.en.md) with a verification status;
- comparisons against the surrogate-gradient baseline to include a **quantitative gap
  report** — the project plan §9 lists "a quantitative gap report against the
  surrogate-gradient baseline" as a phase-2 acceptance item, not a requirement to close
  the gap.

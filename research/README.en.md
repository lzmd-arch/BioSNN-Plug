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

## What will be here

Following the phase breakdown in the project plan §7, this directory is filled in starting
from **phase 1**:

| Phase | Contents | Plan reference |
| :--- | :--- | :--- |
| Phase 1 (months 1-3) | e-prop cognitive core (Trace Propagation + ALIF + adaptive threshold); kernelized IB-Hebbian perception layer (including divisive normalization); R-STDP execution layer (including the Critic) | §7 phase 1, §3 |
| Phase 2 (months 4-8) | Dual-rule coordination, TAAF fast fine-tuning, ES meta-learning arbiter | §7 phase 2 |
| Phase 3 (months 9-15) | Three-rule closed loop, continual-learning benchmark | §7 phase 3 |
| Phase 4 onward | Neurogenesis, scaling up, metacognitive gating | §7 phase 4 |

The docstring header of every module must state the grace period and its expiry date.

## Directory conventions

```text
research/
├─ <phase-or-topic>/
│  ├─ README.md
│  ├─ <implementation>.py
│  └─ tests/
└─ ...
```

One README per topic, stating three things plainly: **what claim is being tested**,
**the current conclusion**, and **how to reproduce it**. The evaluation-metrics table in
the project plan §9 is the acceptance criterion for those conclusions.

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

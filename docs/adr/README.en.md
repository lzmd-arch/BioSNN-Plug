# Architecture decision records (ADR)

[中文](README.md) · [日本語](README.ja.md)

> The project plan §12.4 lists ADRs as "the core means of countering bus factor = 1". This
> project is maintained by one person: the code can show "how it is done", but it cannot say
> "**why it is done this way**" or "**what else was tried**" — ADRs fill in that part.

## What deserves an ADR

One per **significant trade-off**. The test is simple: if someone later asks "why not write it
that way", and the answer is not in the code, it needs an ADR.

## Format

This uses Michael Nygard's classic four-part structure:

| Section | Contents |
| :--- | :--- |
| **Context** | The constraints and the background at the time. An ADR that does not state its constraints cannot be re-evaluated. |
| **Decision** | What was decided. Use declarative sentences, writing "we did X". |
| **Consequences** | The benefits, the costs, and **which downsides were accepted**. |
| **Alternatives** | The options that were seriously considered and rejected, and the reasons they were rejected. |

## Status

`Proposed` / `Accepted` / `Deprecated` (naming the ADR that replaced it) / `Superseded`.

**Once an ADR is recorded, its body is not edited.** When you change your mind, write a new one
and link the two to each other.

## Index

| Number | Title | Status |
| :--- | :--- | :--- |
| [ADR-0001](ADR-0001-skeleton-as-separate-library.en.md) | Skeleton library split out as `biosnn-bus` | Accepted |
| [ADR-0002](ADR-0002-numpy-core-torch-optional.en.md) | Skeleton library core on numpy, torch as an optional bridge | Accepted |
| [ADR-0003](ADR-0003-entry-point-plugin-discovery.en.md) | Plugin discovery via entry points rather than a global registry | Accepted |
| [ADR-0004](ADR-0004-monorepo-uv-workspace.en.md) | Adopt monorepo + uv workspace rather than two repositories | Accepted |
| [ADR-0005](ADR-0005-versioning-policy.en.md) | Versioning policy: semver for the skeleton library, a 12-month grace period for research code | Accepted |
| [ADR-0006](ADR-0006-plugin-interface-fidelity.en.md) | `ModalityPlugin` keeps the original signature from §2.3 of the project plan | Accepted |
| [ADR-0007](ADR-0007-td-ltp-critic-provenance.en.md) | Critic uses TD-LTP, with provenance in Frémaux et al. (2013) | Accepted |
| [ADR-0008](ADR-0008-spikingjelly-license-and-python-floor.en.md) | SpikingJelly uses OIOSL 1.0, raising the development environment Python floor | Accepted |
| [ADR-0009](ADR-0009-w3-behaviour-policy-and-trace-centring.en.md) | W3's behaviour policy becomes Boltzmann sampling, with the trace centred by the sampling probability | Accepted |
| [ADR-0010](ADR-0010-eprop-quadratic-storage-and-trace-propagation.en.md) | e-prop's quadratic storage holds, but §3.1's remedy (adopting Trace Propagation) is not adopted | Accepted |
| [ADR-0011](ADR-0011-gas-modality-as-out-of-repo-plugin.en.md) | The gas (chemical sensing) modality ships as an out-of-repo third-party package, not under packages/ | Accepted |
| [ADR-0012](ADR-0012-taaf-locality-and-bus-output-contract.en.md) | TAAF's locality red line, the open projection-layer training rule, and the `SpikeBus` output-shape contract | Partially accepted |
| [ADR-0013](ADR-0013-synergy-benchmark-priority-and-arbiter-scope.en.md) | The synergy benchmark becomes P0; the shared-interface plan is rejected (counterfactual arm only); three-way mixing stays in phase five | Accepted |
| [ADR-0014](ADR-0014-adaptive-arbitration-unmeasurable-and-fit-retained.en.md) | Adaptive arbitration is not measurable (not disproved); `ESArbiter.fit` kept on standby; P4 headline changes | Accepted |
| [ADR-0015](ADR-0015-world-model-as-phase-four-component.en.md) | The world model becomes a formal phase-four component (schedule unchanged) | Accepted |
| [ADR-0016](ADR-0016-arousal-channel-and-consolidation-experiment.en.md) | The arousal channel becomes a component (naming instinct) and the consolidation experiment is scheduled | Accepted |
| [ADR-0017](ADR-0017-cl-main-route-switch-and-replay-first.en.md) | Main CL route switches to Route R (replay first); Route W retained; phase-4 entry redefined | Accepted |
| [ADR-0018](ADR-0018-task-aware-cl-gate.en.md) | The phase-3 CL gate becomes task-aware CIL (task-incremental); pure-CIL negatives kept | Accepted |
| [ADR-0019](ADR-0019-stage-four-entry-decoupled-from-relearn.en.md) | Phase-4 entry decoupled from relearn 3×; 3× recorded as unmet and open | Accepted |
| [ADR-0020](ADR-0020-phase-five-m0-readonly-mcp.en.md) | Phase 5 M0: read-only MCP skeleton (no LLM, no writes) | Accepted |

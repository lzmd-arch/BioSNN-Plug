# ADR-0011: The gas (chemical sensing) modality ships as an out-of-repo third-party package, not under packages/

[中文](ADR-0011-gas-modality-as-out-of-repo-plugin.md) · [日本語](ADR-0011-gas-modality-as-out-of-repo-plugin.ja.md)

- **Status**: accepted
- **Date**: 2026-10-06
- **Related**: project plan §2.3 modality table, §7 phase two, §12.3, §12.5, §12.6;
  [ADR-0001](ADR-0001-skeleton-as-separate-library.en.md),
  [ADR-0003](ADR-0003-entry-point-plugin-discovery.en.md),
  [ADR-0004](ADR-0004-monorepo-uv-workspace.en.md); [docs/references.md](../references.en.md)

## Context

Before v6.3 the §2.3 modality table had only three rows — text / image / audio — corresponding to
"text + image plugins" in §7 phase two and "audio plugin integration" in phase three. The fourth
modality is **gas (MOS array / electronic nose)**, and its implementation comes from outside this
repository.

The repository already has two landing paths, and they were designed for **different purposes**:

- `packages/*` — uv workspace members ([ADR-0004](ADR-0004-monorepo-uv-workspace.en.md)), first-class
  citizens. They are **hard-coded** into the root `pyproject.toml`'s `[tool.pytest.ini_options]
  testpaths` and `[tool.coverage.run] source`, the `test` and `package` jobs in `ci.yml`, and the
  release chain in `release.yml`; `packages/*/README.md` falls under the three-language check, and
  the Python code blocks in those READMEs are **actually executed** by CI.
- entry points ([ADR-0003](ADR-0003-entry-point-plugin-discovery.en.md)) — the third-party plugin
  channel, group `biosnn_bus.plugins`. `CONTRIBUTING.md` states "writing a modality plugin does not
  require a pull request here. A standalone package is enough", and gives the
  `[project.entry-points]` form.

## Decision

The gas modality's implementation lives as a **standalone distribution package outside this
repository**, wired in through an entry point (group `biosnn_bus.plugins`), with **zero changes to
this repository's code**. On the plan side only three things are registered: one row in the §2.3
modality table, the phase-two schedule in §7, and one revision entry in the version notes.

## Consequences

The accepted costs first:

- It is **not covered by this repository's CI** — the "what CI actually runs" list in §12.3 does not
  reach it, and it is not released with this repository;
- It depends on `biosnn-bus` from PyPI rather than the workspace path version, so if the skeleton
  library makes a breaking change, **it will not go red in this repository's CI first**; the semver
  ADR-0005 grants the skeleton library is its only protection;
- The §12.3 row "minimal reproduction script" requires one `examples/paper_<name>.py` per core
  reference, and there is **no counterpart in this repository** — that script lands inside the
  out-of-repo package. This is an **explicitly recorded deviation**, not an omission.

Then the benefits:

- Zero changes to this repository: none of the hard-coded paths in `testpaths` / `coverage.source` /
  `ci.yml` / `release.yml` need touching;
- It is the **first real trial** of the third-party channel promised by `CONTRIBUTING.md` and
  [ADR-0003](ADR-0003-entry-point-plugin-discovery.en.md) — until now "third-party packages plug in
  with zero changes" existed in this repository only as prose, with no test ever running
  `discover_plugins()` against a real third-party entry point;
- It is the first instance of §12.6's "month 24: externally contributed modality plugins ≥ 1", and it
  does **not** change phase three's "three modalities integrated" success criterion — that one is a
  self-developed project deliverable, a different thing from an external plugin.

## Alternatives

- **Put it under `packages/biosnn-plug-gassensor/`**: **rejected**. `CONTRIBUTING.md`'s guidance for
  third-party plugins says the opposite in so many words ("do not open a pull request here"); and it
  would drag in a whole set of hard obligations — three-language README, README code blocks executed
  by CI, `testpaths` and `coverage.source` in the root `pyproject.toml`, the `test` and `package`
  jobs in `ci.yml`, the release chain in `release.yml`, and registration in the plan's §12.3 table.
  Those obligations are designed for **self-developed project deliverables**; imposing them on one
  external modality is out of proportion.
- **Put it under `research/gas_sensor/`**: **rejected**. `research/` holds the three **self-developed**
  validation lines (W1/W2/W3), enjoys a 12-month breaking-change grace period (ADR-0005), and is
  currently fully decoupled from the skeleton library; putting an external implementation there would
  confuse both "who is validating what" and "who enjoys the grace period".
- **Schedule the fourth modality into §7 phase three (alongside audio)**: **rejected**. Phase three's
  task is "audio plugin integration" (single modality, self-developed); phase two's "modality plugin
  framework and dual-channel fusion" is where a new modality belongs.

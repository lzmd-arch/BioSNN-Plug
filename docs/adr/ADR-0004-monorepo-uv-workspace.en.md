# ADR-0004: Adopt monorepo + uv workspace rather than two repositories

[中文](ADR-0004-monorepo-uv-workspace.md) · [日本語](ADR-0004-monorepo-uv-workspace.ja.md)

- **Status**: Accepted
- **Date**: 2026-09-21
- **Related**: project plan §12.2 (layered open-source route); [ADR-0001](ADR-0001-skeleton-as-separate-library.en.md), [ADR-0005](ADR-0005-versioning-policy.en.md)

## Context

[ADR-0001](ADR-0001-skeleton-as-separate-library.en.md) decides to extract the skeleton library
into a separate distribution package. "Separate distribution package" is not the same thing as
"separate repository" — these are two different matters, and they are easily conflated.

The layered open-source route in project plan §12.2 says "the skeleton library (biosnn-bus) goes
open source first", which on the face of it looks more like two repositories. But the project plan
also states plainly that this project is **maintained by one person** (§12.5: no SLA on issue
response, made explicit in the README).

Single-maintainer is the decisive constraint here. Two repositories mean: two sets of CI, two
commits, two issue queues, and — most troublesome of all — **cross-repository changes**. Change one
interface in the skeleton library and the research code has to change with it: two PRs to open,
two CI runs to wait for, and commit hashes cited from one side to the other.

## Decision

**A single monorepo**, with the skeleton library as a subpackage:

```text
BioSNN-Plug/
├─ packages/biosnn-bus/    separate distribution package: its own pyproject, its own version number, its own tests
├─ research/               research code (from phase 1 onward)
└─ docs/  examples/  scripts/
```

Managed with a **uv workspace**: the root `pyproject.toml` declares the members, a single
`uv sync` installs the development environment, and the skeleton library is installed in editable
mode, so that changing the code takes effect immediately.

The key point: **it is still an independently publishable distribution package**. The wheel that
`uv build --package biosnn-bus` produces is exactly the same as one built from a separate
repository, and can be published to PyPI on its own. Repository form and distribution form are
decoupled.

## Consequences

**Benefits**

- One commit can change the skeleton library and the research code at the same time, and CI runs
  once;
- One issue queue, one set of governance files;
- A single `uv sync` installs all development dependencies;
- The skeleton library's "independence" is guaranteed by the **dependency graph** (it does not
  import the cognitive core, and CI asserts that no torch is present after the wheel is installed),
  not by its directory location — the latter is form, the former is substance.

**Costs (accepted)**

- The repository looks bloated to anyone who only cares about the skeleton library.
  `packages/biosnn-bus/README.md` being separate from the README published to PyPI mitigates this
  somewhat;
- The skeleton library's CI and the research code's CI are in the same workflow. They are
  currently separated by job; if the research code's CI becomes very heavy in the future (GPU
  tests, long training runs), path filtering needs to be added;
- If active third-party maintainers do appear in the future, the skeleton library may need to be
  split out. **Split it then** — splitting is far easier than recombining.

**What this does not settle**

- GitHub's contributor statistics and star counts are computed per repository, so the skeleton
  library cannot accumulate those metrics on its own. Given that project plan §12.6 states plainly
  that "being cited by a peer-reviewed paper" is truer than star count, this loss is acceptable.

## Alternatives

**A. Two repositories: `biosnn-bus` and `BioSNN-Plug` kept apart.**
Rejected (at this stage). Under single-maintainer, the friction of cross-repository changes would
genuinely slow development down, and the narrative effect it buys — "the skeleton library goes
open source first" — is equally achievable with a subpackage in a monorepo.
**This decision is reversible**: the `packages/biosnn-bus/` directory is itself a complete,
publishable package, so when the split is really wanted, `git subtree split` will do it, and the
history comes along.

**B. A single repository, but with the skeleton library not packaged separately (a top-level `biosnn_bus/` directory instead).**
Rejected. That goes back to the option [ADR-0001](ADR-0001-skeleton-as-separate-library.en.md)
rejected: it cannot be published independently, and outside users cannot install it.

**C. Attach the skeleton library as a git submodule.**
Rejected. A submodule's day-to-day operations (updating the pointer, detached HEAD, forgetting
`--init`) are pure friction for contributors, and the payoff is only that it "looks like two
repositories".

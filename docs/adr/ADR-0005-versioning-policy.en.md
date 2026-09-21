# ADR-0005: Versioning policy — semver for the skeleton library, a 12-month grace period for research code

[中文](ADR-0005-versioning-policy.md) · [日本語](ADR-0005-versioning-policy.ja.md)

- **Status**: Accepted
- **Date**: 2026-09-21
- **Related**: project plan §7 phase 0 task 5, §12.2, §12.5; [ADR-0001](ADR-0001-skeleton-as-separate-library.en.md), [ADR-0004](ADR-0004-monorepo-uv-workspace.en.md)

## Context

The project plan §12.5 contains a sentence that sounds harsh but is honest:

> **Managing response expectations**: maintained by one person, no SLA on issue responses, stated
> plainly in the README — managing expectations builds more community trust than pretending there
> is a team.

The same honesty has to apply to API stability. In a 36-month research project, the first 12
months keep overturning your own design: interfaces deform because an experiment needs them to,
directories get reorganised, and an abstraction that looks elegant today may be proven wrong
tomorrow.

Promising API stability for **all** the code is not achievable in a research project like this
(the reasons for rejecting it are in Alternatives A).

But the skeleton library is a different matter: its whole reason to exist is to be depended on by
others ([ADR-0001](ADR-0001-skeleton-as-separate-library.en.md)). Nobody will write plugins
against a library that works today and breaks tomorrow. The hard metric in the project plan §12.6
— "month 8: the skeleton library is imported by ≥ 1 project not written by the author" —
presupposes API stability.

## Decision

**A two-layer policy, written into the documentation and written into CI:**

### Skeleton library `biosnn-bus` — strict semver

- A breaking change is allowed only when the major version number changes;
- 0.x phase (currently `0.1.0`): under the semver convention, the minor version of a 0.x release
  (`0.1` → `0.2`) may contain breaking changes. **This is deliberate** — the skeleton library is
  not yet depended on externally, and at this point iterating quickly is worth more than
  pretending to be stable. It goes to `1.0.0` when the first externally depended-on version is
  released, and from then on the convention is followed strictly;
- `scripts/check_version_consistency.py` guarantees in CI that the version numbers in
  `pyproject.toml` and `__init__.py` are always the same — the installed version and the version
  the library reports at runtime disagreeing is one of the most time-wasting classes of problem
  when you are debugging.

### Research code `research/` — a 12-month grace period for breaking changes

- The API is unstable for the first 12 months, and **no compatibility promise is made**;
- It takes no part in the version number consistency check;
- The top of each module's docstring records the grace period and the expiry date.

### Project plan document — the file name and the version number in the body must agree

The project plan §12.3 calls out the need to block "the file name and the version number in the
contents being out of step": a reader looks for v6 by the file name and gets the contents of v6.2.

## Consequences

**Benefits**

- Contributors know which APIs they can safely depend on (the skeleton library) and which they
  should not (the research code);
- The skeleton library can iterate quickly during the 0.x phase, without paying the price for an
  external dependency that has no users yet;
- Version number consistency is guaranteed by CI, not by good intentions.

**Costs (accepted)**

- The 0.x phase may still break things, and early adopters need to be prepared for that. The
  README states this plainly;
- The grace period on the research code means that **results from this phase are hard for a third
  party to reuse directly**. This is exactly why the project plan §12.2 schedules the "reference
  implementation layer" for months 4-8 — first let the reusable part stabilise, then open up the
  research code.

**What happens when the grace period expires**

When the 12-month grace period ends (expected 2027-09), the research code is either refactored
into the skeleton library as a stable API, or labelled "experimental, no compatibility promise".
**This ADR needs review when it expires.**

## Alternatives

**A. Promise semver for all the code.**
Rejected. Either it ties your hands (the self-imposed review cost during the research phase is
extremely high), or you break the promise (which is worse).

**B. Promise no stability at all.**
Rejected. The skeleton library's whole reason to exist is to be depended on. Without a stability
promise, "imported by ≥ 1 project not written by the author" in the project plan §12.6 can never
be met.

**C. Include the research code in the version number consistency check too.**
Rejected. The research code adds and removes things one experiment at a time, and the benefit
of enforcing version number discipline is far smaller than the friction.

**D. Ship 1.0.0 from the start to signal stability.**
Rejected. 0.x is the standard way in the Python ecosystem to express "this can change", and at the
moment it genuinely will change. Shipping 1.0 early would only be deceiving users.

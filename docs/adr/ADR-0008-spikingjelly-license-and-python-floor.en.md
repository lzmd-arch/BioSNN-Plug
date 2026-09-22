# ADR-0008: SpikingJelly is licensed under OIOSL 1.0, and the development environment's Python floor rises accordingly

[中文](ADR-0008-spikingjelly-license-and-python-floor.md) · [日本語](ADR-0008-spikingjelly-license-and-python-floor.ja.md)

- **Status**: Accepted
- **Date**: 2026-09-22
- **Related**: project plan §1.4, §七 phase 1, §八, §12.1, §12.3; the v6.3 correction table in [docs/references.md](../references.en.md)

## Context

§1.4 and §八 of the project plan name **SpikingJelly** as this project's framework choice. In the license strategy table of §12.1, the source-code row reads:

> Source code | **Apache-2.0** | Compatible with the SpikingJelly ecosystem (same license); the patent grant terms are friendly to industrial users

Phase 1's first task is "environment setup" — actually installing SpikingJelly. Before installing it, we checked its license. The conclusion does not match that sentence.

### Fact one: SpikingJelly is not Apache-2.0; it is the Open-Intelligence Open Source License 1.0

**The Open-Intelligence Open Source License, Version 1.0 (OIOSL)**. The verifiable evidence:

- The repository root has a `LICENSE` (the license's authoritative Chinese text) and a `LICENSES/` directory (translations plus a usage and distribution guide); the License section of the README reads "SpikingJelly is distributed under the Open-Intelligence Open Source License Version 1.0";
- The GitHub API's `license` field is `NOASSERTION` — licensee could not identify it as a standard license;
- On PyPI, the stable release `0.0.0.0.14` carries the classifier `License :: Other/Proprietary License`; `2.0.0rc1` has an empty License field.

According to the license text, the actual conditions of OIOSL 1.0 include:

- **Use and redistribution are both permitted**, in source and executable form, provided the license conditions, the license notice, and the disclaimer are retained;
- **Use or redistribution for commercial purposes requires prior public disclosure on the AITISA website** (the New Generation Artificial Intelligence Industry Technology Innovation Strategic Alliance, aitisa.org.cn), including at least the name of the user/redistributor, contact details, address, telephone, email, and purpose of use;
- The patent grant sits in a "voluntary patent declaration": if the licensor holds patents covering the software, then for users holding no relevant patents it grants one of FRAND-RF / joining the patent pool / FRAND; for users who do hold relevant patents, the licensor "has the right but no obligation" to license on a reciprocal basis.

### Fact two: this repository's existing license audit silently lets it through

`scripts/check_licenses.py` compares license names against `FORBIDDEN_PREFIXES = ("AGPL", "GPL", "SSPL", "BUSL", "CPAL", "OSL")` by **prefix**. `Open-Intelligence Open Source License` matches none of those six prefixes; and SpikingJelly's wheel metadata has an empty License field, so it lands in the `unknown` bucket — and CI invokes the script without `--strict`, meaning **unknown is only printed, never failed**.

§12.3 of the project plan says the license audit exists to catch exactly this kind of problem. It does not catch this one.

### Fact three: SpikingJelly 2.0.0rc1 requires Python >= 3.11

`2.0.0rc1` declares `requires_python` of `>=3.11` and depends on `torch>=2.6`. And it is the **only** release line that supports a modern torch: the previous stable release, `0.0.0.0.14`, dates from 2023-03, predating numpy 2 and torch 2.14. The root project was then `requires-python = ">=3.10"`, and the local Python is 3.10.11.

### One thing checked along the way

The other component named in §八 of the project plan, **eprop-PyTorch, is Apache-2.0** — the license itself is compatible. But its last commit is dated 2022-02-18.

## Decision

**1. Keep SpikingJelly and accept OIOSL 1.0's conditions.** Replacing it would mean abandoning the stack §1.4/§八 designates, and it serves only two purposes here: supplying LIF/ALIF neuron primitives, and acting as the surrogate-gradient BPTT control baseline. This project is a research prototype and non-commercial, so the commercial-use disclosure obligation is not triggered in its current form; but it is an **obligation downstream users inherit**, so it belongs in the documentation rather than in a corner nothing detects.

**2. The root project's Python floor rises to >= 3.11**, with SpikingJelly pinned to `==2.0.0rc1`.

The cost is that CI's pytest matrix narrows from 3.10/3.11/3.12 to 3.11/3.12. **`biosnn-bus` keeps its own `requires-python >=3.10`** — it genuinely depends only on numpy and does run on 3.10, and that public claim should not be dragged along by the root development environment's floor. So that the claim does not become empty, CI's `package` job grows to two legs, 3.10 and 3.12: install the wheel in a clean venv, then run the skeleton library's test suite. **What changes is the development environment's floor, not the distribution package's.**

**3. eprop-PyTorch is a reading reference and cross-validation source only; it does not become a dependency.** Its license is compatible, but it has been unmaintained since 2022 and pins torch APIs that do not match 2.14. e-prop is implemented in-house; for the hardest part (ALIF eligibility traces) its source serves as a reading companion to check item by item, and **where they conflict, the original Bellec et al. 2020 text governs**.

**4. The license audit tightens: "undetectable" becomes "must be explicitly declared".** `scripts/check_licenses.py` gains an allowlist registering **known and accepted** non-standard licenses (each with a reason and a link to this ADR), plus a new `--strict-unknown` flag: any unknown not registered in the allowlist fails. CI switches to `--strict-unknown`, and the audit's `uv sync` gains `--group research` — otherwise torch, SpikingJelly, and gymnasium never enter the audit's scope at all.

The allowlist is a hole, so every entry must state its reason. That is the same rule §12.5 sets for the `EXEMPT` table.

**5. Project plan §12.1 needs a v6.3 correction**, replacing "compatible with the SpikingJelly ecosystem (same license)" with a factual statement and adding OIOSL's commercial-use disclosure obligation. Until that correction lands, this ADR governs wherever §12.1 is cited.

## Consequences

**Benefits**

- Environment setup no longer rests on an unverified license assertion;
- The license audit moves from "blocks strong copyleft only" to "unknown fails, acceptance must be registered". A case like OIOSL — **neither copyleft nor a standard open-source license** — finally has a home;
- The skeleton library's public Python support claim gains real coverage (the CI 3.10 leg installs the wheel and runs the test suite) instead of being lifted along with the root;
- eprop-PyTorch's role is written down, so "why not just use it?" has an answer.

**Costs (accepted)**

- **OIOSL's commercial-use disclosure obligation travels with distribution.** This repository's own research use does not trigger it; but anyone using this project (and thus the SpikingJelly dependency) for commercial purposes must bear the disclosure obligation to AITISA themselves. That belongs in the README's dependency notes, not only in an ADR.
- **Depending on a pre-release.** `spikingjelly==2.0.0rc1` is pinned exactly; when it reaches a stable release this ADR must be re-evaluated — **especially whether the license terms changed**.
- CI's pytest loses its 3.10 leg; skeleton tests on 3.10 move to the `package` job, where coverage shifts from "run in the source tree" to "install the wheel, then run" — not fully equivalent.
- `uv.lock` now records two sets of torch wheels, CPU and cu130, making the file larger and the lock diff noticeably noisier on torch upgrades.

**What this ADR explicitly does not claim**

1. **It does not claim OIOSL 1.0 "is not an open-source license".** What is recorded here are **verifiable facts**: it is not Apache-2.0, GitHub's licensee cannot identify it (`NOASSERTION`), the PyPI stable release is marked `Other/Proprietary`, and it carries a commercial-use disclosure obligation. **The full official OSI list was not checked item by item** (the license list page at opensource.org is not readable by scripts), so no claim is made about its OSI status.
2. **It does not claim a legal review was performed.** The above is a reading of the license text, not legal advice. If the project's shape changes (for instance, accepting commercial sponsorship or offering a hosted service), it should be re-evaluated.
3. **It does not claim in-house neuron primitives would be an equivalent substitute.** e-prop's correctness depends on an exact implementation of ALIF dynamics, and implementing that in-house carries its own risk; keeping SpikingJelly is precisely how this ADR avoids importing that risk.

## Alternatives

**A. Implement LIF/ALIF neuron primitives in-house and drop SpikingJelly entirely (roughly 200–300 lines).**

Rejected. The benefits are clear: no license entanglement, no Python floor change, all three CI legs preserved. The cost is that the correctness of ALIF eligibility traces rests entirely on us — and that is precisely the part of phase 1 that most needs an external reference. §1.4/§八 designating SpikingJelly points the same way. If OIOSL's terms ever tighten beyond acceptability, this is the cheapest exit path.

**B. Pin the old stable release `spikingjelly==0.0.0.0.14` and leave the Python floor alone.**

Rejected. That release dates from 2023-03, predating numpy 2 and torch 2.14; installing it would either fail at import or force torch down to an old version too — and old torch does not support the RTX 5060's sm_120. This is not "conservative"; it pins the environment to a combination that cannot run this project's hardware.

**C. Switch to an Apache/MIT-licensed SNN framework.**

Deferred, not rejected. It would solve both the license and the Python floor at once, but it departs from the project plan's technology choices, and phase 1's §七 neuron primitives and surrogate-gradient baseline would all have to be re-checked. **This ADR does not assess that route's feasibility** — if OIOSL's commercial-use disclosure obligation becomes a real obstacle, a new ADR should compare it seriously.

**D. Leave the audit script as is and merely note SpikingJelly's license in the documentation.**

Rejected. That is exactly the "undetectable corner" — §12.3 lists the license audit as a CI item precisely so that it does not depend on someone remembering. Documentation drifts; `--strict-unknown` does not.

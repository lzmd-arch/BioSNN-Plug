# ADR-0001: The skeleton library as a separate `biosnn-bus`

[中文](ADR-0001-skeleton-as-separate-library.md) · [日本語](ADR-0001-skeleton-as-separate-library.ja.md)

- **Status**: Accepted
- **Date**: 2026-09-21
- **Related**: project plan §7 (phase 0), §12.2 (layered open-source route); [ADR-0002](ADR-0002-numpy-core-torch-optional.en.md), [ADR-0004](ADR-0004-monorepo-uv-workspace.en.md)

## Context

The project plan bets the whole project on one high-risk technical route: **give up surrogate
gradients entirely**, and compose cognitive ability out of three purely local rules — e-prop +
kernelized IB-Hebbian + R-STDP. The project plan itself admits that this route "does carry
higher technical risk" (§13), and it reserves several fallback paths (§11).

In other words: **the core research may well fail, or be substantially downgraded.**

At the same time, the route contains one part that **does not depend on whether the research
succeeds**: the modality plugin interface contract, the registration and discovery mechanism,
and the seams of the spike bus. Whether or not e-prop works, whether or not TAAF can align
modalities, this part is needed either way, and its value is unrelated to whether the learning
rules succeed.

If it were written into the same package as the cognitive core, it would carry the full risk of
the research code along with it: an API that keeps changing, dependencies that keep growing, and
a pile of things you do not need pulled in by `pip install`.

## Decision

Extract the plugin framework into a **separate distribution package**, `biosnn-bus`, living in
`packages/biosnn-bus/`:

- it **does not import** the cognitive core, the learning rules, or any neuron model;
- it has its own version number, its own tests, its own README;
- it is available to the outside world before any research result exists.

The project plan §12.2 states this principle in one sentence: **the research may fail; the
skeleton must be usable.**

## Consequences

**Benefits**

- The research code can change freely, and the skeleton library's API promises are unaffected;
- `pip install biosnn-bus` pulls only numpy, consistent with the hard success criterion of "no
  GPU dependency, runs on Colab";
- It can accumulate users, credibility and external review before any research result exists —
  the project plan §12.6 defines "being cited by a peer-reviewed paper as a baseline or a tool"
  as a truer measure of success than star count, and that route is only open if the skeleton
  library exists.

**Costs (accepted)**

- One more layer of abstraction: for the cognitive core to use a plugin, it has to go through the
  `SpikeBus` interface. Some experiments will want the raw spike data directly, and bypassing the
  bus is less work — **such bypassing is allowed**; the plugin interface does not force you
  through the bus;
- Maintaining two version numbers and two READMEs. See [ADR-0005](ADR-0005-versioning-policy.en.md);
- The skeleton library has its own CI, which slightly increases build time.

**What this does not settle**

- Whether the skeleton library's interface is well designed can only be falsified by outside
  users. There are no outside users yet — that is the question the month-8 metric in the project
  plan §12.6 has to answer, and this ADR cannot settle it.

## Alternatives

**A. A single package, with the plugin framework as a submodule.**
Rejected. Then `pip install biosnn-plug` would drag in torch, experiment dependencies, and a pile
of code abandoned partway through the research — in direct conflict with "get the Colab demo
running within 5 minutes".

**B. Write the research code first, and extract the library once the interface has stabilised.**
Rejected. This is the most common order, but it reverses the premise that "the skeleton must come
first". The project plan §12.2 judges that the skeleton library needs no research to succeed
first, yet can accumulate users and outside eyes before the research succeeds. Extracting it only
after the research is done means giving up what those years would have accumulated.

**C. Skip the skeleton library entirely and go all-in on the research.**
Rejected. Same reason as above, and the hard metrics in the project plan §12.6 (by month 2, the
skeleton library installable with `pip install`, CI all green) would immediately be missed.

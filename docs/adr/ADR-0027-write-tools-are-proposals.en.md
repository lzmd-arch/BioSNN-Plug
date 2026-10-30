# ADR-0027: write tools are proposals, not writes

[中文](ADR-0027-write-tools-are-proposals.md) · [日本語](ADR-0027-write-tools-are-proposals.ja.md)

- **Status**: accepted
- **Date**: 2026-10-31
- **Related**: [ADR-0020](ADR-0020-phase-five-m0-readonly-mcp.en.md) (the read-only boundary),
  [ADR-0026](ADR-0026-mcp-write-tools-and-explicit-arming.en.md) (the 9-tool surface),
  `packages/biosnn-mcp`

## Context

[ADR-0026](ADR-0026-mcp-write-tools-and-explicit-arming.en.md) added five "write tools" to the MCP
server. Their real scope is the **in-memory stand-in core** - they never touch a real memory store
(the Axo store is locked read-only separately, see `docs/security.md`: SQLite `mode=ro`, writes
refused by the engine).

That leaves a name-versus-behaviour mismatch: a tool called `memory.write` that **never writes any
memory store**. By this repository's rule that a doc/behaviour mismatch is a bug, a footnote saying
"it is fake" is not an acceptable fix.

## Decision

1. **Rename the five tools into a `propose.` namespace**: `propose.memory.write`,
   `propose.memory.consolidate`, `propose.trigger.arm`, `propose.trigger.disarm`,
   `propose.trigger.feed`; the four read-only tools (`state.query`, `memory.search`,
   `trigger.status`, `trigger.poll`) keep their names.
2. **Write the contract into the name and the payload**: every `propose.*` tool is permanently
   scoped to the stand-in core; payloads carry `"scope": "stand-in"` and `state.query` reports
   `"write_scope": "stand-in"`.
3. **The write tools are never "promoted" to real writes.** Their value is twofold: early adopters
   can exercise the whole tool surface, and conformance/integration tests have a target. They are
   **interface scaffolding**, not half-finished wiring waiting for a switch.
4. **If proposals must one day produce real memory effects**, the answer is not to wire these tools
   up but to add a **dynamics-owned write path**: a `propose` queue plus consolidation-gated
   consumption - proposals enqueue, the dynamics dispose of them, everything audited as JSONL.
   These tools then retire, replaced by `biosnn_propose_*`.
5. **Guard**: tests assert that any non-read-only tool starts with `propose.` and mentions
   `stand-in` in its description.

## Consequences

- The name is the contract: what a client reads already says it will not write any real store;
- Any new write-flavoured tool must pass the naming guard first (the test has to be changed);
- The version stays **0.2.0** - the rename happened before the first publication, so no published
  contract breaks;
- The Axo read-only decision is unaffected: `axo-mcp`'s tool surface is unrelated to this ADR, and
  its store stays read-only at the engine level;
- Cost: `propose.*` reads more verbosely than `write.*`. That is deliberate - verbosity for honesty.

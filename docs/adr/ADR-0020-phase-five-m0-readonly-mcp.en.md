# ADR-0020: Phase 5 M0 — a read-only MCP skeleton; interfaces before autonomy

[中文](ADR-0020-phase-five-m0-readonly-mcp.md) · [日本語](ADR-0020-phase-five-m0-readonly-mcp.ja.md)

- **Status**: Accepted
- **Date**: 2026-10-29
- **Related**: plan §VII phase 5, [ADR-0007](ADR-0007-td-ltp-critic-provenance.en.md)
  (the phase-5 provenance gate), [ADR-0019](ADR-0019-stage-four-entry-decoupled-from-relearn.en.md),
  [stage-4 status](../../docs/stage4_status.en.md)

## Context

Phase 4 is closed under the strict reading: E2E v18 is 18/18 at the state level, P3 v6
rolling-horizon closed loop is ready, and WM v9 / episodic v5 / metacognitive v11 are ready;
**the 500-step pure-imagination P3 loop remains a proven structural boundary**. Phase 5
(LLM integration and coordination) has not started: the repository has no MCP Server, LLM
adapter, trigger module, or dialogue trace.

Writing a script that calls an LLM on a timer would create a system that looks autonomous but
is unauditable: no tool schema, no trigger definition, no per-turn record of state/retrieval/
call/consolidation. Phase-5 success criteria (first autonomous call within 10 turns, trigger
accuracy 70%+, consolidation 80%+, task-completion gain 50%+) all require an observable pipeline.

## Decision

1. **Start phase-5 M0: a read-only MCP skeleton**. Add `packages/biosnn-mcp`, a dependency-free
   JSON-RPC 2.0 over stdio server exposing three read-only tools: `state.query`,
   `memory.search`, `trigger.status`; the cognitive core is an `InMemoryCore` stand-in.
2. **M0 does not connect an LLM, write memory, or arm the trigger**. `trigger.status` always
   returns unarmed; no "autonomous trigger" claim is valid at M0.
3. **The LLM is an external inference engine only**: it never enters a synaptic learning path
   or receives gradients; API keys never enter the repository; every tool call must be
   recordable and replayable.
4. **Phase-5 acceptance remains defined by new M1–M3 preregistrations**: M1 trigger module +
   mock LLM adapter, M2 result re-encoding/consolidation, M3 dialogue evaluation; M0 does not
   count toward phase-5 success criteria.
5. **The phase-4 boundary is unchanged**: the 500-step pure-imagination P3 loop stays open;
   starting phase 5 does not write it up as solved.

## Consequences

- We get a minimal interface that an MCP client can enumerate and tests can drive, with no
  network dependency; the tool schema and read-only boundary are fixed before the real core
  is plugged in;
- Cost: M0 demonstrates no "intelligence"; `trigger.status` is always false and can be
  misread as "missing feature". This is deliberate — "unarmed" is the default so scripted
  calls are not mistaken for autonomous triggering;
- The next milestone must add a mock LLM adapter and a dialogue trace before a real LLM;
- The new package must be tested in CI; it does not affect the independence of `biosnn-bus`
  (ADR-0001).

## Alternatives

- **Connect a real LLM and call it on a timer now**: fastest to show "a call", but cannot
  distinguish autonomous triggering from a schedule, and the key/audit questions are open;
  rejected.
- **Write only scripts/functions, no MCP**: less code, but no tool schema or client contract,
  so the phase-5 "replaceable inference engine" has no interface; rejected.
- **Start with an HTTP service**: MCP stdio is smaller and has no port/auth surface; HTTP can
  be evaluated at deployment; rejected for M0.
- **Wait for the pure-imagination P3 loop to be solved before starting**: that boundary is
  unreachable for any non-bit-exact model (see the P3 v6 README); blocking phase 5 on it would
  wait forever, so we record the boundary and start M0 rather than pretend it is solved.

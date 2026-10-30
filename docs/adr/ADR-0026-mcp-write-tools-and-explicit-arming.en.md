# ADR-0026: MCP write tools and explicit arming (M0 -> M1)

[中文](ADR-0026-mcp-write-tools-and-explicit-arming.md) · [日本語](ADR-0026-mcp-write-tools-and-explicit-arming.ja.md)

- **Status**: accepted
- **Date**: 2026-10-30
- **Related**: [ADR-0020](ADR-0020-phase-five-m0-readonly-mcp.en.md) (the M0 read-only boundary),
  plan section 7 (phase five), [phase-five gap list](../../sweep_results/stage1_5_gap_plan.md)

## Context

[ADR-0020](ADR-0020-phase-five-m0-readonly-mcp.en.md) fixed M0 as a read-only skeleton: three
read-only tools, `trigger.status` permanently unarmed. Everything from M1-M3 has since been
implemented (mock and real LLM adapters, the dialogue trace, sleep consolidation, the spiking
trigger, real embeddings), but **none of it is an MCP tool** - an external client could look but
not write, so it could never actually be used.

ADR-0020 also set two hard constraints: **the LLM is only an external inference engine** (never in
the learning path, keys never committed) and **every tool call must be recordable and replayable**.
Extending the tool surface has to satisfy both.

## Decision

1. **The tool surface grows to 9**, in two layers:
   - read-only (available in any mode): `state.query`, `memory.search`, `trigger.status`,
     `trigger.poll`;
   - write (requires `--allow-write`): `memory.write`, `memory.consolidate`, `trigger.arm`,
     `trigger.disarm`, `trigger.feed`.
2. **Read-only by default.** The write tools still appear in `tools/list` (a client can discover
   them), but calling one in read-only mode returns `isError: true` with an error text saying
   `--allow-write` is required. `InMemoryCore(writes_enabled=...)` decides the capability; the
   server exposes it as a CLI flag.
3. **Arming requires a reason.** `trigger.arm` demands a non-empty `reason` string and records it
   in the status; `disarm` drops pending events. `trigger.feed` advances the in-package
   `SpikingTrigger` by one step with a salience in `[0, 1]`; `trigger.poll` drains and returns the
   events (**read-only**, callable in any mode).
4. **Auditable**: the server writes one JSONL line per `tools/call` (`ts` / `tool` / `ok` / `error` /
   `arguments`), at the path given by `--audit-log`; without a path the entries stay in memory
   (for tests).
5. **Consolidation uses deterministic hash embeddings** (`HashEmbedding`) so `memory.consolidate`
   works without Ollama; threshold, rounds, noise and seed are tool arguments, defaulting to
   `SleepConsolidator`'s values.
6. **Version 0.2.0** (`biosnn-mcp`).

## Consequences

- External clients can now **write, arm and consolidate**, and by default nothing mutates silently:
  writing needs the switch, arming needs a reason, every call leaves a trace;
- `state.query` reports `milestone` `M1` (was `M0`) and `read_only` follows the switch; the old
  "M0 is always read-only" assertion in the tests is replaced by "write tools are refused in
  read-only mode";
- **The LLM is still not connected**: `llm` stays `not_connected`, and this ADR puts no LLM call
  behind a tool;
- **`armed=true` must not be written up as "autonomous triggering works"**: arming is only a state;
  the quality of triggering is still decided by the M3k-M3p evidence (passes on the synthetic
  stream, fails on a real one - see the per-run READMEs);
- The write tools mutate an **in-memory stand-in core**, not a real memory store; replacing it with
  the Axo adapter or a real store must reuse the same protocol and the same audit convention.

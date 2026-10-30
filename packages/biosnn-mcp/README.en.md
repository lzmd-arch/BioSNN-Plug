# biosnn-mcp

[中文](README.md) · [日本語](README.ja.md)

The **phase-five MCP server** of [BioSNN-Plug](https://github.com/lzmd-arch/BioSNN-Plug):
JSON-RPC 2.0 over stdio, no third-party dependencies, exposing the cognitive state, episodic
memory, sleep consolidation and the spiking trigger as MCP tools.

> **Boundary**: the LLM is only an **external inference engine** - this package calls no model,
> never reads an API key into memory and never enters a synaptic learning path. The tool surface is
> **read-only by default**; writes need an explicit `--allow-write` and **every call leaves one
> audit line** ([ADR-0026](../../docs/adr/ADR-0026-mcp-write-tools-and-explicit-arming.en.md)).

## Install

Not on PyPI yet; install from the repository:

```bash
pip install "biosnn-mcp @ git+https://github.com/lzmd-arch/BioSNN-Plug.git#subdirectory=packages/biosnn-mcp"
```

Or run it in place (point `PYTHONPATH` at the sources when it is not installed):

```bash
PYTHONPATH=packages/biosnn-mcp/src python -m biosnn_mcp --allow-write
```

## Quick start

```bash
# read-only (default)
biosnn-mcp

# writable + audited
biosnn-mcp --allow-write --audit-log ~/biosnn-mcp-audit.jsonl
```

MCP client configuration (Claude Desktop or any stdio client):

```json
{
  "mcpServers": {
    "biosnn": {
      "command": "biosnn-mcp",
      "args": ["--allow-write", "--audit-log", "/tmp/biosnn-mcp-audit.jsonl"]
    }
  }
}
```

A hand-written JSON-RPC line works too:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"state.query","arguments":{}}}' | biosnn-mcp
```

## Tools

| Tool | Layer | Arguments | Purpose |
| :--- | :--- | :--- | :--- |
| `state.query` | read | - | phase/milestone, entry count, `read_only`, armed, pending triggers |
| `memory.search` | read | `query`, `top_k` (1-50) | token-overlap search over memory entries |
| `trigger.status` | read | - | armed flag, arming reason, fed samples |
| `trigger.poll` | read | - | drain and return queued trigger events |
| `memory.write` | write | `text`, `id?`, `tags?` | store one entry; id conflicts are rejected |
| `memory.consolidate` | write | `threshold`, `rounds`, `noise_sigma`, `seed` | one sleep-consolidation pass (deterministic hash embeddings) |
| `trigger.arm` | write | `reason` | arm the spiking trigger; **a reason is mandatory** |
| `trigger.disarm` | write | - | disarm and drop pending events |
| `trigger.feed` | write | `salience` (0-1) | advance the trigger population by one step; hits are queued while armed |

In read-only mode the write tools **still appear in `tools/list`** (clients can discover them) but
calling one returns `isError: true` with a message saying `--allow-write` is required - deliberately,
so that "the tool does not exist" cannot be misread as "the capability is missing".

## Audit log

With `--audit-log PATH`, every `tools/call` appends one JSONL line:

```json
{"ts":"2026-10-30T06:30:00Z","tool":"memory.write","ok":true,"error":null,"arguments":{"text":"..."}}
```

Arguments are recorded verbatim (no tool here takes a sensitive field); without a path the entries
stay in memory for tests to assert on.

## State and milestones

`state.query` reports `milestone` `M0` in read-only mode and `M1` when writes are enabled;
`read_only` follows the switch. **`armed=true` only means the trigger is armed** - the quality of
triggering is decided by the phase-five M3k-M3p evidence (passes on the synthetic stream, fails on
the real dialogue stream) and must not be read as "autonomous triggering works".

## What it does not do

- call any LLM, read an API key, or touch the network;
- compute gradients or enter a synaptic learning path;
- write a real memory store - it mutates an in-memory stand-in core; wiring a real store must reuse
  the same protocol and the same audit convention.

## Tests

```bash
uv run pytest packages/biosnn-mcp/tests -q
```

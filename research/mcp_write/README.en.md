# MCP write mode: system-level acceptance (P2)

[中文](README.md) · [日本語](README.ja.md)

**2026-10-31 -- 14 of 14 checks passed, in one run.** The script is
[`scripts/e2e_write_mcp.py`](../../scripts/e2e_write_mcp.py); the criteria are W1-W6 as
pre-registered in the plan (S3.4, frozen 2026-10-31 per S9.2).

## How to run

```bash
python scripts/e2e_write_mcp.py            # temp directory, removed afterwards
python scripts/e2e_write_mcp.py --keep     # keep it, to read store.json / audit.jsonl
```

The script starts a real `python -m biosnn_mcp write` child process and speaks JSON-RPC over
stdin/stdout -- exactly what an MCP client sees, not a direct Python-API call.

## Result (2026-10-31, Python 3.13 / Windows 11)

| Rule | Check | Result |
| :--- | :--- | :--- |
| W6 | Tool table has 13 entries, the 9 read-only ones first | PASS |
| W1 | Proposal is judged `accepted` | PASS |
| W1 | The entry is searchable afterwards | PASS |
| W1 | All three audit lines exist (`proposal.received` / `gate.accepted` / `memory.applied`) | PASS |
| W2 | A duplicate is `rejected` with a duplicate reason | PASS |
| W2 | The store stays byte-identical after the rejection | PASS |
| W3 | Replaying the same `proposal_id` adds nothing | PASS |
| W4 | A second entry is accepted | PASS |
| W4 | The revert succeeds | PASS |
| W4 | After the revert the store is byte-identical to before the write | PASS |
| W5 | A proposal without `proposal_id` is refused, not written | PASS |
| - | `gate.configure` takes effect (hourly quota set to 0) | PASS |
| - | A proposal over quota becomes `pending`, not dropped | PASS |
| - | Queue counts are consistent (accepted 1 / rejected 1 / reverted 1 / pending 1) | PASS |

## What this run does **not** show

- **No Axo**: the script only touches the biosnn-side persistent file; the real Axo library is out
  of scope for P1-P6 (decision 2).
- **No real model**: the `source` field is filled in by the script. The `llm` source weight is
  exercised only through the constants; no qwen3-4b or cloud model produced a proposal here.
- **Hash embedding stand-in** (64 dimensions): the behaviour of the 0.90 duplicate threshold under
  a real embedding is not covered. When a real embedder is wired in, this table must be re-run and
  recorded separately.

## Reproduction block

| Item | Value |
| :--- | :--- |
| Code version | `83a4530` (P1) plus the commit carrying this script |
| Command | `python scripts/e2e_write_mcp.py` |
| Result | 14/14 passed, exit code 0 |
## P5: packaging check (2026-10-31)

Not a release -- the point is that **one wheel carries both entry points and both work**.

| Step | Result |
| :--- | :--- |
| `uv build --wheel packages/biosnn-mcp` | `biosnn_mcp-0.3.0-py3-none-any.whl` |
| install into a clean venv | both `biosnn-mcp.exe` and `biosnn-mcp-write.exe` appear |
| `biosnn-mcp-write --help` | fine (`--state-dir / --store / --queue / --audit-log`) |
| stdio session through the installed entry point | `tools/list` returns 13 tools; the proposal is `accepted`; search hits; `memory.revert` returns `reverted: true`; 7 audit lines |

A CI leg was added ("write entry smoke"): build the wheel, install into a clean venv, run
propose -> accept -> revert through `biosnn-mcp-write`, asserting 13 tools and both verdicts.
**Nothing was published** -- the version number and the tag are the user's call.
## Isolated end-to-end on the Axo side (2026-10-31, closes gap B2)

Until now this directory only recorded the biosnn-side acceptance; the Axo write tier had unit
tests but nobody had driven the installed program end to end. That is now covered, and **everything
runs in a temporary directory**: store, queue and audit all live under `mkdtemp()` and are removed
afterwards. The write tier is switched on through `AXO_PROPOSE_SETTING`, which is only honoured when
`--data-dir` is given as well -- so the user's real settings and real library were never touched.

Script: `tools/_smoke_mcp_write.py` in the Axo repository, driving the installed
`dist/_mcp/axo-mcp-write.exe`.

| Check | Result |
| :--- | :--- |
| Tool table has 8 entries (including `queue.approve`) | PASS |
| Proposal is judged accepted | PASS |
| Searchable afterwards (multi-word query) | PASS |
| Duplicate proposal is rejected | PASS |
| Same `proposal_id` is idempotent | PASS |
| Revert takes the store back to zero | PASS |
| Audit events complete (received / accepted / applied / revert) | PASS |

**This run found two real problems**

1. **The read-only connection could not see the entry it had just written**: the proposal was
   accepted and `state.query` showed one entry, but `memory.search` returned nothing. Search used a
   read-only connection opened at start-up while writes went through another connection. Fixed by
   reopening the reader after every accept.
2. **Single-word queries do not hit**: `"gate"` returns 0 while `"accepted this line"` returns 1 for
   the same memory. That is a property of the lexical encoder in the recall layer, not a defect of
   the write tier -- but users will run into it, so it is recorded here.

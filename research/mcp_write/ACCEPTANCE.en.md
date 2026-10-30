# System acceptance (P6, S1-S4)

[中文](ACCEPTANCE.md) · [日本語](ACCEPTANCE.ja.md)

**2026-10-31.** Criteria come from S7 of the plan and were frozen on 2026-10-31; the scripts are
[`scripts/accept_system.py`](../../scripts/accept_system.py) and
[`scripts/e2e_write_mcp.py`](../../scripts/e2e_write_mcp.py).

## Results

| Rule | Check | Result |
| :--- | :--- | :--- |
| S1 | Both entry points of one package install in a clean environment and answer `--help` | Pass (local Windows plus the CI legs on Linux 3.10/3.12) |
| S1 | Separate installers for the two Axo tiers | **Not done**: the tiers share one package, the UI switch works (P4), but no installer was built |
| S2 | One stdio end-to-end script per version, tool tables as expected | Pass: 9 read-only / 13 in write mode |
| S3 | Synthetic conversation -> trigger -> memory -> consolidation -> retrieval, asserted one by one, audit complete | Pass (8 of the 13 checks) |
| S4 | After a revert the store file hashes back to the pre-write state, and the audit shows the revert | Pass |
| Regression | Read-only tool table and refusal semantics match 0.2.0 | Pass |

Full script output: `13/13 passed` (`python scripts/accept_system.py`).

## One thing the acceptance run found and fixed

**S4 failed the first time**: the store file did not exist before the write, and the revert left an
empty JSON behind -- same content, different hash. S4 asks for the file hash to return to the
pre-write state, so `_restore` now **deletes the file** when the pre-image is empty instead of
writing an empty store. A test pins that behaviour
(`test_revert_of_the_first_write_leaves_no_store_file`).

## What this run does not show

- **No Axo installers**: the two tiers share one code base and one package (decision 1, S4.1); the
  UI switch (P4) works, but PyInstaller packaging and the two-tier distribution notes are not done.
- **The write target is still the biosnn-side stand-in store**: the real Axo library is outside
  P1-P6 (decision 2).
- **No real model**: the `source` field is filled in by the script; salience is computed from
  constants only.
- **Linux covered the wheel only**: the Axo tiers were exercised on Windows.

# persona-core

[中文](README.md) · [日本語](README.ja.md)

A **general** persona-and-memory core. It imports no host application (no `axo`) and uses only
the standard library.

A fused model's capability splits in two:

| Half | Where | Shippable with the package |
| :--- | :--- | :--- |
| General capability (common sense / ethics / psychology) | **in the weights** | yes |
| This user's memories and persona | **outside the weights** (local files) | **no** |

So the package must ship a command that builds the second half; otherwise the user gets only half.

## Usage

```bash
python -m persona_core init-memory --root ~/persona-core
python -m persona_core manifest   --root ~/persona-core
```

`init-memory` creates:

```
<root>/memory/mem.db           the memory store (schema shared with memcore)
<root>/identity/instance.json  instance id, birth date, age at birth
```

and writes one **pinned built-in persona anchor** (`kind=persona_anchor`,
`source_ref=persona://builtin`). **Idempotent**: re-running writes nothing new.

`manifest` records what she is made of into `<root>/MANIFEST.json` (per-file sha256) for
packaging and verification.

## Generality constraints

- The persona text comes from the package's `persona.md`; hosts may **append** style via
  `--extra-style`, but cannot replace the identity — identity is meant to live in the weights;
- The data root is whatever `--root` says; **no product-specific path is hard-coded**;
- The memory schema matches `memcore`, so a host that installs `memcore` can read the same store.

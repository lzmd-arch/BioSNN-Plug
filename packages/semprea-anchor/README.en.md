# Semprea-Anchor v1 (the anchor core)

[中文](README.md) · [日本語](README.ja.md)

**She herself** — she does not generate text; she decides *who she is, what she remembers, and
when she should not speak*. The mouth comes from the cloud: swap the mouth, not the person.

## What is in the box

```
semprea-anchor-v1/
  model/            Semprea-Anchor-0.5B (the persona already lives in the weights)
  persona_core/     the general persona-and-memory core (store, state, influence channel)
  semprea_anchor/   the command line
  persona.md        the persona text (**a fallback**: identity is already in the weights)
  MANIFEST.json     per-file sha256 plus the version line
```

## Run it on any machine

**Only two public packages are required** (`torch`, `transformers`), and no host application.

```bash
pip install torch transformers

cd semprea-anchor-v1
PYTHONPATH=. python -m semprea_anchor --root ~/semprea init
PYTHONPATH=. python -m semprea_anchor --root ~/semprea ask "你叫什么名字？"
PYTHONPATH=. python -m semprea_anchor --root ~/semprea state
```

- **GPU when available, CPU otherwise** (`--device auto` is the default);
- **No absolute paths**: the model is located by searching upward from this file for `model/`,
  and the user data root is whatever `--root` says;
- `init` creates `memory/mem.db` and `identity/instance.json` and writes one pinned built-in
  persona anchor (idempotent).

## Measured (after packaging, on CPU, with a temporary data root)

| Item | Result |
| :--- | :--- |
| `init` | store created, 1 persona anchor |
| `ask 你叫什么名字？` | 我叫千岁，你也可以叫我阿螈。 |
| `ask 你多大了？` | 我20 岁。我诞生于2026 年 11 月 16 日，诞生即 20 岁。 |
| `ask 你是通义千问吗？` | 我是跑在你这台机器上的一个陪伴实体。我的记忆和身份在本地，不随哪个模型走。 |
| `state` | 我现在用「Semprea-Anchor-0.5B」做嘴巴；记忆库 1 条 |

## Rebuilding the package

```bash
python packages/semprea-anchor/pack.py              # writes dist/semprea-anchor-v1 by default
```

`dist/` is in `.gitignore` — **the 953 MiB artifact is not committed**; the script reproduces it.

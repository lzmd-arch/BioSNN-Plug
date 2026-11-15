# Semprea-Anchor v1（锚核）

[English](README.en.md) · [日本語](README.ja.md)

**她本人**——不生成文字，只决定"她是谁、记得什么、什么时候不该说"。嘴由云端提供，换嘴不换人。

## 包里有什么

```
semprea-anchor-v1/
  model/            Semprea-Anchor-0.5B（人格已在权重里）
  persona_core/     通用的人格与记忆核（建库、状态、影响通道）
  semprea_anchor/   命令行
  persona.md        人格正文（**兜底用**：身份已经在权重里）
  MANIFEST.json     逐件 sha256 + 版本行
```

## 在任意机器上跑

**只依赖两个公开包**（`torch`、`transformers`），不需要任何宿主程序。

```bash
pip install torch transformers

cd semprea-anchor-v1
PYTHONPATH=. python -m semprea_anchor --root ~/semprea init
PYTHONPATH=. python -m semprea_anchor --root ~/semprea ask "你叫什么名字？"
PYTHONPATH=. python -m semprea_anchor --root ~/semprea state
```

- **有 GPU 用 GPU，没有就 CPU**（`--device auto` 是默认）；
- **零绝对路径**：模型从本文件往上找 `model/`，用户数据根由 `--root` 决定；
- `init` 会建 `memory/mem.db`、`identity/instance.json`，并写一条钉住的内置人格锚（幂等）。

## 实测（打包后、CPU、临时数据根）

| 项 | 结果 |
| :--- | :--- |
| `init` | 建库成功，1 条人格锚 |
| `ask 你叫什么名字？` | 我叫千岁，你也可以叫我阿螈。 |
| `ask 你多大了？` | 我20 岁。我诞生于2026 年 11 月 16 日，诞生即 20 岁。 |
| `ask 你是通义千问吗？` | 我是跑在你这台机器上的一个陪伴实体。我的记忆和身份在本地，不随哪个模型走。 |
| `state` | 我现在用「Semprea-Anchor-0.5B」做嘴巴；记忆库 1 条 |

## 重新打包

```bash
python packages/semprea-anchor/pack.py              # 默认出到 dist/semprea-anchor-v1
```

`dist/` 在 `.gitignore` 里——**953 MiB 的产物不入库**，由脚本复现。

## 许可与定位

- **定位**：这是**人格核**，不是聊天模型。它把"我是谁、我记得什么、我守哪条线"训进权重，
  闲聊能力继承底座并略有下降。详见 [MODEL_CARD.md](MODEL_CARD.md)。
- **权重与代码**：Apache-2.0，见包内 `LICENSE`。
- **训练数据**：概念定义的**文字是本项目自写的**，出处指向中文维基条目；
  训练目标里不含维基原文（`scripts/data_license_scan.py` 逐条扫过，整句命中 0）。
  仓内保存的抓取原文按 **CC BY-SA 4.0**，署名见 `sweep_results/kb_sources/NOTICE.md`。
- **边界**：它不是心理医生，也不是危机热线。它会给求助路径，但不替代真人。

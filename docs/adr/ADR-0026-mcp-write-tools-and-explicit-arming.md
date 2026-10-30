# ADR-0026：MCP 写工具与显式武装（M0 → M1）

[English](ADR-0026-mcp-write-tools-and-explicit-arming.en.md) · [日本語](ADR-0026-mcp-write-tools-and-explicit-arming.ja.md)

- **状态**：已采纳
- **日期**：2026-10-30
- **相关**：[ADR-0020](ADR-0020-phase-five-m0-readonly-mcp.md)（M0 只读边界）、
  计划书 §七 第五阶段、[阶段五缺口清单](../../sweep_results/stage1_5_gap_plan.md)

## Context

[ADR-0020](ADR-0020-phase-five-m0-readonly-mcp.md) 把 M0 定成只读骨架：三个只读工具、
`trigger.status` 永远未武装。此后 M1–M3 的东西都实现了（mock/真 LLM adapter、对话 trace、
睡眠巩固、脉冲触发、真实嵌入），但**没有一个是 MCP 工具**——外部客户端能看、不能写，
也就不可能真的"用起来"。

ADR-0020 同时立了两条硬约束：**LLM 只是外部推理引擎**（不入学习路径、key 不入库）、
**所有工具调用必须可记录、可回放**。扩工具面必须同时满足这两条。

## Decision

1. **工具面扩到 9 个**，分两层：
   - 只读（任何模式可用）：`state.query`、`memory.search`、`trigger.status`、`trigger.poll`；
   - 写（需 `--allow-write`）：`memory.write`、`memory.consolidate`、`trigger.arm`、
     `trigger.disarm`、`trigger.feed`。
2. **默认只读**。写工具照常出现在 `tools/list` 里（客户端可发现），但在只读模式下调用返回
   `isError: true`，错误文本写明「需要 --allow-write」。构造 `InMemoryCore(writes_enabled=...)`
   决定能力，服务器用 CLI 开关控制。
3. **武装必须带 reason**。`trigger.arm` 要求非空 `reason` 字符串并记进状态；`disarm` 清空待处理
   事件。`trigger.feed` 每步喂一个 `[0,1]` 的显著性、推进包内 `SpikingTrigger` 一步；
   `trigger.poll` 排空并返回事件（**只读**，任何模式可调）。
4. **可审计**：服务器把每次 `tools/call` 记一行 JSONL（`ts` / `tool` / `ok` / `error` /
   `arguments`），路径由 `--audit-log` 指定；不给路径时只留在内存里（测试用）。
5. **巩固用确定性哈希嵌入**（`HashEmbedding`），使 `memory.consolidate` 在没有 Ollama 的环境
   也能跑；阈值、轮数、噪声、种子都是工具参数，默认值与 `SleepConsolidator` 一致。
6. **版本 0.2.0**（`biosnn-mcp`）。

## Consequences

> **后续修订**：五个写工具已于同日改名到 `propose.*` 命名空间——它们永远只改内存替身核，
> 叫 `write` 是名字与行为不符。见 [ADR-0027](ADR-0027-write-tools-are-proposals.md)。

- 外部客户端现在**能写、能武装、能巩固**，且默认不会悄悄改东西——写要先开开关，武装要写理由，
  每次调用留痕；
- `state.query` 的 `milestone` 从 `M0` 变为 `M1`、`read_only` 随开关变化；旧的"M0 永远只读"
  断言在测试里已更新为"只读模式下写工具被拒"；
- **仍不接 LLM**：`llm` 字段仍是 `not_connected`，本 ADR 不把任何 LLM 调用接进工具面；
- **不能把 `armed=true` 写成"自主触发成立"**：武装只是状态，触发质量仍由 M3k–M3p 的证据决定
  （合成流通过、真实流未过，见各 README）；
- 写工具改变的是**内存里的替身核**，不是真实记忆库；替换成 Axo 适配器或真实库时要复用同一
  协议与同一条审计约定。

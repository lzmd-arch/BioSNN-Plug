# ADR-0027：写工具是提案，不是写

[English](ADR-0027-write-tools-are-proposals.en.md) · [日本語](ADR-0027-write-tools-are-proposals.ja.md)

- **状态**：已采纳
- **日期**：2026-10-31
- **相关**：[ADR-0020](ADR-0020-phase-five-m0-readonly-mcp.md)（只读边界）、
  [ADR-0026](ADR-0026-mcp-write-tools-and-explicit-arming.md)（工具面扩到 9 个）、
  `packages/biosnn-mcp`

## Context

[ADR-0026](ADR-0026-mcp-write-tools-and-explicit-arming.md) 给 MCP 加了五个"写工具"。
它们的实际作用域是**服务器内存里的替身核**——永远不碰真实记忆库（Axo 侧的库另由"只读"决定锁死，
见 `docs/security.md`：SQLite `mode=ro`，引擎层拒绝写入）。

于是出现一个名字与行为不符的问题：一个叫 `memory.write` 的工具，**从不真正写入任何记忆库**。
按本仓"文档与行为不一致即 bug"的规矩，这不能靠文档补一句"其实是假的"糊过去。

## Decision

1. **五个工具改名到 `propose.` 命名空间**：
   `propose.memory.write`、`propose.memory.consolidate`、`propose.trigger.arm`、
   `propose.trigger.disarm`、`propose.trigger.feed`；四个只读工具（`state.query`、
   `memory.search`、`trigger.status`、`trigger.poll`）不变。
2. **契约写进名字与数据**：所有 `propose.*` 的作用域**永远是替身核**；payload 带
   `"scope": "stand-in"`，`state.query` 报 `"write_scope": "stand-in"`。
3. **写工具不"转正"**。它们的价值是两件事：让尝鲜者把工具面试通、让一致性/接入测试有对象可测。
   这就够了——它们是**接口脚手架**，不是待接线半成品。
4. **将来若要让提议产生真实记忆效果**，接法不是把这些工具接线，而是新增一条**动力学独占写**的
   路径：`propose` 队列 + 巩固门控消费——提议先入队，由动力学处置，全程 JSONL 审计。
   届时这些工具退役，换成 `biosnn_propose_*`。
5. **守卫**：测试断言"凡不是只读的工具，名字必须以 `propose.` 开头，且描述里写明 `stand-in`"。

## Consequences

- 名字即契约：客户端读到的名字直接说明它不会写任何真实存储；
- 新增任何带写语义的工具，必须先过命名守卫（改测试才能加）；
- 版本仍是 **0.2.0**——改名发生在首次发布之前，没有已经发布出去的契约被破坏；
- Axo 侧的只读决定不受影响：`axo-mcp` 的工具面与本 ADR 无关，它的库仍在引擎层只读；
- 代价：`propose.*` 读起来比 `write.*` 啰嗦；这是刻意的——啰嗦换诚实。

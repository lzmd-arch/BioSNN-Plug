# biosnn-mcp

[English](README.en.md) · [日本語](README.ja.md)

[BioSNN-Plug](https://github.com/lzmd-arch/BioSNN-Plug) 的**阶段五 MCP 服务器**：
JSON-RPC 2.0 over stdio，零第三方依赖，把认知状态、情景记忆、睡眠巩固与脉冲触发暴露成 MCP 工具。

> **边界**：LLM 只是**外部推理引擎**——本包不调用任何模型、不把 API key 读到内存里、
> 不参与任何突触学习路径。工具面**默认只读**；写操作要显式 `--allow-write`，并且**每次调用
> 都会留一行审计**（[ADR-0026](../../docs/adr/ADR-0026-mcp-write-tools-and-explicit-arming.md)）。

## 安装

```bash
pip install biosnn-mcp
```

要从 `main` 装**尚未发布**的改动时走 Git：

```bash
pip install "biosnn-mcp @ git+https://github.com/lzmd-arch/BioSNN-Plug.git@main#subdirectory=packages/biosnn-mcp"
```

或在本仓库里直接跑（未安装时把源码目录放进 `PYTHONPATH`）：

```bash
PYTHONPATH=packages/biosnn-mcp/src python -m biosnn_mcp --allow-write
```

## 快速开始

```bash
# 只读（默认）
biosnn-mcp

# 可写 + 审计
biosnn-mcp --allow-write --audit-log ~/biosnn-mcp-audit.jsonl
```

MCP 客户端配置（Claude Desktop / 任何 stdio 客户端）：

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

手工发一条 JSON-RPC 也能测：

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"state.query","arguments":{}}}' | biosnn-mcp
```

## 写版本（`biosnn-mcp-write`）

同一个包、同一版本号，多一个入口。区别只有一句：**提议不再只落在内存替身核，而是进队列，
由巩固门控决定要不要记**。

| | `biosnn-mcp`（默认） | `biosnn-mcp-write` |
| :--- | :--- | :--- |
| 工具数 | 9 | 13 |
| 写工具去向 | 内存替身核 | 队列 → 门控 → 持久化文件 |
| 撤销 | 没有这一层 | `memory.revert` |
| 审计 | 可选 `--audit-log` | 默认 `<state>/audit.jsonl` |
| 适合 | 让别的程序看、试工具面 | 让提议真的成为记忆 |

```bash
biosnn-mcp-write --state-dir ~/.biosnn-mcp    # 队列 / 存储 / 审计默认都在这个目录
```

| 环节 | 做什么 |
| :--- | :--- |
| `propose.memory.write` | 带客户端生成的 `proposal_id`（幂等键）进队列，**不直接落库** |
| 门控 | 余弦 ≥ 0.90 判重复；显著度 ≥ 0.60 且配额未满自动接受；0.30–0.60 或超配额转待审；< 0.30 拒绝 |
| 落库 | 只有门控接受的那一条会写进 `<state>/store.json`，并留前像 |
| `memory.revert` | 用前像把库恢复原样，撤销本身也进审计 |
| `queue.status` / `queue.list` / `gate.configure` | 看队列、看判定理由、改阈值（要 reason） |

审计默认写 `<state>/audit.jsonl`：`proposal.received` / `gate.accepted` / `memory.applied` /
`memory.revert` / `gate.configure` 各占一行。

**只读版一个字没改**：`biosnn-mcp` 的工具表、拒绝语义、审计字段与 0.2.0 相同，由测试钉住。

## 工具

| 工具 | 层 | 参数 | 说明 |
| :--- | :--- | :--- | :--- |
| `state.query` | 只读 | — | 阶段/里程碑、条目数、`read_only`、是否武装、待处理触发数 |
| `memory.search` | 只读 | `query`、`top_k`(1–50) | 按词重叠检索内存条目 |
| `trigger.status` | 只读 | — | 是否武装、武装理由、已喂样本数 |
| `trigger.poll` | 只读 | — | 排空并返回待处理的触发事件 |
| `propose.memory.write` | 提议 | `text`、`id?`、`tags?` | 向替身核提议一条记忆；`id` 冲突会被拒 |
| `propose.memory.consolidate` | 提议 | `threshold`、`rounds`、`noise_sigma`、`seed` | 提议跑一次睡眠巩固（确定性哈希嵌入） |
| `propose.trigger.arm` | 提议 | `reason` | 提议武装替身核的脉冲触发；**必须写理由** |
| `propose.trigger.disarm` | 提议 | — | 提议解除武装并清空待处理事件 |
| `propose.trigger.feed` | 提议 | `salience`(0–1) | 提议推进一步；武装时命中的事件入队 |

**为什么叫 `propose.*` 而不是 `write.*`**（[ADR-0027](../../docs/adr/ADR-0027-write-tools-are-proposals.md)）：
这些工具**永远不会写真实记忆库**，只改服务器内存里的替身核。名字即契约——一个不真实写入的
`write_*` 工具本身就是文档与行为不符。payload 里带 `"scope": "stand-in"`，`state.query` 报
`"write_scope": "stand-in"`。它们**不"转正"**：将来若要让提议产生真实记忆效果，接法是新增
**动力学独占写**的 `propose` 队列 + 巩固门控消费，而不是把这些工具接线。

提议类工具在只读模式下**仍然出现在 `tools/list` 里**（客户端可发现），但调用会返回
`isError: true`，错误文本写明需要 `--allow-write`——这是刻意的，避免"工具不存在"被误读成
"能力不具备"。

## 审计日志

`--audit-log PATH` 打开后，每次 `tools/call` 追加一行 JSONL：

```json
{"ts":"2026-10-30T06:30:00Z","tool":"memory.write","ok":true,"error":null,"arguments":{"text":"..."}}
```

参数原样记录（本包的工具参数没有敏感字段）；不给路径时只留在内存里，供测试断言。

## 状态与里程碑

`state.query` 的 `milestone` 在只读模式是 `M0`、可写模式是 `M1`，`read_only` 跟着开关走。
**`armed=true` 只表示武装状态**——触发质量由阶段五 M3k–M3p 的证据决定
（合成流通过、真实对话流未过），不能读成"自主触发已成立"。

## 不做什么

- 不调用任何 LLM、不读 API key、不联网；
- 不做梯度、不进突触学习路径；
- 不写真实记忆库——现在改的是内存替身核；接真实库要复用同一协议与同一条审计约定。

## 测试

```bash
uv run pytest packages/biosnn-mcp/tests -q
```

# biosnn-mcp 0.2.0

BioSNN-Plug 阶段五的 MCP 服务器，**首次发布**。JSON-RPC 2.0 over stdio，零第三方依赖。

## 工具（9 个）

- **只读（4）**：`state.query`、`memory.search`、`trigger.status`、`trigger.poll`；
- **提议（5）**：`propose.memory.write`、`propose.memory.consolidate`、`propose.trigger.arm`、
  `propose.trigger.disarm`、`propose.trigger.feed`。

**为什么叫 `propose.*` 而不是 `write.*`**：这些工具**永远不会写真实记忆库**，只改服务器内存里的
替身核。名字即契约——一个不真实写入的 `write_*` 工具本身就是文档与行为不符。它们不"转正"：
将来若要让提议产生真实记忆效果，接法是新增**动力学独占写**的 `propose` 队列 + 巩固门控消费，
而不是把这些工具接线。见 ADR-0027。

## 安全默认

- **默认只读**：提议类工具照常出现在 `tools/list`，但只读模式下调用返回 `isError` 并提示
  `--allow-write`；
- **武装必须带 reason**；
- `--audit-log PATH` 把每次 `tools/call` 记一行 JSONL（`ts`/`tool`/`ok`/`error`/`arguments`）；
- 不调用任何 LLM、不读 API key、不联网、不进任何突触学习路径。

## 安装

```bash
pip install biosnn-mcp
biosnn-mcp --help
```

## 复现

```bash
pip install "biosnn-mcp @ git+https://github.com/lzmd-arch/BioSNN-Plug.git@mcp-v0.2.0#subdirectory=packages/biosnn-mcp"
```

源码与测试：<https://github.com/lzmd-arch/BioSNN-Plug/tree/main/packages/biosnn-mcp>

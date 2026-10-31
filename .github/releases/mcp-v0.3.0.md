# biosnn-mcp 0.3.0

同一个包，**两个入口**：

- `biosnn-mcp` —— 与 0.2.0 完全一致（只读 + 5 个提议工具，工具表、拒绝语义、审计字段一字未改）；
- `biosnn-mcp-write` —— **新增**。提议不再只落在内存替身核，而是进队列，由巩固门控决定
  要不要成为记忆；落库可撤销，全程 JSONL 审计。

## 写版本做了什么

```
MCP 客户端 ──propose.*──▶ 提议队列（SQLite，本机）
                              │ pending
                              ▼
                     巩固门控（唯一写入口）
                     ├─ 重复：与现有条目相似度 ≥ 0.90 → 拒绝
                     ├─ 显著度 ≥ 0.60 且配额未满 → 自动接受
                     ├─ 0.30–0.60 或超配额 → 待审（不丢弃）
                     └─ < 0.30 → 拒绝
                              │ accepted
                              ▼
                    持久化存储（<state>/store.json），保留前像
                              │
                              └── 审计 <state>/audit.jsonl
```

- **提议不是写入**：`propose.memory.write` 需要客户端生成的 `proposal_id`（幂等键），
  重复提交只返回同一条；
- **门控是唯一写入口**：整个包里只有门控的 `_apply` 会改存储，`WriteCore.write()` 直接抛错；
- **可回滚**：`memory.revert <seq>` 用前像把库恢复原样；写之前库是空的，撤销后连文件都不留；
- **配额**：每小时 20 条、每天 200 条，超了转待审而不是丢弃。

## 新增的 4 个工具（写版本）

| 工具 | 说明 |
| :--- | :--- |
| `queue.status` | 队列深度、配额余量、当前门控配置 |
| `queue.list` | 提议与判定理由（可按状态过滤） |
| `memory.revert` | 撤销一条已接受的写入（要 `reason`，进审计） |
| `gate.configure` | 改阈值/配额（要 `reason`，进审计） |

阈值冻结在项目计划书 §9.1（2026-10-31）：重复 0.90 / 自动接受 0.60 / 下限 0.30 / 20·200。
显著度 = 来源权重（user 0.45 · client 0.25 · llm 0.15）+ 长度带 0.20 + 新颖度 0.25×(1−最大相似度)
+ 实体或数字 0.10。

## 状态与路径

```bash
biosnn-mcp-write --state-dir ~/.biosnn-mcp
# 默认：<state>/store.json、<state>/proposals.sqlite3、<state>/audit.jsonl
```

也可以用 `--store` / `--queue` / `--audit-log` 分别指定。

## 验收

- **W1–W6**（计划书 §3.4，预注册）逐条有测试：端到端 + 三条审计、重复拒绝且库逐字节不变、
  幂等、撤销后逐字节回到写前、无旁路（AST 守卫）、只读版回归；
- 系统级验收 `scripts/accept_system.py`：**13/13**（触发 → 记忆 → 巩固 → 查回 → 撤销）；
- 端到端脚本 `scripts/e2e_write_mcp.py`：**14/14**，走真的 stdio 子进程。

## 已知边界

- 写的是 **biosnn 侧的持久化替身文件**，不是任何真实记忆库；
- 默认嵌入是 64 维哈希替身：0.90 这个重复阈值在真嵌入下的行为**未标定**；
- 验收只覆盖合成数据，没有真实使用记录。

## 安装

```bash
pip install biosnn-mcp==0.3.0
biosnn-mcp --help          # 只读 + 提议
biosnn-mcp-write --help    # 队列 + 门控 + 撤销
```

## 复现

```bash
pip install "biosnn-mcp @ git+https://github.com/lzmd-arch/BioSNN-Plug.git@mcp-v0.3.0#subdirectory=packages/biosnn-mcp"
```

源码与测试：<https://github.com/lzmd-arch/BioSNN-Plug/tree/main/packages/biosnn-mcp>

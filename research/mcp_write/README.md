# MCP 写版本：系统级验收（P2）

[English](README.en.md) · [日本語](README.ja.md)

**2026-10-31，一次跑通 14/14。** 脚本 [`scripts/e2e_write_mcp.py`](../../scripts/e2e_write_mcp.py)，
判据是计划书 §3.4 预注册的 W1–W6（冻结于 2026-10-31，§9.2）。

## 跑法

```bash
python scripts/e2e_write_mcp.py            # 临时目录，跑完即删
python scripts/e2e_write_mcp.py --keep     # 保留目录，用来翻 store.json / audit.jsonl
```

脚本起一个真的 `python -m biosnn_mcp write` 子进程，用 stdin/stdout 说 JSON-RPC——
和真实 MCP 客户端看到的完全一样，不是直接调 Python API。

## 结果（2026-10-31，Python 3.13 / Windows 11）

| 判据 | 检查项 | 结果 |
| :--- | :--- | :--- |
| W6 | 工具表 13 个，只读 9 个在前 | PASS |
| W1 | 提议被判 `accepted` | PASS |
| W1 | 落库后可检索 | PASS |
| W1 | 审计三条齐全（`proposal.received` / `gate.accepted` / `memory.applied`） | PASS |
| W2 | 重复提议判 `rejected`，理由含 duplicate | PASS |
| W2 | 拒绝后库逐字节未变 | PASS |
| W3 | 重放同一 `proposal_id` 不产生新条目 | PASS |
| W4 | 第二条被接受 | PASS |
| W4 | 撤销成功 | PASS |
| W4 | 撤销后库逐字节回到写前 | PASS |
| W5 | 缺少 `proposal_id` 时拒绝而不是落库 | PASS |
| — | `gate.configure` 生效（配额压到 0） | PASS |
| — | 超配额的提议转 `pending`，不丢弃 | PASS |
| — | 队列统计自洽（accepted 1 / rejected 1 / reverted 1 / pending 1） | PASS |

## 这一轮**没有**证明的事

- **没走 Axo**：脚本只碰 biosnn 侧的持久化文件；Axo 真库不在 P1–P6 范围（决策 2）。
- **没接真模型**：提议的 `source` 是脚本自己填的，`llm` 来源的显著度只由常量算出，
  没有用 qwen3-4b 或任何云端模型产生过提议。
- **嵌入是哈希替身**（64 维）：重复判定的 0.90 阈值在真嵌入下的行为未被这轮覆盖；
  接真嵌入时要重跑这张表，并把结果另记一份。

## 复现块

| 项 | 值 |
| :--- | :--- |
| 代码版本 | `83a4530`（P1 落地）+ 本脚本所在提交 |
| 命令 | `python scripts/e2e_write_mcp.py` |
| 结果 | 14/14 通过，退出码 0 |
## P5：打包验证（2026-10-31）

不是为了发版，是为了证明**同一个 wheel 里两个入口都在、都能用**。

| 步骤 | 结果 |
| :--- | :--- |
| `uv build --wheel packages/biosnn-mcp` | `biosnn_mcp-0.3.0-py3-none-any.whl` |
| 装进全新 venv | `biosnn-mcp.exe` 与 `biosnn-mcp-write.exe` 两个入口都出现 |
| `biosnn-mcp-write --help` | 正常（`--state-dir / --store / --queue / --audit-log`） |
| 通过装好的入口跑 stdio 会话 | `tools/list` 返回 13 个工具；提议被判 `accepted`；检索命中；`memory.revert` 返回 `reverted: true`；审计 7 行 |

CI 里另加了一条腿「写版本入口的 smoke」：构建 wheel、装进干净 venv、用
`biosnn-mcp-write` 走一遍 propose → accept → revert，断言 13 个工具与两条判定。
**发布本身没有做**——版本号与 tag 由用户决定。
## Axo 侧的隔离端到端（2026-10-31，补 B2 缺口）

之前这里只记了 biosnn 侧的验收；Axo 的写档位只有单元测试，没人用装好的程序走一遍。
这次补上，且**全程临时目录**：库、队列、审计都在 `mkdtemp()` 里，跑完即删；
写档位开关靠 `AXO_PROPOSE_SETTING` 覆盖，而这个覆盖只在同时给了 `--data-dir` 时被采信，
所以用户的真实设置与真实库一次都没被碰。

脚本：Axo 仓 `tools/_smoke_mcp_write.py`，打的是装好的 `dist/_mcp/axo-mcp-write.exe`。

| 判据 | 结果 |
| :--- | :--- |
| 工具表 7 个 | PASS |
| 提议被判 accepted | PASS |
| 落库后可检索（多词查询） | PASS |
| 重复提议判 rejected | PASS |
| 同一 `proposal_id` 幂等 | PASS |
| 撤销后库回到 0 条 | PASS |
| 审计事件齐全（received / accepted / applied / revert） | PASS |

**这一轮抓到两个真问题**

1. **只读连接看不见刚落库的条目**：提议被接受、`state.query` 也显示 1 条，但
   `memory.search` 返回空。原因是检索用启动时打开的只读连接，写走另一个连接。
   修法：每次 accepted 之后重开只读连接。
2. **单字查询命中不了**：`"gate"` 0 条、`"accepted this line"` 1 条（同一条记忆）。
   这是召回层词法编码的性质，不是写档位的缺陷——但用户会撞上，记录在案。

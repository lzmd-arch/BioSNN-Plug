# MCP 写版本：系统级验收（P2）

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

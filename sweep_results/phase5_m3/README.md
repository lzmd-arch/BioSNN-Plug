# 阶段五 M3a：mock 结果回注与巩固协议（2026-10-29）

**预注册：** [`../phase5_m3_prereg.md`](../phase5_m3_prereg.md)。
12 个 knowledge 问题；规则触发 + MockLLM；回答写入可变 `DictMemory`，下一轮按 id 取回；
trace 记录 `consolidation`。**这是协议冒烟，不是真 LLM 巩固能力。**

## 结果

| 指标 | 实测 | 判据 | 结果 |
| :--- | ---: | :--- | :--- |
| 巩固成功率 | **1.000** | ≥0.80 | 过（平凡） |
| trace 完整 | 12/12 | 12/12 | 过 |
| 网络/key | 0 | 0 | 过 |

`m3a_ready=true`，但 **1.000 是“写进去再按 id 取出来”的必然结果**，不能当语义巩固。

## 读法

- M3a 只证明管道：调用 → 写记忆 → 下一轮可取回 → trace 有 consolidation 字段；
- 没有睡眠期噪声巩固、没有跨会话持久化、没有语义保真/改写；
- MockLLM 回答是 `[mock] + question`，不是知识；
- 真 LLM 的巩固成功率、对话任务完成率、首次触发等仍是 M3b。

## 边界

- 合成 12 问；不外推；
- 不把 M3a 的 1.000 写进阶段五验收；
- 写入/取回是同一进程内字典，不是 memcore/SQLite。

## 自证

`summary.json` / `trace.jsonl` 由 `scripts/run_phase5_m3.py` 生成。

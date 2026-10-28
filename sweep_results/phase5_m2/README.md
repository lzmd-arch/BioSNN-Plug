# 阶段五 M2：触发策略 + mock LLM 对话 trace（2026-10-29）

**预注册：** [`../phase5_m2_prereg.md`](../phase5_m2_prereg.md)。
30 个脚本化场景（12 knowledge / 12 memory / 6 chat），意图给定；检索分数由
`InMemoryCore` 实际计算；`MockLLM` 确定性、无网络；结果写 `trace.jsonl` + `summary.json`。

## 结果

| 指标 | 实测 | 判据 | 结果 |
| :--- | ---: | :--- | :--- |
| 触发准确率 | **0.733** | ≥0.70 | 过（薄） |
| 首次正确调用 | **第 2 轮** | ≤10 | 过 |
| trace 字段完整性 | 30/30 轮 | 每轮完整 | 过 |
| 网络/key | 0 | 0 | 过 |

`m2_ready=true`（仅 M2 mock 协议）。实际只调用了 **4 次** MockLLM——规则触发不是“12 个
knowledge 全调”，有 8 个 knowledge 因词面撞车得分 ≥0.25 被判为“检索足够”。

## 读法

- 这是**意图给定**的规则门，不是从 SNN 状态自主推断意图；M2 不宣称自主触发；
- 0.733 只比门槛高 3.3 pp，且 30 个场景是合成集；不能外推到真实对话；
- 4 次调用是预期的：规则只在“knowledge 且低分”时调用，不是按意图无脑调用；
- trace 记录每轮 state/retrieved/trigger/llm_response/consolidation，可回放；
  巩固字段为空，M3 才实现。

## 边界

- 不接真 LLM、不读 key、不联网；
- MockLLM 的回答不能用于质量/能力结论；
- 阈值 0.25/0.10 先写死，没有在测试集上调；
- “首次第 2 轮”只表示脚本集中第 2 个 knowledge 场景触发，不是真实用户对话延迟。

## 自证

`summary.json` 与 `trace.jsonl` 由 `scripts/run_phase5_m2.py` 生成；命令：
`python scripts/run_phase5_m2.py`。

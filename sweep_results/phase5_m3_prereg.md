# 预注册：阶段五 M3a——mock 结果回注与巩固协议（2026-10-29）

**性质：M3 的协议层，不是真 LLM 对话验收。** 用 `MockLLM` 验证“调用结果回注到记忆、
下一轮能检索到”的管道；真 LLM/对话集留到 M3b。

## 协议

- 12 个 knowledge 问题，M2 的规则触发（knowledge + 低检索分）；
- MockLLM 回答 `"[mock] " + question`；
- 每轮把回答写入一个可变的 `DictMemory`（id=`answer:<turn>`）；
- 紧接着用问题本身（或 `"what did you answer for <question>"`）检索该记忆；
- trace 的 `consolidation` 字段记录 `memory_id` 与 `stored=True`。

## 判据（先写死）

1. **巩固成功率**：12 个回答中，下一轮检索能取回正确条目的比例 ≥ **80%**；
2. trace 12/12 轮含 consolidation 字段；
3. 0 网络/0 key；
4. 三条全过 → `m3a_ready=true`；否则报边界，不调场景/阈值/判据。

## 边界

- MockLLM 回答不是能力；巩固是字符串写入/检索，不是睡眠期噪声巩固；
- 12 个合成问题，不外推真实对话；
- 真 LLM、跨会话持久化、巩固质量（语义保真）未测；
- 阶段五的“巩固成功率 80%+”只是**协议层**通过，不写成端到端协同能力。

## 产物

- `scripts/run_phase5_m3.py`；
- `sweep_results/phase5_m3/`（summary + trace）。

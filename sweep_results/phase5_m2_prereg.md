# 预注册：阶段五 M2——触发策略与 mock 对话协议（2026-10-29）

**性质：M2 协议冒烟，不是真 LLM 能力实验。** 用固定意图/检索分数与 `MockLLM` 验证
“该调用时调用、不该调用时不调用”的可观测管道；真 LLM 对话留到 M3。

## 触发策略

```text
intent ∈ {knowledge, memory, chat}
score  = max retrieval fused score（0..1）
uncertainty = 1 − score

call iff:
  intent == "knowledge" and score < 0.25
  or intent == "unknown" and score < 0.10
otherwise skip
```

- 阈值先写死；不在测试集上调；
- `chat` 永远 skip（寒暄不需要外部推理）；
- `memory` 永远 skip（记忆库内应可答；若检索为空，下一版再定义“记忆缺失”意图）。

## 场景集（先写死）

30 个脚本化场景：
- 12 个 `knowledge`：与库内记忆无关的外部知识问题；期望调用；
- 12 个 `memory`：与库内某条记忆词面命中；期望不调用；
- 6 个 `chat`：寒暄/情绪；期望不调用。
每个场景带 `should_call`；检索分数由 `InMemoryCore` 的 `memory.search` 实际计算，
不是手填分数。

## 判据（先写死）

1. 触发准确率 ≥ **70%**（30 个场景）；
2. 首次正确触发 ≤ **10 轮**；
3. trace 每轮有 `turn/user/state/retrieved/trigger/llm_response/consolidation` 字段；
4. 全程 **0 次网络/0 个 key**（MockLLM 确定性）；
5. 四条全过 → `m2_ready=true`；否则报边界，不调阈值/场景/判据。

## 边界

- 这是**意图给定**的规则触发，不是从 SNN 状态自主推断意图；M2 不宣称“自主触发”；
- mock LLM 的回答不能用于质量/能力结论；
- 30 个场景是合成集，不外推到真实对话；
- 巩固（consolidation）只记录空字段，M3 才实现。

## 产物

- `packages/biosnn-mcp/src/biosnn_mcp/trigger.py`；
- `packages/biosnn-mcp/tests/test_m2.py`；
- `scripts/run_phase5_m2.py` + `sweep_results/phase5_m2/`（summary + README）。

# 预注册：阶段五 M1——mock LLM、对话 trace 与触发协议（2026-10-29）

**性质：M1 接口/协议，不是能力实验。** 本预注册只定义 M1 的退出条件与 M2 的触发评测口径；
M1 不接真 LLM、不报触发准确率、不宣称自主。

## M1 交付

1. **Mock LLM adapter**：`LLMAdapter.generate(prompt, context)` 协议 + `MockLLM` 确定性替身；
   无网络、无 key、无外部依赖。
2. **对话 trace**：每轮记录 `turn / user / state / retrieved / trigger / llm_response /
   consolidation`；JSONL 追加、可回放、schema_version=1。
3. **触发协议（本文件）**：先写死“何时应调用”的标签与指标，M2 才实现/评测。

## 触发定义（M2 将实现）

每一轮，SNN 认知核产出：
- `state`：WM/情景/元认知状态摘要；
- `retrieved`：关联检索结果及分数；
- `uncertainty`：元认知门的置信度（v8/v11 口径）；
- `intent`：对话意图（规则/分类器，先固定）。

触发策略输入 `[max_retrieval_score, uncertainty, intent]`，输出 `call / skip`。
阈值只在 dev 对话集上选，test 固定。

## 标签与指标（M2 评测，先写死）

- **should_call**：用户请求需要外部知识/生成，且检索最高分低于阈值；
- **should_not_call**：寒暄/记忆内可答/明确不需要外部生成；
- **触发准确率** = 正确 call/skip 的比例，目标 **≥70%**；
- **首次触发延迟**：从对话开始到第一次正确 call 的轮数，目标 **≤10 轮**；
- **巩固成功率**：call 返回后写入情景记忆且下一轮可检索到的比例，目标 **≥80%**（M2/M3）；
- **任务完成率提升**：相对纯 SNN 基线 ≥50%（M3，需真 LLM）。

## M1 退出条件（不涉及准确率）

1. `MockLLM` 确定性、无网络；测试覆盖空 prompt/context；
2. trace 写入→读取→回放逐字段一致；重复 turn 拒绝；
3. 触发协议文档冻结（字段、标签、阈值规则、指标）；
4. 所有测试不依赖真 API key；
5. 四条全过 → `m1_ready=true`；否则报缺口，不进入 M2。

## 边界

- M1 的 mock 输出**不能**用于任何能力/质量结论；
- 触发准确率/首次延迟在 M2 才测；M1 不写“自主触发已实现”；
- 真 LLM（本地或云端）在 M2 才接；key 不入库，adapter 可替换；
- 阶段四 500 步纯想象 P3 边界与本 M1 无关。

## 产物

- `packages/biosnn-mcp/src/biosnn_mcp/llm.py`、`trace.py`；
- `packages/biosnn-mcp/tests/test_m1.py`；
- M2 触发实现与评测 runner（未开始）。

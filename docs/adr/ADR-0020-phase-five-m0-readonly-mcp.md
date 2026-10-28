# ADR-0020：阶段五 M0——只读 MCP 骨架，先立接口再谈自主

[English](ADR-0020-phase-five-m0-readonly-mcp.en.md) · [日本語](ADR-0020-phase-five-m0-readonly-mcp.ja.md)

- **状态**：已采纳
- **日期**：2026-10-29
- **相关**：计划书 §七 第五阶段、[ADR-0007](ADR-0007-td-ltp-critic-provenance.md)
  （阶段五立项出处门槛）、[ADR-0019](ADR-0019-stage-four-entry-decoupled-from-relearn.md)、
  [阶段四状态](../../docs/stage4_status.md)

## Context

阶段四在严格口径下已收口：E2E v18 状态级 18/18、P3 v6 滚动时域闭环 ready、WM v9/情景 v5/
元认知 v11 ready；**500 步纯想象 P3 仍是已证明的结构性边界**。阶段五（LLM 集成与协同）
尚未开始，仓库里没有 MCP Server、LLM adapter、触发模块或对话 trace。

直接写一个“定时调用 LLM”的脚本会制造一个看似自主、实际不可审计的系统：没有工具 schema、
没有触发定义、没有每轮状态/检索/调用/巩固记录，无法判断“该调用时调用”。阶段五的成功
标准（≤10 轮首次触发、触发准确率 70%+、巩固 80%+、任务完成率 +50%）都需要先有一个
可观测的管道。

## Decision

1. **启动阶段五 M0：只读 MCP 骨架**。新增 `packages/biosnn-mcp`，用无第三方依赖的
   JSON-RPC 2.0 over stdio 暴露三个只读工具：`state.query`、`memory.search`、
   `trigger.status`；认知核是 `InMemoryCore` 替身。
2. **M0 不连接 LLM、不写记忆、不武装触发**。`trigger.status` 固定返回未武装；任何“自主
   触发”的结论在 M0 都不成立。
3. **LLM 只作外部推理引擎**：不进入任何突触学习路径，不做梯度回传；API key 不入库；
   所有工具调用必须可记录、可回放。
4. **阶段五正式验收仍由 M1–M3 的新预注册决定**：M1 触发模块 + mock LLM adapter，M2
   结果回注/巩固，M3 对话评测；M0 不计入阶段五成功标准。
5. **阶段四的边界不改**：500 步纯想象 P3 仍是开放边界；阶段五启动不把它写成已解决。

## Consequences

- 得到一个可被 MCP 客户端枚举、可测试、无网络依赖的最小接口；工具 schema 与只读边界
  先固定，后续换真实认知核时工具名/参数不变；
- 代价：M0 没有任何“智能”可展示；`trigger.status` 永远 false，容易被误读为“功能缺失”。
  这是刻意的——先把“未武装”写成默认，避免脚本调用被当成自主触发；
- 下一里程碑必须先补 mock LLM adapter 与对话 trace，再谈真实 LLM；
- 新增包需要随 CI 跑测试；不影响 `biosnn-bus` 的骨架独立性（ADR-0001）。

## Alternatives

- **直接接真 LLM + 定时调用**：能最快看到“调用”，但无法区分自主触发与脚本定时，且
  key/审计问题未定；否决。
- **只写脚本/函数，不立 MCP**：实现更少，但没有工具 schema 与客户端契约，阶段五的
  “可替换推理引擎”无从谈起；否决。
- **先做 HTTP 服务**：MCP 的 stdio 传输更小、无端口/鉴权面，M0 先用 stdio；HTTP 留到
  部署阶段再评估。
- **等阶段四纯想象 P3 解决再启动**：该边界对非逐位模型不可达（见 P3 v6 README）；继续
  阻塞会让阶段五无限期等待，故记录边界后启动 M0，而不是假装它已解决。

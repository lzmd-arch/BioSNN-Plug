# ADR-0013：协同性基准升为 P0、共享接口方案否决、三方混合维持阶段五

[English](ADR-0013-synergy-benchmark-priority-and-arbiter-scope.en.md) · [日本語](ADR-0013-synergy-benchmark-priority-and-arbiter-scope.ja.md)

- **状态**：已采纳
- **日期**：2026-10-18（裁决）· 2026-10-24（落库）
- **相关**：计划书 §12.7（外部参考）；[`sweep_results/arbiter_synergy_task_spec.md`](../../sweep_results/arbiter_synergy_task_spec.md)（任务规格与判据演化）；[research/arbiter](../../research/arbiter/README.md)（实现与结论）；[ADR-0010](ADR-0010-eprop-quadratic-storage-and-trace-propagation.md)

## Context

P4 论文骨架讨论中提出两条会改变路线图的方案：

1. **共享接口**：让 Hebbian 与 e-prop 通过一个显式的共享接口协同，而不是共享同一张权重；
2. **优先级前移**：把阶段五的三方混合与调度维度提到当前阶段做。

两条都牵涉"协同"这个词的语义边界。若不先钉死，后续所有"协同"数字都会混着两种不同的
机制一起报——而它们的可证伪性完全不同。

## Decision

**① 共享接口方案否。** 语义变形——它测的是"两套表示之间的桥"，不是"两条规则在同一
组突触上合作"。该方案**只可作为标注清晰的 counterfactual arm**（反事实臂），不得与
原生点结果并列表述。

**② 优先级反转：协同性基准是 P0。** 基准的点位是 **Hebbian ↔ e-prop 的原生点**——
两条线本来就共享的那一处权重（输入 → 隐藏）。判据的原意是"单方失败、混合才行"。
e-prop ↔ R-STDP 的反事实臂**排在基准之后**，等有基准结果再设计。

**③ 三方混合与调度维度维持阶段五不动。** 本阶段不因基准需要而提前。

**④ 保留项**：43 条测试通过的基础设施保留；冲突判定的 per-row cosine 与 retention
**联合报告**照旧。

## Consequences

- **"原生点"的形状由任务决定**：sMNIST 每步只喂一行像素，因此共享权重是 `28 → 256`，
  不是静态 MNIST 的 `784 → 256`。这不是缩小模型，是任务形态决定的。
- **判据演化必须留痕**：原判据要求"H 单独失败、E 单独失败"，实测不成立——纯 Hebbian
  把 Fisher 比从 0.309 抬到 0.472。2026-10-18 改为**协同量判据**（`Fisher(MIX) − max(Fisher(H), Fisher(E)) ≥ δ = 0.05`），
  **写于看到 MIX 数字之前**。改判据这件事本身不写进 ADR 就无法解释为什么前后结论不同。
- **反事实臂不会消失，但会被标注**：共享接口路线仍可做，结论必须写成 counterfactual，
  且不得用它回答"协同是否成立"。
- **阶段五的边界保持清晰**：三方混合与调度维度不在本阶段的任何实验里出现。

⚠️ 本 ADR 只固化裁决与随之而来的边界，不新增实证主张。基准本身的数字以
[`sweep_results/arbiter_synergy_task_spec.md`](../../sweep_results/arbiter_synergy_task_spec.md)
与结果目录为准。

# ADR-0022：判据重述仪表板

[English](ADR-0022-restatement-dashboard.en.md) · [日本語](ADR-0022-restatement-dashboard.ja.md)

- **状态**：已采纳
- **日期**：2026-10-29
- **相关**：[ADR-0021](ADR-0021-criterion-death-conditions.md)、计划书台账 148

## Context

判据重述散落在计划书、结果 README 与 ADR 里，没有人能一眼回答「这条线被重述过几次、为什么」。
ADR-0021 要求每次重述登记，但登记到哪里需要一个单一入口；否则「登记」本身又会漂移。

## Decision

本文末尾的仪表板是**唯一**的重述登记处。每次重述追加一行：日期、判据、从什么改成什么、
原因、链接（ADR/预注册/台账）。不修改旧行；更正追加新行。

## Dashboard

| 日期 | 判据 | 从 → 到 | 原因 | 链接 |
| :--- | :--- | :--- | :--- | :--- |
| 2026-10-26 | 跨模态检索 | R@1 ≥ 50% → 恢复率 ≥ 50% | 合成集上 R@1 与恢复率测的不是同一件事 | 计划书 ㉟ |
| 2026-10-25 | TAAF 无需微调 | 原样过线 → 消解（非达成） | 定位与代码契约冲突，标准不可测 | [ADR-0012](ADR-0012-taaf-locality-and-bus-output-contract.md) |
| 2026-10-26 | 仲裁器冲突减少 ≥50% | → P4 协同性基准（静态混合 +0.179） | 冲突事件计数不可复现，改测混合增益 | [ADR-0014](ADR-0014-adaptive-arbitration-unmeasurable-and-fit-retained.md)、台账 ㊾ |
| 2026-10-29 | 阶段三「最好评测」 | best_steps → 末次/末三段 | 32 点取最大把峰值当水平 | 台账 148、`stage3_sustained_audit.md` |

## Consequences

- 重述变成可数的；「第几次」不再靠记忆；
- 代价：每次重述多一步登记；漏登记会被 reviewer 发现（但 CI 暂不强制）。
- 仪表板本身不判断重述对不对，只记录发生过。

## Alternatives

- **把登记散在各 README**：已经散过，找不到全局视图；否决。
- **用 issue/PR 记录**：单人仓、无 PR 流程；否决。

# ADR-0017：CL 主路线切换为 Route R，回放先行

[English](ADR-0017-cl-main-route-switch-and-replay-first.en.md) · [日本語](ADR-0017-cl-main-route-switch-and-replay-first.ja.md)

- **状态**：已采纳
- **日期**：2026-10-25
- **相关**：计划书 §七 第三/四阶段、§5.2；[ADR-0015](ADR-0015-world-model-as-phase-four-component.md)；[预注册 v2](../../sweep_results/continual_v2_replay_prereg.md)；[formal_cil](../../sweep_results/continual_cifar100/formal_cil/README.md)

## Context

阶段三的持续学习正式运行（W1 特征 + delta 读出，预注册 CIL 口径）给出 **ACC 0.0876 / AF 0.290**；
同一份特征上**提供任务 id 的 oracle 上界也只有 0.322**。失败方式是灾难性遗忘（末行旧任务全零），
但更根本的是**表示层封顶**：oracle 都过不了 65%，回放修不了"特征本身分不开 100 类"。

探索期的 Route R（冻结 ResNet18 avgpool + 类原型）给出 ACC 0.658 / AF 0.127，但这两个数是在
**旧的累计类集矩阵**上算的，不是 CIL 口径——必须重测后才能当基线。

## Decision

1. **CL 研究主路线切换为 Route R**（冻结强表示 + 抗遗忘）；**Route W 保留为已声明的原始路线**，
   继续如实报告，不删除、不改写。
2. 机制按**归因链**排序：**回放先行**，不够再归一化，再不够参数隔离；**一次只上一个机制**。
3. **阶段四入口改写为「Route R + 回放：AF < 8% 且 ACC ≥ 65%」**（CIL 口径）；具体门槛数字从
   Step 0 重测基线推导，并在结果 README 里注明日期——**不用旧的 0.658/0.127 直接套**。
4. **阶段三拆分**：3a（今日关闭，原样封存）/ 3b（回放臂；「重学快 3×」设计排在回放结果之后；
   脉冲 W3 判别实验的阈值写死后才占 GPU）。
5. 旧结果与旧预注册不删改；路线切换的日期、理由与本 ADR 一起构成"不是为过门而换路线"的证据。

## Consequences

- 需要实现 latent replay（每类定额的特征缓冲，ρ ∈ {0.1, 0.3}），并**先**重测 Route R 基线；
- 「重学快 3×」与脉冲诊断的排期不变，但分别排在回放结果之后 / 阈值写死之后；
- 阶段四的世界模型、神经发生等仍受阶段三验收门控；世界模型的打底（一步动力学、rollout、
  Acrobot）不改变门的状态。

## Alternatives

- **先上归一化**：会污染回放的归因，且它是通用 trick——先上它等于放弃"哪个机制真的起作用"的答案；
- **先上参数隔离（EWC 族）**：与 P4 §2.4（正则化管不了无任务边界的冲突）的结论有张力，
  同一份论文不能一边引用一边用；
- **留在 Route W**：oracle 上界 32.2%，65% 的门在表示层就过不去，回放修不了特征。

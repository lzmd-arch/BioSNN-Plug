# 阶段四组件状态与路线（2026-10-27）

**性质：状态收口，不是验收通过。** 阶段四入口按 [ADR-0018](adr/ADR-0018-task-aware-cl-gate.md)
（任务感知 CIL）与 [ADR-0019](adr/ADR-0019-stage-four-entry-decoupled-from-relearn.md)
（与重学 3× 解耦）打开。本文记录截至 2026-10-27 的组件状态、边界和开放研究项，
供阶段四后续排期使用；不替代任何预注册，也不把“组件 ready”写成“阶段四端到端成立”。

## 1. 记分卡

| 组件 | 状态 | 关键数字 | 边界/口径 |
| :--- | :--- | :--- | :--- |
| 多时间尺度工作记忆 | **v4 ready** | 三线索 R² **0.9994 / 0.9982 / 0.9994**，均值 0.9990 > ridge 0.9931 | 局部 RLS 读出（rank-one，无反向传播）；脉冲 v5 不 ready |
| 元认知门控 | **v3 + v4 ready（仅尺度）** | v3 CIFAR：选择性 80% **0.5423**、AUROC **0.7530**、ECE **0.0296**；v4 e2e v3：弃权 **0.8456**、AUROC **0.9003**、覆盖 **0.8028** | v4 增益来自 `state_norm` 对 A/B×3 尺度 OOD 的检测。v5 旋转不构成 OOD；v6 量化 OOD 有效（误差 6.38×）但弃权 0.1716/AUROC 0.4876，门失效。不是通用 OOD；v7 高阶矩 0.2048/0.5024，仍失效 |
| 世界模型 P3 | **部分** | 闭环 H=10 NMSE ~2e-8；首发散步 **115.5**；终止闭环 AUROC **0.632–0.665**；v2 多步想象 **0.6469** | 状态/闭环稳定过；终止头仍不过 0.75，`p3_ready=false`；v2 标签修订与结果见 README |
| 情景记忆 | **DPAR 非局部可用** | DPAR（K=8、σ=0.1）：字面 NN **0.938**、pseudo **0.828**、Hebbian **0.311**；状态控制 **0.121**（chance 0.125） | 只有非局部伪逆过线；局部 Hebbian 仍不可用；容量边界 K=8–16；原 CartPole 任务边界保留 |
| 阶段四集成 demo | **可运行** | WM/世界模型逐位等价；元认知回归 AUROC **0.562**；情景检索 MSE 1.194 | 组合/等价性成立；端到端元认知未继承，不能当能力结论 |
| 端到端任务（延迟问答 + OOD 弃权） | **4/5 判据过** | R² **0.9994**、选择性 MSE **0.00053**、OOD AUROC **0.927**、弃权 **0.887**；情景 MSE **0.822** | WM + 元认知 + 世界模型惊喜能协同；**情景检索是唯一未过项**，`e2e_ready=false` |
| 端到端 v2（四线索 + 离散情景） | **5/5 ready** | R² **0.9983**、选择性 MSE **0.00172**、OOD AUROC **0.8895**、弃权 **0.824**、E 检索 **1.000** | 五条全过；但 E 检索是“管道通”不是难记忆，seed 2 弃权仅 0.746 |
| 端到端 v3（容量/噪声/延迟压力） | **4/5 判据** | 主档 K=8/σ=0.3/t=100：R² **0.9092**、选择性 MSE **0.0675**、OOD AUROC **0.8350**、弃权 **0.7056**、E 检索 **0.9040** | 弃权未过（五种子全 <0.80）；容量边界 K=8–16（K=16 仅 0.7736）；v2 的弃权余量本就薄。元认知 v4 的密度门在同一数据上到 0.8456，但依赖尺度特征，不改变本行判定 |
| 端到端 v4（DPAR + WM 共存） | **4/6 判据** | WM R² **0.6558**、DPAR **0.9628**、状态控制 **0.2652**、OOD AUROC **0.8612**；选择性 MSE **0.3507**、弃权 **0.7648** 未过 | DPAR 与 WM 共存互相干扰；预注册 R²≥0.50 与 MSE≤0.10 本身不一致，本轮判据不可达；`e2e_v4_ready=false` |
| 端到端 v5（两段式 WM → DPAR） | **6/6 ready** | WM R² **0.99995**、选择性 MSE **6.16e-05**、DPAR **0.9584**、状态控制 **0.2472**、OOD AUROC **0.9572**、弃权 **0.9492** | 顺序组合：WM 在 t=35 先测，DPAR 在 t=85 后测，不共享最终状态；v4 的共存干扰原样保留；不是阶段四整体验收 |
| 端到端 v6（单 bank / 双 bank 共存） | **两臂均未 ready** | single：R² **0.527**、MSE **0.339**、OOD 弃权 **0.793**；dual：R² **0.994**、MSE **0.0046**、OOD 弃权 **0.622**；DPAR 0.962、状态控制 0.235 | 路由修好 WM 但打坏 OOD（AUROC 0.874→0.782）；共存与弃权未同时满足 |
| 端到端 v7（双 bank + bank W 置信度） | **5/6** | WM R² **0.9938**、MSE **0.0045**、DPAR **0.962**、状态控制 **0.235**、OOD AUROC **0.882**、弃权 **0.786** 未过 | 把 bank D 从置信度移除后弃权 0.622→0.786、AUROC 0.782→0.882；仍差 1.4 pp；t=85 bank W 衰减是剩余边界 |
| 端到端 v8（共享状态 + bank W 尺度门） | **6/6 ready** | WM R² **0.9938**、MSE **0.0045**、DPAR **0.962**、状态控制 **0.235**、OOD AUROC **0.9562**、弃权 **0.9368** | 共享最终状态；增益由 bank W 的 `state_norm` 闭合（gauss2 0.823→gauss3 0.937）；只对 A/B×3 尺度 OOD；不是通用弃权，也不是阶段四整体验收 |
| 端到端 v9（v8 稳健性） | **主档 ready，稳健 false** | scale 1.5/2/3/5 弃权 **0.456 / 0.720 / 0.937 / 0.996**；K2/4/8 WM R² **0.998 / 0.994 / 0.89975**（K8 差 0.00025）；σ0.1/0.3/0.5 弃权 **0.945 / 0.937 / 0.919** | `v9_main_ready=true`、`v9_robust=false`、`v9_ready=false`；v8 只覆盖 **s≥3、K≤4、σ≤0.5**，s=2 就掉到 0.720 |
| 端到端 v10（收缩协方差门） | **s2/s3/s5 ready** | s2 覆盖 **0.7900**、弃权 **0.8276**、AUROC **0.8937**；s3/s5 弃权 **0.9764 / 0.9984**；K8 WM R² **0.89975**、覆盖 **0.7532** | `v10_ready=true`：s2 从 0.720 提到 0.828；K8 仍是容量/覆盖率边界；s=1.5 未测 |
| 端到端 v11（K=8 三臂） | **三臂未 ready** | baseline R² **0.8926**/覆盖 **0.7740**；snapshot85 R² **0.9936**/覆盖 **0.7552**；longtau R² **0.90145**/覆盖 **0.7776**；DPAR 0.934、状态 0.12 | 唯一失败项是 K=8 的 test coverage（<0.78）；WM 容量已被 snapshot85/longtau 修好，剩余是门校准/覆盖率 |
| 端到端 v12（K=8 diag 门） | **两臂 ready** | snapshot85+diag：R² **0.9936**、覆盖 **0.7820**、弃权 **0.9292**；longtau+diag：R² **0.90145**、覆盖 **0.8052**、弃权 **0.9244**；shrunk 仍 **0.7552 / 0.7776** | `v12_ready=true`：K8 的边界是协方差估计，不是 WM；两臂均过，但 snapshot 覆盖 0.7820 只比 0.78 高 0.002 |
| 端到端 v14（18 格矩阵） | **14/18 ready** | s3/s5 全过；K4-s2 σ0.1/0.3 过；失败 **K4-s2-σ0.5 弃权 0.7756**、**K8-s2 三档 0.7656/0.7448/0.7024** | `v14_ready=false`；失败项只有 s=2 的 OOD 弃权；WM/DPAR/状态控制/AUROC 都过 |

## 2. “ready”是什么意思

- 只表示**在该组件的预注册判据下**通过（见各结果 README），不包括：
  - 阶段四端到端任务；
  - 与总线/插件的集成；
  - 脉冲化（工作记忆）或全局部化（元认知标定头）；
- 所有“负结果/边界”原样保留，没有事后改判据；计划书台账 ㊽–108 是完整索引。

## 3. 开放研究项（按建议顺序）

1. **端到端 v3（压力测试）已完成，4/5**：主档 K=8/σ=0.3/t=100 只挂 OOD 弃权
   （0.7056 < 0.80），五种子全低于线；OOD 排序本身仍可用（AUROC 0.835）。容量边界在
   K=8–16（K=16 E top-1 0.7736）。v2 的 0.824 只比门槛高 2.4 pp，本就薄。**v4 已把
   DPAR 接入**：DPAR 0.9628 过，但 WM 被干扰（R² 0.6558、MSE 0.3507），且预注册
   R²≥0.50 与 MSE≤0.10 不一致；下一版先修判据/拆分目标，不要把 v4 改判成 ready。
   **v5 已用两段式做到 6/6**（WM 在 t=35、DPAR 在 t=85，不共享最终状态），但这只是
   顺序组合上界，不能推翻 v4 的共存干扰。**v6 的双 bank 路由**把 WM R² 从 0.527 修到
   0.994，但 OOD 弃权从 0.793 掉到 0.622；两臂都未 ready，共存和弃权仍未同时满足。
   **v7 只把 bank W 用于置信度**：弃权 0.622→0.786、AUROC 0.782→0.882，仍差 1.4 pp
   才到 0.80；差距来自 t=85 的 A/B/Q 衰减，不是阈值位置。**v8 加入 bank W 的
   `state_norm`** 后弃权 0.9368、AUROC 0.9562，六条全过；但这是**尺度 OOD 专用**，
   不是通用弃权，也不改变 v6/v7 的信息丢失型 OOD 边界。**v9 稳健性扫描**：s=2 弃权
   0.720、K=8 WM R² 0.89975，`v9_robust=false`；v8 的 ready 只能写成 s≥3、K≤4。
   **v10 用 bank W 收缩 Mahalanobis** 把 s=2 的弃权提到 0.8276（覆盖 0.7900、AUROC
   0.8937），s3/s5 也过；K=8 仍是容量/覆盖率边界（0.89975 / 0.7532）。
   **v11 把 K=8 拆成三臂**：snapshot85 的 R² 0.9936、longtau 0.90145，但 coverage
   0.7552/0.7776 都低于 0.78；K8 的剩余边界是**门校准**，不是 WM 容量。**v14 全矩阵
   14/18**：s3/s5 全过，失败只剩 s=2 的弃权（K4-σ0.5 0.7756、K8 三档 0.766/0.745/0.702）。
   **v12 换 diag 门**：snapshot85 0.7820、longtau 0.8052，两臂七项全过；shrunk 仍不过。
   K8 的边界是 **28×28 协方差估计**，不是 WM 或 DPAR。
2. **工作记忆 v6（脉冲）**：v5 的失败在**表示**（60 步计数只有个位数），不在读出；
   v6 需改膜电位/亚阈值状态、更长积分窗口或 rate 输入。注意 v5 跑了 2598.7 s，下一轮
   要减种子/窗口或并行。
3. **元认知 v4 已完成（有边界）**：独立校准集 + 三维密度门（disagreement / surprise /
   state_norm）把 e2e v3 的弃权从 0.7056 提到 **0.8456**、AUROC 0.8350→**0.9003**，
   选择性 MSE 基本不变。但 **gauss2 反而更差**（0.6984），增益来自 `state_norm` 对
   A/B×3 尺度偏移的检测；需要**与尺度无关的 OOD**（协方差/频率/相位偏移）再验，
   不能把这条当通用弃权能力。v3 的分类标定头仍未迁移到回归。**v5 的保范正交旋转经
   全量误差检查不是 OOD**（0.0679 vs in-dist 0.0681），所以 v5 无泛化结论；下一步必须
   先过“OOD 误差 ≥2× in-dist”的有效性门，再谈弃权。**v6 量化 OOD 过了有效性门**
   （6.38×），但 gauss3 弃权 0.1716、AUROC 0.4876，四个门全失效；**v7 换高阶矩门**
   （mean z²/|z|/z⁴/尾比例）仍是 0.2048 / 0.5024，尺度 sanity 也掉到 0.7544。所以 v4
   只能限定在尺度偏移，不能写成通用弃权能力。
4. **世界模型 P3 后续**：终止闭环的分布偏移（离线 0.999 → 闭环 0.665）没有解决；
   方向是更强状态模型（把发散步从 115 推远）或与想象分布匹配的终止头；一步想象增广
   已试，0.632；**v2 多步想象 + 真实标签 0.6469**，仍不够；下一步需要时间/风险特征或
   更强状态模型。
5. **情景记忆**：DPAR（v4）证明任务确实需要绑定（状态控制 0.121 ≈ chance）；**非局部
   伪逆**在 K=8/σ=0.1 过线（0.828），字面 NN 0.938 更高，**局部 Hebbian 仍只有 0.311**，
   容量边界 K=8–16。原 CartPole 任务/噪声/时间顺序边界保留。下一步应试局部或低维键，
   或接受查表基线；不要继续调 k/β/编码维度。
6. **工作记忆 v5 的替代**：如果项目坚持“纯脉冲计数”，需要先解决低计数表示问题；
   否则接受 v4 的速率漏积分器作为阶段四的可用局部表示。

## 4. 与阶段三的关系

- 阶段三 CL 门按 ADR-0018 为任务感知 CIL 已过（构造性 AF=0，不是抗遗忘算法）；
- **重学 3× 未达**，按 ADR-0019 与阶段四入口解耦，仍是开放项；
- 本文件**不改变**上述两条，也不把阶段四组件 ready 写成阶段三全部验收。

## 5. 索引

- 工作记忆：[v4 结果](../sweep_results/working_memory_v4/README.md)、
  [v5 结果](../sweep_results/working_memory_v5/README.md)
- 元认知：[v3 结果](../sweep_results/metacognitive_gate_v3/README.md)、
  [v4 风险控制](../sweep_results/metacognitive_gate_v4/README.md)、
  [v5 旋转有效性检查](../sweep_results/metacognitive_gate_v5/README.md)、
  [v6 量化 OOD](../sweep_results/metacognitive_gate_v6/README.md)、
  [v7 高阶矩门](../sweep_results/metacognitive_gate_v7/README.md)
- 世界模型：[P3 闭环](../sweep_results/world_model_closed_loop_h500/README.md)、
  [终止头集成](../sweep_results/world_model_closed_loop_head/README.md)、
  [想象状态头](../sweep_results/world_model_closed_loop_imagined/README.md)、
  [终止头 v2](../sweep_results/world_model_termination_v2/README.md)
- 情景记忆：[v3 结果](../sweep_results/episodic_memory_v3/README.md)、
  [DPAR 结果](../sweep_results/episodic_paired_associate/README.md)
- 集成 demo：[结果](../sweep_results/stage4_integration_demo/README.md)
- 端到端：[v2 结果](../sweep_results/stage4_e2e_v2/README.md)、
  [v3 压力测试](../sweep_results/stage4_e2e_v3/README.md)、
  [v4 DPAR 共存](../sweep_results/stage4_e2e_v4/README.md)、
  [v5 两段式](../sweep_results/stage4_e2e_v5/README.md)、
  [v6 单/双 bank](../sweep_results/stage4_e2e_v6/README.md)、
  [v7 bank W 置信度](../sweep_results/stage4_e2e_v7/README.md)、
  [v8 bank W 尺度门](../sweep_results/stage4_e2e_v8/README.md)、
  [v9 稳健性扫描](../sweep_results/stage4_e2e_v9/README.md)、
  [v10 收缩协方差门](../sweep_results/stage4_e2e_v10/README.md)、
  [v11 K8 三臂](../sweep_results/stage4_e2e_v11/README.md)、
  [v12 K8 diag 门](../sweep_results/stage4_e2e_v12/README.md)、
  [v14 18 格矩阵](../sweep_results/stage4_e2e_v14/README.md)
- 技术报告：[paper/stage4/README.md](../paper/stage4/README.md) · Zenodo
  [10.5281/zenodo.23696829](https://doi.org/10.5281/zenodo.23696829)（v1.0，2026-10-28）

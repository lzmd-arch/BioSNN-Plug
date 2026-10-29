# 阶段四组件状态与路线（2026-10-27）

**性质：状态收口，不是验收通过。** 阶段四入口按 [ADR-0018](adr/ADR-0018-task-aware-cl-gate.md)
（任务感知 CIL）与 [ADR-0019](adr/ADR-0019-stage-four-entry-decoupled-from-relearn.md)
（与重学 3× 解耦）打开。本文记录截至 2026-10-27 的组件状态、边界和开放研究项，
供阶段四后续排期使用；不替代任何预注册，也不把“组件 ready”写成“阶段四端到端成立”。

## 1. 记分卡

| 组件 | 状态 | 关键数字 | 边界/口径 |
| :--- | :--- | :--- | :--- |
| 多时间尺度工作记忆 | **v4 + v9 ready** | v4 三线索 R² **0.9994 / 0.9982 / 0.9994**；v9 脉冲迹 A/B **0.9257 / 0.9281**（LIF）、**0.9261 / 0.9292**（ALIF） | v4 局部 RLS；v9 独立传入群体 + 脉冲迹（臂差中位数 0.130，非平凡）；v5/v6/v7 负结果，v8 平凡解不作证据；v9 是脉冲计数/低通码，不要求时序 |
| 元认知门控 | **v3 + v8 + v11 ready** | v3 CIFAR：选择性 80% **0.5423**、AUROC **0.7530**、ECE **0.0296**；v8（种子 5–9）：覆盖 **0.810**、选择性 MSE **0.033**、量化弃权 **0.898**、AUROC **0.925**、尺度弃权 **0.851**、有效性 **6.43×** | v4 仅尺度；v5 旋转不构成 OOD；v6/v7 量化门失效；v8 是重建一致性门；**v11 三维门（max/mean/norm）四族全过**：time_shift 0.998、time_reverse 0.990、dim_dropout 0.858、correlated 0.987，尺度 0.855；v9（均值）0.596/0.539、v10（max）dim 0.762 的负结果保留。仍只覆盖输入变换型 OOD，不是通用弃权 |
| 世界模型 P3 | **状态/开环/滚动闭环终止 ready；纯想象闭环开放** | 闭环 H=10 NMSE ~2e-8；散步 115.5；开环 v5 AUROC **0.999996**、FP **0.000504**、命中 **0.998**；滚动闭环 v6 AUROC **1.0**、FP **0.000912**、命中 **1.0**、终止步误差 **1.14e-5**；纯想象 **0.634–0.665** | 开环与滚动闭环都 ready；**500 步纯想象闭环仍失败**（v2 0.6469、v6 对照 0.645），根因是动作反馈 + float32 逐位敏感性；`p3_closed_loop_imagination_ready=false` |
| 情景记忆 | **DPAR 非局部 + 局部 delta 可用** | v4：字面 NN **0.938**、pseudo **0.828**、Hebbian **0.311**；v5 局部 delta K=8 **0.805**、K=4 **0.922**、状态控制 **0.123** | 局部 delta 逼近伪逆（0.805 vs 0.816），K8 薄过（3/5 种子<0.80），K4 稳定；容量 K=8–16；Hebbian 一次性外积仍不可用 |
| Predictive E-prop | **1/3 种子全过；`ready=false`** | 重建 0.011–0.022×（三种子全过）；自由运行幅度 **0.584 / 0.536 / 0.119**、DTW **0.088 / 0.137 / 0.231**（对照 0.228–0.268） | 自持振荡跨种子不稳定；未测 Lorenz/噪声/插值与 TBPTT 对照；按描述改编非逐位复现 |
| 规模/神经发生/能效 | **规模/事件优先级/神经发生均有边界数据；能效仅代理** | 8 GB 上限约 **1.35k**；500k 前向 **5.02 s**、单 update **5.27 s**；10k 6000 样本 1 epoch **205.6 s / acc 0.520**、完整 60k 1 epoch **2,397 s / acc 0.697**（单 epoch、非验收）；事件优先级选择 **0.339** 但精度 **0.6213 vs 0.7748**；神经发生线性 0.432/0.310、NCM 0.766/0.277；实测 sparsity **0.8758 → 代理 41.2×** | 500k 训练未验；能效无硬件；事件优先级/NCM 均未 ready |
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
| 端到端 v15（补 s=2 四格） | **0/4** | K4-σ0.5 longtau **0.7868**；K8-s2 snapshot **0.7700/0.7416/0.7044** | s=2 的换臂（longtau/snapshot85）仍不过 0.80；边界未补 |
| 端到端 v16（输入尺度门） | **18/18 ready** | K4/K8 × s2/3/5 × σ.1/.3/.5：覆盖 **0.796–0.810**、弃权 **0.882–1.000**、AUROC **0.928–1.000**；WM/DPAR/状态全过 | **输入级** input_norm 门，不是状态级元认知；v14 状态级仍 14/18；只对尺度 OOD |
| 端到端 v17（状态级重建门） | **12/18 ready** | 种子 10–14：s3/s5 全过；s2 六格弃权 **0.620–0.736** 全失败；其余 R² 0.906–0.994、DPAR 0.930–0.964、状态 0.125–0.243 | 状态级 kNN[errors,state_norm] 未关闭 s=2，比 v14 的 14/18 少两格；v16 输入级 18/18 不混报 |
| 端到端 v18（状态级预测范数门） | **18/18 ready** | 种子 15–19：s=2 弃权 **0.837–0.907**、覆盖率 **0.795–0.814**、AUROC **0.900–0.942**；WM R² 0.906–0.994、DPAR 0.937–0.958 | 状态级门，不用 input_norm；v14 14/18、v17 12/18 原样保留 |

## 2. “ready”是什么意思

- 只表示**在该组件的预注册判据下**通过（见各结果 README），不包括：
  - 阶段四端到端任务；
  - 与总线/插件的集成；
  - 全局部化（元认知标定头）；v9 的脉冲工作记忆是计数/低通码，时序编码未验；
- 所有“负结果/边界”原样保留，没有事后改判据；计划书台账 ㊽–110 是完整索引。

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
   **v15 换臂后仍 0/4**（longtau 0.7868；snapshot 0.770/0.742/0.704），s=2 需换更强的尺度特征。
   **v16 换输入尺度门 `max(||A||,||B||)` 后 18/18 全过**；但这是输入级门，不是状态级
   元认知，v14 的 14/18 仍是状态级结论。**v17 把元认知 v8 的 kNN[errors,state_norm] 接到 18 格，只到 12/18**：s=3/5 全过，s=2 六格弃权 0.620–0.736 全失败，比 v14 少两格；状态级 s=2 仍未关闭。**v18 换读出预测范数 z 门后 18/18**：s=2 六格弃权 0.837–0.907、覆盖率 0.795–0.814、AUROC 0.900–0.942，状态级 s=2 关闭。
   **v12 换 diag 门**：snapshot85 0.7820、longtau 0.8052，两臂七项全过；shrunk 仍不过。
   K8 的边界是 **28×28 协方差估计**，不是 WM 或 DPAR。
2. **脉冲工作记忆 v5–v9（已收口）**：v5 计数 0.217/0.187、v6 膜电位 0.309/0.121、
   v7 群体膜电位 0.788——瓶颈从“5 个伯努利样本”转到“脉冲复位擦除”；v8 的
   `v+thr·trace` 代数抵消复位，LIF/ALIF 逐位相同（≤2e-7），是平凡解、不作证据；
   **v9 用独立传入群体 + 纯脉冲迹 `s=αs+z`**，LIF/ALIF 0.9269/0.9276，臂差中位数
   0.130，四条判据全过。边界：v9 是脉冲计数/低通码，不要求时序；读出是测量头。
3. **元认知 v4 已完成（有边界）**：独立校准集 + 三维密度门（disagreement / surprise /
   state_norm）把 e2e v3 的弃权从 0.7056 提到 **0.8456**、AUROC 0.8350→**0.9003**，
   选择性 MSE 基本不变。但 **gauss2 反而更差**（0.6984），增益来自 `state_norm` 对
   A/B×3 尺度偏移的检测；需要**与尺度无关的 OOD**（协方差/频率/相位偏移）再验，
   不能把这条当通用弃权能力。v3 的分类标定头仍未迁移到回归。**v5 的保范正交旋转经
   全量误差检查不是 OOD**（0.0679 vs in-dist 0.0681），所以 v5 无泛化结论；下一步必须
   先过“OOD 误差 ≥2× in-dist”的有效性门，再谈弃权。**v6 量化 OOD 过了有效性门**
   （6.38×），但 gauss3 弃权 0.1716、AUROC 0.4876，四个门全失效；**v7 换高阶矩门**
   （mean z²/|z|/z⁴/尾比例）仍是 0.2048 / 0.5024，尺度 sanity 也掉到 0.7544。所以 v4
   只能限定在尺度偏移，不能写成通用弃权能力。**v8 换成重建一致性门**
   （kNN(5) on [errors, state_norm]）：种子 5–9 覆盖 0.810、量化弃权 0.898/AUROC
   0.925、尺度 0.851、有效性 6.43×，`v8_ready=true`；边界是只覆盖输入变换型 OOD，
   覆盖率单种子出带。
4. **世界模型 P3（拆分收口）**：闭环状态/稳定性过（H=10 NMSE ~2e-8、散步 115.5），
   但**闭环终止**仍失败：v1 0.616、dedicated 0.665、一步想象 0.632、v2 多步 0.6469，
   v3/v4/v5 的闭环对照也只有 0.634–0.640。根因是动作反馈：平衡策略在 `θ+0.5θ̇` 阈值
   附近翻转动作，微小模型误差改变动作序列，115 步想象发散；DAgger 不能修（一步误差已
   1e-5）。**开环固定动作序列**的终止风险可用：v3 AUROC≈1 但全或无分离差 1.4e-4，
   v4 率判据 FP 0.00244 未过，**v5 校准阈值后 AUROC 0.999996、FP 0.000504、命中
   0.998、终止步误差 0.00305**。`p3_v5_ready=true` 仅限开环；**v6 滚动时域闭环**（每步用真实观测出动作、模型只预测一步）AUROC 1.0、FP 0.000912、命中 1.0、终止步误差 1.14e-5，`p3_v6_ready=true`。**500 步纯想象闭环仍开放**，`p3_closed_loop_imagination_ready=false`。
5. **情景记忆**：DPAR（v4）证明任务确实需要绑定（状态控制 0.121 ≈ chance）；**非局部
   伪逆**在 K=8/σ=0.1 过线（0.828），字面 NN 0.938 更高，**局部 Hebbian 仍只有 0.311**。
   **v5 的局部 delta 规则**把 K=8 提到 **0.805**、K=4 **0.922**，逼近伪逆 0.816；
   边界：K8 均值过线但 3/5 种子 <0.80，K=16 掉到 0.531，σ=0.3 掉到 0.553。容量边界
   K=8–16；原 CartPole 任务/噪声/时间顺序边界保留。
6. **工作记忆收口**：v4（速率）与 v9（脉冲迹）都是 ready；v9 的边界是计数/低通码、
   不要求时序。若后续要主张时序编码，需要新的预注册与时间置换守卫。

## 4. 与阶段三的关系

- 阶段三 CL 门按 ADR-0018 为任务感知 CIL 已过（构造性 AF=0，不是抗遗忘算法）；
- **重学 3× 未达**，按 ADR-0019 与阶段四入口解耦，仍是开放项；
- 本文件**不改变**上述两条，也不把阶段四组件 ready 写成阶段三全部验收。

## 5. 索引

- 工作记忆：[v4 结果](../sweep_results/working_memory_v4/README.md)、
  [v5 结果](../sweep_results/working_memory_v5/README.md)、
  [v6 膜电位](../sweep_results/working_memory_v6/README.md)、
  [v7 群体膜电位](../sweep_results/working_memory_v7/README.md)、
  [v8 平凡解](../sweep_results/working_memory_v8/README.md)、
  [v9 脉冲迹](../sweep_results/working_memory_v9/README.md)
- 元认知：[v3 结果](../sweep_results/metacognitive_gate_v3/README.md)、
  [v4 风险控制](../sweep_results/metacognitive_gate_v4/README.md)、
  [v5 旋转有效性检查](../sweep_results/metacognitive_gate_v5/README.md)、
  [v6 量化 OOD](../sweep_results/metacognitive_gate_v6/README.md)、
  [v7 高阶矩门](../sweep_results/metacognitive_gate_v7/README.md)、
  [v8 重建一致性门](../sweep_results/metacognitive_gate_v8/README.md)
- 世界模型：[P3 闭环](../sweep_results/world_model_closed_loop_h500/README.md)、
  [终止头集成](../sweep_results/world_model_closed_loop_head/README.md)、
  [想象状态头](../sweep_results/world_model_closed_loop_imagined/README.md)、
  [终止头 v2](../sweep_results/world_model_termination_v2/README.md)、
  [终止头 v3](../sweep_results/world_model_termination_v3/README.md)、
  [终止头 v4](../sweep_results/world_model_termination_v4/README.md)、
  [终止头 v5](../sweep_results/world_model_termination_v5/README.md)、
  [终止头 v6 滚动闭环](../sweep_results/world_model_termination_v6/README.md)
- 情景记忆：[v3 结果](../sweep_results/episodic_memory_v3/README.md)、
  [DPAR 结果](../sweep_results/episodic_paired_associate/README.md)、
  [局部 delta v5](../sweep_results/episodic_paired_associate_v5/README.md)
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
  [v14 18 格矩阵](../sweep_results/stage4_e2e_v14/README.md)、
  [v15 补 s=2 四格](../sweep_results/stage4_e2e_v15/README.md)、
  [v16 输入尺度门](../sweep_results/stage4_e2e_v16/README.md)、
  [v17 状态级重建门](../sweep_results/stage4_e2e_v17/README.md)、
  [v18 预测范数门](../sweep_results/stage4_e2e_v18/README.md)
- 技术报告：[paper/stage4/README.md](../paper/stage4/README.md) · Zenodo
  [10.5281/zenodo.23696829](https://doi.org/10.5281/zenodo.23696829)（v1.0，2026-10-28）

# CL v2：Route R 基线重测、回放与归一化臂（2026-10-25 起）

**预注册见 [`../continual_v2_replay_prereg.md`](../continual_v2_replay_prereg.md) 与
[`../continual_v2_replay_prereg_amend1.md`](../continual_v2_replay_prereg_amend1.md)（均先于数据）。**
路线切换的正式记录见 [ADR-0017](../../docs/adr/ADR-0017-cl-main-route-switch-and-replay-first.md)。

## 协议

| 项 | 取值 |
| :--- | :--- |
| 特征 | **冻结 ResNet18**（ImageNet1K V1，avgpool，512 维），CIFAR resize 224 |
| 数据 / 切分 | CIFAR-100，10×10，按类 id 升序，全量 |
| 评估 | **CIL**：样本按任务切、argmax 在全部已见类、不提供任务 id |
| 种子 | 0–4 |
| 读出 | NCM+白化（参考）或 delta 线性 |
| 回放 | 每类 20 条**特征**，minibatch B=128，`n_replay = round(ρ·B)` |

## Step 0：Route R 基线（先于回放）

| 臂 | 读出 / batching | ACC | AF |
| :--- | :--- | ---: | ---: |
| R-NCM | 类原型 + 对角白化 | **0.6583**（5 种子同值） | **0.1276** |
| R-delta-full | delta，全批量（v1） | 0.2358（同值） | 0.7933 |
| **R-delta-mini** | delta，minibatch B=128 | **0.2466**（0.2460–0.2469） | **0.8047**（0.8016–0.8066） |

R-NCM 与探索期的 0.658 / 0.127 一致——**旧数字不是累计矩阵造成的假象**；但 NCM 的 AF
仍然 12.8%，离 8% 有距离。delta 读出即使在冻结的强特征上也会灾难性遗忘（AF 约 0.80）。

## 回放臂（对照 = R-delta-mini）

| 臂 | ρ | ACC | AF | 相对对照 |
| :--- | ---: | ---: | ---: | :--- |
| R-delta-mini | 0 | 0.2466 | 0.8047 | — |
| replay-ρ0.1 | 0.1 | 0.2608（0.2597–0.2617） | 0.8013（0.8003–0.8020） | AF −0.3 pp，ACC +1.4 pp |
| replay-ρ0.3 | 0.3 | **0.3630**（0.3612–0.3655） | **0.7399**（0.7383–0.7416） | **AF −6.5 pp，ACC +11.6 pp** |

**判据（预注册）：AF < 8% 主、ACC ≥ 65% 且不低于对照 1 pp 次。**

- **两档都没过**：AF 80.1% / 74.0%，ACC 26.1% / 36.3%；
- 回放**方向正确**（ρ=0.3 把 AF 拉低 6.5 个百分点、ACC 抬高 11.6 个百分点），
  但离 8% / 65% 的门还差一个数量级；
- 按预注册：**回放不够 → 下一份预注册才轮到归一化**；不回头调 ρ、不改判据。

## 归一化臂（对照 = R-delta-mini）

预注册：[`../continual_v2_normalization_prereg.md`](../continual_v2_normalization_prereg.md)
（先于数据）。唯一机制是**任务 1 统计量冻结的逐维标准化**；两个臂把"逐维几何"与
"整体步长"分开。无回放。

| 臂 | 变换 | ACC | AF | 相对参考 |
| :--- | :--- | ---: | ---: | :--- |
| R-delta-mini | 无 | 0.2466 | 0.8047 | — |
| **N1** | 逐维标准化（native） | 0.2686 | **0.9033** | ACC +2.20 pp，AF **+9.86 pp（更差）** |
| **N1-scale** | 标准化后对齐任务 1 行范数 | **0.3151** | 0.7917 | ACC **+6.85 pp**，AF −1.30 pp |

预注册的机制信号要求 **AF −10 pp 且 ACC ≥ 参考 −1 pp**；两臂都卡在 AF 上 →
**`no_signal`**。阶段门（AF<8% / ACC≥65%）也仍远未达到。

关键读数：本仓 ResNet18 特征的行范数恒为 1.0，所以 **N1 同时把有效步长放大了约 22.5 倍**
（标准化后行范数 22.545，η 不变）——它的 AF 恶化不能单独归给"归一化"。把尺度对齐后
（N1-scale）几何本身让 ACC 涨 6.9 pp、AF 只降 1.3 pp：**归一是 ACC 的小杠杆，不是 AF 的杠杆**。

按预注册判读表：两臂都不过 → **归一化不是这个瓶颈的杠杆；下一机制是参数隔离**，
需要新预注册（未开始）。结果目录：[`normalize_diag/`](normalize_diag/README.md)、
[`normalize_diag_scaled/`](normalize_diag_scaled/README.md)。

## 参数隔离臂 P1（对照 = R-delta-mini）

预注册：[`../continual_v2_isolation_prereg.md`](../continual_v2_isolation_prereg.md)（先于数据）。
只更新当前任务 10 个类的输出行，其余 90 行逐位冻结；无回放、无归一化。

| 臂 | ACC | AF | 相对参考 |
| :--- | ---: | ---: | :--- |
| R-delta-mini | 0.2466 | 0.8047 | — |
| **P1 current-rows** | **0.5426** | **0.1739** | ACC **+29.60 pp**，AF **−63.08 pp** |

**但不能宣称"输出行隔离有效"**：

- **"不是少学"不过**：逐任务对角线均值 **0.534**，参考臂 0.808，门槛是 0.758——
  隔离保住了旧行，却让当前行要在冻结旧行之上重新争 argmax，新任务越往后越难学；
- **对齐检验无法判定**：随机行控制臂 5 个种子**全部作废**——随机行大多不是当前类、
  目标为 0 且权重为 0，误差恒为 0，跑门出现 0.0/NaN（见
  [`isolation_random_rows/`](isolation_random_rows/README.md)）。这是控制设计的问题，
  不是 P1 的代码问题。

所以准确读数是：**只更新当前行大幅压低遗忘，但同时显著降低获取能力**——用获取换保持，
不是免费的隔离。下一步若继续，必须与回放叠加（回放把获取拉回来），按新预注册重跑；
在叠加结果之前，不把参数隔离写成 AF 的杠杆。结果：
[`isolation_current_rows/`](isolation_current_rows/README.md)、
[`isolation_random_rows/`](isolation_random_rows/README.md)。

## 隔离 + 回放（P1+R）

预注册：[`../continual_v2_isolation_replay_prereg.md`](../continual_v2_isolation_replay_prereg.md)。
隔离（只更新当前行）+ latent replay ρ0.3，5 种子。

| 臂 | ACC | AF |
| :--- | ---: | ---: |
| replay ρ0.3 | 0.3630 | 0.7399 |
| P1（只隔离） | 0.5426 | 0.1739 |
| **P1+R** | **0.4310** | **0.0291** |

- **AF 第一次过 8% 线（2.91%）**，但 ACC 43.10% 不过 65%；阶段门只过一半；
- 相对 replay-alone 过（AF −71.1 pp、ACC +6.8 pp），但相对 P1 **ACC 掉了 11.2 pp**、
  **"不是少学"不过**（对角线 0.328）——回放在隔离下救不回获取；
- 原因：回放样本是旧类，隔离只让当前行更新，回放无法重训旧行；它还挤掉了当前样本的
  更新量（每批 128 → 90）。

结论：**隔离与回放不叠加**。下一机制按预注册写死：**部分隔离**（旧行 `α·η` 小步长，
α=0.1 / 0.3），先预注册再跑。结果：
[`isolation_replay/`](isolation_replay/README.md)。

## 部分隔离（α = 0.1 / 0.3）

预注册：[`../continual_v2_partial_isolation_prereg.md`](../continual_v2_partial_isolation_prereg.md)。
当前行全步长、旧行 α·η，无回放。5 种子。

| α | ACC | AF | 对角线 |
| ---: | ---: | ---: | ---: |
| 0（P1） | 0.5426 | 0.1739 | 0.534 |
| **0.1** | **0.2610** | **0.7951** | **0.808** |
| 0.3 | 0.2512 | 0.8017 | 0.808 |
| 1（参考） | 0.2466 | 0.8047 | 0.808 |

**没有中间态**：α 从 0 跳到 0.1，AF 立刻从 17.4% 回到 79.5%、对角线从 0.534 回到 0.808。
两条判据（机制信号、获取恢复）都不过，"不是少学"过——因为它已经回到全量更新的行为。
按预注册判读表：**输出行轴走到头**，不调第三个 α；下一机制转向特征级隔离或任务推断。
结果：[`partial_isolation_alpha010/`](partial_isolation_alpha010/README.md)、
[`partial_isolation_alpha030/`](partial_isolation_alpha030/README.md)。

## 特征级掩码（F1 Fisher / 随机控制）

预注册：[`../continual_v2_feature_isolation_prereg.md`](../continual_v2_feature_isolation_prereg.md)。
每任务在训练特征上取 top-k（k=256）掩码并冻结；每个类只看 `m_t ⊙ x`；无任务 id。5 种子。

| 臂 | ACC | AF | 掩码 Jaccard |
| :--- | ---: | ---: | ---: |
| R-delta-mini | 0.2466 | 0.8047 | — |
| **F1 Fisher** | **0.2343** | **0.7727** | **0.425** |
| **F1-control random** | **0.2342** | **0.7788** | **0.333** |

两条都几乎等于参考，Fisher 与随机掩码也几乎一样（AF 差 0.61 pp）：
- 机制信号（vs P1：ACC ≥59.3、AF ≤7.4）**不过**；
- 学到的掩码检验（AF 低 ≥5 pp）**不过**；
- "不是少学"过（对角线 0.775 / 0.778 ≥0.758），但阶段门远不过（ACC 23.4%）。

任务掩码之间的 Jaccard 是 0.33–0.43——**特征子空间重叠很高**，限制 50% 的特征既没
恢复获取，也没保住旧知识。按预注册判读表：**特征轴不是杠杆** → 启动**任务推断保底臂**
（新预注册），测量"任务边界信息是否必须以某种形式进入推理"。结果：
[`feature_mask_fisher/`](feature_mask_fisher/README.md)、
[`feature_mask_random/`](feature_mask_random/README.md)。

## 跑门

每个任务 `‖ΔW‖/‖W‖`（t=0 从零初始化，分母不存在，不参与判定）：

- 所有种子 **`void_seeds = []`**；t≥1 的最小值 0.34（ρ=0.3）——没有"更新没进权重"的运行。

## 这**不**说明什么

- 不说明"回放无效"：它确实减轻了遗忘，只是幅度不足；归一化已按顺序测过（`no_signal`），
  下一步是参数隔离；
- 不说明 Route R 不可行：NCM 已过 ACC 65%，只差 AF；但 NCM 不是回放的作用对象；
- 不改阶段四入口；「重学快 3×」的**标准设计**已在
  [`../relearn_3x_prereg.md`](../relearn_3x_prereg.md)（先于数据），尚未运行。

## 自证

- R-NCM：commit `1b9bea6`，工作区干净；
- 其余四臂：commit `60211a7`，工作区干净；
- 归一化两臂：commit `912a97b`，工作区干净。
- ⚠️ 首跑的四个 delta/回放目录因**输出目录未跟踪**导致自证块报"工作区不干净"，
  已移出仓外留档；本目录是干净重跑的结果（数字与首跑一致，确定性）。

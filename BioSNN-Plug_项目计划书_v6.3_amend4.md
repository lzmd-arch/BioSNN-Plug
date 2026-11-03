# 计划书 v6.3 修订之四：SEL 收口与主线转线 A

**状态**：状态页。amend2 是登记页，amend3 是 SEL 分阶段状态；本页只记 P4 之后的收口与
主线变更。
**决策**：[ADR-0034](docs/adr/ADR-0034-sel-distillation-archival.md)。
**Paper C 结构**：[`paper/c/OUTLINE.md`](paper/c/OUTLINE.md)。

## 结论

**SEL 不是失败，是特定技术路径（TI-LIF + CWSSA + SSM + KD）的效率不达标。**

| 层级 | 状态 |
| :--- | :--- |
| SEL 概念（换模型语气不变） | ✅ 仍然成立，产品痛点真实 |
| TI-LIF/ALIF 神经元（biosnn-bus） | ✅ 已发布，数值正确 |
| CWSSA/SSM + KD 蒸馏 | ✗ 停止，P4-3 不过（1.148× > 0.90×） |
| 产品端 | Phase 2（模板 + 模型混合渲染）为回退；默认仍关，T4 未过 |
| 研究端 | 唯一活跃线：线 A 脑区特化（预注册已锁，已起跑，结果待出） |
| Paper C | §4.1 SoS 归档 / §4.2 容量轴归档 / §4.3 线 A / §5 SEL 尝试与反思 |

## P4 五种子（最终）

| 臂 | test top-1 中位数 [范围] | test CE 中位数 [范围] | 中位秒 |
| :--- | ---: | ---: | ---: |
| bigram（分母） | 0.6457 | — | — |
| GRU(256) | 0.8831 [0.8795, 0.8846] | 0.5203 [0.4686, 0.5540] | 83 |
| ALIF(512) | 0.8685 [0.8634, 0.8729] | **0.4594** [0.4304, 0.5518] | 620 |
| TI-LIF(512, D=3) | 0.8641 [0.8532, 0.8685] | 0.5274 [0.4987, 0.8482] | 946 |

P4-1（≥0.7457）过；P4-2（≥0.8331）过；**P4-3 不过**（1.148×，要求 ≤0.90×）；
P4-4 N/A（无 tokenizer 对齐）；P4-5 未做；P4-6 过。

## 资源去向

- **P5 在线 R-STDP 取消**；CWSSA/SSM 不实现；不再补 KD；
- **P3 的 391 对教师数据**保留在 `D:/AxoData/sel/pairs/v1_amend3`，可转 SFT 语料，
  另写数据卡与预注册，不做蒸馏；
- **ALIF-only SEL** 作为远期备选，现在不写预注册、不分配算力；
- **产品**继续 Phase 2（模板 + 模型混合渲染），默认关；不接脉冲表达层；
- **线 A** 是唯一活跃研究线：预注册
  [`research/brainregions/prereg-brain-region-specialization.md`](research/brainregions/prereg-brain-region-specialization.md)，
  2 臂 × 3 种子 ≈ 1.8 小时；2026-11-04 已起跑，结果待出；
- **Paper C** 按 [`paper/c/OUTLINE.md`](paper/c/OUTLINE.md) 写，SEL 章节定位为
  "尝试与技术路径反思"，不写成"实现"。

## 四条硬边界

纯 CIL（ADR-0029）、脉冲 SoS（ADR-0031）、容量轴（ADR-0033）、SEL 脉冲蒸馏（ADR-0034）。
在重开条件满足并另写预注册之前，不再为这四条分配算力。

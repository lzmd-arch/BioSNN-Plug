# 预注册：元认知 v6——量化 OOD 与有效性门（2026-10-27）

**定稿时间：2026-10-27，在本轮任何数据之前。** v5 的正交旋转经全量误差检查发现
**不是 OOD**（旋转后 0.0679 ≈ in-dist 0.0681），所以作废。本预注册改用**会丢信息的
非线性量化**作为尺度无关 OOD，并**先设有效性门**：如果 OOD 全量误差没有显著升高，
本轮同样作废，不解释弃权率。

## OOD 操作

对 A/B 线索做逐元素**符号量化**：`a' = sign(a)`，`b' = sign(b)`；查询/Q 与事件/E 不变。
**目标仍是原始的 a / b**（不是量化后的值），因此模型必须从 `sign(a)` 恢复 `a` 的幅度——
信息已丢失，任务误差必须升高。`sign(a)` 的 L2 范数恒为 2，不是 A/B ×3 这种全局尺度偏移。

对照 sanity：A/B ×3 的尺度 OOD（v3/v4 原口径），应复现 v4 的 0.8456/0.9003。

## 数据

与 e2e v3 主档相同：K=8、σ=0.3、t=100；train 2000 / calibration 500 / test 500 /
OOD 500；5 个种子。阈值仍只在 calibration（`seed+30_000`）上取。

## 有效性门（先于弃权）

**OOD 全量 MSE ≥ 2 × in-dist 全量 MSE**。in-dist 全量约 0.068，因此 OOD 需 ≥0.136；
若量化没有把误差推过这条线，本轮不得解释弃权/AUROC，直接记“操作化失败”。

## 门与判据

门与 v4/v5 相同：主门 **gauss3**（disagreement / surprise / state_norm 三维高斯密度），
对照 zsum / gauss2 / knn3；阈值只在 calibration 取。

主门判据（5 种子均值）：

1. **有效性**：量化 OOD 全量 MSE ≥2× in-dist 全量 MSE；
2. test 覆盖率 **0.78–0.82**，选择性 MSE **≤0.10**；
3. 量化 OOD 弃权 **≥0.80**，AUROC **≥0.75**；
4. 尺度 sanity：gauss3 弃权 **≥0.80**；
5. 五条全过 → `quantization_ood_ready=true`；否则报边界，不换门、不调阈值/特征。

**报告义务**：四个门在量化 OOD 上的结果全报；若主门失败而对照门过，只写探索性。
若有效性门失败，`quantization_ood_ready` 必须为 false，并明确写“本轮无效”。

## 不做什么

- 不在量化 OOD 上拟合均值/协方差/阈值；
- 不调符号函数、特征集合、覆盖率、阈值分位；
- 不把“量化后任务变难”当成 OOD 检测成功：判据看弃权/AUROC；
- 不把合成量化说成自然 OOD。

## 产物

- `research/cognitive/run_metacognitive_gate_v6.py`；
- `sweep_results/metacognitive_gate_v6/summary.json` + README。

# 预注册：元认知 v7——高阶矩门 vs 信息丢失 OOD（2026-10-27）

**定稿时间：2026-10-27，在本轮任何数据之前。** v6 的量化 OOD 通过了有效性门（6.38×），
但 v4 的 gauss3 只有 AUROC 0.4876。原因很可能是：`sign(a)` 与 `a` 的**均值/协方差相同**
（都是零均值、单位方差），而 v4 的特征（disagreement / surprise / state_norm）只用了
二阶统计量。本预注册换一个**高阶矩**门：不改 OOD，不改任务，只换状态特征。

## 任务与 OOD（复用 v6）

- e2e v3 主档：K=8、σ=0.3、t=100；train 2000 / calibration 500 / test 500 / OOD 500，
  种子 0–4；
- **quantized OOD**：A/B 输入取 `sign`，目标保留原始幅度（v6 原口径）；
- **scale OOD**：A/B ×3（v4/v6 sanity）；
- 阈值仍只在 calibration（`seed+30_000`）上取。

## 主门：moment4

对每个状态向量，用 calibration 的逐维均值/标准差标准化 `z = (h - μ)/σ`，计算四个
**逐样本**的矩统计量：

```text
mean(z²), mean(|z|), mean(z⁴), mean(|z| > 1.5)
```

在 calibration 的这 4 维矩向量上拟合均值与协方差（加 1e-6 对角正则），置信度 =
负 Mahalanobis 距离；阈值 = calibration 置信度 20% 分位（80% 覆盖）。

**为什么是它**：`sign` 保留一、二阶矩，但把连续分布压到离散格点；`mean(z⁴)` 与尾比例
应下降。若这仍然检测不到，说明问题不在“简单高阶矩”，需要更强的生成模型/可逆性特征。

对照（只报告，不参与主判据）：

- `moment2`：只用 `mean(z²), mean(|z|)` 的二维矩门；
- `gauss3`：v4/v6 的 `[disagreement, surprise, state_norm]` 三维高斯；
- `zsum`、`gauss2`、`knn3`：v4/v5 同口径。

## 判据（先写死，5 种子均值）

1. **有效性**：quantized OOD 全量 MSE ≥2× in-dist 全量 MSE（复算，不引用旧值）；
2. test 覆盖率 **0.78–0.82**，选择性 MSE **≤0.10**；
3. quantized OOD 弃权 **≥0.80**，AUROC **≥0.75**；
4. scale OOD 弃权 **≥0.80**，AUROC **≥0.75**（主门也要在尺度上不退化）；
5. 五条全过 → `moment_gate_ready=true`；否则报边界，不换特征、不调阈值/正则。

**报告义务**：moment4 / moment2 / gauss3 / zsum / gauss2 / knn3 在两种 OOD 上全报；
若 moment4 失败而某个对照过，只写探索性。若有效性门失败，本轮无效，不解释弃权。

## 不做什么

- 不加“输入是否是 ±1”这种直接看 OOD 变换的 lattice 特征；主门只用状态的高阶矩；
- 不在 quantized/scale OOD 上拟合均值、协方差、阈值；
- 不调矩的阶数、尾阈值 1.5、覆盖率或正则；
- 不把“高阶矩过线”写成通用 OOD：本轮只有两种合成偏移。

## 产物

- `research/cognitive/run_metacognitive_gate_v7.py`；
- `sweep_results/metacognitive_gate_v7/summary.json` + README。

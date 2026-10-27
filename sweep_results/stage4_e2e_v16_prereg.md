# 预注册：阶段四端到端 v16——输入尺度门补齐 18 格（2026-10-28）

**定稿时间：2026-10-28，在本轮任何数据之前。** v14 的 18 格中 14 格 ready；v15 的
snapshot85/longtau 臂没有补上 s=2 的 4 格（0.704–0.787）。s=2 的剩余问题不是 WM/DPAR，
而是**弱尺度偏移在状态摘要里不够显眼**。本预注册直接用**输入线索的尺度**做门：

```text
input_norm = max(||A||_2, ||B||_2)     # 测试时可见的 A/B 线索
confidence = -zscore(input_norm, calibration)
threshold = calibration 20% 分位
```

这是**输入级尺度门**，不是状态级元认知；OOD 仍是 A/B × s，s=2/3/5。

## 矩阵与结构

- 18 格：K∈{4,8} × s∈{2,3,5} × σ∈{0.1,0.3,0.5}；
- K=4 用 bank W 最终 28 维；K=8 用 longtau 32 维；bank D/DPAR 不变；
- 门固定为 **input_norm**（校准集拟合均值/方差，阈值 20% 分位）；
- 5 个种子；train 2000 / calibration 500 / test 500 / OOD 500。

## 判据（每格七项，5 种子均值）

1. WM R² ≥0.90；
2. WM 选择性 MSE ≤0.10；
3. DPAR 最好臂 ≥0.80；
4. 状态控制 ≤1/K + 0.10；
5. OOD AUROC ≥0.75；
6. OOD 弃权 ≥0.80；
7. test 覆盖率 0.78–0.82。

`v16_ready` = 18 格全过。

## 边界与报告

- **这是尺度 OOD 专用**：input_norm 直接测 A/B 的幅度；不声称通用 OOD，也不改变
  v6/v7 信息丢失型 OOD 的边界；
- 报告完整 18 格；不调 s/σ/K/阈值/覆盖率；
- 必须写清这是**输入级**门，不是状态级元认知；阶段四的“元认知”主张不能因此升级；
- 与 v14 的对照：v14 是状态级（K4 shrunk/K8 diag），v16 只换门。

## 产物

- `research/cognitive/run_e2e_delayed_recall_v16.py`；
- `sweep_results/stage4_e2e_v16/summary.json` + README。

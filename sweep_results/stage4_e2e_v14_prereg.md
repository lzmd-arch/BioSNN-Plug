# 预注册：阶段四端到端 v14——按 K 选门后的全矩阵（2026-10-28）

**定稿时间：2026-10-28，在本轮任何数据之前。** v13 用 diag 统一门，前两格
（K4-s2-n0.1、K4-s2-n0.3）弃权 0.772/0.747，未过；v10 的 K4-s2 用 shrunk 是 0.8276。
本预注册把门按 K 固定：**K=4 用 shrunk（v10 口径），K=8 用 diag（v12 口径）**，其余结构、
矩阵、判据与 v13 完全相同。18 格全过才 `v14_ready=true`。

## 矩阵与结构

- K∈{4,8} × s∈{2,3,5} × σ∈{0.1,0.3,0.5}；
- K=4 用 bank W 最终 28 维；K=8 用 longtau 32 维（τ=400）；bank D/DPAR 不变；
- 门：K=4 shrunk（λ=0.1）；K=8 diag；
- 5 个种子；train 2000 / calibration 500 / test 500 / OOD 500；
- 阈值只在 calibration 上取。

## 判据（每格七项，5 种子均值）

1. WM R² ≥0.90；
2. WM 选择性 MSE ≤0.10；
3. DPAR 最好臂 ≥0.80；
4. 状态控制 ≤1/K + 0.10；
5. OOD AUROC ≥0.75；
6. OOD 弃权 ≥0.80；
7. test 覆盖率 0.78–0.82。

## 报告义务

- 18 格全表；任何失败写明判据；
- 必须报 v13 已跑的两格（K4-s2-n0.1/.3，diag）作为对照；
- 不调门/τ/阈值/覆盖率；K=8 的门只固定 diag。

## 不做什么

- 不做 v13 未跑格的 diag 统一门；
- 不把某一 K 的门选择说成通用；
- 不把尺度 OOD 结论外推到信息丢失型 OOD。

## 产物

- `research/cognitive/run_e2e_delayed_recall_v14.py`；
- `sweep_results/stage4_e2e_v14/summary.json` + README。

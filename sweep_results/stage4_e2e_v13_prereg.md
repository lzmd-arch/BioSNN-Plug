# 预注册：阶段四端到端 v13——全矩阵验收（2026-10-28）

**定稿时间：2026-10-28，在本轮任何数据之前。** 用户选择的口径是 **E2E + 全部组件**。
本预注册只覆盖 E2E 全矩阵：K∈{4,8} × s∈{2,3,5} × σ∈{0.1,0.3,0.5}，共 18 格；
每格七项全过才算该格 ready。组件（P3 终止、脉冲 WM、通用 OOD、局部情景）另行预注册，
不混进本表。

## 结构

- 双 bank：bank W 只收 A/B/Q，bank D 只收 K 对键值 + 查询；
- WM 读出/成员/世界模型/OOD 只读 bank W；DPAR 每幕独立；状态控制用拼接状态；
- **门固定为 diag**（v12 证明 K8 的 shrunk 协方差估计会压 coverage）；
- K=4 用 bank W 最终状态（t=85）；K=8 用 **longtau**（t=125，τ 集合加 400）；
- 5 个种子；train 2000 / calibration 500 / test 500 / OOD 500；
- 阈值只在 calibration 上取，覆盖率目标 80%。

## 矩阵

```text
K=4: s=2/3/5 × σ=0.1/0.3/0.5   = 9 格
K=8: s=2/3/5 × σ=0.1/0.3/0.5   = 9 格
```

OOD：A/B 乘 s；in-dist A/B 加 σ；DPAR 查询噪声固定 0.1。

## 判据（每格七项，5 种子均值）

1. WM R² ≥0.90；
2. WM 选择性 MSE ≤0.10；
3. DPAR 最好臂 ≥0.80；
4. 状态控制 ≤1/K + 0.10（K=4 0.35、K=8 0.225）；
5. OOD AUROC ≥0.75；
6. OOD 弃权 ≥0.80；
7. test 覆盖率 0.78–0.82。

`cell_ready` = 七项全过；`v13_ready` = 18 格全部 ready。

## 报告义务

- 18 格全表：R²/MSE/覆盖/弃权/AUROC/DPAR/状态控制/ready；
- 任何一格失败都写明是哪条，不隐藏；
- 若 K=8 的 s=2 或 σ=0.5 失败，写清边界；不调覆盖率/λ/τ；
- 不把某一格的通过写成整个矩阵通过。

## 不做什么

- 不调门（diag 固定）、τ（longtau 固定）、K/σ/s、阈值、覆盖率；
- 不在 test/OOD 上拟合；
- 不把本矩阵的尺度 OOD 结论外推到信息丢失型 OOD。

## 产物

- `research/cognitive/run_e2e_delayed_recall_v13.py`；
- `sweep_results/stage4_e2e_v13/summary.json` + README。

# 预注册：阶段四端到端 v12——K=8 用对角协方差门（2026-10-28）

**定稿时间：2026-10-28，在本轮任何数据之前。** v11 把 K=8 的 WM 修好了（snapshot85
R² 0.9936、longtau 0.90145），但 shrunk 全状态协方差门的 coverage 只有 0.7552/0.7776，
低于 0.78。v10 的 K=8 参照里，**diag**（逐维方差）的 coverage 是 0.7808、弃权 0.9188、
AUROC 0.9503——它没有 28×28 协方差的估计负担。本预注册检验：把 K=8 的门换成 **diag**，
能否在 WM 已修好的两臂上过 coverage。

## 结构（复用 v11）

- 三臂：baseline（bank W t=125）、snapshot85（t=85）、longtau（t=125 + τ=400）；
- bank D、DPAR、状态控制不变；K=8、s=3、σ=0.3；5 个种子；
- 每臂同时报两个门：**diag（主门）** 与 **shrunk（v10/v11 对照）**；
- 阈值仍只在 calibration 上取，覆盖率目标 80%。

## 主门：diag

```text
置信度 = -Σ_i (h_i - μ_i)^2 / σ_i^2
```

μ、σ 只在 calibration 上取；阈值 = calibration 置信度 20% 分位。没有协方差矩阵，
所以不会因 28/32 维、500 幕的协方差估计误差把 coverage 拉低。

## 判据（每臂七项，5 种子均值）

1. WM R² ≥0.90；
2. WM 选择性 MSE ≤0.10；
3. DPAR 最好臂 ≥0.80；
4. 状态控制 ≤1/8 + 0.10 = 0.225；
5. OOD AUROC ≥0.75；
6. OOD 弃权 ≥0.80；
7. test 覆盖率 0.78–0.82。

`arm_diag_ready` = 七项全过。`v12_ready = snapshot85_diag.ready or longtau_diag.ready`
（baseline 只作参照）。shrunk 的七项也全部报，不参与 `v12_ready`。

## 解释规则

- diag 过、shrunk 不过 → K8 的 coverage 失败是**协方差估计**，不是 WM 或 DPAR；
- 两门都不过 → K8 的 coverage 是更深的分布问题；不调覆盖率/λ；
- 两门都过 → 报两者，不把 diag 说成“必须”；
- baseline diag 若 R² 仍 <0.90，仍不 ready；WM 修复仍归 snapshot85/longtau。

## 报告义务

- 三臂 × 两门的 R²/MSE/coverage/弃权/AUROC/DPAR/状态控制全部单列；
- 必须把 v11 的 shrunk 数字（0.7552/0.7776）放在表里，不能只报 diag；
- 不把 K8 的通过写成 K>8 或通用 OOD；
- 不改变 v10 的 s≥2/K≤4 结论。

## 不做什么

- 不调覆盖率、阈值、λ、τ、窗口；
- 不在 test/OOD 上拟合均值/方差；
- 不把 diag 的 coverage 0.78 说成“任意 K 都稳”。

## 产物

- `research/cognitive/run_e2e_delayed_recall_v12.py`；
- `sweep_results/stage4_e2e_v12/summary.json` + README。

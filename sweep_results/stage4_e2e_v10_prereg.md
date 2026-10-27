# 预注册：阶段四端到端 v10——收缩全状态协方差门，冲 s=2（2026-10-28）

**定稿时间：2026-10-28，在本轮任何数据之前。** v9 显示 v8 的 bank W 尺度门在 s=2 只有
弃权 0.720，s=3 才 0.937。v8 的探索臂里，bank W 全状态 Mahalanobis 的 AUROC 是 0.991，
但 coverage 只有 0.761。本预注册检验：**用收缩的全状态协方差把 coverage 校准回 0.80，
能否让 s=2 过线**；K=8 仍单独报容量，不混进主判据。

## 结构（固定）

- v8/v9 的双 bank：bank W 只收 A/B/Q，bank D 只收键值/查询；56 维拼接；
- WM 读出/成员/世界模型只在 bank W；DPAR 每幕独立；状态控制用 56 维；
- 阈值只在 calibration（`seed+30_000`）上取；5 个种子。

## 门

主门 **shrunk-mahalanobis**：

1. 用 calibration 的 bank W 状态逐维标准化（`z = (h - μ) / σ`）；
2. 计算 28 维协方差 `C`，做固定收缩
   `C' = (1 - λ) C + λ * (trace(C) / 28) * I`，**λ = 0.1 预先写死**；
3. 置信度 = 负 Mahalanobis 距离；
4. 阈值 = calibration 置信度的 20% 分位（目标 80% 覆盖）。

对照（只报告，不参与主判据）：

- `gauss3`：v8 的 `[disagreement, surprise, state_norm]` 三维高斯；
- `diag`：逐维方差的对角 Mahalanobis；
- `norm`：只用 `state_norm` 的一维高斯。

## 条件

| 条件 | K | σ | OOD 尺度 s | 角色 |
| :--- | :--- | :--- | :--- | :--- |
| **s2（主档）** | 4 | 0.3 | **2.0** | 主判据；v9 的失败点 |
| s3 | 4 | 0.3 | 3.0 | 已知应过；sanity |
| s5 | 4 | 0.3 | 5.0 | 已知应过；sanity |
| K8 | 8 | 0.3 | 3.0 | 容量单独报告，不参与主判据 |

## 判据（每条件 5 种子均值）

1. WM R² ≥0.90；
2. WM 选择性 MSE ≤0.10；
3. DPAR 最好臂 ≥0.80；
4. 状态控制 ≤1/K + 0.10（K=4 0.35，K=8 0.225）；
5. OOD AUROC ≥0.75；
6. OOD 弃权 ≥0.80；
7. test 覆盖率 **0.78–0.82**（与 80% 阈值口径一致；否则“全弃权”也会过）。

**主档 s2** 七项全过 → `v10_s2_ready=true`。

> **修订 1（2026-10-28，正式数据前）**：20 幕 smoke 里 shrunk 门的 test coverage 只有
> 0.10、弃权 1.00——只看弃权会把“几乎全弃权”误判成好门。正式跑之前把
> **test coverage 0.78–0.82** 加为第 7 条判据；这不是按性能调参，是防平凡解。
> 原六条不变。

**尺度 sanity**：s3、s5 也用主门七项全过 → `v10_scale_ready=true`。
`v10_ready = v10_s2_ready and v10_scale_ready`。
K8 只报七项与 ready，不计入 `v10_ready`；它的 WM 容量失败不追认。

## 报告义务

- 四个条件全部报；主门与三个对照的 coverage/abstain/AUROC 全部列出；
- 若 s2 过但 coverage 偏离 0.78–0.82，写清“阈值口径偏了”，不得只报弃权；
- 若 s2 仍不过，写“收缩协方差不够”，不调 λ；
- 不把 K8 的 DPAR/OOD 高分掩盖 WM R² 0.89975 的容量边界。

## 不做什么

- 不调 λ、标准化方式、覆盖率、特征；
- 不在 test/OOD 上拟合均值/协方差/阈值；
- 不把 s=2 的成功外推到 s=1.5 或信息丢失型 OOD；
- 不改变 v8/v9 的结果。

## 产物

- `research/cognitive/run_e2e_delayed_recall_v10.py`；
- `sweep_results/stage4_e2e_v10/summary.json` + README。

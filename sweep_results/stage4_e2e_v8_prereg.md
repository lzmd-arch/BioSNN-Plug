# 预注册：阶段四端到端 v8——bank W 的尺度特征（2026-10-27）

**定稿时间：2026-10-27，在本轮任何数据之前。** v7 双 bank 的 WM/DPAR/状态控制都过，
唯一未过是 OOD 弃权 **0.7860**，差 1.4 pp。v7 的置信度只用 `disagreement + surprise`；
v4 的经验是 `state_norm` 对 A/B×3 尺度偏移有效（gauss3 0.8456 vs zsum 0.7208）。本预注册
在 **bank W** 上加入状态范数，检验这 1.4 pp 是否由此闭合。

## 任务与结构（复用 v7）

- 双 bank：bank W 只收 A/B/Q，bank D 只收键值/查询；56 维拼接；
- WM 读出/成员/世界模型只读 bank W；DPAR 每幕独立；状态控制用 56 维；
- OOD 为 A/B ×3；t=85；阈值只在 calibration（`seed+30_000`）上取；5 个种子。

## 门

| 门 | 特征 | 角色 |
| :--- | :--- | :--- |
| **gauss3（主门）** | `[disagreement, surprise, state_norm]` 三维高斯密度 | v4 的尺度门，本轮主判据 |
| zsum（v7 基线） | `-z(disagreement) - z(surprise)` | 对照 |
| gauss2 | `[disagreement, surprise]` 二维高斯 | 对照 |
| norm-only | `state_norm` 一维高斯 | 对照，检查“只靠范数” |
| mahalanobis（探索） | bank W 28 维状态的完整高斯 | 只报告，不参与主判据 |

`state_norm = ||bank W state||_2`；所有统计量只在 calibration 上拟合。

## 判据（主门 gauss3，5 种子均值）

1. WM R² **≥0.90**；
2. WM 选择性 MSE **≤0.10**；
3. DPAR 最好臂 **≥0.80**；
4. 状态控制 **≤0.35**；
5. OOD AUROC **≥0.75**；
6. OOD 弃权 **≥0.80**；
7. 六条全过 → `e2e_v8_ready=true`；否则报边界，不换门、不调阈值/特征。

**报告义务**：五个门全部报；若 gauss3 失败而 mahalanobis 过，只写探索性；若 norm-only
单独过，说明这 1.4 pp 只靠一个尺度特征，不能写成通用 OOD。

## 不做什么

- 不在 test/OOD 上拟合均值/协方差/阈值；
- 不调特征集合、覆盖率、阈值分位；
- 不把 A/B×3 尺度偏移说成通用 OOD；v6/v7 已表明信息丢失型 OOD 检测不到；
- 不改变 v7 的 5/6 结果。

## 产物

- `research/cognitive/run_e2e_delayed_recall_v8.py`；
- `sweep_results/stage4_e2e_v8/summary.json` + README。

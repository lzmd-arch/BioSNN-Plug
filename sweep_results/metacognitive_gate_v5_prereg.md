# 预注册：元认知 v5——尺度无关 OOD（正交旋转）检验（2026-10-27）

**定稿时间：2026-10-27，在本轮任何数据之前。** v4 的密度门把 e2e v3 的弃权从 0.7056
提到 0.8456，但增益来自 `state_norm`，而 v3 的 OOD 正是 A/B ×3 的**尺度偏移**。本预注册
检验：当 OOD **不改变范数**时，v4 的主门还能不能工作。

## 任务与数据

与 e2e v3 主档相同：K=8、σ=0.3、t=100；train 2000 / calibration 500 / test 500，
种子 0–4。

| OOD 分片 | 变换 | 作用 |
| :--- | :--- | :--- |
| **primary：旋转** | 每个种子抽一个随机正交 `Q`（4×4，QR 得到），把 A/B 线索都乘 `Q` | 改变方向、**保持每个线索的 L2 范数**；`state_norm` 原理上不再直接指示 OOD |
| reference：尺度 | A/B ×3（v3 原 OOD） | 复现 v4 的 0.8456，作 sanity |

查询/Q 与事件/E 不变；旋转只作用于 A/B。500 幕 OOD，种子 `seed+20_000`。

## 门

与 v4 完全相同的四个门，阈值仍只在 in-dist calibration（`seed+30_000`）上取：

- **主门 gauss3**：`[disagreement, surprise, state_norm]` 的三维高斯密度；
- 对照 `gauss2` / `zsum` / `knn3`（kNN 用 leave-one-out）。

## 判据（先写死，5 种子均值）

1. test 覆盖率 **0.78–0.82**，选择性 MSE **≤0.10**（与 v4 相同）；
2. **旋转 OOD** 上 gauss3 弃权 **≥0.80**，AUROC **≥0.75**；
3. **尺度 OOD** 上 gauss3 弃权 **≥0.80**（sanity，应复现 v4）；
4. 三条全过 → `scale_independent_ready=true`；否则报边界，不换门、不调阈值/特征。

**解释规则**：

- 若旋转 OOD 上 gauss3 掉到 0.80 以下，而 `gauss2`/`zsum` 也掉，说明 v4 的增益主要来自
  尺度特征；写成“尺度型 OOD 专用”，不写成通用弃权。
- 若 gauss3 掉但某个对照门过，仍只写**探索性**，不能追认主门；主门是 gauss3。
- 若旋转 OOD 上 gauss3 仍过，说明 disagreement/surprise 也携带了旋转信号，v4 的结论
  可以扩到“至少这一类保范 OOD”，但仍不外推到真实分布偏移。

## 不做什么

- 不在 rotation OOD 上拟合任何均值/协方差/阈值；
- 不调旋转角度、Q 的分布、特征集合、覆盖率或分位；
- 不把“旋转后任务变难”当成 OOD 检测成功：判据看弃权/AUROC，不看任务误差；
- 不把合成旋转说成自然 OOD。

## 产物

- `research/cognitive/run_metacognitive_gate_v5.py`；
- `sweep_results/metacognitive_gate_v5/summary.json` + README。

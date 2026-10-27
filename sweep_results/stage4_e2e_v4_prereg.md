# 预注册：阶段四端到端 v4——DPAR + 工作记忆共存（2026-10-27）

**定稿时间：2026-10-27，在本轮任何数据之前。** v2 的 5/5 来自分离良好的 4-bit E；
v3 加压力后 4/5；v4 的端到端任务还没有把**真正需要绑定的 DPAR** 放进去。本预注册把
DPAR 的 K 对键值写进同一条时间线，与 A/B/Q 工作记忆共存，检验组件组合而不是单任务。

## 任务

每幕：

```text
A:  [0, 4]        B: [15, 19]       Q: [30, 34]      （A/B 加 σ=0.3 噪声）
DPAR: K=4 对 (key_i, value_i)，key_i ∈ [40+10i, 44+10i]，
      value_i ∈ [45+10i, 49+10i]，i=0..3
查询键: [80, 84]；test_step=85
```

键/值均为 4 维 `N(0, I)`；查询下标 q 均匀；DPAR 查询键加 σ=0.1 噪声（记忆侧），
状态时间线用干净查询键。目标：

- 工作记忆：重建 A/B/Q（12 维）；
- DPAR：从 K 个值里 top-1 选 `value_q`（chance 0.25）；
- 状态控制：从最终 28 维状态线性读出 `value_q`，应接近 chance（证明必须绑定）。

OOD：A/B ×3；DPAR 对不变；500 幕。5 个种子；train 2000 / calibration 500 / test 500。

## 组件

- WM：局部 RLS 读出（δ=1000），5 成员 bootstrap；
- 世界模型：ridge `previous_state → state`；置信度 = 成员分歧 + 一步惊喜；
- DPAR 记忆：`SparseEncoder(512, k=32)`，每幕独立建库；Pseudo-inverse / Raw NN 为主，
  Hebbian 单列报告；
- 阈值只在 calibration（`seed+30_000`）上取，覆盖率 80%。

## 判据（先写死，5 种子均值）

1. WM 的 A/B/Q 平均 R² **≥0.50**；
2. DPAR 最好臂（Pseudo / Raw NN）top-1 **≥0.80**；
3. 状态控制 top-1 **≤0.35**（chance 0.25；否则任务不是“必须绑定”，本轮作废）；
4. OOD AUROC **≥0.75**；
5. OOD 弃权 **≥0.80**；
6. 80% 覆盖选择性 MSE **≤0.10**；
7. 六条全过 → `e2e_v4_ready=true`；否则报边界，不调 K/σ/阈值/特征。

## 报告义务

- DPAR 三臂（Pseudo / Raw NN / Hebbian）分开报，不能让 Pseudo 掩盖局部 Hebbian 的失败；
- 状态控制是**有效性检查**，不是性能臂；它过不了 0.35 就写“任务无效”，不解释 e2e；
- WM 与 DPAR 是两种目标，**不能把 DPAR top-1 混进 WM R²**；
- 若 OOD 弃权不过，明确写 `e2e_v4_ready=false`，不追认 v3。

## 不做什么

- 不调 K、时间窗、σ、读出、阈值或覆盖率；
- 不把 DPAR 的每幕独立建库说成终身情景记忆；
- 不把 A/B×3 的尺度 OOD 说成通用 OOD；
- 不把本任务写成阶段四整体验收。

## 产物

- `research/cognitive/run_e2e_delayed_recall_v4.py`；
- `sweep_results/stage4_e2e_v4/summary.json` + README。

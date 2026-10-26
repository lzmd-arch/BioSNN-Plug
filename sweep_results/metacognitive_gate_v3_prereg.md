# 预注册：元认知门控 v3——ensemble 不确定性组合（2026-10-27）

**定稿时间：2026-10-27，在 v3 的任何数据之前。** v1 的单读出 AUROC 0.745、ECE 0.132；
v2 的 ensemble 均值 logits 反而 0.738/0.177。v3 不再只换温度，而是把**ensemble 的分歧**
作为第二路信号，并只在校准集上拟合一个 P(正确) 的标定头。

## 机制

```text
K=5 个 bootstrap(80%) delta 读出（与 v2 相同）
每样本三特征：
  M = mean_i max_j p_ij          （平均最大概率）
  S = std_i  max_j p_ij          （成员间分歧）
  G = top1 − top2 of mean logits （平均 logit margin）
标定头：类平衡 L2 逻辑回归（IRLS）在**校准半集**上拟合 P(正确)；评估半集不参与
置信度 = P(正确)；阈值只在校准集取 80% 覆盖
```

- 标定头是测量头（IRLS，不反向传播）；特征/读出/数据/切分与 v2 相同；
- 对照：只用 M 的 Platt 标定（同一个逻辑回归，只喂 M）。

## 判据（与 v0–v2 相同，先写死）

1. 选择性准确率（80% 覆盖）比 base **≥ +5 pp**；
2. 正确性 AUROC **≥0.75**；
3. 标定后 ECE（10 箱）**≤0.10**；
4. 三条全过 → `v3_ready`；否则报边界，不调 K/特征/L2/阈值。

## 不做什么

- 不换读出/特征/切分/数据；不调 K/bootstrap/L2；
- 不做 stacking 以外的新模型、不做 OOD/LLM 触发；
- 不因为组合头好看就只报它；M-only 对照必须同表。

## 产物

- `research/cognitive/metacognitive_gate.py` 增加通用 `LogisticCalibrator`；
- `research/cognitive/run_metacognitive_gate_v3.py`；
- `sweep_results/metacognitive_gate_v3/summary.json` + README。

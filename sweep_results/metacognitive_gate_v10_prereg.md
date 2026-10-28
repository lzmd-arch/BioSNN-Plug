# 预注册：元认知 v10——逐维最大重建误差门（2026-10-29）

**性质：v9 失败项的定向修复，不换任务。** v9 的 `dim_dropout`（弃权 0.596）与
`correlated`（0.539）都有很高全量误差（4.7×/15.2×），但**全局均值 `errors` 与 in-dist
高误差样本重叠**。v10 只把门的误差维换成**逐维重建误差的最大值** `max_errors`；门仍为
kNN(5)，`state_norm` 保留，阈值只在 calibration 上取。

## 机制

```text
readout：state → 预测 [A,B,Q,E]
residual_t = (预测 − 观测目标)²，16 维
errors     = mean(residual_t)        ← v8/v9 用的
max_errors = max(residual_t)         ← v10 新增
门：kNN(5) on [max_errors, state_norm]
```

- `max_errors` 对“某一维被丢掉/两 cue 相关”的方向性破坏更敏感；对单点噪声也更敏感，
  覆盖率是否仍稳是判据之一。
- 不训练新读出；用同一个 RLS；阈值只在 calibration 上取。

## OOD 家族（与 v9 相同）

`time_shift`、`time_reverse`、`dim_dropout`、`correlated`；目标保留原始 A/B/Q/E；
每个家族先过有效性门（OOD 全量 MSE ≥2× in-dist）。

## 判据（与 v8/v9 相同，先写死）

- 每个有效家族：覆盖 0.78–0.82、选择性 MSE ≤0.10、弃权 ≥0.80、AUROC ≥0.75；
- 尺度 sanity 弃权 ≥0.80；
- 至少 3 个家族有效，且所有有效家族五条全过、尺度 sanity 过 → `v10_ready=true`；
- 若 v9 已过的 `time_shift/time_reverse` 在 v10 掉线，即使 `dim_dropout/correlated`
  修好，也按“顾此失彼”记录，不写成通用 OOD。

## 数据独立性

- v10 最终用**新种子 15–19**；v8/v9 的种子 5–9/10–14 不混报。
- 不调 k/特征/阈值/覆盖率/判据；若 `max_errors` 只修好一个家族，照实记录，不继续拼接。

## 边界

- 仍是输入变换型 OOD；不覆盖标签漂移/零空间变换。
- `max_errors` 是同一个任务读出的逐维误差，不是独立自编码器；读出失效时门语义会变。
- 单维噪声可能提高误报；覆盖率判据必须同报。

## 产物

- `research/cognitive/run_metacognitive_gate_v10.py`；
- `sweep_results/metacognitive_gate_v10/summary.json` + README。

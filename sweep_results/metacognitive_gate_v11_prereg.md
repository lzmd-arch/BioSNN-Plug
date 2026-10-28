# 预注册：元认知 v11——max/mean/norm 组合门（2026-10-29）

**性质：v10 的最后一档定向修复。** v9 的均值门漏掉 `dim_dropout`/`correlated`；v10 的
`max_errors` 修好 `correlated`（0.983），但 `dim_dropout` 仍 0.762 < 0.80。v11 只把
`errors` 加回门，形成 `[max_errors, errors, state_norm]` 三维 kNN(5)；不再新增特征。

## 机制

```text
readout: state → 预测 [A,B,Q,E]
errors     = mean(residual²)      ← 整体能量
max_errors = max(residual²)       ← 局部丢维/方向破坏
state_norm = ‖state‖              ← 尺度
门：kNN(5) on [max_errors, errors, state_norm]，阈值只在 calibration 上取
```

- 不训练新读出；不调 k；不换任务。
- 四家族与 v9/v10 相同：`time_shift`、`time_reverse`、`dim_dropout`、`correlated`。

## 判据（与 v9/v10 相同，先写死）

- 每个有效家族：覆盖 0.78–0.82、选择性 MSE ≤0.10、弃权 ≥0.80、AUROC ≥0.75；
- 尺度 sanity 弃权 ≥0.80；
- 四个家族都有效且五条全过、尺度过 → `v11_ready=true`；
- 任何一族掉线（包括 v9/v10 已过的）→ 记“顾此失彼”，不写成通用 OOD。

## 数据独立性

- 新种子 **20–24**；v8/v9/v10 的种子不混报。
- 不调特征/阈值/判据；若 v11 仍不过，停止拼特征，转为逐维/协方差级专门门的新预注册。

## 边界

- 仍只覆盖输入变换型 OOD；不覆盖标签漂移/零空间变换。
- 三维门仍可能对单维噪声敏感；覆盖率同报。
- 这是组合门，不是“通用 OOD”的证明；四家族之外不外推。

## 产物

- `research/cognitive/run_metacognitive_gate_v11.py`；
- `sweep_results/metacognitive_gate_v11/summary.json` + README。

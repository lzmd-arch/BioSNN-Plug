# 元认知 v10：逐维最大重建误差门（2026-10-29）

**预注册：** [`../metacognitive_gate_v10_prereg.md`](../metacognitive_gate_v10_prereg.md)。
门 = kNN(5) on `[max_errors, state_norm]`；新种子 15–19；干净工作区 commit `56c5f14`。

## 结果：correlated 修好，dim_dropout 仍差一点

| 家族 | 有效性 | 弃权 | AUROC | 判定 |
| :--- | ---: | ---: | ---: | :--- |
| `time_shift` | 11× | 0.998 | 0.994 | 过 |
| `time_reverse` | 28× | 0.980 | 0.986 | 过 |
| `dim_dropout` | 4.9× | **0.762** | 0.866 | **未过** |
| `correlated` | 15.6× | **0.983** | 0.986 | 过 |
| 尺度 sanity | — | 0.850 | — | 过 |

`v10_ready=false`：只有 `dim_dropout` 的弃权 0.762 < 0.80。

## 读法

- `max_errors` 确实抓到了 `correlated`（v9 的 0.539 → 0.983），说明“局部方向破坏”需要
  逐维最大误差；
- 但只看最大维丢掉了整体能量，`dim_dropout` 仍不够；
- v11 把 `errors` 加回形成三维门，`dim_dropout` 升到 0.858；v10 的负结果保留。

## 边界

- 只覆盖输入变换型 OOD；不覆盖标签漂移/零空间变换。
- 覆盖率 0.797；单维噪声敏感性由 v11 的覆盖率继续同报。
- 不调 k/特征/阈值；v10 的失败不删。

## 自证

`summary.json` 复现块：commit `56c5f14`，工作区干净；耗时 87.7 s。

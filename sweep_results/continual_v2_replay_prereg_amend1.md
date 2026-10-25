# 预注册 v2 修订 1：Step 0 的 batching 对照（2026-10-25）

**修订时间：2026-10-25，在任何 v2 数据之前。** 原预注册
[`continual_v2_replay_prereg.md`](continual_v2_replay_prereg.md) 的判据与机制不变；
本文件只补两件实现层必须先说清的事。

## 1. Route R（NCM）的矩阵口径也错了，已修

`run_with_frozen_frontend`（Route R + 类原型）和 W1 的 delta 分支一样，填的是
**累计类集**准确率。它同样不是 CIL 口径——探索期的 **0.658 / 0.127** 因此也不能当基线。
本次已改成：样本按任务切、在**全部已见原型**里取最近，再按任务 i 的准确率填 `a[t][i]`。
Step 0 用修正后的函数重测，结果以本次为准。

## 2. Step 0 的主对照是"同 batching 的 ρ=0"，不是 v1 全批量

回放按预注册 v2 用 **minibatch**（B=128）混合；而 v1 的 R-delta 基线是**全批量**。
直接比会把"batching 差异"算进回放的账——这是配方不平等的另一种形态。

所以 Step 0 跑三条：

| 臂 | 读出 | batching | ρ | 作用 |
| :--- | :--- | :--- | ---: | :--- |
| R-NCM | 类原型 + 白化 | — | — | Route R 参考上界（不参与回放对照） |
| R-delta-full | delta | 全批量（v1） | 0 | 与历史协议对齐的参考 |
| **R-delta-mini** | delta | minibatch B=128 | 0 | **回放臂的主对照** |

回放臂（ρ=0.1 / 0.3）与 **R-delta-mini** 比；R-delta-full 只作参考，不用它判回放的成败。

## 3. 回放的 batching 细节（写死）

- 每个 minibatch 固定 **B = 128**；回放条数 `n_replay = round(ρ·B)`，当前任务条数
  `B − n_replay`；
- 回放缓冲：每类 20 条**特征**，任务结束时从当前任务训练特征里按固定种子抽；
- 跑门：每个任务 `‖ΔW‖ / max(‖W_before‖, 1e-12) ≥ 0.05`；低于则**该种子作废**，
  作废数写进 summary 的 `void_seeds`，不进入均值；
- 判据、种子（0–4）、评估口径（CIL）与 v2 相同。

## 产物

- `sweep_results/continual_v2/step0_ncm/`
- `sweep_results/continual_v2/step0_delta_full/`
- `sweep_results/continual_v2/step0_delta_minibatch/`
- 回放：`replay_rho010/`、`replay_rho030/`

每个目录自带 README；Step 0 完成后才根据其实测推导门槛数字（不套旧 0.658/0.127）。

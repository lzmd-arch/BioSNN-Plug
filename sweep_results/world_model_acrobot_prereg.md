# 预注册：世界模型打底的非线性环境（Acrobot，2026-10-25）

**定稿时间：2026-10-25，在任何 Acrobot 数据之前。** 这是打底的第 P2 级台阶：
把"一步动力学可测"从线性环境（CartPole）挪到一个**线性解不掉**的环境。入口条件不变。

## 为什么是 Acrobot

- 离散动作（3 个），与现有基线（one-hot 动作）兼容；
- 6 维状态、强非线性（摆的耦合动力学）、终止是一个稀疏事件；
- 便宜、经典、gymnasium 自带，不需要新依赖。

Pendulum 的连续动作会逼我们换一套动作编码，属于另一件事；本轮不选。

## 协议

| 项 | 取值 |
| :--- | :--- |
| 环境 | `Acrobot-v1`，均匀随机动作 |
| 数据 | 训练 200 回合、测试 100 回合，按回合切分；种子 0–4 |
| 模型 | `copy_last` / `mean_delta` / `per_action_mean` / `ridge_action` / `ridge_quadratic`（主） |
| 指标 | 状态 NMSE（分母 = 训练集 `Var(s')`）、动作敏感度（3 个动作两两配对）、终止 AUROC |
| 判据 | 与打底原预注册相同：NMSE ≤ 0.90×`copy_last`；动作敏感度 ≥ 0.01；终止 AUROC ≥ 0.75 |

## 终止事件：不可评则如实标"不可评"

随机策略在 Acrobot 上**极少到达目标**（smoke 里 5 回合、终止率 0，AUROC 为 `nan`）。
所以在看到正式数据之前先定死：

- 若**测试集终止率 < 1%**，终止 AUROC 记为 **`not_evaluable`**，不参与"任务可测"的判定；
  结果 README 必须写明"这个行为策略下终止事件太稀疏"；
- 若终止率 ≥ 1%，AUROC 判据照常；
- **不许**为了凑出终止事件去换行为策略——那是改协议，要另写修订。

## 判据的预期

- `ridge_action`（线性）**很可能过不了** NMSE 判据：Acrobot 的动力学不是 `[s,a,a·s]` 的线性函数；
- 若线性与二次项都过不了，结论是"**这个任务需要非线性模型**"——这正是 P2 的用途，
  不是失败；它给下一级（P3 学习规则 / 神经世界模型）提供了非平凡的目标；
- 若二次项意外过了，先查是不是只学到了短时近似（看逐维 NMSE 与 horizon 曲线），再下结论。

## 产物

- 输出目录 `sweep_results/world_model_acrobot/`，不覆盖任何已有结果；
- 命令：`uv run python -m research.cognitive.run_world_model --seeds 0-4 --env Acrobot-v1 --primary-model ridge_quadratic --out sweep_results/world_model_acrobot`

# 预注册：世界模型多步 rollout（CartPole，2026-10-25）

**定稿时间：2026-10-25，在任何 rollout 数据之前。** 这是打底的第 P1 级台阶，不是阶段四启动；
入口条件（阶段三验收）不变。

## 要回答什么

一步误差已经很低（`ridge_quadratic` 的 NMSE ≈ 4.9×10⁻⁸ × `copy_last`）。**一步低不等于
k 步不发散**：自由 rollout 每一步都吃自己的预测，误差会累积。本预注册只问两件事：

1. 自由 rollout 的误差随 k 怎样增长？
2. 它比教师强制（每一步从真实状态出发）差多少——即**误差累积本身**有多大？

## 协议

| 项 | 取值 |
| :--- | :--- |
| 环境 / 数据 | CartPole-v1，均匀随机动作；训练 200 回合、测试 100 回合，按回合切分 |
| 种子 | 0–4 |
| 模型 | `ridge_quadratic`（主）与 `copy_last`（下界）；沿用 amendment 1 的拟合配置（λ=1.0） |
| 窗口 | 测试回合内长度 k 的连续窗口；**不跨回合边界** |
| horizon | k = 1 … 10 |
| 指标 | 逐 horizon 的 NMSE = `mean_d((pred − true)² / Var_train(s'_d))`；自由与教师强制各一条 |
| 归一化 | 分母只用**训练集**的 `Var(s')` |

## 判据（先写死）

- **不设"通过/不通过"的门**——这是一条描述性曲线，用途是判断"能不能开始谈规划"；
- 两条解释性阈值，写在看到数字之前：
  1. **可用**：`free_NMSE(k=10) ≤ 1.0`（误差仍在状态方差之内）；
  2. **累积主导**：`free_NMSE(k=5) / teacher_NMSE(k=5) ≥ 2`——此时下一步该修的是
     rollout 稳定性（噪声注入 / 多步训练），而不是换更大的模型。
- 若 `free_NMSE(k=1)` 与 `teacher_NMSE(k=1)` 不相等，先查实现（两者在 k=1 时应逐位相同）。

## 已知的平凡性

CartPole 的一步动力学是线性的（欧拉积分），所以 `ridge_quadratic` 很可能在 10 步内都不发散。
**这是任务的性质，不是世界模型的成就**——结果 README 必须把这句写在数字旁边。真正的多步
考验留给 P2 的非线性环境。

## 产物

- runner 已有 `--rollout-horizon`（`research/cognitive/run_world_model.py`）；
- 输出目录 `sweep_results/world_model_rollout/`，不覆盖 `world_model_groundwork/`；
- 命令：`uv run python -m research.cognitive.run_world_model --seeds 0-4 --rollout-horizon 10 --primary-model ridge_quadratic --out sweep_results/world_model_rollout`

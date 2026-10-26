# W3 脉冲化诊断（2026-10-26）

**诊断，不是验收。** 预注册阈值见
[`../../research/rstdp_spiking/diagnosis-plan.md`](../../research/rstdp_spiking/diagnosis-plan.md)：
③ 评测噪声的 95% CI 半宽 ≤ 中位数的 20% 才算"评测可用"。

## ③ 评测噪声：**通过（非平凡）**

| 种子 | 中位数 | 范围 | 95% CI | 半宽 / 中位数 | 判定 |
| ---: | ---: | :--- | :--- | ---: | :--- |
| 0 | 73.8 | 60.9–90.5 | [69.67, 77.40] | **5.2%** | usable |
| 1 | 76.5 | 51.1–99.9 | [72.73, 82.79] | **6.6%** | usable |

两份策略都确实学到了（>70 步，随机约 9–10），所以这不是退化策略上的平凡通过。
**评测口径不是瓶颈**——按诊断计划，下一步是 ②（熵崩塌）与 ①（Critic 信号）。

结果目录：[`noise_lr1e-4/`](noise_lr1e-4/README.md)、
[`noise_seed1_lr1e-4/`](noise_seed1_lr1e-4/README.md)。

## ② 策略熵崩塌：**未崩塌**

| 种子 | 第 100 回合前最后 10 回合平均熵 | 判定 |
| ---: | ---: | :--- |
| 0 | **0.662 nats** | no_collapse |
| 1 | **0.590 nats** | no_collapse |

两动作的熵上界是 `log 2 ≈ 0.693`；两个种子都接近上界，说明第 100 回合时策略仍在
均匀探索。**② 不是三臂都学不动的解释。** 结果与曲线见 [`entropy/`](entropy/README.md)。

## ① Critic 信号：**分档结论**

| Critic | seed 0 下降 | seed 1 下降 | 判定 |
| :--- | ---: | ---: | :--- |
| 线性 TD | 13.5% | 17.8% | no_signal |
| 群体（gradient / 均匀 / 归一化） | **36.8%** | **31.1%** | signal（seed 1 仅高阈值 1.1 pp） |

① 只在**线性 TD** 上成立；**群体 Critic 有信号**，不解释三臂共同的失败。
结果：[`critic_td/`](critic_td/README.md)、[`critic_pop/`](critic_pop/README.md)。

## 三条判别实验的合并结论

- **③ 评测噪声**：不是瓶颈（CI 半宽 5.2% / 6.6%）；
- **② 熵崩塌**：不成立（第 100 回合 0.662 / 0.590 nats）；
- **① Critic 信号**：线性 TD 没有，群体有（勉强）。

所以三臂的共同失败**不能归到这三处**。按诊断计划的排序，剩下最可疑的是
**Actor 的信用分配**（痕迹更新 → 成功信号 → 权重的路径），以及**固定策略本身太弱**
（① 里训练最好只有 25–27 步）。下一份诊断要先写死这两条的判据，再动 GPU。

## 方法记录：两次作废的前置尝试（原样保留）

1. **学习率写错**：第一版把 `learning_rate` 写成 `1e-2`（发布配置是 `1e-4`），输出层几乎
   沉默、greedy 恒选动作 0；两个种子的 20 个评测值**逐位相同**，看起来像"评测很稳"。
   这是"达标数字可能是平凡解"的活例。目录 [`noise/`](noise/README.md) 与
   [`noise_seed1/`](noise_seed1/README.md) 保留并加了更正横幅。
2. **评了最终权重而不是最好权重**：`train` 只记 `best_steps`、不存权重，于是诊断评的是
   训练结束时的游走权重。已加"选中最好一步时快照 actor 权重"（commit `fbf9b22`）。

修正后的两条结果用发布配置 + 最好权重快照重跑，自证块均为 commit `fbf9b22`、工作区干净。

## 下一步：Actor 信用分配 A1 已执行，**失败**

预注册见 [`../rstdp_spiking_actor_diagnosis_prereg.md`](../rstdp_spiking_actor_diagnosis_prereg.md)，
结果见 [`../rstdp_spiking_actor_diagnosis/README.md`](../rstdp_spiking_actor_diagnosis/README.md)：
**7/20** 个窗口的被选列增量更大（4 个平局；其中 3 个窗口被选列为负）→
`credit_assignment_inverted`。按预注册 **A2（oracle EMA）不跑**。

**第二因子与动作选择不同向**：`pair_based_stdp = LTP − LTD`，发放多的动作也拿到更多
LTD，净增量可以更小甚至为负；而 `apply_success_signal` 用全局标量 S 乘整条痕迹。
修复（动作门控 + 拆开 LTP/LTD + 按 S 的符号选相位）已经实现并测过：A1′ 修订口径 18/18，
阶段 2 配对中位数 +17.95 步（门槛 +20，不过）、8/10 变好、验收 0/10。见
[`../rstdp_spiking_actor_fix/README.md`](../rstdp_spiking_actor_fix/README.md)。

另一条线——**固定策略太弱**（① 里训练最好只有 25–27 步）——尚未处理。

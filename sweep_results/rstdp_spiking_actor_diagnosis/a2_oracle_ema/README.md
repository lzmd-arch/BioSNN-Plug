# A2：oracle EMA 信号能不能让 Actor 学起来（2026-10-26）

**诊断，不是验收。** 预注册见
[`../../rstdp_spiking_actor_diagnosis_prereg.md`](../../rstdp_spiking_actor_diagnosis_prereg.md)
的 A2 节，执行条件见
[`../../rstdp_spiking_actor_diagnosis_prereg_amend1.md`](../../rstdp_spiking_actor_diagnosis_prereg_amend1.md)
（A1′ 通过后补跑；两臂都用 sign-consistent 痕迹）。

## 协议

| 项 | 取值 |
| :--- | :--- |
| 信号 | `ema` = 存活步数 − EMA 基线；`td` = 线性 TD Critic |
| 痕迹规则 | sign-consistent（两臂相同） |
| 回合数 / Actor lr | 300 / 1e-4 |
| 特征 / 步长 | 64 / 25 |
| 评测 | 每 25 回合、10 个贪心回合；`best_steps` 是训练中的诊断值 |
| 种子 | 0、1 |
| 设备 | CPU |

## 结果：两条判据都没过

| 种子 | EMA `best_steps` | TD `best_steps` | EMA − TD | EMA ≥ 30 |
| ---: | ---: | ---: | ---: | :--- |
| 0 | 38.6 | 29.7 | **+8.9** | 是 |
| 1 | **10.0** | 39.5 | **−29.5** | 否 |

预注册判据：

1. 两个种子上都满足 `ema ≥ td + 20` → **不过**（+8.9、−29.5）；
2. 且 `ema ≥ 30` → **不过**（seed 1 只有 10.0）。

所以 A2 的结论是：**把 Critic 换成一个更简单的因果信号，Actor 仍没有学起来**。

## 这说明什么 / 不说明什么

- 说明：问题不只在"Critic 不够准"。seed 1 上 EMA 甚至**低于** TD（10.0 vs 39.5），
  说明"换一个更干净的信号"本身不保证 Actor 路径能学；
- 不说明 EMA 信号一定更差：300 回合、lr 1e-4、两个种子，这是原文写死的诊断规模；
- 不说明 Actor 不可学：A2 的 EMA 仍然是**回合级标量**，不解决"一个 S 摊到整回合所有
  决策"的信用时域问题——这正是判读表里说的下一嫌疑；
- 不覆盖默认群体 Critic 的 10 种子验收（那套是 800 回合、lr 1e-3，见
  [`../../rstdp_spiking_actor_fix/README.md`](../../rstdp_spiking_actor_fix/README.md)）。

## 判读

A1′ 通过（痕迹方向正确）→ A2 不过（oracle EMA 仍学不起来）→ **Actor 路径本身仍可疑**。
下一嫌疑按预注册写死：**信用时域**（episode 级 S 摊到所有决策）或策略参数化；需要新预注册。
不回头调学习率、回合数或种子来"救"A2。

## 自证

两臂的复现块：commit `108cc2c`，工作区干净；耗时 37.6 s / 28.2 s。
原始数值：[`ema.json`](ema.json)、[`td.json`](td.json)。

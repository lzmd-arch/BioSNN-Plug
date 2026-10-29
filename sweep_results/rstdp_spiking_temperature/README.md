# W3 脉冲执行层 D6：脉冲策略的温度调度（2026-10-29）

**预注册：** [`../rstdp_spiking_temperature_prereg.md`](../rstdp_spiking_temperature_prereg.md)。
只改采样温度：`logit_scale` 按回合从 `start` 指数降到 `end`；资格、Critic、更新规则、
回合数不变。10 种子、800 回合、count-aligned、CPU；每 100 回合在极角 ±0.2 采窗口探针。

## 结果

| 臂 | start → end | 末期熵中位 | 熵 ≥0.3 的种子 | 末次中位 | best 中位 | slope >0 的种子 | 三条判据 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| A（对照） | 1.0 → 1.0 | 0.001 | 0/10 | 9.5 | 9.5 | 0/10 | 否 |
| B | 0.1 → 0.02 | 0.691 | 10/10 | 9.5 | 37.95 | 0/10 | 否（1 过、2/3 不过） |
| C | 0.03 → 0.01 | 0.693 | 10/10 | 9.5 | 27.6 | 1/10 | 否（1 过、2/3 不过） |

选择规则（预注册）：按末次中位选臂——三臂末次中位都是 9.5，**按规则无法分出最好臂**；
没有任何臂同时满足三条 → `temperature_fix_ready=false`。

## 读法

- **温度确实解开了熵坍缩**：对照 A 的末期熵中位 0.001，B/C 在 0.69 附近（上界 ln2≈0.693），
  熵 ≥0.3 的种子各为 10/10；
- **训练期的学习恢复了**：B 的 best 中位 **37.95**、C **27.6**（A 9.5）；评测曲线里出现过
  数十到一百多步的峰（B seed6：300 回合 86.0、325 回合又回到 9.5；C seed1：750 回合 159.7）；
- **但末期没有守住**：三臂的末次中位都是 **9.5**，极角探针的 slope 在 B 全为 0、C 只有 1 个
  种子为正——温度修好了「一开始就压成一边」，没有修好「峰值之后退化」；
- 结论：**温度是必要条件，不是充分条件**。峰值是真的（与口径复核一致），末期退化才是
  剩下的问题；下一步按 D7 预注册做归因。

## 边界

- 只改温度；不改规则/宽度/回合；
- 熵是 Boltzmann 采样分布的熵；评测（greedy）与温度无关，三臂的评测曲线比较的是同一套
  贪心规则；
- slope 来自每 100 回合的窗口探针（极角 ±0.2、其余状态为 0），与真实状态分布不同；
- best 是 32 个评测点取最大，末次是最后一次评测（噪声口径见
  [`../stage3_sustained_audit.md`](../stage3_sustained_audit.md)）；
- 单任务 CartPole，不外推。

## 自证

运行输出先写到仓外 `D:/Projects/d6_final/`（三臂顺序跑），跑完拷入本目录；复现块在**开跑前**
采集工作区状态（commit `c484082` 起的修法），三臂均显示干净。命令为 `cartpole.py --seeds 0,1,...,9 --episodes 800
--learning-rate 1e-3 --critic pop --critic-rate 0.1 --pop-post gradient --pop-readout uniform
--pop-normalize --pop-units 64 --pop-bias 0.4 --pop-gain 8 --warmup 10 --trace-rule
count-aligned --spiking-temperature --entropy-log --probe-every 100 --device cpu` 生成，
温度分别为 A 1.0→1.0、B 0.1→0.02、C 0.03→0.01。

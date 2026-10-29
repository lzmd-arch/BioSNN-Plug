# 预注册：W3 脉冲执行层 D3——连续速率读出（2026-10-29）

**状态：判据先落盘，实现与运行随后。** 结果不得回改判据。

## 要检验什么

计数同向（count-aligned）实验把资格方向修对了，但 10 种子中位数仍只有 **9.5 步**
（D1 rate-features 62.15、sign-consistent 75.65）。那份 README 把下一嫌疑指向
**计数策略本身的表达力**：窗口内的硬脉冲计数噪声大，策略可能几乎恒定。

D3 只换**策略读出**这一处：把 logits 从「窗口内硬计数」换成「窗口末的**低通滤波发放率**」
（同一批后突触脉冲，连续值），采样与贪心评测都用这个连续读出。资格/信用分配不变
（仍是 count-aligned）。

## 协议

- 完全复用 count-aligned 的 harness 与超参：CartPole、800 回合、Actor lr 1e-3、
  critic_rate 0.1、gradient critic、units 64、bias 0.4、gain 8、warmup 10；
- 滤波常数 **decay = 0.9**（与 `trace_decay` 同值，不调）；
- 10 种子（0–9）、CPU；对照直接引用：D1 62.15、count-aligned 9.5、sign-consistent 75.65。

## 判据（全部满足才 `spiking_rate_readout_ready=true`）

1. **10 种子中位数 ≥ 150 步**（阶段三「200 步以上」的中间线，和此前各轮同一口径）；
2. **配对 vs D1**：中位差 ≥ **+20 步**，且 **≥7/10** 种子比 D1 好；
3. 守卫：原有 count-aligned 的符号测试与轨迹测试全过（不得为了跑通改判据）。

任一不过 → 记负结果，不调 decay、不换 critic、不追加轮次。

## 边界

- 只换读出；仍是单层 LIF + 局部三因子规则，没有 TD 反向传播；
- 连续读出**不是**「脉冲网络策略就是这么算的」——它是对脉冲率的显式估计，
  写进 README 的差异里；
- 如果 D3 也不动，下一步应转向「单层网络的表达力」，而不是继续换读出代数。

## 产物

- `sweep_results/rstdp_spiking_rate_readout/README.md` 与 `summary.json`（复现块）；
- 逐种子 `best_steps`、与 D1 的配对差。

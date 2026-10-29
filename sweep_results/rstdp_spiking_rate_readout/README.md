# W3 脉冲执行层 D3：连续速率读出（2026-10-29）

**预注册：** [`../rstdp_spiking_rate_readout_prereg.md`](../rstdp_spiking_rate_readout_prereg.md)。
唯一改动：策略 logits 从「窗口内硬脉冲计数」换成「窗口末低通发放率」（decay 0.9）；
资格/信用分配仍是 count-aligned。10 种子、CPU、800 回合；**干净工作区**（复现块 commit
`dfae8e1`，工作区干净）。

## 结果（两种口径并列）

| 口径 | D3 本臂 | count-aligned | D1 rate-features |
| :--- | ---: | ---: | ---: |
| best 中位（预注册主判据） | **65.2** | 9.5 | 62.15 |
| **末次中位** | **9.55** | 9.5 | 13.05 |
| 末三段中位 | 11.42 | 9.5 | 20.8 |
| 配对 vs D1（best）中位差 | **+5.95** | — | — |
| 比 D1 好的种子数（best） | **7/10** | — | — |

逐种子（best / 末次 / 末三段）：
0：77.4 / 9.2 / 15.4；1：56.7 / 14.6 / 11.4；2：42.1 / 41.8 / 21.7；3：75.7 / 9.2 / 9.6；
4：73.7 / 15.9 / 12.0；5：30.0 / 9.3 / 10.7；6：78.8 / 9.5 / 10.1；7：25.8 / 9.5 / 13.1；
8：74.8 / 9.6 / 9.5；9：34.8 / 11.0 / 11.5。

**判据：** best 中位 ≥150 → **未过**；配对中位差 ≥+20 → **未过**（+5.95）；
≥7/10 变好 → 过。`spiking_rate_readout_ready = false`。

## 读法

- **best 口径下连续速率读出明显好于 count-aligned（65.2 vs 9.5），但末次口径下两者相同
  （9.55 vs 9.5）**。也就是说「窗口内硬计数」不是持续控制的瓶颈——换了读出之后，
  策略仍然没有在训练末期保住成绩；
- 与 D1（rate-features）比：best 中位差只有 +5.95、7/10 变好——**没有超过**
  预注册的 +20 门槛，不能说速率读出优于 D1 的资格规则；
- **口径警告**：`best_steps` 是 32 次评测取最大，记录的是「出现过的最好一刻」。峰值本身
  不是噪声（诊断 ③ 重测最好快照 20 个评测种子中位 73.8/76.5），但末期退化是事实；
  全臂复核见 [`../stage3_sustained_audit.md`](../stage3_sustained_audit.md)；
- 结论：**读出不是主要瓶颈**。下一步应查「为什么末期退化」——与 D4（状态依赖性诊断）
  和熵轨迹一起看，而不是继续换读出代数。

## 边界

- 只换读出；仍是单层 LIF + 局部三因子规则，无 TD 反向传播；
- 评测 10 回合/点、冻结随机流；噪声偏大（见口径复核）；
- 本轮 summary 已有 `last_steps` / `final3_mean_steps`，但没有 `eval_noise`（最好快照的
  20 种子重测）——这一项留给下一轮；
- 单任务 CartPole，不外推。

## 自证

`summary.json`（含复现块，commit `dfae8e1`、工作区干净）由
`cartpole.py --seeds 0-9 --episodes 800 --learning-rate 1e-3 --critic pop --critic-rate 0.1
--pop-post gradient --pop-readout uniform --pop-normalize --pop-units 64 --pop-bias 0.4
--pop-gain 8 --warmup 10 --trace-rule count-aligned --readout rate --device cpu` 生成。

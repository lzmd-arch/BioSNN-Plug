# 阶段三口径复核：best_steps 之外的「持续学会」（2026-10-29）

**性质：后验分析，不产生新数据。** 只读 `sweep_results/**/summary.json`，脚本
[`../scripts/stage3_sustained_audit.py`](../scripts/stage3_sustained_audit.py)。
不改变任何既有预注册判定（它们本来就全部未过），只回答一件事：
**「最好成绩」和「最后还在不在学」是不是同一个数。**

## 为什么查这个

`best_steps` 的定义是「32 个评测点（每点 10 回合）里取最大」，它回答的是
**「训练过程中有没有出现过好的一刻」**，不是「训练结束时还剩多少」。D3 的连续速率读出把
best 中位从 9.5 抬到 65.2，但**末次评测中位仍是 9.55**——两者差得太大，促成了这次复核。

⚠️ **不要据此把 best 说成纯噪声**：诊断 ③（`sweep_results/rstdp_spiking_diagnosis/`）
把**最好快照**用 20 个新的评测种子重测过，lr=1e-4/TD 配置下中位 **73.8 / 76.5**
（CI 半宽 5.2% / 6.6%），说明那个峰值是真的。能下的结论是：
**峰值存在，但训练末期没有保住它**——问题在稳定性/遗忘，而不是测量造假。

## 全表（全部历史臂，同一口径）

| 结果 | 种子 | best 中位 | 末次中位 | 末三段中位 |
| :--- | ---: | ---: | ---: | ---: |
| `sweep_results/rstdp_spiking/pop/summary.json` | 5 | 15.7 | 9.2 | 9.2 |
| `sweep_results/rstdp_spiking/pop_gradient/summary.json` | 5 | 14.3 | 9.2 | 9.2 |
| `sweep_results/rstdp_spiking/pop_gradient_10seeds/summary.json` | 10 | 55.0 | 9.4 | 9.5 |
| `sweep_results/rstdp_spiking/pop_gradient_seeded/summary.json` | 5 | 65.6 | 9.2 | 9.2 |
| `sweep_results/rstdp_spiking/pop_homeo005/summary.json` | 5 | 72.9 | 9.2 | 9.2 |
| `sweep_results/rstdp_spiking/pop_homeo005_seeded/summary.json` | 5 | 40.1 | 9.2 | 14.1 |
| `sweep_results/rstdp_spiking/pop_mask50_seeded/summary.json` | 5 | 30.9 | 9.2 | 9.3 |
| `sweep_results/rstdp_spiking/pop_nonorm/summary.json` | 5 | 63.1 | 10.2 | 9.5 |
| `sweep_results/rstdp_spiking/pop_nonorm_seeded/summary.json` | 5 | 58.1 | 9.3 | 9.3 |
| `sweep_results/rstdp_spiking/pop_opg16/summary.json` | 5 | 72.4 | 9.2 | 9.2 |
| `sweep_results/rstdp_spiking/pop_opg16_10seeds/summary.json` | 10 | 37.1 | 9.4 | 9.7 |
| `sweep_results/rstdp_spiking/pop_opg16_seeded/summary.json` | 5 | 35.2 | 9.2 | 9.2 |
| `sweep_results/rstdp_spiking/pop_opg16_signed/summary.json` | 5 | 9.3 | 9.2 | 9.2 |
| `sweep_results/rstdp_spiking/pop_opg16_signed_seeded/summary.json` | 5 | 9.4 | 9.2 | 9.3 |
| `sweep_results/rstdp_spiking/pop_seeded/summary.json` | 5 | 20.4 | 9.2 | 9.2 |
| `sweep_results/rstdp_spiking/pop_signed/summary.json` | 5 | 9.2 | 9.2 | 9.2 |
| `sweep_results/rstdp_spiking/pop_signed_nonorm/summary.json` | 5 | 33.9 | 9.2 | 9.2 |
| `sweep_results/rstdp_spiking/pop_signed_nonorm_seeded/summary.json` | 5 | 9.3 | 9.2 | 9.2 |
| `sweep_results/rstdp_spiking/pop_signed_seeded/summary.json` | 5 | 22.9 | 9.3 | 9.3 |
| `sweep_results/rstdp_spiking/td_10seeds/summary.json` | 10 | 50.4 | 9.5 | 9.5 |
| `sweep_results/rstdp_spiking/td_seeded/summary.json` | 5 | 17.3 | 9.3 | 9.3 |
| `sweep_results/rstdp_spiking_actor_fix/seeds_pair_stdp/summary.json` | 10 | 55.0 | 9.4 | 9.5 |
| `sweep_results/rstdp_spiking_actor_fix/seeds_sign_consistent/summary.json` | 10 | 75.7 | 18.3 | 16.4 |
| `sweep_results/rstdp_spiking_count_aligned/summary.json` | 10 | 9.5 | 9.5 | 9.5 |
| `sweep_results/rstdp_spiking_credit_diagnosis/rate_features/summary.json` | 10 | 62.1 | 13.0 | 20.8 |
| `sweep_results/rstdp_spiking_policy_control/linear_rate/summary.json` | 10 | 179.8 | 9.6 | 14.3 |

## 读法

1. **没有一条臂在末次口径上超过 20 步**：最高是 sign-consistent 的 **18.3**，
   其余 9.2–13.0。目标线是 200 步。
2. **best 与末次的差最大的是 D1 rate-features（62.1 → 13.0）与 D2 线性策略
   （179.8 → 9.6）**——这不是「学得快慢」的差别，是「峰值 vs 末期」的差别。峰值可复现
   这一点只在诊断 ③ 的两个种子上验过；其余臂没有 `eval_noise`，不能假定 best 可复现。
3. **历史比较要重新读**：例如「sign-consistent 75.65 > pair-stdp 55.05」在 best 口径下
   成立，末次口径是 18.3 vs 9.4——差仍在，但两个数都离 200 很远，
   不能把它写成「规则选对了，只是差一点」。
4. **count-aligned 的 9.5 不是「突然塌了」**：它没有出现任何评测尖峰，
   而别的臂有过尖峰；末次口径下两者本来就都在 9–10 附近。

## 对后续的意义

- 新实验一律**同时报 best / 末次 / 末三段**（`cartpole.py` 已加字段）；
- 关键臂应补 `eval_noise_seeds`（最好快照在 20 个新评测种子上的分布），否则 best 只能当
  「出现过」的证据，不能当「学会了」的证据；
- 未来的预注册应以**末三段中位**为主判据，best 只作「是否有过任何成功回合」的辅助；
- 评测噪声本身要压：10 回合均值太薄，后续轮次应提高 eval_episodes 或改用冻结种子的
  更多回合；这需要新一轮预注册，不在本复核里改。

## 自证

本文件由 `python scripts/stage3_sustained_audit.py` 的输出直接生成（表未手改）。

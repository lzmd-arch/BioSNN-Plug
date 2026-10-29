# 阶段三脉冲执行层：读出诊断汇总（D3–D15，2026-10-29）

**这份文件把散在台账 147–169 与主 README 那一行里的诊断结论集中起来**，主 README 只留一句
结论 + 链接。每条都指向自己的结果目录；判据与修正见各自的预注册文件。

## 一句话结论

> **在 L1 归一化（每列 L1=1）+ 阈值 1.0 的工作点上，R-STDP 类规则产生的差方向（相对幅度
> 0.5–7.3%）比任何读出能分辨的尺度小一个数量级。** 计数、低通率、膜电位、首发放潜伏期
> 四种读出都分不开；唯一有效的杠杆是**平均多个独立窗口**（`snr ∝ K^0.6`），代价是每步 K 倍
> 的窗口评估。

## 链条

| 轮次 | 问的问题 | 答案 | 结果目录 |
| :--- | :--- | :--- | :--- |
| D3 | 换读出（rate vs count）能不能救 | 不能（65.2 / 9.55） | [`rstdp_spiking_rate_readout`](rstdp_spiking_rate_readout/README.md) |
| D4 | 策略是否依赖状态（slope） | 训练后 slope 全 0 | [`rstdp_spiking_state_dependence`](rstdp_spiking_state_dependence/README.md) |
| D5 | 输出是不是坍缩 | 熵坍缩 | [`rstdp_spiking_collapse`](rstdp_spiking_collapse/README.md) |
| D6 | 温度能不能修熵 | 熵修好，末次仍 9.5 | [`rstdp_spiking_temperature`](rstdp_spiking_temperature/README.md) |
| D7 | 退化能否归因（熵/范数/Critic） | 5/5 `unattributed` | [`rstdp_spiking_degradation`](rstdp_spiking_degradation/README.md) |
| D8 | 末期读出还有多少状态信息 | 5/5 `no_state_information`（Δ 极差 0–0.4 计数，信噪比 11–23） | [`rstdp_spiking_readout`](rstdp_spiking_readout/README.md) |
| D10 | 差方向存在吗？ | **在**（4/5，同范数随机方向的 2.95–13.4 倍），但相对幅度只有 **0.5–7.3%**；三种判定 2/2/1，无多数 | [`rstdp_spiking_direction`](rstdp_spiking_direction/README.md) |
| D11 | 调阈值 / 不应期 / 偏置 / 窗口 | 不够：54 组最好 0.7–3.45 计数（门槛 2），5 种子全可用 **0** | [`rstdp_spiking_readout_gain`](rstdp_spiking_readout_gain/README.md) |
| D12 | 抬每列预算 / 去正负抵消 | 不够：λ=0.30（差方向占预算 30%）最好 **0.80** 计数 | [`rstdp_spiking_readout_gain_scale`](rstdp_spiking_readout_gain_scale/README.md) |
| D13 | 换读出（低通率 / 膜电位） | 不够：λ=0.10 上 S 只有 0.58–0.65，且 **S 不随 λ 涨** | [`rstdp_spiking_readouts`](rstdp_spiking_readouts/README.md) |
| D14 | K 窗平均 / 加长窗口 / 去编码噪声 | 平均有用（0.57→2.12，K=1→8）但 5/5 只到 3/5；**加长窗口无收益**；**无噪声前端 ⇒ 计数差恒 0** | [`rstdp_spiking_denoise`](rstdp_spiking_denoise/README.md) |
| D15 | 首发放潜伏期（真实幅度） | 在计数为 0 的种子上有信号（0.75 / 0.54），但仍 <2 | [`rstdp_spiking_latency`](rstdp_spiking_latency/README.md) |

## 机制（量出来的，不是猜的）

1. **净每步驱动 ≈ 0.047，阈值 1.0**：列 L1 归一化后正负权重互相抵消 95%+，稳态膜电位 ≈0.96
   贴着阈值；发放节律由 `baseline_period=5` 的偏置撑着，输入只负责推移相位（D11）。
2. **计数是整数格点上的量**：差方向小到改变不了整数个脉冲，于是 Δ 计数极差停在 0.2–0.8（D12）。
3. **信号本身是涨落驱动的**：把伯努利编码换成期望输入（无噪声），各状态上的计数差**恒为 0**
   （D14，4/5 种子）。无噪声前端会让这个读出变成纯时钟。
4. **换读出改变不了斜率**：λ 从 0.05 加到 0.30，四种读出的 S 几乎不动——分子分母一起涨（D13）。

## 口径教训（写下来免得重犯）

D10 与 D14 各踩了两次「判据先验不可达 / 恒成立」：

- **比值型判据**要先问分母会不会过零或变号：`(d₀−d₁)/((|d₀|+|d₁|)/2)` 被三角不等式钉在
  [−2, 2]，两列驱动异号时恒为 ±2（D10 修正 1→2）；
- **固定分母**会让「平均 K 个窗口」这一维白做：`spread/sd_ref` 对 K 完全不敏感（D14 修正 2）；
- **问问题的位置**要选在信号存在的地方：λ=0.05 处确定性档恒为 0，那里问降噪是 0/0（D14 修正 1）。

## 下一步（按优先级）

1. **D9（已在 GPU 队列）**：`contrastive_gamma` 0 / 0.1 / 0.5——减掉两列共模成分，直接抬高
   差方向的相对幅度，是本表缺的那一样东西；
2. 若 D9 能把相对幅度抬到 ~0.3 量级，再谈 D14 的 K 窗平均（K=8，代价 8×）；
3. 潜伏期读出 + 平均的组合是第三条候选，但要先证明潜伏期在训练后的真实列上也一样敏感。

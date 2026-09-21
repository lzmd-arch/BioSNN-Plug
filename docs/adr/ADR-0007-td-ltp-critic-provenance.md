# ADR-0007：Critic 采用 TD-LTP，出处为 Frémaux et al. (2013)

- **状态**：已采纳
- **日期**：2026-09-21
- **相关**：计划书 §3.2、§十一 P0 待办；[docs/references.md](../references.md)

## Context

计划书 §3.2 把执行层 Critic 的训练规则写作 **"TD-LTP"**，并标注"⚠️ 出处待补"，
同时在 §十一 把它列为 **P0 待办**，并规定：

> TD-LTP 文献考证 | Critic 的训练规则是整个 R-STDP 闭环局部性声明的基石，
> **无出处则第五阶段不立项**。

也就是说，这个名词有没有出处，决定了第五阶段立不立项。

同时，计划书 §3.2 把 "TD-LTP" 与 **[3] Frémaux, Sprekeler & Gerstner (2010)**
绑定引用——而 2010 那篇讲的是 R-STDP 的无监督偏差与"必须有刺激特异性奖励预测系统"，
它**不提出也不命名** TD-LTP。这个配对是否成立，需要考证。

## Decision

**采用 TD-LTP，出处是 Frémaux, Sprekeler & Gerstner (2013), *PLoS Computational
Biology* 9(4):e1003024。P0 待办结项，第五阶段立项门槛通过。**

原文逐字（Critic learning 一节，对应 Eq. 17）：

> Because it has, roughly, the form of "TD error signal × Hebbian LTP", we call this
> learning rule **TD-LTP**.

Figure 2A 图注："TD-LTP is the learning rule given in Eq. 17."。"TD-LTP" 在全文出现
40 次。该文开放获取，可自行复核。

**规则本体**（三因子，仅计 pre-before-post）：

Δw ∝ δ(t) · κ ∗ [ x_i(t) · y_j(t) ]

- 因子 1、2：pre 与 post 脉冲串通过**仅计 pre-before-post** 的符合窗（post-before-pre
  配对被忽略——这是 TD-LTP 与 TD-STDP 的实质差别）；
- 过滤核 κ 充当资格痕迹（原文称其"serves a role similar to the eligibility trace"）；
- 因子 3：δ(t)，**全局标量** TD 误差，经多巴胺式广播承载。

**非局部量只有这一个标量**，不含误差向量的反向传播，与计划书的"纯局部"声明相容。

**计划书的引用配对需要拆成两条**：

| 内容 | 应引 |
| :--- | :--- |
| TD-LTP 的名称与规则本体 | Frémaux et al. (2013), PLoS Comput Biol 9(4):e1003024 |
| R-STDP 的无监督偏差、必须引入刺激特异性奖励预测 | Frémaux et al. (2010), J Neurosci 30(40):13326-13337（该文 Figure 3 的主题） |

## Consequences

**好处**

- §十一 的 P0 难题有确定答案，第五阶段不再悬着；
- 架构选择有外部先例支撑：[13] Tihomirov et al. (2025) 把 TD-LTP 用作脉冲
  actor-critic 的 Critic，与 §3.2 的设计同构，是"这条路走得通"的独立证据；
- 规则满足项目最硬的约束——非局部信号只有一个标量。

**代价（已接受）**

- **规则形式只能按二手描述实现。** 2013 那篇开放获取、可核；但应用线
  （Tihomirov/Rybka 的期刊版）在付费墙后。因此本 ADR 记录的是"命名与规则骨架"，不是
  逐字转录的方程。第一阶段实现时若发现细节对不上，以 2013 原文为准。
- 计划书正文需要出 v6.3 更正引用配对。在更正之前，任何引用计划书 §3.2 的场合都要
  以本 ADR 为准。

**明确不声称的（避免过度引用）**

1. Frémaux 2013 中 "no back-propagation signal has been observed in experiments"
   **不能**用来论证"生物网络不做误差向量反向传播"。该句的 back-propagation 指的是
   TD 误差沿**时间**的信用分配特征（原文另称 "this back-propagation phenomenon is a
   signature of TD learning algorithms"），与本项目的局部性主张无关。要论证局部性，
   应引三因子形式本身与原文 "the global signal" 的表述。
2. TD-STDP 与 TD-LTP 的关系，原文措辞是 "behaves similarly"、"only slightly worse"。
   **不写成"功能等价"。**
3. 本 ADR 未逐字核验付费墙后的期刊版符号体系。凡涉及具体实现，以 2013 原文 Eq. 17
   与 Figure 2A 为准。

## Alternatives

**A. 判定为无出处，触发 §十一 的降级路径（滑动平均基线 Critic）。**
否决。考证结论是有出处，且是作者正式命名（有命名语句 + 公式编号 + 图注三重证据）。
降级的前提不成立。

**B. 采用 Tihomirov/Rybka et al. (2025) 作为 TD-LTP 的第一出处。**
否决。两篇都真实存在，但 2025 那篇是**使用** TD-LTP，2013 那篇是**命名并定义**它。
引命名出处，不引使用出处。
（核验过程中确实有两个检索角度把归属给了 2025，对抗性复核推翻——复核者承认其检索
未回溯到 2013。这次分歧本身值得记住：**一次检索没找到不等于不存在**。）

**C. 不落地 TD-LTP，直接用 TD-STDP（改动最小，与既有 R-STDP 代码差异仅一处）。**
暂缓，不是否决。原文验证两者行为相近，TD-STDP 只需把奖励调制项 S=R−⟨R⟩ 换成 δ(t)。
若第一阶段实现 TD-LTP 遇到障碍，这是成本最低的回退。届时另开 ADR。

**D. 整体改用 e-prop 的 RL 形式（Bellec et al. 2020 已含 actor-critic）。**
暂缓。它可整体替代"R-STDP + 独立 Critic"，理论上更统一，但会把执行层与认知层绑到
同一套规则上，削弱"三种局部规则组合"这条主线。留待第二阶段评估规则冲突时再议。

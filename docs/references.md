# 引用清单

> **这份文件是引用信息的唯一真相源。**
>
> 计划书正文里的 `[n]` 编号对应本表的"编号"列。新增引用必须登记到这里，
> 否则 `scripts/check_references.py` 与 CI 里的 lychee 都看不到它。
>
> 检查分工：**本表的结构**（编号连续、URL 格式、每条都有核验状态）由
> [`scripts/check_references.py`](../scripts/check_references.py) 在提交时校验；
> **链接是否存活**由 CI 里的 lychee 校验。两者都不检查引用内容是否正确——
> 那是"核验状态"列要如实反映的东西。

## 已发生过的错误

计划书 §12.3 把"引用链接存活检查"的用途写得很具体：它是拦截 **"Khacef 署名"类
错误的入口**。那个错误在本仓库真实发生过——参考文献 [8] 的作者被误署为
"Khacef et al."，v6.1 才更正为 Hajizada et al.。

## 核验状态

| 状态 | 含义 | 该做什么 |
| :--- | :--- | :--- |
| `verified` | 逐条核验过：文献存在、元数据正确、正文归因于它的**具体数字**能在原文找到 | 无 |
| `partial` | 文献存在、元数据正确，但部分归因数字未在原文中确认 | 补核那些数字 |
| `metadata-error` | 文献存在，但登记的作者 / 年份 / 卷期页 / 文章号有误 | **改书目** |
| `unverified` | 尚未核验。**对外引用前必须先升级** | 去核 |

## 清单

| 编号 | 文献 | URL | 核验状态 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Bellec, G., Scherr, F., Subramoney, A., Hajek, E., Salaj, D., Legenstein, R., & Maass, W. (2020). A solution to the learning dilemma for recurrent networks of spiking neurons. *Nature Communications*, 11, 3625. | https://www.nature.com/articles/s41467-020-17236-y | verified | 作者、卷、文章号全部吻合。三条归因均确认。⚠️ 两处表述建议按原文精确化：原文式 (28) 为 ΔW_ji = −η Σ_t L_j^t ē_ji^t，**带负号、对时间求和**，且学习信号索引的是突触后神经元 j（计划书写成 L_i(t) 且为瞬时形式）。 |
| 2 | Pes, L., Yin, B., Stuijk, S., & Corradi, F. (2025). Traces Propagation: Memory-Efficient and Scalable Forward-Only Learning in Spiking Neural Networks. *arXiv preprint* arXiv:2509.13053. | https://arxiv.org/abs/2509.13053 | metadata-error | ⚠️ **数据集名称错误**：计划书正文写"在 MNIST 和 SHD 上"，原文是 **N-MNIST**（事件相机数据集），二者不是一回事；原文全文从未使用 MNIST。⚠️ **出处不完整**：该文已发表于 *Neuromorphic Computing and Engineering* **6(1):014002 (2026)**，DOI 10.1088/2634-4386/ae2ef9，建议补上期刊版。另："按突触存储导致 O(N²)"这一条对资格痕迹类规则成立，但原文把该复杂度明确归属给 OSTTP/OSTL，并未单独给出 e-prop 的空间复杂度——计划书把它单独归给 e-prop 属引申。 |
| 3 | Frémaux, N., Sprekeler, H., & Gerstner, W. (2010). Functional Requirements for Reward-Modulated Spike-Timing-Dependent Plasticity. *Journal of Neuroscience*, 30(40), 13326-13337. | https://www.jneurosci.org/content/30/40/13326 | verified | 作者、卷期页全部吻合，无署名错误。5 条归因中 4 条确认。⚠️ URL 对浏览器正常，但对脚本化客户端返回 **403**（Cloudflare）——CI 的 lychee 已用 `--accept ...403` 放行，这是预期行为不是死链。归因原文为 "±25% of the SD (σR)"（双向）。 |
| 4 | Pogodin, R., & Latham, P. E. (2020). Kernelized information bottleneck leads to biologically plausible 3-factor Hebbian learning in deep networks. *Advances in Neural Information Processing Systems*, 33. | https://proceedings.nips.cc/paper/2020/hash/517f24c02e620d5a4dac1db388664a63-Abstract.html | partial | 三因子结构、第三因子无需自顶向下传递、需要除法归一化——三条均确认。⚠️ **"不遭受深度表征瓶颈"未在原文中找到**（`not_found`）：原文的说法是该规则族**避免了**深度网络的表征瓶颈问题，与"不遭受"是不同强度的表述，建议按原文改写。 |
| 5 | Confavreux, B., Agnes, E. J., Zenke, F., Sprekeler, H., & Vogels, T. P. (2025). Balancing complexity, performance and plausibility to meta learn plasticity rules in recurrent spiking networks. *PLoS Computational Biology*, 21(4), e1012910. | https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012910 | verified | 元数据吻合。ES 元学习、复杂规则开始失败、先验设计损失函数困难——三条确认。"稳健稳定全部四种突触类型"为 `close`（原文的稳健性结论有条件限定，非无条件成立）。 |
| 6 | Shen, J., Xie, Y., Xu, Q., Pan, G., Tang, H., & Chen, B. (2025). Spiking Neural Networks with Temporal Attention-Guided Adaptive Fusion for imbalanced Multi-modal Learning. *Proceedings of the 33rd ACM International Conference on Multimedia*. | https://dl.acm.org/doi/10.1145/3746027.3755622 | partial | TAAF 的动态重要性分配与时间异构分层整合——两条确认。⚠️ **"提供时间融合与语义融合双通道"未在原文中找到**（`not_found`）。计划书 §2.2 的"双通道融合"是**本项目自己的架构设计**，不应归因给 TAAF。ACM DL 对脚本化客户端返回 403，lychee 已放行。 |
| 7 | Hu, K., Wen, L., Zhang, T., & Zhang, H. (2026). PS-SNN: pattern separation learning for expandable spiking neural networks in class-incremental learning. *Scientific Reports*, 16, Article 12653. | https://www.nature.com/articles/s41598-026-42970-6 | metadata-error | ⚠️ **文章号错误**：计划书写 "Article 42970"，权威来源（Nature 出版页与 Crossref）均为 **Article number 12653**。42970 是 DOI 尾段，不是文章号。76.42% 的增量准确率与正交类中心两条归因均确认。 |
| 8 | Hajizada, E., Rager, D., Shea, T., Campos-Macias, L., Wild, A., Hüllermeier, E., Sandamirskaya, Y., & Davies, M. (2026). Online Continual Learning on Intel Loihi 2 via a Co-designed Spiking Neural Network. *arXiv preprint* arXiv:2511.01553. | https://arxiv.org/abs/2511.01553 | metadata-error | 署名已确认正确（v6.1 把 "Khacef et al." 更正为 Hajizada et al. 是对的，应予保留）。⚠️ **年份错误**：arXiv 编号 2511 即 2025 年 11 月，首次提交 2025-11-03；2026 只是 v2 修订年。⚠️ **版本与数据错配**：计划书引的是 v2 的标题，却用 v1 的指标（70× / 23.2ms / 5,600× / 281mJ）——v2 已改为 113× / 37.3ms / 6,600× / 333mJ。二者取一，不要混用。 |
| 9 | Savage, W. (2026). EMBER: Autonomous Cognitive Behaviour from Learned Spiking Neural Network Dynamics in a Hybrid LLM Architecture. *arXiv preprint* arXiv:2604.12167. | https://arxiv.org/abs/2604.12167 | verified | **计划书 §四 的整个证据基础，也是优先级最高的核验对象。** 元数据吻合，14 条归因数字**全部逐条确认**：82.2% / 83.8% 区分度、s=0.14、σ=0.1 与 0.9Hz、7 轮对话首次触发、3 天 52 条消息 23 次调用（1 reach_out + 22 journal）、3 倍冲动阈值与 5 分钟 3 次、15 分钟 24 个侧向脉冲、突触连接 0→10,843→53,992→201,394、1.6% 衰减、64 节点 124 边、Claude Sonnet 4.6、220K 神经元 / RTX 5070 Ti 与 RTX 4060 Ti。无一条落空。 |
| 10 | tfatykhov. (2026). MEMBRAIN: Neuromorphic Memory Bridge for LLM Agents. *GitHub Repository*. | https://github.com/tfatykhov/membrain | metadata-error | ⚠️ **作者项错误**：计划书把项目名 "MEMBRAIN" 当作作者/机构；这是个人仓库，作者是 GitHub 账号 **tfatykhov**。FlyHash 编码（1536→20,000 维、int8 随机投影 + WTA、约 30MB）、Nengo + Voja、睡眠期噪声巩固——四条归因均确认。 |
| 11 | christophejlegros-lgtm. (2026). ASTRA: Unified Research Lab + MCP Server. *GitHub Repository*. | https://github.com/christophejlegros-lgtm/ASTRA-Unified-ResearchLab-MCP-v2.7 | metadata-error | ⚠️ **作者项错误**：同上，把项目名 "ASTRA" 当作作者；实际是 GitHub 账号 **christophejlegros-lgtm**，非组织。⚠️ 计划书称其"验证了 SNN 引擎可通过 MCP Server 暴露给 Claude Desktop 等客户端"——仓库确实这么做，但"已验证可用"是比"提供了该接口"更强的说法，建议按实际证据强度改写。 |
| 12 | Frémaux, N., Sprekeler, H., & Gerstner, W. (2013). Reinforcement Learning Using a Continuous Time Actor-Critic Framework with Spiking Neurons. *PLoS Computational Biology*, 9(4), e1003024. | https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003024 | verified | **P0 待办（TD-LTP 考证）的结论所在**，详见下节。原文逐字："Because it has, roughly, the form of 'TD error signal × Hebbian LTP', we call this learning rule TD-LTP."（Critic learning 一节，对应 Eq. 17 与 Figure 2A 图注 "TD-LTP is the learning rule given in Eq. 17."）该文为开放获取，全文可核。**尚未引入计划书**，需 v6.3 补入。 |
| 13 | Tihomirov, Y., Rybka, R., Serenko, A., & Sboev, A. (2025). Combination of reward-modulated spike-timing dependent plasticity and temporal difference long-term potentiation in actor-critic spiking neural network. *Cognitive Systems Research*, 90, 101334. | https://doi.org/10.1016/j.cogsys.2025.101334 | verified | TD-LTP 在脉冲 actor-critic 上的后续应用线。会议版：Tihomirov, Rybka, Serenko & Sboev (2024), BICA 2024, *Studies in Computational Intelligence*, pp. 411-415, DOI 10.1007/978-3-031-76516-2_41。另有同组开放获取前置工作 Vlasov et al. (2024), *Moscow University Physics Bulletin* 79(S2):S944-S952, DOI 10.3103/S0027134924702400，含 CartPole 细节。**尚未引入计划书**。 |
| 14 | fangwei123456. (2026). SpikingJelly: An open-source deep learning framework for spiking neural networks based on PyTorch. *GitHub Repository*. | https://github.com/fangwei123456/spikingjelly | verified | **许可证事实核验**：计划书 §12.1 写"与 SpikingJelly（同一许可证）"，事实**不是**——SpikingJelly 不是 Apache-2.0。其 LICENSE 为**启智开源许可证 1.0**（Open-Intelligence Open Source License, OIOSL）；GitHub API 的 license 字段是 `NOASSERTION`；PyPI 稳定版 0.0.0.0.14 的 classifier 是 `License :: Other/Proprietary License`。商业使用或再发布前须在 AITISA（aitisa.org.cn）声明披露；专利许可为自愿声明且带专利池 / FRAND 条件。此外 2.0.0rc1 要求 Python >= 3.11。裁决与取舍见 `docs/adr/ADR-0008` |

## 待办

### ✅ P0：TD-LTP 出处考证 —— 已完成，结论是"有出处"

计划书 §3.2 与 §十一 把 Critic 的训练规则写作 "TD-LTP" 并标注"⚠️ 出处待补"，
§十一 规定 **"无出处则第五阶段不立项"**。考证结论：

**TD-LTP 是一个被作者正式命名的学习规则，出处确切，P0 门槛通过。**

- **命名出处**：Frémaux, Sprekeler & Gerstner (2013), *PLoS Comput Biol* **9(4)**:e1003024
  —— 即本表新增的 [12]。原文含明确的命名语句、Eq. 17 编号与 Figure 2A 图注，
  不是泛称被包装成正式命名。该文开放获取，"TD-LTP" 在全文出现 40 次，可自行复核。
- **计划书的错误是引用配对，不是捏造**：计划书把 "TD-LTP" 与 **[3] Frémaux et al. 2010**
  绑定，但命名出自 **2013** 那篇。两篇分工不同，应拆开引用：
  - **TD-LTP 的名称与规则本体** → [12] Frémaux et al. 2013；
  - **R-STDP 存在无监督偏差、必须引入刺激特异性奖励预测** → [3] Frémaux et al. 2010
    （这正是该文 Figure 3 的主题）。
- **应用先例**：[13] Tihomirov et al. 2025 把 TD-LTP 用作脉冲 actor-critic 的 Critic，
  与计划书 §3.2 的架构选择同构，可作为该路线可行的外部证据。

**核验过程中出现的分歧（如实记录）**：三个独立检索角度中有两个把 TD-LTP 归给
Tihomirov et al. 2025，与 [12] 冲突。对抗性复核推翻了这两个归属——复核者承认
"未发现 Tihomirov/Rybka 之前的 TD-LTP 用法"，即其检索未回溯到 2013。而 [12] 的
归属被复核者独立抓取 PLoS 全文后**逐字复现**。故采信 [12]。这也说明：
**一次检索没找到不等于不存在**，归属类结论必须做对抗性复核。

**复核同时指出的记录纪律**（写进 ADR 时可避免过度声称）：

1. Frémaux 2013 中 "no back-propagation signal has been observed in experiments"
   一句指的是 TD 误差沿**时间**的信用分配特征，**不是**"生物网络不做误差向量反向传播"。
   用它论证"纯局部"属于断章取义；应改引三因子形式本身（pre×post → κ 滤波 → 乘标量 δ）
   与原文 "the global signal" 的表述。
2. TD-STDP 与 TD-LTP 的关系，原文措辞是 "behaves similarly" / "only slightly worse"，
   **不宜表述为"功能等价"**。

### 第一阶段候选实现基线（尚未引入计划书，故未编号）

以下文献在核验过程中被确认存在且描述准确，但尚未被计划书引用，暂不编号：

- Chung & Kozma (2020), *Reinforcement Learning with Feedback-modulated TD-STDP*,
  arXiv:2008.13044 —— 开放获取，含 CartPole-v1 与 LunarLander 数字，
  消融证明去掉 feedback modulation 就学不会。**本地复现成本最低的起点。**
  注意其 critic 是**神经元群体**而非单个神经元（该组数字出自 Vlasov et al. 2024）。
- Bellec et al. (2020) 的 e-prop RL 部分（本表 [1]）已含 actor-critic，可整体替代
  "R-STDP + 独立 Critic"，是另一条值得评估的路线。

### 核验流程

对每条引用，核验四件事：

1. **存在性**——URL / DOI / arXiv 编号可达；
2. **元数据**——作者、年份、期刊或会议、卷期页/文章号与登记一致；
3. **归属**——没有把 A 的结论记到 B 头上（[8] 与 [10]/[11] 的历史教训）；
4. **数字**——计划书正文里归因于它的具体数字能在原文中找到。找不到或对不上的，
   状态只能给 `partial`，并在备注里写明哪一项没对上。

**不要用记忆代替检索。** 核验不到就如实标 `unverified` 或 `partial`。

## 计划书正文待更正项

以下问题属于 **`BioSNN-Plug_项目计划书_v6.2.md` 正文**，不在本文件的可改范围内。
更正正文应走 v6.3，而不是就地修改一份带版本号的正式文档。

| 位置 | 问题 | 建议改法 |
| :--- | :--- | :--- |
| §3.2、§六表格、§十、§十一、§十二 | "TD-LTP Critic" 绑定 [Frémaux et al., 2010] | 拆成两条：TD-LTP → 2013 PLoS Comput Biol 9(4):e1003024；R-STDP 偏差 → 2010 J Neurosci。并**删除"出处待补"标注与 §十一 的 P0 待办** |
| §3.1 关键修正 1 | "在 MNIST 和 SHD 上" | 改为 **N-MNIST** 和 SHD；补期刊版出处 |
| §5.2、§七 第三阶段 | "PS-SNN……76.42%" 所在行的出处 | 文章号 42970 → **12653** |
| §6.1、§七 第六阶段 | Hajizada et al. "2026" + 70×/5,600× | 年份改 **2025**；指标与所引版本对齐（v2 为 113×/6,600×） |
| §3.3 | "不遭受深度表征瓶颈" | 按原文改写为"避免了深度网络的表征瓶颈问题" |
| §2.2 | "双通道融合"归因给 TAAF | TAAF 提供的是时间注意力引导的自适应融合；**双通道融合是本项目自己的设计**，不应归因 |
| §4.6、参考文献 [11] | ASTRA 作者写为项目名 | 改为 GitHub 账号 christophejlegros-lgtm |
| §4.3、参考文献 [10] | MEMBRAIN 作者写为项目名 | 改为 GitHub 账号 tfatykhov |
| §3.2 | Critic 训练规则的具体形式 | 按 [12] 2013 原文写：Δw ∝ δ(t)·κ∗[x_i·y_j]（**仅计 pre-before-post**），δ 为全局标量 TD 误差。并避免用"no back-propagation signal"一句论证局部性 |
| §12.1、§1.4、§八 | 许可证表把 SpikingJelly 写成"与本项目同一许可证（Apache-2.0）" | 按事实改写：SpikingJelly 用**启智开源许可证 1.0**（OIOSL），并补上其商业使用披露义务。§1.4/§八 的技术选型行注上"该依赖的许可条款，以及它对开发环境 Python 下限的影响，见 `docs/adr/ADR-0008`" |

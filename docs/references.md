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

本仓库还发生过一类**不涉及结论、却会让内容凭空消失**的错误：引用 [4] 那行的列结构
说明里用了三个未转义的 `|`。GFM 里表格的 `|` **即使在反引号内也仍是分隔符**，必须
写成 `\|`；多出来的单元格在渲染时**被直接丢弃**——那一行从第三个竖线到行尾的 877 个
源字符（Table 4/5 的 MNIST 数值、两句逐字 HSIC 引文）整段不显示，备注格的可见文本
从 1,089 字掉到 260 字，而读原文文件一切正常。2026-09-25 三语一并改正。教训是：**核对这张表不能只读原文文件，
要看它渲染出来的样子**，或至少数一数每行的单元格数是否与表头相同。

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
| 1 | Bellec, G., Scherr, F., Subramoney, A., Hajek, E., Salaj, D., Legenstein, R., & Maass, W. (2020). A solution to the learning dilemma for recurrent networks of spiking neurons. *Nature Communications*, 11, 3625. | https://www.nature.com/articles/s41467-020-17236-y | verified | 作者、卷、文章号全部吻合。三条归因均确认。⚠️ 两处表述建议按原文精确化：原文式 (28) 为 ΔW_ji = −η Σ_t L_j^t ē_ji^t，**带负号、对时间求和**，且学习信号索引的是突触后神经元 j（计划书写成 L_i(t) 且为瞬时形式）。 **2026-09-25 补核（「此文没有 sMNIST 基线」这一条的出处）**：正文（去标签后 146,503 字符）里 `MNIST` / `mnist` / `permuted` / `sequential` / `SHD` **各 0 次**，出现的任务只有 TIMIT（8）/ ATARI（1）/ temporal credit assignment（9）；补充材料 PDF（MOESM1，提取文本 81,610 字符）里 `shd` / `heidelberg` / `spike-hearing` 也**全为 0**。同一工作的预印本（arXiv:1901.09049）摘要页 `MNIST` 亦为 0 次。**此文没有给出任何 MNIST 或 SHD 结果**——凡把这两个数据集的成绩归给此文者（含经第三方表格转载的），都必须先回到这一条查真实出处。 |
| 2 | Pes, L., Yin, B., Stuijk, S., & Corradi, F. (2025). Traces Propagation: Memory-Efficient and Scalable Forward-Only Learning in Spiking Neural Networks. *arXiv preprint* arXiv:2509.13053. | https://arxiv.org/abs/2509.13053 | metadata-error | ⚠️ **数据集名称错误**：计划书正文写"在 MNIST 和 SHD 上"，原文是 **N-MNIST**（事件相机数据集），二者不是一回事；**原文从未把 MNIST 用作评测数据集**（该词出现 3 次：摘要的 `NMNIST`，以及 §3.1.1 说明 N-MNIST 的来源）。⚠️ **出处不完整**：该文已发表于 *Neuromorphic Computing and Engineering* **6(1):014002 (2026)**，DOI 10.1088/2634-4386/ae2ef9，建议补上期刊版。另：**「按突触存储导致 O(N²)」这一条成立，而且原文单独给出了 e-prop 的空间复杂度**——Table 3 有独立的 `E-prop [2]` 行、Space Complexity = `LH²`，§1.3.1 还点名 e-prop 作为该族属性的实例。**本条此前记的「属引申」是错的，已撤回**（当初只读了散文：v2 §2.3 那句只点了 OSTTP；v1 那句更明确，说 ETLP 与 OSTTP「based on E-prop [2] and OSTL [8]」）。裁决与逐格证据见 `docs/adr/ADR-0010`。 **2026-09-25 补核（SHD 数字的出处）**：其 Table 1 的 SHD 区块里有**两行 e-prop**——前馈 LIF 450 = **63.04%**、循环 LIF 450 = **80.79%**——脚注逐字写着「1 Results from [10].」。⚠️ **该脚注不止标在 e-prop 那两行上**——按表格结构逐格取，SHD 区块挂 `1` 的是 **5 行**（`eProp [2]` 前馈 / `DECOLLE [12]` 前馈 / `eProp [2]` 循环 / `ETLP [10]` 循环 / `DECOLLE [12]` 循环；N-MNIST 区块另有 2 行）。也就是说 [2] 是把**一整块对比数字**从 [10] 转过来的，e-prop 两行在其中——所以它们既不是 [1] 自报（[1] 里 `SHD` 零命中，见上条），也不是 [2] 自己跑出来的。同表 SHD 行中除本文自己的前馈行是 400 神经元外，其余均为 **450 神经元 / 100 时间步 / 100 epoch**；正文另述「for all fully connected architectures, a batch size of 128 is used」。⚠️ 该文全文搜不到 `Adam` 或 `optimizer`。 |
| 3 | Frémaux, N., Sprekeler, H., & Gerstner, W. (2010). Functional Requirements for Reward-Modulated Spike-Timing-Dependent Plasticity. *Journal of Neuroscience*, 30(40), 13326-13337. | https://www.jneurosci.org/content/30/40/13326 | verified | 作者、卷期页全部吻合，无署名错误。⚠️ URL 对浏览器正常，但对脚本化客户端返回 **403**（Cloudflare）——CI 的 lychee 已用 `--accept ...403` 放行，这是预期行为不是死链。**2026-09-23 逐字复核（PMC 全文 PMC6634722）**：本条此前记「5 条归因中 4 条确认」，但**没写是哪一条未确认**，而状态列给的是 `verified`（该状态的定义是「归因数字都能在原文找到」）——两处自相矛盾，那个「4/5」不可追溯、已作废。计划书 §3.2 归因给该文的**三句话逐字全部命中**：① 「~25%（σR）」← 原文 "Figure 2A shows that success offsets of a magnitude of ∼25% of the SD (σR) of the success signal are sufficient to prevent R-STDP from learning a target spike train in response to a given input spike pattern."；② 「S̄ < −0.4σR 时学习后低于学习前」← 原文 "Moreover, for a success offset S̄ < −0.4σR (i.e., the average success signal is negative) (Fig. 2A, green points), the performance after learning is even below the performance before learning (Fig. 2A, dotted horizontal line). Hence, R-STDP not only fails to learn the task, but sometimes even leads to unlearning of the task."；③ 「无法通过调整 STDP 窗口参数或更换权重依赖模型解决」← 原文 "The strong sensitivity of R-STDP to success offsets is not a property of this particular model of R-STDP, but rather a general one. Performance remains just as low for a weight-dependent model of STDP (van Rossum et al., 2000) (Fig. 2D) and cannot be increased by altering the balance between pre-before-post and post-before-pre windows in STDP (Fig. 2E)."⚠️ 但 §3.2 的一句**中文转述过读**——见下方待更正表（v6.3 已按原文改写）。 |
| 4 | Pogodin, R., & Latham, P. E. (2020). Kernelized information bottleneck leads to biologically plausible 3-factor Hebbian learning in deep networks. *Advances in Neural Information Processing Systems*, 33. | https://proceedings.nips.cc/paper/2020/hash/517f24c02e620d5a4dac1db388664a63-Abstract.html | partial | 三因子结构、第三因子无需自顶向下传递、需要除法归一化——三条均确认。⚠️ **"不遭受深度表征瓶颈"未在原文中找到**（`not_found`）：原文的说法是该规则族**避免了**深度网络的表征瓶颈问题，与"不遭受"是不同强度的表述，建议按原文改写。 **2026-09-23 逐格复核（arXiv:2006.07123 的 HTML 全文，附录 Table 3/4/5）**：W1 的消融对照要逐臂对应论文的列，而仓库此前**没有**这几列的取值记录，等于无据可依。现逐格取回。列结构（Table 3 与 Table 4 同构）为 `backprop×2 \| last layer×2 \| pHSIC: cossim×3 \| pHSIC: Gaussian×3`，子列依次是 `[—, div] [—, div] [—, grp, grp+div] [—, grp, grp+div]`——"—"是该项目的基础变体，对 pHSIC 而言即既无分组也无除法归一化。MNIST 行按「**准确率（Table 4，5 种子均值）/ 极差（Table 5）/ η_l / c^k**」记：Gaussian plain **94.6 / 0.2 / 0.6 / ——**、Gaussian grp **98.4 / 0.3 / 1.0 / 32**、Gaussian grp+div **98.1 / 0.2 / 1.0 / 32**、cossim plain **94.9 / 1.4 / 0.5 / ——**、cossim grp **95.8 / 0.5 / 0.6 / 16**、cossim grp+div **96.3 / 0.6 / 0.4 / 16**、backprop **98.6 / 0.2 / —— / ——**、last layer **92.0 与 95.4 / 0.3 与 0.3**。⚠️ 论文自身的一处排版错位在此一并确认：Table 3 的 `c^k` 行在 backprop 与 last layer 那两列印着 16（CIFAR10 区块的那两列印的是 32），可这两列没有核、本不该有 `c^k`——与 CIFAR10 那处同源。⚠️ HSIC 那一组**没有任何数值**：§5.1 与附录 D.8 只给定性结论，逐字为 "Optimizing HSIC instead of our approximation, pHSIC, didn't improve performance" 与 "training with HSIC instead did not lead to a significant change in the results (not shown)"。 |
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
| 15 | Bellec, G., Scherr, F., Subramoney, A., Hajek, E., Salaj, D., Legenstein, R., & Maass, W. (2019–2020). eligibility_propagation: 论文 [1] 的官方实现。*GitHub Repository*. | https://github.com/IGITUGraz/eligibility_propagation | verified | **许可证核验**：LICENSE 正文是标准 **BSD-3-Clause**（源码保留声明、二进制再现声明、不得背书；GitHub API 标 `NOASSERTION` 只因版权头格式非标准，不是许可证不明）。与 Apache-2.0 兼容。**仅作阅读参考与交叉验证，不进依赖**（与 [14] SpikingJelly 同样的处理）。本次核对了它的动力学（`EligALIF.__call__`：输入电流上无 `(1−α)` 因子、复位项用基阈值而脉冲判据用适应后阈值）、式 (25) 的资格痕迹实现、以及它的 `numerical_verification_eprop_factorization_vs_BPTT.py` |

| 16 | BEKO2210. (2026). Javis: An associative SNN memory co-processor for LLM agents. *GitHub Repository*. | https://github.com/BEKO2210/Javis | verified | **同类开源先例（只读参考，不进依赖）**。定位：把 LLM agent 的记忆做成脉冲网络里的**细胞集群**，查询当部分线索、靠模式补全重新激活集群，只把少数概念喂给 LLM，自称 token 比朴素 RAG 省 35–45%。学习规则全是 STDP 家族（pair / iSTDP / 三相 / 奖励调制 STDP / BCM / SFA / 结构可塑性…），**没有 e-prop、没有反向传播**；纯 Rust，不依赖任何 SNN 框架。⚠️ **许可证是 PolyForm Noncommercial 1.0.0**——非 OSI 认可、非 SPDX 标准，附研究用途 addendum（禁止生产部署与对外服务）：**任何代码都不能并入本仓库**，处理方式与 [15] 相同（只读、只引结论、不进依赖）。⚠️ **同名混淆**：GitHub 上叫 Javis 的还有 Julia 动画库 `JuliaAnimators/Javis.jl`（**MIT**）、`JavisVerse/JavisGPT`、`JavisVerse/JavisDiT` 等——许可审计时极易把 MIT 安到它头上。⚠️ **无配套论文/DOI**；0 star、2026-05 后停更；README 里的数字（自召回 100%、token 省 35–45%、联想召回约 2%、容量约 50 概念）全部出自作者自建小语料基准、**未经第三方复现**，只能当「同类先例」引用，不能当外部证据 |
| 17 | Korcsák-Gorzo, A., Espinoza Valverde, J. A., Stapmanns, J., Plesser, H. E., Dahmen, D., Bolten, M., van Albada, S. J., & Diesmann, M. (2025). Event-driven eligibility propagation in large sparse networks: efficiency shaped by biological realism. *arXiv preprint* arXiv:2511.21674. | https://arxiv.org/abs/2511.21674 | verified | **e-prop 的事件驱动实现**（不是「事件优先级」）：把「每个时间步同步更新」改成「突触收到脉冲事件时才更新」，接进 NEST，在模式生成 / 证据累积 / N-MNIST 上复现原版性能，弱/强 scaling 到 **2 百万神经元**。为遵守**严格局部性**还改掉了原版一处违例——资格痕迹滤波器依赖输出神经元的时间常数（原文：for synapses to compute their weight updates, they must know the time constant of the output neuron, which violates the principle of locality），这一句对「纯局部」的叙事最有用。⚠️ 全文 `priorit*` **零命中**，该文**没有**「事件优先级」这一机制，勿混。⚠️ 姓氏拼写：arXiv 与出版社一律写 `Korcsak-Gorzo`（不带重音），文献通行拼法为 Korcsák-Gorzo；⚠️ 同名混淆：粒子物理界的 Katherine Korcsak-Gorzo 是另一个人。⚠️ 其博士论文（RWTH Aachen 2025，D 82，CC BY 4.0）第 3 章即本工作全文，作者贡献页写 Submitted to Nature Computational Science，但截至 2026-09-23 未见期刊正式版 |
| 18 | Millidge, B. (2025). Generalizing E-prop to Deep Networks. *arXiv preprint* arXiv:2512.24506. | https://arxiv.org/abs/2512.24506 | verified | **纯数学笔记，无任何实验**：把 e-prop 的资格痕迹递归从「单层循环网络」推广到任意深度网络乃至任意 DAG，论证复杂度随深度保持线性。作者自己在 Discussion 里写着 we have performed no experiments demonstrating that good credit assignment across depth works in practice——**不得用它支撑任何性能数字**。它自列的局限反而是本项目在线学习主张的反证材料：e-prop does not actually perform online weight updates（要等 episode 结束才更新）、每组参数各存一套痕迹在深层 quickly unmanageable。⚠️ 题名拼写两版并存：论文 PDF 与 arXiv HTML 用 `Generalizing`，arXiv 元数据 / abs 页 / OpenAlex 用 `Generalising`；本表按论文本体登记并注明差异。⚠️ 与 Predictive E-prop **无作者关系**（后者作者里没有 Millidge，也没引这篇） |
| 19 | Noè, D., Yamamoto, H., Katori, Y., & Sato, S. (2026). Predictive E-prop: A biologically inspired approach to train predictive coding-based recurrent spiking neural networks. *bioRxiv preprint* 2026.02.12.705507. | https://doi.org/10.64898/2026.02.12.705507 | verified | **e-prop 的预测编码变体**：把「第三因子」从任务专用外部信号换成**预测编码自身的局部预测误差**。作者自我定位逐字：We term the resulting model Predictive Eprop, emphasizing its role as **a learning principle rather than a task specific model**。两个动力学系统（正弦极限环、Lorenz）上的三个任务与 truncated BPTT 相当（p > 0.05），但收敛 epoch 少 **70%**（约 23 vs 约 80）；σ_in ≤ 0.2 时性能不显著劣化。⚠️ **全文 0 次 metacognition / forgetting / working memory**——计划书 v6.2 把它挂在第四阶段「元认知门控」名下属误引，v6.3 已改。⚠️ 许可 `cc_no`（All rights reserved），**图不能复用**。⚠️ 近名混淆：同组 2025 年另有一篇 e-prop 论文（*Neuromorphic Computing and Engineering* 5(4) 044002，主题是连接度与内禀噪声分离），勿互记结论；Ororbia 的 spiking neural predictive coding 是另一条线 |
| 20 | Graf, L., Su, Z., & Indiveri, G. (2024). EchoSpike Predictive Plasticity: An Online Local Learning Rule for Spiking Neural Networks. *arXiv preprint* arXiv:2405.13976. | https://arxiv.org/abs/2405.13976 | verified | **ESPP = EchoSpike Predictive Plasticity**：把「上一个样本的整段脉冲活动」当预测目标（echo），同标签拉近、异标签推远；预测编码 + 对比编码式的**层间局部规则**，全程不用自动微分。SHD 上 84.32%（自报）。**「事件优先级」这个中文提法的原文出处就在 §III-C**：用输入活动阈值 + 损失阈值自己决定哪些时间步才更新权重，原文 ESPP intrinsically has the ability to selectively choose those time steps that matter the most，实测只执行 **18%–27%** 的时间步且随训练递减。⚠️ 它是**另一条规则**，不是 e-prop 的变体（同 ADR-0010 对 TP 的处理），别写进 §3.1 的 e-prop 链。⚠️ 论文脚注的 `largraf/EchoSpike` **已 404**，权威仓库是 [Zhe-Su/ESPP](https://github.com/Zhe-Su/ESPP)（**Apache-2.0**，但 LICENSE 的版权人一行仍是模板占位符）；⚠️ 「ESPP」是高频缩写（员工购股计划 / 欧洲粒子物理战略 / 欧洲可持续磷平台 / Espressif 组件库），SNN 语境下才是这一个 |
| 21 | Frenkel, C. (2022). eprop-PyTorch: PyTorch implementation of the eligibility propagation (e-prop) learning algorithm. *GitHub Repository*. | https://github.com/ChFrenkel/eprop-PyTorch | verified | 计划书 §1.4/§八 点名的「e-prop 实现参考」。**Apache-2.0**（版权人 University of Zurich；LICENSE 是未改动的原版模板，Appendix 仍是 `Copyright [yyyy] [name of copyright owner]` 占位符）。**非官方**——官方是 [15]；**非 PyPI、无 tag/release**，只能锚 commit `0f32a8f2`（2022-02-18 单一提交，全仓 7 个文件）。**只有 LIF 且已显式移除 ALIF**（`main.py` 写着 Support for the ALIF neuron model has been removed.），任务只有证据累积一种；**无依赖清单 / 无 CI / 无测试**，`setup.py` 两处 `np.int` 在 NumPy ≥ 1.24 已报错。⚠️ 因此 **ALIF 的对照物只能用 [15] 与 [1] 原文**——ADR-0008 决策 3 里「拿它的源码核对 ALIF 资格痕迹」一句已在该 ADR 的后续更正注记里撤回 |
| 22 | Bellec, G., Salaj, D., Subramoney, A., Legenstein, R., & Maass, W. (2018). Long short-term memory and learning-to-learn in networks of spiking neurons. *Advances in Neural Information Processing Systems*, 31 (NeurIPS 2018). | https://arxiv.org/abs/1803.09574 | verified | **sMNIST 上可比的「同族基线」的出处——但它不是 e-prop**。本条目的意义是：e-prop 原文 [1] 没有 sMNIST，同实验室把 sMNIST 做掉的是这篇 LSNN 论文，而它的训练算法是 **BPTT + DEEP R**。**2026-09-25 逐字复核**（实际抓取 arXiv PDF，4,339,505 字节，`pdftotext -layout` 抽文）：正文逐字 "A performance comparison is given in Fig. 1B. LSNNs achieve **94.7%** and **96.4%** classification accuracy on the test set when every pixel is presented for 1 and 2ms respectively. An LSTM network achieves 98.5% and 98.0% accuracy on the same task setups." ⚠️ **这两个数都是 `max`，不是均值**——Table S1（1 ms）`LSNN 100(A), 120(R) 12% 8185 (full 68210) 12 94.2% 0.3% 94.7%`（12 次运行，均值 94.2 ± 0.3，max 94.7）；Table S2（2 ms）同行 `12 93.8% 5.8% 96.4%`（均值 93.8 ± 5.8，max 96.4）。同表另有单次运行的大网络：`LSNN 350(A), 350(R) 12% 66360 (full 553000) 1 - - 96.1%`（1 ms）/ `- - 97.1%`（2 ms），**# runs = 1，无标准差**。⚠️ **LSTM 对照极不稳定，不可当「论文水平」引用**：1 ms 档 `LSTM 128 100% 67850 12 79.8% 26.6% 98.5%`（均值仅 79.8、std 26.6），2 ms 档 `12 48.2% 39.9% 98.0%`（均值 48.2）。纯脉冲 LIF 220 对照为 60.9/63.3（1 ms）与 34.6/51.8（2 ms）。⚠️ 官方实现 `IGITUGraz/LSNN-official` 的 README 说「achieve above **96%** accuracy on the sequential MNIST task」——与上表 2 ms 稀疏档的 max 96.4% 对得上，**这是同一件事的仓库说法，不是第二个独立来源**。 **反向核查（2026-09-25）：在我查过的引用文献里，也没有任何一篇把 sMNIST 数字归给 [1]。** 两个样本。(a) **BNTT**（Kim & Panda, *Front. Neurosci.* 15:773954, 2021；PMC8695433）的 Table 3 标题逐字为 `Classification accuracy (%) on sequential MNIST`，表内三行逐字为 `LIF (Bellec et al., 2018) 63.3` / `LSNN (Bellec et al., 2018) 93.7` / `DEEP R LSNN (Bellec et al., 2018) 96.4`——**三行全归 2018**；该文全文 `17236` / `3625` / `e-prop` / `learning dilemma` **各 0 次**，**它根本不是 [1] 的引用文献**。⚠️ 顺带可见转抄已经失真：这三个数就是本条目 **2 ms** 档的值，而转抄把均值 `93.8` 写成 `93.7`、把 max `96.4` 与均值并列。(b) **DelayNet**（Balafrej et al., arXiv:2310.19067）**确实引用 [1]**（在该文参考文献里是 [6]），但它的对比表 Table 1 标题逐字为 `Comparison of test accuracy on psMNIST`——是 **psMNIST，不是 sMNIST**（全文独立出现的 `sMNIST` **0 次**，两处命中都只是 `psMNIST` 的词尾）；表内 `SRNN [4] 63.3` / `LSNN [4] 93.3` / `LSNN + Deep-R [4] 94.7` 三行的 [4]，按该文参考文献是 Bellec et al. 2018（*NeurIPS* 31, pages 787–797）——**不是 [1]**。 |

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

#### 核验用的全文副本存在哪里

上表凡标 `verified` 的"逐字复核"，都基于一份本地留存的全文副本。这些副本**不入库**——第三方论文全文的
再分发受各自许可约束（[3] 是 J Neurosci 的 all-rights-reserved，[19] 是 `cc_no`）。按与 §12.1
"不提供数据镜像、只提供获取方式"相同的原则，这里登记**来源与指纹**：

| 引用 | 来源 | 本地副本 | 提取方式 |
| :--- | :--- | :--- | :--- |
| [1] Bellec et al. 2020 | `nature.com/articles/s41467-020-17236-y` | 97,228 B · sha256 `6a02005749ee0949…` | 出版社 HTML 去标签 |
| [3] Frémaux et al. 2010 | `jneurosci.org/content/30/40/13326` | 90,510 B · sha256 `86e2fe258282f113…` | 同上（另与 PMC6634722 交叉核对） |
| [4] Pogodin & Latham 2020 | `arxiv.org/abs/2006.07123` | 125,749 B · sha256 `c860d802e48557a0…` | arXiv LaTeXML HTML **逐格解析** |
| [19] Noè et al. 2026 | `biorxiv.org/content/10.64898/2026.02.12.705507v1.full` | 52,690 B · sha256 `5255b7fda6ed6e5e…` | 出版社 HTML 去标签（首抓被 Cloudflare `error code: 1015` 拦下，2026-09-25 重抓） |

三点必须写明，否则这些指纹会被读成比实际更强的东西：

1. **不是 PDF 文本流。** [4] 的 Table 3/4/5 用 `pdftotext -layout` 读会串行错位（本仓库在这上面栽过一次），
   所以表格走的是 HTML 逐格解析。
2. **本地副本未必是产生上表那些数字的同一份提取。** 例：[1] 的备注写"正文去标签后 146,503 字符"，
   而本地这份 `.txt` 是 97,002 字符——两者不是同一个产物（很可能一个只含正文、一个含参考文献）。
   所以**哈希标识的是本地这份文件，不是上表结论的唯一依据**；结论的依据是备注里那些**逐字引文本身**，
   它们可以脱离副本、直接回到公开来源核对。
3. **"逐字"本身也可能在细微处不逐字。** [19] 的作者自我定位那一句，原文写的是 `Predictive Eprop`——
   **没有连字符**；带连字符的 `Predictive E-prop` 是论文标题和全文其余 42 处的写法。台账初版把这处引成了
   带连字符的形式，直到 2026-09-25 重抓副本、逐字比对才发现。同一次比对覆盖了三语台账里标了逐字的英文引用
   （副本能比对的共 8 条），**除这一条外全部命中**。引文里的连字符、上下标、希腊字母要照抄原文，
   不要按习惯规范化——否则"逐字"二字就不再成立。

## 计划书正文待更正项

以下问题属于 **`BioSNN-Plug_项目计划书_v6.2.md` 正文**，本文件改不了它。**2026-09-23：这些更正已全部并入 [`BioSNN-Plug_项目计划书_v6.3.md`](../BioSNN-Plug_项目计划书_v6.3.md)**——本表保留作为改动记录，**引用计划书时以 v6.2 为准；需要更正后的表述时用 v6.3**。

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
| §三、§八 选型表、§十一 风险表、§3.1 关键修正 1、**§6.2、§十、§12.4、§十三** | 把 Trace Propagation 写成「e-prop 的存储优化：O(N²) → O(N)」 | 按 [ADR-0010](adr/ADR-0010-eprop-quadratic-storage-and-trace-propagation.md) 改写：TP 是**另一条规则**（痕迹按神经元而非按突触存储），不是 e-prop 的省内存版本；两者并列而非替代。同时删去 §3.1 把它列为本条补救措施的表述。**2026-09-25 补记**：本行原先只列了前四处，v6.3 的第一遍更正因此**漏掉了 §6.2 的清单项与显存约束声明、§十 风险表两行、§12.4 的 ADR 举例、§十三 的创新点行**——那五处当时仍在教读者「上 TP 换内存」。现已一并改掉；**§6.2 的显存约束声明同时做了重算**（原声明的两个数漏乘了批大小那一维，推导见 v6.3 §6.2）。教训：更正一处**定位错误**时，要按"这个说法还在哪里出现过"全文搜一遍，而不是按当初列出的位置清单逐条改 |
| §12.1、§1.4、§八 | 许可证表把 SpikingJelly 写成"与本项目同一许可证（Apache-2.0）" | 按事实改写：SpikingJelly 用**启智开源许可证 1.0**（OIOSL），并补上其商业使用披露义务。§1.4/§八 的技术选型行注上"该依赖的许可条款，以及它对开发环境 Python 下限的影响，见 `docs/adr/ADR-0008`" |
| §七 第四阶段任务行 | 把 **Predictive E-prop** 与「元认知门控」并列 | 该文全文 **0 次** metacognition / forgetting / working memory，挂在这里是误引。v6.3 已改：元认知门控保留（依据是本项目自己的 §2.2/§3.4），Predictive E-prop 按论文自定位（a learning principle）移入 §3.1 的 e-prop 变体，任务行补入**事件优先级**（ESPP 的选择性时间步更新）|
| §3.1、§八 | 「事件优先级」若被写成某篇论文的术语 | 2025–2026 脉冲网络文献里没有这个术语（七篇逐字核过，`priorit*` 命中 0/0/0/1/0/0/0）。v6.3 已按 ESPP 的机制写，并注明「事件优先级」是本项目的中文转述而非文献术语 |
| §3.2 | 「R-STDP 不仅无法学习，还会导致**遗忘已学会的技能**」（中文转述） | 原文 Figure 2A 的基准是**未训练**的均匀权重，现象是「学习后的表现跌到学习前之下」，原文自用词为 "unlearning of the task"（该文对它的定义："If the network performs worse than this level after learning, it has effectively unlearned."）。应改成「学习后的表现**跌到未训练水平之下**」，**不要写成「遗忘已学技能」**——那是灾难性遗忘，Figure 2A 不支撑。v6.3 §3.2 已按此改写 |

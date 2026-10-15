# 2026-10-15 协议级标准专项检索

> 这是 [`2026-10-14-local-learning-landscape.md`](2026-10-14-local-learning-landscape.md) 那份触发档案所要求的**重跑**：
> 带完整日志、可被第三方照着复现的一次检索。它比 10-14 那份强的地方只有一处——**过程可审查**；
> 它比 10-14 那份弱的地方也有一处，见第四节：**这次检索找到了近邻，"没有对应物"这句话因此必须收窄**。

## 一、要回答什么，以及判据

目标断言（`paper/main.tex` §2.4）：

> To our knowledge no equivalent protocol-level standard exists for local-rule SNN training.

判据沿用 10-14 那份，三条**同时**满足才算命中：

1. 它是**规范**（清单、政策、投稿必填项），不是某篇论文自己那套做法；
2. 它针对**局部规则 SNN 的训练**（放宽一点：SNN 训练）；
3. 它规定**一次报告必须包含什么**（预算、种子数、优化器、评测协议……）。

## 二、检索记录（逐字）

**日期**：2026-10-15（全部条目均在这一天取得）。
**源**：本 agent 可用的 `WebSearch` 索引（**美国区索引**，见第五节局限）与对具体 URL 的抓取。
**计数口径**："返回"是链接条目数（同一查询内不同区块会重复同一条，故为约数）；"纳入"是进入
第三节判定的条数。

| # | 源 | 检索式 / URL（逐字） | 返回 | 纳入 | 处置 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| S1 | WebSearch | `reproducibility checklist spiking neural network training standard` | ≈10 | 2 | 纳：TechRxiv 的 SNN 实践指南、Pauli et al. 2018。其余多为**论文自带的 NeurIPS checklist 页**（是投稿模板的产物，不是规范本体） |
| S2 | WebSearch | `neuromorphic benchmarking standard checklist NeuroBench reporting protocol` | ≈10 | 1 | 纳：NeuroBench。其余为其不同镜像版本 |
| S3 | WebSearch | `local learning rules spiking neural networks benchmark comparison protocol budget seeds reporting` | ≈10 | 2 | 纳：NeuroSeqBench、STEP。其余为单篇论文 |
| S4 | WebSearch | `"spiking neural network" benchmark reporting requirement multiple seeds training budget local learning rule standard` | ≈10 | 2 | 纳：arXiv:2402.01782（变局部性基准）、NeuroEval。其余为单篇论文自带的 checklist 页 |
| F1 | WebFetch | `https://www.techrxiv.org/doi/full/10.36227/techrxiv.175322635.58911591/v1` | — | 0 | **403 Forbidden**，未能核。S1 命中它，但内容没读到，**不得**据标题下结论 |
| F2 | WebFetch | `https://arxiv.org/abs/2304.04640`（NeuroBench） | — | 1 | 取到摘要级信息 |
| F3 | WebFetch | `https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2018.00046/full` | — | 1 | 取到全文级要点 |
| F4 | WebFetch | `https://github.com/liyc5929/neuroseqbench` | — | 1 | 取到 README 级信息 |

**纳入/排除判据**：与"报告该怎么写"无关的条目一律排除——单篇论文里出现的 NeurIPS checklist
页、代码仓库的顺手引用、与可复现性无关的 SNN 论文，都不进第三节。**排除不等于不存在**：
它们只是不构成判据 1 意义上的"规范"。

## 三、命中项的逐条判定

| 命中 | 是什么 | 判据 1 规范？ | 判据 2 局部规则 SNN 训练？ | 判据 3 规定报告必含什么？ |
| :--- | :--- | :--- | :--- | :--- |
| NeuroBench（Yik et al.，100 位作者，arXiv:2304.04640 / Nature 合集，2023–2025） | 神经形态计算的基准框架，分 algorithm track（硬件无关）与 system track；给任务、指标与基线 | 半：框架 + 指南，但**不是**投稿强制项 | ✗：按系统/任务组织，**摘要里不涉及** e-prop、STDP、Hebbian 这类学习规则 | 半：规定它自己那套任务要报的指标（准确率、时延、能耗），**不含**训练预算、种子数、优化器 |
| NeuroSeqBench（liyc5929，GPL-3.0，24 star） | SNN 时序任务的基准套件，**明确把 `E-prop` 与 STBP/OTTT/SLTT 并列比较**，并称提供"保证公平一致评估"的指南 | ✗：代码套件 + 指南 | ✓：这是本次检索里**判据 2 最接近命中**的一条 | 半：报准确率、PPL、参数量、时延、训练速度、显存、能效，并接 NeuroBench 指标；**可见内容未规定种子数或固定预算** |
| arXiv:2402.01782《Benchmarking Spiking Neural Network Learning Methods with Varying Locality》 | **按局部性分层**比较 SNN 学习方法的基准论文 | ✗：是一篇论文 | ✓：主题几乎正对 | ✗：论文自己那套做法 |
| STEP（OpenReview） | 脉冲 Transformer 的"公平可复现基准"评估平台 | ✗：平台 | ✗：面向 Transformer 训练，不是局部规则 | 半：平台内建，非对外规范 |
| NeuroEval（Zenodo 报告） | 用 LLM 评估论文的数据透明度、可复现性与神经编码严谨性的**元评估**框架 | ✗：评估工具与方法 | 半：以计算神经科学论文为对象 | 半：给评分项，不是投稿要求 |
| Pauli, Weidel, Kunkel & Morrison 2018（*Front. Neuroinform.* 12:46） | 脉冲网络**模型**可复现性的最佳实践指南（以 polychronization 复现为例） | ✗：指南，非规范 | ✗：对象是**仿真模型**的可复现性，不是训练基准 | 半：建议共享代码、版本控制、记录软件版本、多次运行；⚠️ 它自己报告了**种子敏感**（87% vs 13% 落到不同动力学态）——这条**支持**我们的主张，而非反驳 |
| Pineau et al. 2021（*JMLR* 22(164)，台账 [27]） | ML 领域的协议级做法：代码提交政策、复现挑战、投稿必填清单 | ✓ | ✗：针对 ML 整体 | 半：规定报告要素，但不含局部规则训练的特有项 |

## 四、结论：没有命中的是什么，以及这句话现在该怎么说

**三条同时满足的条目：0 条。** 最接近的是 **NeuroSeqBench**（判据 2 ✓、判据 3 半）与
**NeuroBench**（判据 1 半、判据 3 半）——两者都是"基准套件 + 自家指南"，没有一条是**投稿强制
或社区采纳的规范**，也没有一条规定局部规则训练该报的预算、种子数与优化器敏感性。

因此**这句话可以升级，但升级后的句子必须点名近邻**，不能只说"找到了 0 条"：

> A search of the sources and query strings recorded in Appendix X on 2026-10-15 returned no
> protocol-level standard for local-rule SNN training. The nearest work is benchmark-oriented:
> suites that fix tasks and metrics for comparing SNN learning algorithms, including e-prop,
> and reproducibility guides aimed at spiking model *simulation*; none of them prescribes the
> budget, seed and optimizer reporting this paper controls.

⚠️ **升级尚未落地，原因有两条，都不是技术性的**：

1. 那个句子要点名近邻，点名就要**引用**（NeuroBench、NeuroSeqBench、Pauli 2018 至少三条）。
   而 `paper/refs.bib` 顶部的规则是：**本文件不新增任何仓库里没有的引用**——它们得先登记进
   [`docs/references.md`](../references.md) 台账并核验元数据，这是需要单独裁决的一步。
   本文件记下它们**不等于**已经登记：本文件里的这几条**都不是台账条目**。
2. "a search of [源] on [日期] with [检索式]" 这种写法要把检索规格写进论文（附录一节），
   否则审稿人无从核对——那是 §2.4 与附录的结构性改动，不是换一句话。

**在这两条完成之前，§2.4 保持防御式措辞。** 10-14 那份档案的结论不变。

## 五、局限，以及别人怎么重跑

**覆盖范围不是穷举。** 本次检索用的是本 agent 可及的 `WebSearch` 索引（**美国区**）与逐个 URL
的抓取，**没有** ACM DL / IEEE Xplore / Scopus / Web of Science 的库内检索权限，也没有
Google Scholar 的完整分页。所以本文件支撑的是"**用下列检索式、在这一天、在这个索引上，没有
找到**"，**不是**"该领域不存在"。任何"exhaustive / systematic review"的说法都不成立。

**重跑方式**：照抄第二节每一行的检索式与 URL，在任意搜索引擎或数据库重跑，逐条对照第三节的
纳入/排除判据。判据写在第一节，就是为了让重跑者能得出**与我不同**的结论。

**已知的未竟项**：F1（TechRxiv《Practical Guidelines for Spiking Neural Networks》）返回 403，
内容未读。它是标题上最像"规范"的一条，**值得下一次优先核**——若它其实规定了报告要素，第三节的
结论要改。

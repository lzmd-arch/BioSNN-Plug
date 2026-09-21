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

## 为什么较真

计划书 §12.3 把"引用链接存活检查"的用途写得很具体：它是拦截 **"Khacef 署名"类
错误的入口**。那个错误在本仓库真实发生过——参考文献 [8] 的作者被误署为
"Khacef et al."，v6.1 才更正为 Hajizada et al.。

一个署错的作者、一个指向别处的 DOI、一个对不上的数字，都会让整篇论述的可信度
打折。对一个以"公开可复现"为核心主张的项目（§1.2 第 5 条），这是最贵的错误。

## 核验状态

| 状态 | 含义 |
| :--- | :--- |
| `verified` | 逐条核验过：文献存在、元数据正确、正文归因于它的**具体数字**能在原文找到 |
| `partial` | 文献存在且元数据正确，但部分归因数字未在原文中确认 |
| `unverified` | 尚未核验。**对外引用前必须先升到 `verified` 或 `partial`** |

## 清单

| 编号 | 文献 | URL | 核验状态 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Bellec, G., Scherr, F., Subramoney, A., Hajek, E., Salaj, D., Legenstein, R., & Maass, W. (2020). A solution to the learning dilemma for recurrent networks of spiking neurons. *Nature Communications*, 11, 3625. | https://www.nature.com/articles/s41467-020-17236-y | unverified | 计划书 §3.1 的 e-prop 三因子规则、ALIF 资格痕迹推导出处 |
| 2 | Pes, L., Yin, B., Stuijk, S., & Corradi, F. (2025). Traces Propagation: Memory-Efficient and Scalable Forward-Only Learning in Spiking Neural Networks. *arXiv preprint* arXiv:2509.13053. | https://arxiv.org/abs/2509.13053 | unverified | 计划书 §3.1 关键修正 1：资格痕迹存储 O(N²) → O(N) |
| 3 | Frémaux, N., Sprekeler, H., & Gerstner, W. (2010). Functional Requirements for Reward-Modulated Spike-Timing-Dependent Plasticity. *Journal of Neuroscience*, 30(40), 13326-13337. | https://www.jneurosci.org/content/30/40/13326 | unverified | 计划书 §3.2 的 ~25%σR 阈值与 < -0.4σR 导致遗忘；P0 风险项 |
| 4 | Pogodin, R., & Latham, P. E. (2020). Kernelized information bottleneck leads to biologically plausible 3-factor Hebbian learning in deep networks. *Advances in Neural Information Processing Systems*, 33. | https://proceedings.nips.cc/paper/2020/hash/517f24c02e620d5a4dac1db388664a63-Abstract.html | unverified | 计划书 §3.3 的核化 IB-Hebbian 与除法归一化 |
| 5 | Confavreux, B., Agnes, E. J., Zenke, F., Sprekeler, H., & Vogels, T. P. (2025). Balancing complexity, performance and plausibility to meta learn plasticity rules in recurrent spiking networks. *PLoS Computational Biology*, 21(4), e1012910. | https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012910 | unverified | 计划书 §3.4 的 ES 元学习仲裁器；四种突触类型 |
| 6 | Shen, J., Xie, Y., Xu, Q., Pan, G., Tang, H., & Chen, B. (2025). Spiking Neural Networks with Temporal Attention-Guided Adaptive Fusion for imbalanced Multi-modal Learning. *Proceedings of the 33rd ACM International Conference on Multimedia*. | https://dl.acm.org/doi/10.1145/3746027.3755622 | unverified | 计划书 §2.2 的 TAAF；第二阶段任务 |
| 7 | Hu, K., Wen, L., Zhang, T., & Zhang, H. (2026). PS-SNN: pattern separation learning for expandable spiking neural networks in class-incremental learning. *Scientific Reports*, 16, Article 42970. | https://www.nature.com/articles/s41598-026-42970-6 | unverified | 计划书 §5.2 的 76.42% 平均增量准确率 |
| 8 | Hajizada, E., Rager, D., Shea, T., Campos-Macias, L., Wild, A., Hüllermeier, E., Sandamirskaya, Y., & Davies, M. (2026). Online Continual Learning on Intel Loihi 2 via a Co-designed Spiking Neural Network. *arXiv preprint* arXiv:2511.01553. | https://arxiv.org/abs/2511.01553 | unverified | **v6.1 更正**：原误署为 Khacef et al.；计划书 §6.1 的 70× 延迟与 5,600× 能效 |
| 9 | Savage, W. (2026). EMBER: Autonomous Cognitive Behaviour from Learned Spiking Neural Network Dynamics in a Hybrid LLM Architecture. *arXiv preprint* arXiv:2604.12167. | https://arxiv.org/abs/2604.12167 | unverified | 计划书 §四 SNN 调用 LLM 协同模块的整个证据基础；**优先级最高的核验对象** |
| 10 | MEMBRAIN. (2026). Neuromorphic Memory Bridge for LLM Agents. *GitHub Repository*. | https://github.com/tfatykhov/membrain | unverified | 计划书 §4.3 的 FlyHash 编码参考；§4.4 的睡眠期巩固 |
| 11 | ASTRA. (2026). Unified Research Lab + MCP Server. *GitHub Repository*. | https://github.com/christophejlegros-lgtm/ASTRA-Unified-ResearchLab-MCP-v2.7 | unverified | 计划书 §4.6 的 MCP 服务封装先例 |

## 待办

### P0：TD-LTP 出处考证

计划书 §3.2 把 R-STDP 的奖励预测 Critic 的训练规则写作 **"TD-LTP"**，并在 §十一
把它列为 P0 待办：

> TD-LTP 文献考证——Critic 的训练规则是整个 R-STDP 闭环局部性声明的基石，
> **无出处则第五阶段不立项**。

考证结论落地前，本表不新增 TD-LTP 条目；结论出来后按结果处理：

- 若找到正式出处 → 作为第 12 条登记，标 `verified`；
- 若判定为泛称或误记 → 在计划书 §3.2 与 §十一 记录更正，并启用 §十一 的降级
  路径（备选：滑动平均值基线近似 Critic）。

### 核验流程

对每条引用，核验四件事：

1. **存在性**——URL / DOI / arXiv 编号可达；
2. **元数据**——作者、年份、期刊或会议、标题与登记一致；
3. **归属**——没有把 A 的结论记到 B 头上（[8] 的历史教训）；
4. **数字**——计划书正文里归因于它的具体数字（如 76.42%、82.2%、5,600 倍、
   0.05 mJ）能在原文中找到。找不到或对不上的，状态只能给 `partial`，
   并在备注里写明哪一项没对上。

**不要用记忆代替检索。** 核验不到就如实标 `unverified` 或 `partial`——
一份诚实的 `unverified` 比一份假的 `verified` 有价值得多。

# FiLM × SNN × LLM 参考核对

**核对日期**：2026-11-16
**被核对象**：一份外部整理的《FiLM 调制器与 SNN-LLM 融合方案整理》（8,373 字，7 节，
主张「0.5B SNN + 0.5B LLM + FiLM 桥接 = 1B 人格与记忆核心」没有完全相同先例）。

**核对方法**：逐个名字做公开网络检索，记录**能否定位到一手来源**。
**这不是同行评审核验**——"能查到"不等于"结论可靠"。查不到的一律标「未定位」，不推测。

## 结论先说

| 判定 | 数量 |
| :--- | ---: |
| 真实存在，且被正确描述 | 7 |
| 真实存在，但**文档的描述有实质错误** | 4 |
| 真实存在，但**不是研究文献**（商业包 / 博客 / 模型卡 / 数据集） | 8 |
| 未能定位 | 1 |

**最要紧的一条**：23 个里只有 **5 个**出自同行评审场合（ICLR / AAAI / TMLR / CVPR Workshop / MobiSys Poster）。
其余是 arXiv 预印本、Zenodo 自发布、HuggingFace 模型卡、知乎文章、npm 包。
**所以这份文档可以当调研线索，不能当文献综述引用。**

---

## 一、真实且描述正确

| 名字 | 是什么 | 出处 |
| :--- | :--- | :--- |
| **FiLM** | Perez et al., AAAI 2018，特征级线性调制 `γ·F+β` | 经典文献 |
| **Temporal FiLM** | Birnbaum et al., **NeurIPS 2019**，用 RNN 生成时变 γ/β 扩大感受野 | [arXiv:1909.06628](https://arxiv.org/abs/1909.06628) |
| **Nord v4.2** | 618.8M 纯 SNN 语言模型，从零训练、87–93% 稀疏 | [Zenodo 19487510](https://zenodo.org/records/19487510)、[GitHub](https://github.com/gtausa197-svg/-Project-Nord-Spiking-Neural-Network-Language-Model)、[HF](https://huggingface.co/zerdovzad/Nord-AI)（预印本，个人项目） |
| **SpikeWhale-SNN-216M** | 冻结虚拟细胞模型当 bridge，输出 `thr_gain/beta_gain/in_gain` | [HF Quazim0t0/SpikeWhale-SNN-216M](https://huggingface.co/Quazim0t0/SpikeWhale-SNN-216M)（模型仓，**无论文**） |
| **SpikeLLM** | Xingrun Xing et al., **ICLR 2025**，GIF 神经元 + Optimal Brain Spiking | [arXiv:2407.04752](https://arxiv.org/abs/2407.04752) |
| **SpikingLM** | Yu Liang et al., **AAAI 2025 Oral**，DAS + Spike2Max attention | AAAI 2025 |
| **SpikingMamba-1.3B** | Huang/Tang/Wang et al., **TMLR 2026-01**，从 Mamba 蒸馏，SI-LIF | TMLR |

## 二、真实存在，但文档写错了

| 名字 | 文档怎么说 | 实际是什么 |
| :--- | :--- | :--- |
| **FiSTA** | 「FiLM 与 **SNN** 结合用于动态时空特征提取，脉冲发放率降低约 34%」 | **它跟 SNN 没有关系。** FiSTA = FiLM-Modulated Spatio-Temporal Attention，实际发表用途是人形机械臂的肘部姿态预测（[HL-IK](https://hl-ik.github.io/static/pdf/HL-IK.pdf)）。文档把"SCTFA（脉冲时空注意力）"这条相邻工作误当成了 FiSTA。**"降低 34%" 无从核对。** |
| **ConscienceScience_Archon_25k** | 「25,000 条记录」 | 数据集存在（[WithinUsAI/Conscience_25k](https://huggingface.co/datasets/WithinUsAI/Conscience_25k)），但**实际约 1,147 条**。"25k"是名义名，不是行数。 |
| **SymbolicLight** | 「0.5B token 训练预算」 | 主模型是 **194M 参数、3B token**；0.5B token 只出现在**消融实验**的等预算对照里（[arXiv:2605.21333](https://arxiv.org/abs/2605.21333)）。两者不是一回事。 |
| **Time-FiLM（脉冲版）** | 「Time-FiLM 已用于脉冲神经编码等时序任务」 | Temporal FiLM 本身为真（见上），但**"脉冲神经编码版本（Ma et al. 2023）"只出现在二手综述页**（emergentmind），未找到一手论文。判为**未证实**。 |

## 三、真实存在，但不是研究文献

| 名字 | 实际形态 | 能不能当先例 |
| :--- | :--- | :--- |
| **Cortexia** | **npm 商业包**（闭源付费，周下载约 1），宣称 500K 脉冲神经元 + 6 脑区 + 神经递质 | ❌ 不能。它是产品宣传，不是可核验的方法 |
| **Genesis Brain** | **npm 商业包**（v9.0.1，约 220 周下载），但有 OSF 预注册实验（情绪一致性、人格持续性、神经调质状态） | ⚠️ 谨慎。有预注册这一点比 Cortexia 强，但仍是产品方自评 |
| **Nezha（哪吒）** | [Zenodo 18032595](https://zenodo.org/records/18032595) 自发布技术报告 v11.0（SNN System 1 + Active Inference System 2，海马体/杏仁核/神经内分泌） | ⚠️ 描述属实，但**自发布、无同行评审** |
| **EMBER** | [arXiv:2604.12167](https://arxiv.org/abs/2604.12167)，220,000 神经元 SNN + 可替换 LLM，STDP | ✅ 相对可靠（预印本），但作者自述 **N=1、消融为初步结果** |
| **VITA** | **知乎专栏文章**，115,200 神经元 SNN + 冻结 Qwen3-4B 当"语言皮层" | ⚠️ 项目真实，**无任何论文** |
| **arthemis-lm** | [HF rootxhacker/arthemis-lm](https://huggingface.co/rootxhacker/arthemis-lm)，155.8M Llama+SNN+LTC | ⚠️ 只有模型卡，无论文 |
| **HybridIntelligence-0.5B** | [HF Merlin-Research/HybridIntelligence-0.5B](https://huggingface.co/Merlin-Research/HybridIntelligence-0.5B)，Falcon-H1-0.5B + LIF 选择器 | ⚠️ 结果**全部自报**，未找到第三方评测 |
| **Neuromorphic-Memory-Architecture** | [GitHub Papapa1986](https://github.com/Papapa1986/Neuromorphic-Memory-Architecture-for-Conscious-AI-Systems)，"engram fragments" 当不可重写伦理约束 | ⚠️ 思辨性个人仓，**不可当技术依据** |

## 四、真实但需附注

| 名字 | 附注 |
| :--- | :--- |
| **Context as Threshold** | 是 [Lacuna 的 hypothesis 页面](https://lacuna.tiptreesystems.com/hypothesis/context-as-threshold-spiking-networks-whose-firing-rules-encode-biological-c6ae390947e2d494)，**不是论文**。数学本身对：`ReLU(γx+β)=γ·ReLU(x+β/γ)` ⇒ FiLM 等价于改阈值 `θ/γ` 与膜电位偏移 `β/γ`。**但只在 γ>0 时成立**，文档漏了这个前提。 |
| **Firebolt-VL** | [arXiv:2604.04579](https://arxiv.org/abs/2604.04579)，CVPR 2026 Workshop。**arXiv 标注与 2511.11177 有大量文本重叠**，引用前要看这一条。 |
| **CTP-Hybrid** | [Zenodo 20120983](https://zenodo.org/records/20120983) 自发布报告；20M 纯脉冲基线、无自注意力无 LayerNorm，这条描述**属实**。 |
| **EdgeSpike-LoRA** | MobiSys Companion '26 **Poster**（DOI `10.1145/3812835.3814881`），不是长文。 |
| **BrainTransformers** | [arXiv:2410.14687](https://arxiv.org/abs/2410.14687)，BrainGPTForCausalLM。文档未单列，属"SNN 化 LLM"这条路径。 |

## 五、未定位

| 名字 | 情况 |
| :--- | :--- |
| **SpikingLM 的"ICML 2026 扩展版"** | 文档未提，检索中出现，未能独立确认 |

---

## 对我们自己工作的三条直接影响

1. **"没有完全相同先例"这个新颖性主张站不住脚，也不需要它。**
   这份清单里真正同行评审的只有 5 个，而 **VITA 的做法（冻结 Qwen3-4B + SNN 读出调制采样）
   和我们 V1 的做法在结构上高度接近**。写对外材料时**不能声称"未被占据的生态位"**。

2. **文档描述的规模和我们建的差三个数量级。**
   文档是「0.5B SNN + 0.5B LLM」；V1 是**冻结 Qwen2.5-0.5B + 12 维状态向量 → FiLM**，
   状态来自几个 16 神经元的池子。**这份文档不是 V1 的设计依据。**

3. **要引用就引这 5 个**：FiLM（AAAI 2018）、Temporal FiLM（NeurIPS 2019）、
   SpikeLLM（ICLR 2025）、SpikingLM（AAAI 2025）、SpikingMamba（TMLR 2026）。
   其余一律标成"预印本 / 模型卡 / 产品页"，不写成"研究表明"。

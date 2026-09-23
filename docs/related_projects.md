# 相关项目对照

> 这份表是**许可边界**与**证据强度**的台账，不是「相关工作综述」。每个条目都回答三个问题：
> 它的学习规则是什么（纯不纯局部）、**它的代码能不能进本仓库**、它自报的数字有没有第三方复现。

## 为什么需要这张表

三件事逼着我们把它们写下来：

1. **计划书 §八 的技术选型直接来自这些工作**（EMBER → §四 LLM 协同；MEMBRAIN → §4.3 编码替代方案；
   eprop-PyTorch → §1.4/§八 的 e-prop 实现参考）。选型写进计划书，就得能回答「这些东西现在什么状态」。
2. **本仓库是 Apache-2.0，且 CI 里有许可审计**（`scripts/check_licenses.py`、计划书 §12.1）。
   同类项目里既有 MIT（可 vendoring），也有**非商业许可**（一行业都不能抄）。这一栏不能靠印象填。
3. **本仓库的引用纪律**：自报数字与第三方复现过的数字要分开写。这张表里有项目把**目标写成了能力**，
   也有项目**被自己的基准反驳**——那种信息比一个漂亮的准确率更值得记下来。

## 对照表

| 项目 | 类型 | 一句话定位 | 学习规则（是否纯局部） | 实现栈 | 许可证（能否进依赖） | 活跃度与证据强度 | 与本项目的关系 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BioSNN-Plug**（本项目） | 研究原型 | 纯局部学习的 SNN 认知核心 + 模态插件；SNN 决定何时调用 LLM | e-prop（ALIF）+ R-STDP + TD-LTP Critic + 核化 IB-Hebbian；**无代理梯度** | numpy 骨架库 + PyTorch/SpikingJelly 研究线 | 本仓库 Apache-2.0；骨架库 `biosnn-bus` 独立 semver | 第一阶段三条线已验收（W1 97.79% / W2 77.48% / W3 中位数 237.8 步），数字见[复现记录](reproducibility.md) | —— |
| **EMBER**（Savage, 2026，arXiv:2604.12167） | 论文（**无公开仓库**） | SNN 作认知基质、LLM 作可替换推理引擎的混合架构；SNN 决定"何时行动 / 浮现哪些关联"，LLM 决定"做什么动作、生成什么内容" | 抑郁主导 STDP（$A_-=1.05A_+$，$\tau_+=20$ ms / $\tau_-=30$ ms）+ 资格痕迹 × 全局多巴胺门控 + cascade-scaled decay；**无反向传播、无 e-prop** | **自研 LIF，写在 PyTorch 里**；未用 SpikingJelly / snnTorch / Nengo（逐词 0 命中） | **不适用**——没有仓库可审。⚠️ **不要**把 arXiv 的分发许可（CC BY-NC-ND 4.0）或任何同名仓库的许可填进这一栏 | 论文 2026-04 提交，代码 "released at publication"（NeurIPS checklist 第 5 项自答 `[No]`，至 2026-09 仍只有 v1）。**N=1、无误差棒**（checklist 第 7 项 `[No]`）；作者自己标注了 LLM confabulation 风险 | §四 的整个证据基础。**可借鉴**：架构分工、z-score top-k 编码、侧向传播触发；**不可借鉴**：任何代码（没有）、把它的 N=1 数字当外部证据 |
| **MEMBRAIN**（tfatykhov, 2026，GitHub） | 个人仓库（4 天 PoC） | 给 LLM agent 用的"神经形态记忆桥"：FlyHash 稀疏编码 + Nengo 群体 + 吸引子清洗 | Voja（局部无监督，改 encoders）+ PES（误差驱动，改 decoders，默认开）；**无 e-prop、无反向传播** | Nengo（**纯 CPU Simulator**；Loihi/Lava **只存在于规划文档**）+ numpy；**无 torch / SpikingJelly / snnTorch** | **MIT**（标准原文、无附加条款、可 vendoring，须保留版权行）⚠️ 两点审计备注：**(a) MIT 不含专利授予**，而作者在 `AGENTS.md` 写明视"吸引子动力学"为 patent claim 要点；**(b) 不在 PyPI**（PyPI 上的 `membrain` 是另一个无关的冷冻电镜项目） | 2026-01-31 ~ 02-03 共 72 次提交后停更；0 star / 0 fork / 单人 + AI 代理、无外部贡献者。⚠️ **README 宣称与实际不符**：自报"20% 噪声下 100% 补全"，而它自己的基准写着 `MembrainStore` 在 0.2 噪声下 **0.35**（余弦/FAISS 基线 0.55），并自陈 "~5000x slower"；HEAD 的 `recall()` **跳过 SNN 模拟**（PES 发散）。无第三方复现 | §4.3 的编码替代方案。**可借鉴**：`encoder.py` 那约 224 行 numpy FlyHash（int8 投影 + top-k WTA）值得 vendoring；**不建议引入整仓**（停更 + 宣称与实现不符）。**不可引用**它的数字 |
| **eprop-PyTorch**（ChFrenkel, 2022，GitHub） | 第三方复现实现（**非官方**） | e-prop 论文的 PyTorch 复现，作者 Charlotte Frenkel（UZH/ETH，**不在 Bellec et al. 作者名单里**） | 硬编码的 e-prop（式 (4)/(25)），**不依赖 autograd**；**只有 LIF，且 ALIF 被显式移除**（`main.py` 写着 "Support for the ALIF neuron model has been removed."、`models.py` 有 `assert self.model == "LIF"`） | PyTorch（**无依赖清单、无版本约束、无 CI、无测试**）；任务只有证据累积一种 | **Apache-2.0**（标准原文，版权人 University of Zurich；无商业限制，可进依赖）——但工程上**不建议**：不是 PyPI 包、无 tag/release，只能锚 commit `0f32a8f2` | 全仓**一个**初始提交（2022-02-18），68 star / 9 fork；1 条 issue 至今无人回。注意 `updated_at` 2026-08 是元数据变动，**不是代码更新** | §1.4/§八 点名的「e-prop 实现参考」。**实际能对照的只有** LIF + 证据累积那一条链；⚠️ **ALIF 请以官方实现 [15] 与 Bellec et al. 2020 原文为准**（本仓库已移除 ALIF）。ADR-0008 决策 3 里有两句与仓库事实不符，已在该 ADR 的「后续更正」注记中更正 |
| **ESPP**（Graf, Su & Indiveri，2024，arXiv:2405.13976） | 论文 + 官方仓库 | **EchoSpike Predictive Plasticity**：把「上一个样本的整段脉冲活动」当预测目标（echo），同标签拉近、异标签推远；自监督、无全局误差反传 | 预测编码 + 对比编码式的**层间局部规则**（不是资格痕迹路线） | PyTorch + **snnTorch + Tonic**（与本项目既定栈不同，多两个依赖的许可核查成本） | 仓库 [Zhe-Su/ESPP](https://github.com/Zhe-Su/ESPP) 是 **Apache-2.0**（⚠️ LICENSE 版权人一行仍是模板占位符，即版权人未声明；⚠️ 论文脚注给的 `largraf/EchoSpike` **已 404**，权威链接用 Zhe-Su 这个） | 2024-05 预印本（v2，自述 "submitted to IEEE"，至 2026-09 **未检索到正式版**）；仓库 2025-08 建、2025-09 后无提交、0 star。SHD 84.32%（自报，作者自述超过其所知的所有局部规则） | **「事件优先级」的原文出处**：它用输入活动阈值 + 损失阈值**自己决定哪些时间步才做权重更新**（实测 18%–27%，随训练递减），原文逐字 "ESPP intrinsically has the ability to selectively choose those time steps that matter the most."——与本项目 §4.2 的触发动机同源。**但不能替代 e-prop**：它是并列的另一条规则（同 ADR-0010 对 TP 的处理）。另：「ESPP」是高频缩写（员工购股计划、欧洲粒子物理战略…），**SNN 语境下才是这一个** |
| **Javis**（BEKO2210, 2026，GitHub） | 个人仓库（无论文） | 给 LLM agent 用的**联想式 SNN 记忆协处理器**：知识存成细胞集群，查询当部分线索、靠模式补全重新激活，只把少数概念喂给 LLM | 全是 STDP 家族（pair / iSTDP / 三相 / 奖励调制 STDP / BCM / SFA / 结构可塑性，共 12 条可选）；**无 e-prop、无反向传播** | 纯 Rust，**不依赖任何 SNN 框架**（无 Nengo / SpikingJelly / snnTorch） | ⚠️ **PolyForm Noncommercial 1.0.0**——**非 OSI 认可、非 SPDX 标准**，附研究用途 addendum（禁止生产部署与对外服务）：**一行业务代码都不能并入本仓库**，处理方式与 `eligibility_propagation` 相同（只读、只引结论） | 无论文 / 无 DOI；0 star；2026-05 后停更。README 的数字（自召回 100%、token 省 35–45%、联想召回 ≈2%、容量 ≈50 概念）全部出自作者自建小语料，**未经第三方复现** | 与 EMBER 同属"SNN 当基质、LLM 当嘴"路线，但**分工相反**：EMBER 是 SNN 决定**何时**行动，Javis 是 SNN 决定**送什么进上下文**。适合作为 §四 的第二种分工对照 |

## 同名陷阱（填许可前必看）

这个仓库在引用上踩过一次"把 A 的结论记到 B 头上"（Khacef → Hajizada），所以同名项目单独列一节：

- **Javis**：GitHub 上另有一个 **`JuliaAnimators/Javis.jl`（Julia 动画/可视化库，MIT，800+ star）**。
  许可审计时若只写"Javis = MIT"，就是张冠李戴——本表指的是 `BEKO2210/Javis`（PolyForm 非商业）。
  另有 `JavisVerse/JavisGPT`、`JavisVerse/JavisDiT` 等无关项目，以及完全不同的 "JARVIS" 谱系。
- **EMBER**：`Starlight-Unit-Studio/coreui`（"Ember CoreUI … E.M.B.E.R. cognitive architecture"）
  也是本地 LLM WebUI，**无 SNN、无 STDP**，GitHub API 的 license 字段是 `NOASSERTION`——
  这正是「`NOASSERTION` 会骗人」的典型场景。`pandeyAnush/ember`（RAG 助手）同理。
- **MEMBRAIN**：PyPI 上的 `membrain` 是**另一个**项目的包（冷冻电镜膜蛋白定位，BSD-3-Clause），
  与本仓库说的 MEMBRAIN 无关；后者不在 PyPI 上。

## 这张表怎么用

- **要引一条结论** → 先看「活跃度与证据强度」那一栏是不是"未经第三方复现"。是的话，引用时必须写明，
  并按 [`references.md`](references.md) 的规矩登记核验状态。
- **要用一段代码** → 先看「许可证（能否进依赖）」。**非商业 / 非 OSI 许可的，只能读，不能抄**；
  MIT 可以 vendoring，但要保留版权行（并且注意 MIT 没有专利授予）。
- **要判断"这条路走不走得通"** → 看它的学习规则是不是纯局部、跑在什么设备上。本项目的命题是
  **纯局部 + 无代理梯度**，所以「用 BPTT 训出来的漂亮数字」在这里不构成先例。

## 维护

新增同类项目时，四个字段必须来自**实际打开的文件/页面**（LICENSE、`pyproject.toml`、CI 配置、
论文 PDF），不能来自记忆或二手转述；打不开的写"未见"。核验过程本身记进提交信息。

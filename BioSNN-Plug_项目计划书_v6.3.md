# BioSNN-Plug 项目计划书

**面向高生物合理性的全模态脉冲神经网络认知原型**

**版本 6.3 | 纯局部学习规则路线 | SNN 调用 LLM 协同架构 | 学术引用整合 + 开源治理版**

---

## 版本说明

本版在 v6.2 基础上做**引用与定位的更正**，并补入四条新文献与一份项目对照表。**没有改动任何验收标准**——
这一版修的是「文档说的与实际/原文不符」，以及两处被新文献推动的定位调整。

> **v6.3 修订记录**
> ① **Trace Propagation 重新定位**：不再写成「e-prop 的存储优化」。它是**另一条规则**（痕迹按神经元而非按突触），
>    且只在 LIF 上验证过；本项目不采纳它作为 e-prop 的补救措施。涉及 §2.1 架构图、§2.2、§3.1 关键修正 1、§八 选型表。
>    裁决与三条证据见 [ADR-0010](docs/adr/ADR-0010-eprop-quadratic-storage-and-trace-propagation.md)。
> ② **TD-LTP 文献考证结项**：命名与规则本体出自 [12] Frémaux et al. 2013（§3.2 补上逐字出处与规则形式），
>    §十一 的 P0 待办关闭；§7 第三阶段与 §12 的引文一并更正（2010 那篇负责的是「R-STDP 无监督偏差」这一结论）。
> ③ **Hajizada et al. 年份 2026 → 2025**，并把指标换成**所引版本（v2）**的数字（113× / 6,600×）——此前是
>    「v2 的标题配 v1 的数字」。
> ④ **许可事实更正**：SpikingJelly 用启智开源许可证 1.0（OIOSL），**不是** Apache-2.0（§1.4、§12.1，见 [ADR-0008](docs/adr/ADR-0008-spikingjelly-license-and-python-floor.md)）。
> ⑤ **归因更正**：§3.3 按原文改为「避免了深度网络的表征瓶颈问题」；§2.2 的「双通道融合」不再归因给 TAAF
>    （TAAF 提供的是时间注意力引导的自适应融合，双通道是本项目自己的设计）；[7] 文章号 42970 → 12653；
>    [8] 年份；[10] [11] 作者（项目名 → GitHub 账号）。
> ⑥ **新增十条文献**：[12] Frémaux et al. 2013（TD-LTP 命名出处）、[13] Tihomirov et al. 2025、[14] SpikingJelly、
>    [15] eligibility_propagation、[16] Javis、[17] Korcsák-Gorzo et al. 2025、[18] Millidge 2025、[19] Predictive E-prop、
>    [20] ESPP、[21] eprop-PyTorch。
> ⑦ **「事件优先级」与 Predictive E-prop 的重新定位**：§七 第四阶段的任务行**删去「+ Predictive E-prop」**——该文全文
>    0 次 metacognition / forgetting / working memory，挂在「元认知门控」名下是误引；同时补入**事件优先级**，即 ESPP 的
>    选择性时间步更新（实测只跑 18%–27% 的时间步）。Predictive E-prop 按**论文自身定位**（"a learning principle rather
>    than a task specific model"）移入 §3.1 的 e-prop 变体；§3.1 一并补入**深度扩展**（Millidge，纯数学、无实验）与
>    **事件驱动实现**（Korcsák-Gorzo，NEST 上 2 百万神经元），并写明这三样东西**互不相同、不可混引**；§八 选型表补三行。
> ⑧ **新增 `docs/related_projects.md`（三语）**：javis / EMBER / MEMBRAIN / eprop-PyTorch / ESPP 五个同类项目的对照表。
>    重点不是功能罗列，而是**许可边界**（哪个能进依赖、哪个一个字都不能抄）与**证据强度**（哪些数字已被第三方复现、
>    哪些是自报甚至被作者自己的基准反驳）。
> ⑨ v6.2 全部内容原样保留（除上述更正点）。
> ⑩ **§3.2 一句中文转述过读的更正**：原文 Figure 2A 的基准是**未训练**的均匀权重，现象是「学习后的表现跌到学习前之下」（原文自用词 "unlearning of the task"）。此前写成「导致遗忘**已学会的技能**」，把这件事说成了灾难性遗忘——Figure 2A 不支撑后者。已按原文改写。该文的逐字复核（三句话全部命中）见 [`docs/references.md`](docs/references.md) 的 [3] 条。

> ⑪ **速率型 / 脉冲型的现状写清楚，并把两件后续任务排进阶段**：三条验证线里，
> **只有 W2 是真脉冲**（单层 ALIF，256 个神经元）；**W1 是速率型**，而且这不是偷工减料——
> Pogodin & Latham 2020 的网络本身就是 LReLU 速率型，§3.3 选的就是一条速率型规则；
> **W3 也是速率型**，而这一处**是对来源论文的偏离**（Frémaux 2010/2013 是脉冲的），
> `research/rstdp/README.md` 已知边界第 3 条已披露「TD-LTP 与 TD-STDP 那处差别在本实现里
> 无从体现」。对 §九 指标表的含义：**「活跃神经元比例」与「脉冲稀疏度」目前只在 W2 上是
> 原义指标**，W1/W3 报的是速率型类比量（W1 自己标注了这一点）。
> 据此补两条任务：**第二阶段**「感知层接入脉冲总线」（连的是架构，不是把感知层改成脉冲）、
> **第三阶段**「执行层脉冲化」（让 §3.2 的 `STDP(Δt)` 第一次可验，速率版留作对照基线，
> 评测协议为**冻结随机流**）。
> ⚠️ **这两条在第一阶段一律不推进**——第一阶段到此为止，规模、脉冲化、LLM 协同都排在后面。
> ⑫ **§6.2 显存约束声明重算——前提由 TP 换成 e-prop 自身的痕迹存储**：① 那条更正只改到了 §2.2 / §3.1 / §八，
> **§6.2 的清单项与约束声明、§十 风险表两行、§12.4 的 ADR 举例、§十三 的创新点行都还留着 TP**，本版一并改掉。
> 重算的关键发现是：原声明那两个数（"5 万 @ 1% ≈ 0.2 GB；50 万 @ 0.1% ≈ 2 GB"）**恰好等于正确算式在批大小 =1 时
> 取两个痕迹张量的结果**——算式的形式没错，**漏乘了批大小那一维**。按 $b{=}64$ 计同一对配置是 13 GB 与 130 GB。
> 修正后的可行条件是**稀疏度 + INT8 量化 + 批大小 ≤ 8** 三者同时成立，**第三个杠杆是批大小而不是 TP**
> （e-prop 天然支持逐样本在线更新，TP 因为要用 batch 内对比损失**结构上做不到**——方向与原文写反）。
> 推导、算式与反解表见 §6.2；算式已用 `research/eprop/traces.py` docstring 里的两个数字（约 470 MB / 约 33 MB）逐字校验。
> ⚠️ 这只是纸面推导，**无规模实测**，故 §6.2 原降级路径保留不变。
> ⑬ **§3.4 里 [19] 的一句"原文"引用不逐字**：重抓 Noè et al. 2026 的全文副本后逐字比对发现，
> 该文写的是 `Predictive Eprop`——**没有连字符**；带连字符的 `Predictive E-prop` 只出现在标题与其余 42 处。
> 本版 §3.4 与 `docs/references` 三语台账原先把这处引成了带连字符的形式，一并改正。
> 这句是 v6.3 新增的（v6.2 无此句），故不进"计划书正文待更正项"表。来源与哈希见 `docs/references.md`。
> ⑭ **§十三 开头把自己的版本号写成了 v6.2**：那是从 v6.2 §十三 逐字复制过来、没跟着改的一处，
> 本版改为 v6.3。纯机械修正，不涉及任何内容或结论；列在这里是为了让 ⑨ 的"除上述更正点外原样保留"精确成立。
> ⑮ **§12.3 补登记表里漏登的检查项**：这张表本该是"CI 实际跑什么"的清单，却漏了五项——
> `scripts/check_references.py`、`scripts/check_translations.py`、`scripts/check_md_tables.py`、
> 依赖许可审计（`pip-licenses` + `scripts/check_licenses.py`）、wheel 干净环境安装（`uv build` + 全新 venv）。
> 五项都真在 CI 里跑，只是没写进来——"实际有、文档没有"正是这张表要防的状态，本版一并补上，表从 8 行增至 12 行。
> 其中表格结构检查对应的是一次真实事故：GFM 里表格的 `|` **即使在反引号内也仍是列分隔符**，
> 漏转义会让该行超出的单元格在渲染时**被直接丢弃**，而原文文件完好无损、`git diff` 也看不出来
> ——引用台账那行曾因此静默少掉 877 个源字符，见 `docs/references.md` 的「已发生过的错误」一节；
> 该检查已接进 CI 与 pre-commit。
> ⑯ **§12.3「基线差距报告」那一行原先是空头支票**：表里写着「CI artifact」，却**没有任何 job
> 产出它**——全仓 `upload-artifact` 只有许可证清单与 wheel 两处，而真正的差距数字在
> `sweep_results/w2_bptt.log` 那份手工入库的日志里。本版补上 `gap-report` job。它**不重新训练**：
> 那份完整对照在 GPU 上要 306 s 且要先下 MNIST，CI 里跑不动，硬跑还会把一个只在那个配置下
> 成立的数字伪装成 CI 的产物。它做的是把记录现算成报告并上传 artifact，同时核对三语 README
> 引用的差距数字能否追溯到**同一次**运行（同一次比较在不同语言里成了两个数，读者按语言会
> 拿到不同结论）。另配一条缩小规模的冒烟，证明产出那份报告的脚本本身没烂掉——`bptt_baseline.py`
> 此前从没被 CI 碰过。

---

## 版本说明（v6.2，原样保留）

本版在 v6.1 基础上新增**开源治理与社区建设**体系（第十二章），补齐开源项目评价中识别的四块短板，并调整路线图：

> **v6.2 修订记录**
> ① 新增 §十二 开源治理与社区建设：许可证策略、分层开源路线、测试与 CI、文档体系、社区治理、现实版开源成功指标；
> ② 路线图新增**第零阶段（开源基建与骨架库，第 1-2 个月，与第一阶段并行）**：插件框架抽为独立轻量库先行开源；
> ③ 第六阶段"至少 3 个第三方插件"指标改为有机生长目标（不再作为硬性验收）；
> ④ 优先级表新增"P1 开源基建"条目；
> ⑤ 核心目标新增第 5 项"开源可复现"；
> ⑥ v6.1 全部内容（LLM 协同模块、引用更正、显存约束声明、优先级与降级策略等）原样保留。

---

## 一、项目定位

### 1.1 项目名称

**BioSNN-Plug：纯局部学习驱动的可扩展全模态脉冲神经网络认知原型**

### 1.2 核心目标

构建一个基于脉冲神经网络（SNN）的认知智能原型，具备五项核心能力：

1. **纯局部学习**：完全跳过代理梯度，训练过程不依赖全局反向传播，所有突触更新由局部信号驱动。
2. **模态插件化**：文本、图像、音频等模态作为独立插件接入共享认知核心，新增模态无需修改已有架构。
3. **无损规模扩展**：通过神经发生机制动态增长神经元，实现容量随任务复杂度扩展且不遗忘已有知识。
4. **SNN 自主调用 LLM**：SNN 作为系统的认知主体，学会在适当时机调用外部 LLM 进行推理，并整合返回结果。
5. **开源可复现**（v6.2 新增）：所有结论可在公开仓库中复现，骨架框架对他人可用、可扩展、有明确许可证。

### 1.3 设计哲学

> **"局部规则驱动，认知核心恒定，感知外壳可换，外部推理可调，风险显式管理，问题逐项闭环，成果公开可验。"**

本项目刻意跳过代理梯度。BPTT 与生物神经系统的时空局部性观察形成鲜明对比，导致高计算和内存需求，限制了高效训练策略和片上学习 [Bellec et al., 2020]。项目的核心价值不在于复现现有模型的性能，而在于验证：**纯局部学习规则能否通过经验积累实现智能的自然生长，并学会调度外部推理工具来扩展自身能力边界。**

EMBER 架构的核心设计原则为这一方向提供了直接支撑：不是用检索工具增强 LLM，而是将 LLM 作为**可替换的推理引擎**置于一个持久的、生物基础的联想基质**之内** [Savage, 2026]。SNN 决定"**何时行动**"和"**关联什么**"，LLM 仅负责"**选择行动类型并生成内容**" [Savage, 2026]。

### 1.4 技术约束

| 项目 | 约束 |
| --- | --- |
| 硬件 | RTX 5060（8GB 显存） |
| 框架 | SpikingJelly + PyTorch + eprop-PyTorch 参考实现（**SpikingJelly 用启智开源许可证 1.0（OIOSL），不是 Apache-2.0**；其商业使用披露义务与对 Python 下限的影响见 [ADR-0008](docs/adr/ADR-0008-spikingjelly-license-and-python-floor.md)） |
| 首期模态 | 文本、图像、音频 |
| 学习算法 | e-prop + R-STDP + Hebbian（不使用代理梯度） |
| LLM 协同 | MCP 协议 / API 调用模式，LLM 在远程运行 |
| 评估导向 | 在线学习能力、持续学习抗遗忘、学习速度、能效比、调用决策质量 |
| 开源形态 | 代码 Apache-2.0，文档 CC-BY 4.0（详见 §12.1） |

---

## 二、系统架构

### 2.1 四层分离架构（含 LLM 调度层）

```text
┌─────────────────────────────────────────────────────┐
│              LLM 调度与协同层（新增）                   │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│  │ 调用触发模块  │ │ 上下文编码模块│ │ 结果整合模块 │ │
│  │(STDP侧向传播)│ │(z-score top-k)│ │  (记忆巩固)  │ │
│  └──────────────┘ └──────────────┘ └──────────────┘ │
│  ┌──────────────────────────────────────────────┐   │
│  │      LLM 接口（MCP / API，可替换）            │   │
│  └──────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────┤
│                     执行层                            │
│  [决策输出] [动作生成] [模态特定解码器]                │
│       学习规则: R-STDP + TD-LTP Critic               │
│        仲裁: 奖励预测基线网络 (Critic)                 │
├─────────────────────────────────────────────────────┤
│                  认知层 (认知核心)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ 工作记忆  │  │ 情景记忆  │  │ 元认知门控│           │
│  │ (RSNN)   │  │ (可扩展)  │  │ (不确定性)│           │
│  └──────────┘  └──────────┘  └──────────┘           │
│     学习规则: e-prop (ALIF + 自适应阈值)              │
│        仲裁: ES 元学习仲裁器（含调度决策仲裁）          │
│              脉冲总线: TAAF + 双通道融合               │
├─────────────────────────────────────────────────────┤
│                  感知层 (插件化)                        │
│  ┌────────┐  ┌────────┐  ┌────────┐                   │
│  │文本编码 │  │图像编码 │  │音频编码 │  ... 新插件       │
│  └────────┘  └────────┘  └────────┘                   │
│         学习规则: 核化 IB-Hebbian + 除法归一化         │
└─────────────────────────────────────────────────────┘
```

### 2.2 认知核心设计

认知核心采用**循环脉冲神经网络（RSNN）**主干，基于 ALIF 神经元。e-prop 的原始论文正是针对 RSNN 设计的，其核心假设——突触更新由局部资格痕迹和全局学习信号共同决定——与认知核心的功能需求高度匹配 [Bellec et al., 2020]。

**核心组件：**

- **RSNN 主干**：初始约 5 万神经元，支持动态扩展。资格痕迹按突触存储（内存随连接数线性增长）；**不采用** Trace Propagation——它是另一条学习规则，不是 e-prop 的省内存版本（见 §3.1 关键修正 1 与 [ADR-0010](docs/adr/ADR-0010-eprop-quadratic-storage-and-trace-propagation.md)）。
- **脉冲总线**：集成 TAAF 与**本项目自己的**双通道融合。TAAF 提供的是**时间注意力引导的自适应融合**（每个时间步动态分配重要性分数，实现时间异构脉冲特征的分层整合）[Shen et al., 2025]；「时间 / 语义双通道」这套划分是**本项目的设计**，不归因给 TAAF。
- **工作记忆**：包含强抑制-抑制去抑制微电路，维持长时神经元时间尺度。
- **情景记忆**：基于模式分离的正交类中心存储。PS-SNN 在 CIFAR100-B0 上 10 步增量学习中达到 **76.42%** 的平均增量准确率 [Hu et al., 2026]。
- **元认知门控**：监控预测误差和奖励不确定性，动态调节学习率与探索率。在 SNN 调用 LLM 模式下，元认知门控额外承担"**调用触发信号生成**"功能。
- **ES 元学习仲裁器**：使用进化策略优化仲裁参数，不依赖全局反向传播 [Confavreux et al., 2025]。仲裁维度见 §3.4。

### 2.3 模态插件接口规范

```python
class ModalityPlugin(ABC):
    @abstractmethod
    def encode(self, raw_input) -> SpikeTrain: ...

    @abstractmethod
    def get_membrane(self) -> nn.Module: ...

    @abstractmethod
    def decode(self, spike_output: SpikeTrain): ...

    @property
    @abstractmethod
    def modality_name(self) -> str: ...

    @property
    @abstractmethod
    def spike_dim(self) -> int: ...

    @property
    def temporal_scale(self) -> float:
        """该模态的典型时间尺度（毫秒），供 TAAF 模块初始化使用"""
        return 10.0

    @property
    def fusion_channel(self) -> str:
        """返回 'temporal' 或 'semantic'"""
        return 'temporal'
```

| 模态 | 编码方案 | 感知层学习规则 | 融合通道 |
| :--- | :--- | :--- | :--- |
| 文本 | Token 嵌入 + 时间常数编码 | 核化 IB-Hebbian | 语义 |
| 图像 | 差分编码 / DVS 事件流 | 核化 IB-Hebbian + 除法归一化 | 语义 |
| 音频 | 耳蜗模型频率分解 | 核化 IB-Hebbian + 除法归一化 | 时间 |

---

## 三、核心学习算法详解

### 3.1 e-prop：认知核心的在线学习引擎

e-prop 采用三因子学习规则 [Bellec et al., 2020]。对于突触 $j \to i$：

$$\Delta w_{ij} = \eta \cdot L_i(t) \cdot \bar{e}_{ij}(t)$$

其中 $L_i(t)$ 是神经元 $i$ 的学习信号，$\bar{e}_{ij}(t)$ 是该突触的资格痕迹。资格痕迹由突触前脉冲痕迹和突触后伪导数的乘积累积而成，仅依赖突触本地的信息。注意：e-prop 的内存为 **O(突触数)**（随连接数线性增长），"随序列长度 O(1)"仅指时间维度——存储量级与稀疏化的关系见下述关键修正 1 [Pes et al., 2025]。

**关键修正 1：资格痕迹的存储量级（并撤回「上 Trace Propagation」这条补救措施）**

e-prop 的资格痕迹按突触存储，空间复杂度随神经元数量**二次增长**——原文 Table 3 的 `E-prop [2]` 行给出 `LH²`，§1.3.1 也点名 e-prop 属于「按突触存储」这一族 [Pes et al., 2025]。

Trace Propagation（TP）是同一篇论文提出的**另一条**完全局部学习规则：痕迹按**神经元**而非按突触存储（原文给出空间复杂度 `O(LH)`、辅助矩阵 `OH`），在 **N-MNIST** 与 SHD 上报告了与 e-prop 同量级的成绩 [Pes et al., 2025]（期刊版出处见 [2]）。

**但它不是 e-prop 的省内存版本**：论文把两者分列两行、分轴评价（e-prop 时间局部、空间不局部；TP 全 ✓），且 TP 的**全部**实验都在 LIF 上，本项目验收用的 ALIF（β=0.07）没有对应推导。因此本项目**不把 TP 作为 e-prop 的存储优化来落地**。e-prop 一侧真正该做的那一步（`epsilon_v` 的递推不含后突触下标，逐位等价地降一维）已在第一阶段完成。裁决与三条证据见 [`docs/adr/ADR-0010`](docs/adr/ADR-0010-eprop-quadratic-storage-and-trace-propagation.md)；TP 那条规则本身的最小复现在 [`examples/paper_pes2025.py`](examples/paper_pes2025.py)。

**关键修正 2：ALIF 神经元资格痕迹扩展**

ALIF 神经元的资格痕迹涉及**依赖于使用的放电阈值**的时间演化，比 LIF 神经元更复杂。ALIF 资格痕迹的完整推导见 Bellec et al. 2020 原文（其算法与附录部分包含 adaptive threshold 的 eligibility 推导）[Bellec et al., 2020]。

**关键修正 3：死亡神经元防护**

将神经元放电阈值从固定超参数提升为可训练参数，实现环内自适应阈值学习。评估体系中加入"活跃神经元比例"指标，低于 60% 时触发阈值调整。

**e-prop 的三条已知扩展（v6.3 新增；都是并列变体，不是替代品）**

- **深度方向**：e-prop 的资格痕迹框架已被推广到任意深度网络（乃至任意 DAG），做法是在深度方向再引入一层痕迹递归，并论证复杂度随深度保持线性 [Millidge, 2025]。⚠️ **这是一篇纯数学笔记、没有任何实验**（作者自己在 Discussion 里写着 "we have performed no experiments demonstrating that good credit assignment across depth works in practice"），所以它只能支撑"多层 RSNN 用纯局部规则训练在理论上成立"这类方法学断言，**不能用来支撑任何性能数字**；它自列的局限（e-prop 并非真正的在线权重更新——要等 episode 结束才更新；每组参数各存一套痕迹，深层时"很快变得不可控"）反过来是本项目在线学习主张的反证材料。
- **预测编码方向**：把 e-prop 的"第三因子"从任务专用外部信号换成**预测编码自身的局部预测误差**，即 Predictive E-prop。作者自己的定位是"**一条学习原理，而不是某个任务专用模型**"（原文："We term the resulting model 'Predictive Eprop', emphasizing its role as a learning principle rather than a task specific model."）[Noè et al., 2026]。它在两个动力学系统（正弦极限环、Lorenz）上的三个任务里与 truncated BPTT 相当（p > 0.05），但收敛所需 epoch 少 70%（约 23 vs 约 80）。**它是 e-prop 的变体，不是新架构**。
- **事件驱动方向**：把"每个时间步同步更新"改成"**突触收到脉冲事件时才更新**"，并接进大规模仿真平台（NEST）；在模式生成、证据累积、N-MNIST 三个任务上复现原版性能，且在稀疏网络上做到 2 百万神经元的弱/强 scaling [Korcsák-Gorzo et al., 2025]。该文为遵守**严格局部性**还改掉了原版 e-prop 里一处违例（资格痕迹滤波器依赖输出神经元的时间常数——"for synapses to compute their weight updates, they must know the time constant of the output neuron, which violates the principle of locality"），这一句对本项目的"纯局部"叙事最有用。

**注意区分三条不同的东西**（本仓库在引用上踩过"把 A 的结论记到 B 头上"的坑）：e-prop 的**事件驱动版**（Korcsák-Gorzo，是 e-prop 的工程实现变体）≠ **ESPP 的选择性时间步更新**（Graf et al.，是**另一条规则**，预测编码 + 对比编码式的层间局部规则，不属于资格痕迹路线）≠ **"事件优先级"**（本项目对 ESPP 那个机制的中文转述，不是文献术语）。三者在计划书里的归属因此不同：前两条进本节，ESPP 进 §七 第四阶段与 §八 选型表。

### 3.2 R-STDP：执行层的强化学习机制

R-STDP 更新规则：

$$\Delta w_{ij} = \text{STDP}(\Delta t_{ij}) \cdot (R - V(s))$$

**关键修正：TD-LTP Critic 消除无监督偏差**

R-STDP 存在结构性无监督偏差。研究明确显示，成功偏移达到成功信号标准差的 **~25%（σR）** 时，R-STDP 就无法学习目标任务。更严重的是，当偏移 $\bar{S} < -0.4\sigma_R$（平均成功信号为负）时，学习后的性能甚至**低于学习前**——R-STDP 不仅无法学习，还会**把表现压到未训练水平之下**（原文自用词 "unlearning of the task"；基准是**未训练**的均匀权重，**不是**灾难性遗忘）[Frémaux et al., 2010]。该敏感性是**通用特性**，无法通过调整 STDP 窗口参数或更换权重依赖模型解决。

唯一的结构性解决方案是**刺激特异性奖励预测系统**。Critic 为每个输入/输出模式维护独立的奖励预测均值，向 R-STDP 提供无偏的成功信号 $S = R - \langle R \rangle$。只要成功偏移消失（$C/\sigma_R = 0$），学习对 STDP 模型的细节就相对不敏感 [Frémaux et al., 2010]。

Critic 网络使用 **TD-LTP** 训练。该规则的**命名出处已考证**：[12] Frémaux, Sprekeler & Gerstner (2013), *PLoS Computational Biology* **9**(4):e1003024。原文原话是 “Because it has, roughly, the form of ‘TD error signal × Hebbian LTP’, we call this learning rule TD-LTP.”（Critic learning 一节；对应 Eq. 17 与 Figure 2A 图注 “TD-LTP is the learning rule given in Eq. 17.”）。其形式为

$$\Delta w \propto \delta(t) \cdot \kappa * [x_i \cdot y_j]$$

其中 $\delta$ 是**全局标量** TD 误差，$\kappa * [\cdot]$ 是突触前后相关经核 $\kappa$ 滤波，**只计 pre-before-post 的配对**（这正是它与 TD-STDP 的差别）[Frémaux et al., 2013]。

**两篇引文分工不同，不要混引**：TD-LTP 的**名称与规则本体**出自 [12] Frémaux et al. 2013；而「R-STDP 存在无监督偏差、必须引入刺激特异性奖励预测」这一结论出自 [3] Frémaux et al. 2010（该文 Figure 3 的主题）。另外，**不要用 2013 那篇里 “no back-propagation signal has been observed in experiments” 一句论证局部性**——那句讲的是 TD 误差沿**时间**的信用分配特征，不是「生物网络不做误差向量的反向传播」。要引就引三因子形式本身（pre × post → κ 滤波 → 乘标量 $\delta$）。

同时引入**权重归一化**：每个神经元层面保持权重总和恒定，防止突触动态失控。

### 3.3 核化 IB-Hebbian：感知层的无监督特征提取

Pogodin 和 Latham 提出了**核化信息瓶颈（Kernelized Information Bottleneck）**方法，产生了一族**避免了深度网络的表征瓶颈问题**的学习规则 [Pogodin & Latham, 2020]。由此产生的规则具有**三因子 Hebbian 结构**：需要前突触和后突触的放电率，以及一个误差信号——第三因子——由全局教学信号和层特异性项组成，**两者均无需自顶向下传递即可获得** [Pogodin & Latham, 2020]。

**关键修正：除法归一化**

Pogodin 和 Latham 明确指出，为了在困难问题上获得良好性能并保持生物合理性，**该规则需要除法归一化——这是生物网络的一个已知特征** [Pogodin & Latham, 2020]。在感知层每个 Hebbian 层后添加侧抑制模块：

```python
pool = mean(h ** 2)
h_norm = h / sqrt(pool + eps)
out = h_norm * rrms * gain
```

侧抑制权重通过局部 Hebbian 学习获得，保持整体规则的局部性。

### 3.4 ES 元学习仲裁器：四类仲裁维度

Confavreux 等人使用进化策略（ES）元学习**大规模循环脉冲网络中局部共激活可塑性规则**，使用递增复杂度的参数化，发现能够**稳健地稳定所有四种突触类型的网络动力学**（E-to-E、E-to-I、I-to-E 和 I-to-I）[Confavreux et al., 2025]。该研究同时指出，元学习策略对于**越来越复杂的共激活规则开始失败**——先验地设计有效约束动力学的损失函数具有挑战性 [Confavreux et al., 2025]。因此本项目仲裁器从**双规则、简单参数化**起步（见 §七 阶段划分）。

仲裁维度共**四类**：

1. Hebbian ↔ e-prop 冲突检测与权重调节；
2. e-prop ↔ R-STDP 冲突检测与权重调节；
3. Hebbian ↔ R-STDP 冲突检测与权重调节；
4. **调度决策**："SNN 自主处理 vs. 调用 LLM"（详见 §4.5）。

冲突判定：计算两条规则对同一突触权重更新方向的余弦相似度，夹角超过 90° 判定为冲突。

---

## 四、SNN 调用 LLM 协同模块

### 4.1 设计原则

EMBER 架构的核心设计原则是：**不是用检索工具增强 LLM，而是将 LLM 作为可替换的推理引擎置于一个持久的、生物基础的联想基质之内** [Savage, 2026]。SNN 决定"何时行动"和"关联什么"，LLM 仅负责"选择行动类型并生成内容" [Savage, 2026]。

这一设计确保了 SNN 仍然是系统的**认知主体**，纯局部学习规则仍然是核心学习机制。项目的科学命题没有被放弃，而是被重新定义为：**纯局部学习的认知系统能否学会调度外部推理来扩展自身能力边界**。

### 4.2 调用触发模块：STDP 侧向传播

**机制**：EMBER 的核心发现是，**STDP 侧向传播在空闲操作期间可以触发和塑造 LLM 行动，无需外部提示或脚本触发** [Savage, 2026]。当学习到的权重在空闲期通过侧向传播激活时，关联信号会自然涌现。其生物学基础是背景膜噪声（σ = 0.1，约 0.9 Hz 自发放电）——没有噪声，网络在空闲期将保持沉默 [Savage, 2026]。

**关键数据**：从零学习权重开始，**第一次 SNN 触发的行动发生在仅 7 轮对话（14 条消息）之后** [Savage, 2026]。在 8 小时空闲期后，系统基于学习到的人物-话题关联**自主发起了与用户的接触** [Savage, 2026]。3 天脚本基线测试（52 条消息）中，STDP 侧向传播共触发 23 次 LLM 行动选择（1 次 `reach_out` + 22 次 `journal`）[Savage, 2026]。

**实现要点：**

- 在认知核心中维护 STDP 侧向连接，空闲期间持续进行侧向传播；
- **冲动检测（impulse detection）**：概念激活在无直接刺激时超过基线放电率 3 倍记为一次检测；**5 分钟内累积 3 次及以上检测**判定为显著冲动 [Savage, 2026]；
- 触发信号包含关联的**人物概念细胞**（person concept cells）和**话题关联**信息，供上下文编码模块使用 [Savage, 2026]；
- 对话结束后 15 分钟内，背景噪声即可产生 24 个侧向脉冲，证明学习到的权重可以通过空闲传播表达 [Savage, 2026]。

### 4.3 上下文编码模块：z-score 标准化 top-k 群体编码

**机制**：EMBER 提出了一种新颖的 **z-score 标准化 top-k 群体编码**，将文本嵌入编码为 SNN 脉冲，该编码在构造上**与维度无关**，在不同嵌入维度下实现了 **82.2% 的区分度保留率**（1024 维；384 维下为 83.8%，差异仅 1.6%）[Savage, 2026]。

**实现要点：**

- 计算每个感觉神经元偏好方向 $p_i$ 与嵌入 $e$ 的余弦相似度的 z-score：$z_i = (\cos(e, p_i) - \mu) / \sigma$，其中 $\mu, \sigma$ 为全体余弦相似度的均值与标准差 [Savage, 2026]；
- 选取 top-k 神经元（$k = \lfloor s \cdot N \rfloor$，稀疏度 $s = 0.14$，基于稀疏分布记忆理论），按 $r_i = \max(z_i, 0)/\max(z_{\text{top-k}}) \cdot r_{\max}$ 缩放发放率 [Savage, 2026]；
- 编码后**与维度无关**，支持任意 LLM 嵌入维度（4096、1536、1024、768 等），是**模型无关的效率指标** [Savage, 2026]。

**替代方案参考**：MEMBRAIN 使用 **FlyHash 编码**将密集 LLM 嵌入（1536 维浮点）转换为稀疏二进制脉冲序列（20,000 维二进制），使用 int8 随机投影 + Winner-Take-All 抑制，内存占用约 **30 MB** [Membrain, 2026]。注意：MEMBRAIN 基于 Nengo 框架与 Voja 学习规则，与 SpikingJelly 技术栈不同，仅作编码方案参考。

### 4.4 结果整合模块：记忆巩固

**机制**：LLM 返回的结果被重新编码为脉冲，回注到 SNN 的情景记忆中。SNN 需要判断返回结果是否值得巩固。

**关键数据**：EMBER 的权重增长轨迹显示，首次对话（20 分钟）后突触连接从 **0 增长至 10,843**；首次巩固（episodic replay）后跳升 **5 倍至 53,992**；随后在连续夜间睡眠中分别增长 **49% 和 51%**，最终达到 **201,394** [Savage, 2026]。这一**阶跃函数模式**（刺激期间增长、巩固期间稳定、空闲期间衰减）与级联衰减模型（cascade decay）及互补学习系统理论一致 [Savage, 2026]。8 小时空闲期后最大侧向权重仅衰减 **1.6%**，显著连接保持稳定；知识图谱从 0 增长至 **64 节点和 124 边** [Savage, 2026]。

**实现要点：**

- LLM 输出经编码后，通过 STDP 写入 SNN 情景记忆；
- 巩固过程模拟生物睡眠：注入噪声驱动网络进入稳健吸引子状态，**修剪弱的瞬时记忆，强化重要模式** [Membrain, 2026]；
- 巩固后，SNN 的显著连接保持稳定，弱连接被级联衰减清除 [Savage, 2026]。

### 4.5 调度决策仲裁（仲裁维度第 4 类）

ES 元学习仲裁器的第四类仲裁维度：**"SNN 自主处理 vs. 调用 LLM"的调度决策**。

**决策信号：**

- 元认知门控输出的不确定性水平；
- 情景记忆检索到的关联模式强度；
- 当前任务的复杂度评估（由 e-prop 认知核心的预测误差代理）。

**调度策略：**

- 当 SNN 的预测误差低于阈值且关联模式清晰时，**SNN 自主处理**；
- 当不确定性超过阈值或关联模式指向 SNN 无法独立处理的复杂任务时，**触发 LLM 调用**；
- 调用成功后，通过 **R-STDP + TD-LTP Critic** 强化触发该调用的关联通路；调用失败则抑制。

### 4.6 LLM 接口设计

采用 **MCP（Model Context Protocol）** 协议或 **API 调用模式**，LLM 在远程运行，SNN 在本地 RTX 5060 运行。

**MCP 协议参考**：ASTRA 项目已验证，其 SNN 引擎通过 MCP Server 暴露给 Claude Desktop 等客户端使用 [ASTRA, 2026]。BioSNN-Plug 认知核心可封装为 MCP 服务，暴露以下工具接口：

| 工具接口 | 功能 | 输入 | 输出 |
| :--- | :--- | :--- | :--- |
| `query_cognitive_state` | 查询当前认知状态 | 无 | 不确定性水平、活跃关联、工作记忆内容 |
| `retrieve_associations` | 检索关联记忆 | 话题标签 | 关联的脉冲模式、强度 |
| `trigger_llm_call` | 触发 LLM 调用 | 上下文编码 | LLM 返回结果 |
| `consolidate_result` | 巩固 LLM 返回 | 返回结果 + 重要性评分 | 巩固成功/丢弃 |

**LLM 可替换性**：EMBER 使用 Claude Sonnet 4.6 作为推理引擎，但该架构**不依赖于特定 LLM** [Savage, 2026]。可根据需求替换为任意 LLM API。

### 4.7 EMBER 适用边界声明

为保证证据强度评估的诚实性，明确 EMBER 与本项目的差距：

- **模态**：EMBER 为文本单模态；本项目为文本/图像/音频三模态插件体系；
- **实验尺度**：EMBER 为 3 天、52 条消息的初步实验，非受控基准；"首次触发 7 轮对话"为单一个案轨迹，不应直接作为本项目的验收基线（本项目设为"≤ 10 轮对话内首次触发"以保留余量）；
- **学习规则**：EMBER 使用 vanilla STDP + 奖励调制的 eligibility（非 e-prop，非核化 IB-Hebbian）；其证据支持"STDP 基质 + LLM 引擎"的协同可行性，**不直接证明** e-prop/IB-Hebbian 路线下的触发机制同样成立；
- **硬件**：EMBER 的 220K 神经元 SNN 运行于 RTX 5070 Ti（16GB），嵌入模型运行于 RTX 4060 Ti（8GB）；本项目单机 8GB 承载认知核心 + 嵌入编码，规模预算需按 §6.2 重新核算；
- **消融强度**：EMBER 的 SNN-disabled 对照为初步消融（单案例），其跨领域引用等数字应视为方向性证据。

---

## 五、无损扩展与持续学习

### 5.1 神经发生机制

生物系统中，神经发生指**生成新神经元以编码新记忆，同时保持现有神经元完整**的过程，尤其在海马齿状回中支持不破坏旧记忆的新学习 [Hu et al., 2026]。

CLP-SNN 在 Intel Loihi 2 上实现了实时持续学习，集成**神经发生和元可塑性**用于容量扩展和遗忘缓解，神经发生模块在数据流包含不熟悉概念时**按需分配新神经元** [Hajizada et al., 2025]。

```python
class NeurogenesisController:
    def __init__(self, core, max_neurons=500_000):
        self.core = core
        self.max_neurons = max_neurons
        self.contribution_threshold = 0.1
        self.growth_threshold = 0.6  # v6.1: 修复原代码引用未定义 self.threshold 的 bug

    def should_grow(self, task_id, performance) -> bool:
        """新任务性能低于阈值时触发神经发生"""
        return performance < self.growth_threshold

    def grow(self, n_new_neurons: int):
        """在认知核心中新增神经元，旧神经元冻结"""
        new_neurons = self.core.expand(n_new_neurons)
        self.core.freeze_old()
        return new_neurons
```

**多模态场景补充：**

- **模态特异性神经发生**：新模态接入时，新增一组神经元专门处理该模态脉冲模式；
- **跨模态神经发生**：检测到模态间协同模式时，新增同时接收多模态输入的整合单元；
- **调用通路神经发生**：当 SNN 反复学习到"某类关联模式需要调用 LLM"时，**专门生成一组神经元编码这一调用模式**。

### 5.2 持续学习与抗遗忘

- **模式分离学习**：PS-SNN 通过为每个类别预定义固定且相互正交的类中心替代传统可学习分类器，提供稳定的优化目标，防止特征空间冲突并减少任务间干扰 [Hu et al., 2026]；
- **GWR 结构可塑性**：预测误差超阈值时动态添加新神经元而非覆盖旧连接；
- **Ad-STDP**：STDP 学习率根据神经元活动历史自适应调整；
- **压缩潜在回放**：通过时间域压缩将回放数据存储需求降低两个数量级；
- **平均遗忘（AF）**：目标 < 8%（第二阶段），< 5%（第四阶段）；
- **反向迁移（BWT）**：目标 > -10%（第二阶段），> -5%（第四阶段）；
- **调用历史存储**：情景记忆新增"调用历史"，记录每次调用的上下文、LLM 返回结果、以及后续的巩固/丢弃决策。

---

## 六、能效与规模扩展

### 6.1 能效优势

在 Intel Loihi 2 上，CLP-SNN 实现了实时持续学习，相比最佳基线 OCL 方法（Jetson Orin Nano 上的 SLDA）实现了 **113 倍延迟改善**与 **6,600 倍能效提升**（v2 版数据）[Hajizada et al., 2025]。

> ⚠️ **版本注记**：v6.2 及其之前引的是该文 **v2 的标题**却用 **v1 的数字**（70 倍 / 5,600 倍 / 0.33 ms / 281 mJ）。二者取一，本版按 v2 的标题配上 v2 的数字；v1 的数字不再出现。

在 SNN 调用 LLM 模式下，**SNN 的能效优势仍然适用于"SNN 独立运行"的时段**。EMBER 的 SNN 在 8 小时空闲期持续以事件驱动方式运行，仅消耗极低能耗进行 STDP 侧向传播 [Savage, 2026]。只有当关联信号足够强时才触发一次 LLM 调用——系统的**端到端能耗取决于 SNN 触发 LLM 调用的频率**，这是评估指标中新增"调用频率"监控的原因。

### 6.2 RTX 5060 显存优化

- **资格痕迹按突触存储**：e-prop 的痕迹量正比于连接数——原文 Table 3 的 `E-prop [2]` 行给出 `LH²`，§1.3.1 并点名 e-prop 属"按突触存储"那一族 [Pes et al., 2025]。**不采用 Trace Propagation**：它是另一条学习规则而非 e-prop 的省内存版本，见 §3.1 关键修正 1 与 [ADR-0010](docs/adr/ADR-0010-eprop-quadratic-storage-and-trace-propagation.md)；
- **稀疏连接**：连接率控制在 5% 以下（与下方约束声明反解出的上限**取更严者**）；
- **分块训练**：核心分为多个子模块，每次仅训练一个子模块的资格痕迹；
- **梯度累积 + 混合精度**：批大小 2-4，利用 FP4 精度支持；
- **LLM 远程调用**：LLM 通过 API 调用，不在本地占用显存；
- **资格痕迹 INT8 量化**：进一步降低痕迹存储容量。

> **显存约束声明（v6.3 重算：依据由 Trace Propagation 改为 e-prop 自身的痕迹存储）**
>
> **算式**。e-prop 的资格痕迹按突触存储，峰值时刻活着的逐突触张量有四个：`epsilon_a` 与 `e_trace` 各占 `批大小 × 突触数`，`w_rec` 与梯度累加器各占 `突触数`（本实现是朴素 SGD，无优化器状态——见 `research/eprop/eprop.py` 的 `weight.add_(gradient, alpha=-rate)`）。记连接稀疏度为 $c$、神经元数为 $N$、批大小为 $b$、痕迹元素宽度为 $w$（fp32 为 4、INT8 为 1），则
>
> $$\text{峰值} \;\approx\; c\,N^{2}\times\big(2\,b\,w + 8\big)\ \text{字节}$$
>
> 这个算式**与本仓库已有的两个数字逐字吻合**，可作为它正确的旁证：sMNIST 验收形状（$b{=}64$、$N{=}256$、$T{=}28$）下按此式算 `eligibility_traces` 得 469.8 MB，而 `research/eprop/traces.py` 的 docstring 写的是"约 470 MB"；算 `eprop_gradient` 的两张常驻张量得 33.6 MB，docstring 写的是"约 33 MB"。
>
> **订正原声明的两处**。v6.2/v6.3 原写"5 万 @ 1% 稀疏 ≈ 0.2 GB；50 万 @ 0.1% 稀疏 ≈ 2 GB，余量可覆盖激活值与状态"。这两个数**恰好等于上式在 $b{=}1$ 时取两个痕迹张量的结果**（0.10 GB 与 1.00 GB，乘 2 即得）——也就是说，**算式的形式是对的，漏掉的是批大小那一维**，逐样本的数字被写成了总量。按 $b{=}64$ 计，同一对配置是 **13 GB 与 130 GB**，分别超预算 1.6 倍与 16 倍，原声明所称的"余量"并不存在。
>
> **订正后：50 万目标的可行条件是三个杠杆同时成立**——**连接稀疏度 ≤ 0.1%** + **INT8 痕迹量化** + **批大小 ≤ 8**。在 8GB 硬约束下（按 6GB 可用、另 2GB 留给激活值与 CUDA 上下文反解），每个规模各自需要的稀疏度上限：
>
> | 目标规模 | fp32 痕迹 @ $b{=}64$ | INT8 痕迹 @ $b{=}64$ | INT8 痕迹 @ $b{=}4$ |
> | :--- | :--- | :--- | :--- |
> | 5 万 | 0.46% | 1.8% | 3.8% |
> | 10 万 | 0.12% | 0.44% | 0.94% |
> | 50 万 | 0.005% | 0.018% | **0.15%** |
>
> 读法：表格里的数是该列条件下"放得下"的**稀疏度上限**。50 万那一行只有最后一格（INT8 + $b{\le}4$）能容下 0.1% 的稀疏度，前三格都要求比 0.1% 更稀疏一到两个数量级。注意 §6.2 上一条"连接率控制在 5% 以下"是**功能性目标**，与本表反解出的硬上限不是同一个量，取更严者——按本表，5% 这个值在 $b{=}4$ + INT8 下也只在 5 万规模上刚刚可行（上限 3.8%），规模再大就必须更稀疏。
>
> **为什么第三个杠杆是批大小，而不是原声明写的 TP**：e-prop 本身就是在线规则，`eprop_gradient` 逐时刻累积、不保存时间维，**$b{=}1$ 是它的自然工作点**；而 TP 因为要用 batch 内的对比损失，**结构上做不到 $b{=}1$**（见 ADR-0010 末段）。原声明想借 TP 换来的那点内存余量，代价恰恰是丢掉 e-prop 唯一能让批大小降下来的性质——**方向是反的**。TP 另外两条不成立的理由（它是另一条规则、其全部实验是 LIF 而验收配置是 ALIF）见 §3.1 关键修正 1。
>
> ⚠️ **以上是外推，不是实测；且现状比外推更值得警惕**。仓库现有的显存实测里，**最大的一次是 3,220.6 MiB**（`sweep_results/w2_shd/out.log`，W2 的 SHD 旁证运行）——它只有 **256 个神经元**，即目标规模的 **0.05%**，却已占掉 8GB 的 **39%**；其余两次是 W1 的 1024×3 层 1,167.0 MiB 与 W2 的 sMNIST 验收配置 1,104.1 MiB。关键在于：**这三次的开销几乎全是时间步带来的激活值，不是痕迹**（SHD 那次 T=100，sMNIST 是 T=28；N=256 时痕迹按上式只有约 33 MB）。也就是说，**上表算的是痕迹那一项，而当前真正吃掉预算的是另一项**；规模上来时两项会同时增长。**没有针对 10 万 / 50 万预算的探针，也没有断言峰值 < 8GB 的测试**。若第四阶段实测不可达，按原降级路径执行：降为"8GB 约束下的最大可承载规模（预计 10 万–30 万神经元）+ 性能-规模-稀疏度曲线报告"，不设绝对规模门槛。

### 6.3 规模扩展路径

- **神经发生驱动的容量扩展**：任务复杂度增加时动态增加神经元 [Hu et al., 2026]；
- **稀疏性的规模效应**：神经元数量增加，脉冲活动更稀疏，能效比上升。

---

## 七、技术迭代优化路径

### 第零阶段：开源基建与骨架库（第 1-2 个月，与第一阶段并行）【v6.2 新增】

**目标**：在项目研究尚未产出成果之前，先让"插件框架骨架"以独立开源库的形态对外可用，建立仓库治理与 CI 基线。

**任务**：

1. 仓库初始化：LICENSE（Apache-2.0）、README（quickstart + 架构图）、.gitignore、CONTRIBUTING.md、CODE_OF_CONDUCT.md；
2. 骨架库抽离：`ModalityPlugin` ABC、脉冲总线、注册机制、示例编码器（MNIST 级）打包为独立轻量库（建议命名 `biosnn-bus` 或同类），**不依赖认知核心**，与具体硬件无关；
3. CI 基线（见 §12.3）：引用链接存活检查、文档代码块语法检查、版本号一致性检查、pytest 冒烟测试；
4. Colab/在线演示：一个 5 分钟内可跑通的"注册自定义模态插件"演示；
5. 发布节奏约定：骨架库遵循 semver，研究代码明确标注"破坏性变更免责期 12 个月"。

**成功标准**：骨架库可 `pip install` 并通过 CI；演示在 Colab 无 GPU 环境可跑；许可证、行为准则、贡献指南三件套就位。

### 第一阶段：单规则验证与基础设施（1-3 个月）

**任务**：环境搭建；核化 IB-Hebbian 感知验证 [Pogodin & Latham, 2020]；e-prop 认知验证（ALIF + 自适应阈值；计划书原先点名的 Trace Propagation 经考证**不采用**，见 §3.1）[Bellec et al., 2020; Pes et al., 2025]；R-STDP 执行验证（含 TD-LTP Critic；**TD-LTP 文献考证已完成**，见 §3.2）[Frémaux et al., 2013]。

**成功标准**：Hebbian 特征提取器 MNIST 70%+；e-prop RSNN 顺序任务可用，活跃神经元比例 > 60%；R-STDP CartPole 200 步以上，成功偏移 < 10%σR。

### 第二阶段：双规则协同 + 仲裁器（4-8 个月）

**任务**：模态插件框架与双通道融合；文本 + 图像插件；TAAF 快速微调 [Shen et al., 2025]；ES 元学习仲裁器（先仅调节 Hebbian 与 e-prop 两规则）[Confavreux et al., 2025]；**感知层接入脉冲总线**——`research/ib_hebbian/` 改为消费 `biosnn-bus` 插件产出的 `SpikeTrain`（脉冲进、层内速率运算）。⚠️ 这一条补的是**架构连线**，**不是**把感知层换成脉冲神经元：Pogodin & Latham 的网络本身就是 LReLU 速率型，换成脉冲会脱离来源论文。

**成功标准**：插件可独立替换，TAAF 微调样本量 < 从头训练 1/10；跨模态检索 R@1 50%+；仲裁器使双规则冲突事件减少 50% 以上。

### 第三阶段：三规则闭环 + Critic + 仲裁器（9-15 个月）

**任务**：音频插件接入；R-STDP + TD-LTP Critic [Frémaux et al., 2013]；仲裁器扩展至三规则冲突维度（第 1–3 类）[Confavreux et al., 2025]；持续学习基准（CIFAR-100-B0 10 步增量，参考 PS-SNN 的 76.42%）[Hu et al., 2026]；**执行层脉冲化**——把 `research/rstdp/` 的速率型 actor-critic 换成脉冲实现（LIF + **带真实 Δt 窗口的 STDP 资格痕迹** + TD-LTP 群体 Critic），使 §3.2 的 `STDP(Δt_ij)` 公式与「仅计 pre-before-post」那处结构差别**第一次可验**。

> **脉冲化的评测协议（先定死，免得到时候口径对不上）**：脉冲网络的输出**本身是随机的**（输入侧就是伯努利脉冲编码），所以「贪心评测」只能是对**某一个窗口**取 argmax，同一份权重跑两次会得到不同动作。因此评测必须**冻结随机流**：每个评测回合用固定 seed 重采同一串输入脉冲，使同一份权重给出同一个动作。
> 不这么做的话，「贪心评测」量到的是一个**随机策略**——速率版在 `step7_boltz_greedy` 处已经踩过一次同类坑（boltzmann 分支不看 `exploration`，于是「贪心评测」其实是采样），那条教训记在 `sweep_results/README.md` 与本文件 §十一。
> **速率版保留为对照基线**：同一任务、同一规则、脉冲 vs 速率——这本身就是一条有信息量的实验（它回答「脉冲化之后性能掉多少」），而不是把旧结果换掉。

**成功标准**：三模态接入，跨模态检索可用；平均增量准确率 65%+，AF < 8%；规则改变后重新学习速度比从头训练快 3 倍以上。

### 第四阶段：神经发生 + 规模扩展 + 认知增强（16-24 个月）

**任务**：神经发生控制器（多模态场景验证）[Hu et al., 2026]；规模扩展实验（目标 50 万神经元，受 §6.2 约束声明与降级路径约束；**e-prop 的事件驱动实现有先例可循**——NEST 上的事件驱动 e-prop 做到了 2 百万神经元的弱/强 scaling，且在 N-MNIST 上复现了原版性能 [Korcsák-Gorzo et al., 2025]）；多时间尺度工作记忆；元认知门控；**事件优先级——只在"值得算"的时间步上做权重更新**（ESPP 的选择性更新机制，实测只执行 18%–27% 的时间步、且随训练递减 [Graf et al., 2024]）；端到端能效基准测试（Predictive E-prop 作为对照路线之一 [Noè et al., 2026]）。

> **v6.3 对两处措辞的更正（原样保留理由）**
> - **删去「元认知门控 + Predictive E-prop」里的后者**。Predictive E-prop（Noè et al., 2026）全文 **0 次** metacognition / forgetting / working memory——把元认知门控挂在它名下是误引。元认知门控在本项目里的依据是 §2.2 与 §3.4 自己的设计；Predictive E-prop 按**论文自身定位**（"a learning principle rather than a task specific model"）归到 §3.1 的 e-prop 变体，与第一阶段/第三阶段的 e-prop 主线并列。
> - **「事件优先级」按 ESPP 的机制写，不写成某篇论文的术语**。2025–2026 的脉冲网络文献里没有叫 "event priority" 的机制（逐字核过 EMBER／Korcsák-Gorzo／Predictive E-prop／Javis／MEMBRAIN／eprop-PyTorch／ESPP 七篇，`priorit*` 命中 0/0/0/1/0/0/0），最接近的原文表述是 ESPP 的 "ESPP intrinsically has the ability to selectively choose those time steps that matter the most"（§III-C Sparse Weight Updates）。**"事件优先级"是本项目对它的中文转述，不是文献术语**——引用时给 ESPP 的原文与数字，不给一个不存在的术语。

**成功标准**：规模实验达到 §6.2 约束下的目标规模；持续学习 AF < 5%；能效比相比等效 ANN 降低 10 倍以上。跨模态迁移能力以可测量指标（表征相似度 / 线性探测 / 迁移曲线）报告，不设"涌现"类不可验收指标。

### 第五阶段：LLM 集成与协同验证（25-30 个月）

**目标**：将 LLM 作为可替换推理引擎接入 SNN 认知核心，验证"SNN 调用 LLM"的协同能力。

**任务**：

1. **MCP 服务封装**：将认知核心封装为 MCP Server，暴露认知状态查询、关联检索、调用触发等工具接口 [ASTRA, 2026]；
2. **上下文编码模块**：实现 z-score 标准化 top-k 群体编码，将 LLM 嵌入写入 SNN 情景记忆 [Savage, 2026]；
3. **调用触发模块**：实现 STDP 侧向传播驱动的自主触发机制（含背景噪声维持、冲动检测、显著性阈值）[Savage, 2026]；
4. **结果整合模块**：实现 LLM 输出的脉冲编码回注和记忆巩固（含睡眠期噪声巩固）[Membrain, 2026]；
5. **调度决策仲裁**：扩展 ES 元学习仲裁器至第 4 类维度；
6. **协同验证**：在对话场景中验证 SNN 自主触发 LLM 调用的频率、准确性和任务完成率提升。

**成功标准**：

- SNN 在 **≤ 10 轮对话**内产生首次自主触发的 LLM 调用；
- 调用触发准确率（该调用时调用，不该调用时不调用）达到 70%+；
- LLM 返回结果的巩固成功率达到 80%+；
- 端到端任务完成率相比纯 SNN 基线提升 50%+。

### 第六阶段：硬件部署与生态建设（31-36 个月）

**任务**：Loihi 2 部署（注意 CLP-SNN 硬件路径中原型一次性写入后冻结的局限，连续 e-prop 更新的片上支持需预研）[Hajizada et al., 2025]；NIR 兼容；插件生态与文档。

**成功标准**：Loihi 2 上实现实时在线持续学习；插件生态以有机生长为目标（见 §12.6，不作为硬性验收指标）。

---

## 八、关键技术选型汇总

| 技术组件 | 选型 | 理由 | 引用 |
| :--- | :--- | :--- | :--- |
| e-prop 实现 | eprop-PyTorch + SpikingJelly 适配 | 已有 LIF 实现（**SpikingJelly 的许可条款、以及对开发环境 Python 下限的影响，见 ADR-0008**） | Bellec et al., 2020 |
| e-prop 痕迹存储 | **不采用** Trace Propagation | TP 是**另一条规则**（痕迹按神经元而非按突触），不是 e-prop 的省内存版本；两者并列而非替代。裁决见 ADR-0010 | Pes et al., 2025 |
| ALIF 资格痕迹 | Bellec et al. 2020 原文推导 | 自适应阈值时序演化 | Bellec et al., 2020 |
| e-prop 的稀疏更新 | 事件驱动更新（Korcsák-Gorzo）/ 选择性时间步更新（ESPP） | 只在"有事件 / 值得算"的时刻更新：事件驱动版在稀疏网络上做到 2 百万神经元且不损性能；选择性更新实测只跑 18%–27% 的时间步。**两者机制不同、不是同一条** | Korcsák-Gorzo et al., 2025; Graf et al., 2024 |
| e-prop 的深度扩展 | 深度方向的痕迹递归（纯数学，无实验） | 支撑"多层 RSNN 用纯局部规则训练"的方法学论证；**不得用于支撑性能数字** | Millidge, 2025 |
| e-prop 的预测编码变体 | Predictive E-prop | 第三因子换成局部预测误差；与 TBPTT 精度相当但少 70% epoch。论文自定位为**学习原理**，非任务专用模型 | Noè et al., 2026 |
| R-STDP 偏差校正 | TD-LTP Critic | 消除无监督偏差 | Frémaux et al., 2013（规则本体）; Frémaux et al., 2010（偏差结论） |
| 感知层学习 | 核化 IB-Hebbian + 除法归一化 | 避免深度网络的表征瓶颈问题 | Pogodin & Latham, 2020 |
| 三规则仲裁 | ES 元学习仲裁器 | 不依赖全局反向传播 | Confavreux et al., 2025 |
| 模态时间对齐 | TAAF | 时间注意力引导的自适应融合（**双通道融合是本项目自己的设计**，不归因给 TAAF） | Shen et al., 2025 |
| 神经发生 | PS-SNN + CLP-SNN | 模式分离 + 按需扩展 | Hu et al., 2026; Hajizada et al., 2025 |
| LLM 协同架构 | EMBER 式 SNN 基质 + 可替换 LLM | SNN 决定何时行动，LLM 生成内容 | Savage, 2026 |
| 嵌入编码 | z-score top-k / FlyHash | 维度无关编码 / 稀疏二进制 | Savage, 2026; tfatykhov, 2026 |
| 硬件部署 | Intel Loihi 2 + Lava | 实时在线持续学习 | Hajizada et al., 2025 |
| 开源骨架（v6.2 新增） | 独立插件框架库 | 与硬件无关，先行建立社区 | --- |

---

## 九、评估指标体系

| 维度 | 指标 | 第二阶段目标 | 第四阶段目标 | 第五阶段目标 |
| :--- | :--- | :--- | :--- | :--- |
| 在线学习 | e-prop 序列任务 PPL | 可用；**产出与代理梯度基线的量化差距报告** | 差距收窄或有明确归因分析 | --- |
| 持续学习 | 平均增量准确率 | 65%+ | 85%+ | --- |
| 抗遗忘 | 平均遗忘（AF） | < 8% | < 5% | --- |
| 抗遗忘 | 反向迁移（BWT） | > -10% | > -5% | --- |
| **调用决策** | **触发准确性** | --- | --- | **70%+** |
| **调用决策** | **首次触发延迟** | --- | --- | **≤ 10 轮对话** |
| **调用决策** | **巩固成功率** | --- | --- | **80%+** |
| **调用决策** | **任务完成率提升** | --- | --- | **50%+** |
| 网络健康 | 活跃神经元比例 | > 60% | > 80% | > 80% |
| 网络健康 | 脉冲稀疏度 | > 90% | > 98% | > 98% |
| R-STDP 稳定性 | 成功偏移/σR | < 10% | < 5% | < 5% |
| 三规则协同 | 冲突事件比例 | < 20% | < 5% | < 5% |
| 能效 | 推理能效（相对 ANN） | 1x | 10x+ | 10x+（SNN 独立运行时段） |
| 能效 | 端到端能效（含学习 + LLM 调用） | 可观测优势 | 报告调用频率-能耗曲线 | 报告调用频率-能耗曲线 |
| 扩展性 | 神经元规模 | 10 万 | 50 万（受 §6.2 约束） | 50 万（受 §6.2 约束） |
| **开源健康**（v6.2 新增） | **见 §12.6** | 骨架库可用 + CI 绿 | 外部 issue/PR ≥ 10 | 被论文引用 ≥ 1 |
| 生物合理性 | 无全局反向传播 | 完全满足 | 完全满足 | 完全满足（LLM 为外部推理引擎，不参与突触学习） |

---

## 十、风险管理

| 风险 | 概率 | 影响 | 应对策略 | 引用 |
| :--- | :--- | :--- | :--- | :--- |
| R-STDP 无监督偏差 | 高 | 高 | TD-LTP Critic，必须实现 | Frémaux et al., 2010 |
| TD-LTP 规则不收敛 | 中 | 高 | **文献考证已完成**（命名出自 2013 PLoS Comput Biol）；若规则仍不收敛，备选：滑动平均基线（块级奖励均值近似 Critic） | Frémaux et al., 2013 |
| e-prop 存储超限 | 高 | 高 | **稀疏 + INT8 痕迹量化 + 批大小 ≤ 8**（**不是** Trace Propagation——它不是 e-prop 的省内存版本；算式与反解见 §6.2） | Pes et al., 2025（复杂度的出处）; 裁决见 [ADR-0010](docs/adr/ADR-0010-eprop-quadratic-storage-and-trace-propagation.md) |
| 核化 IB-Hebbian 需除法归一化 | 高 | 中 | 侧抑制归一化 | Pogodin & Latham, 2020 |
| ES 仲裁器复杂规则失效 | 中 | 高 | 从双规则简单参数化开始 | Confavreux et al., 2025 |
| TAAF 模态替换破坏对齐 | 高 | 中 | 快速微调 + 对齐锚点 | Shen et al., 2025 |
| SNN 触发频率过高/过低 | 中 | 中 | impulse significance 阈值动态调节 | Savage, 2026 |
| 嵌入维度不匹配 | 低 | 中 | z-score top-k 编码维数无关 | Savage, 2026 |
| EMBER 证据外推失效 | 中 | 中 | §4.7 边界声明；验收基线放宽至 ≤ 10 轮对话 | Savage, 2026 |
| LLM API 延迟/成本失控 | 中 | 中 | 调用频率监控；本地小模型兜底 | --- |
| 8GB 显存不足 | 高 | 高 | 量化 + 分块 + 稀疏 + LLM 远程调用（痕迹存储的算式见 §6.2） | 见 [ADR-0010](docs/adr/ADR-0010-eprop-quadratic-storage-and-trace-propagation.md) |
| **开源维护负担**（v6.2 新增） | 中 | 中 | 骨架库与研究代码分层；破坏性变更免责期；issue 响应不设 SLA | --- |

---

## 十一、优先级与降级策略

| 优先级 | 问题 | 理由 |
| :--- | :--- | :--- |
| ✅ **P0（已完成）** | TD-LTP 文献考证 | 结论：命名与规则本体出自 [12] Frémaux et al. 2013（原文逐字 “we call this learning rule TD-LTP”）。**第五阶段立项门槛通过**；详见 [ADR-0007](docs/adr/ADR-0007-td-ltp-critic-provenance.md) |
| **P0** | R-STDP 无监督偏差 | 不解决则 R-STDP 完全无法工作 |
| **P0** | 核化 IB-Hebbian 除法归一化 | 不解决则感知层不成立 |
| **P1** | e-prop 存储复杂度 | 决定 5 万神经元规模是否可行 |
| **P1** | ES 仲裁器（双规则起步） | 决定多规则能否协同 |
| **P1** | **开源基建与 CI**（v6.2 新增） | 许可证/测试/治理是开源准入证；骨架库先行可提前积累用户与外部 review |
| **P2** | TAAF 模态替换 | 影响插件化实际可用性 |
| **P2** | 显存优化 | 可通过降规模暂时规避（§6.2 降级路径） |
| **P3** | 评估指标完善 | 影响科研质量，不影响运行 |

**降级策略**：

- 若 P0 问题在 6 个月内无解 → 降级为**"半局部学习"**：感知层允许简化全局信号，明确标注为"近似局部学习"；
- 若 50 万神经元实测不可达 → 按 §6.2 降级为最大可承载规模 + 性能-规模曲线；
- 若 SNN 自主触发不可靠（准确率长期 < 50%）→ 降级为"规则触发 + SNN 联想上下文"的半自动模式，保留 SNN 基质但放弃自主触发主张；
- **若维护负担失控（v6.2 新增）** → 骨架库降为"归档维护"状态（仅修关键 bug），研究代码不对外承诺兼容性。

---

## 十二、开源治理与社区建设（v6.2 新增）

### 12.1 许可证策略

| 内容类型 | 许可证 | 说明 |
| :--- | :--- | :--- |
| 源代码 | **Apache-2.0** | 专利授权条款对工业用户友好。**更正：SpikingJelly 不是「与本项目同一许可证」**——它用启智开源许可证 1.0（OIOSL），商业使用或再发布前须在 AITISA 声明披露。裁决见 [ADR-0008](docs/adr/ADR-0008-spikingjelly-license-and-python-floor.md) |
| 文档与计划书 | **CC-BY 4.0** | 署名即可 reuse，适合学术传播 |
| 数据集 | 遵循各数据源许可 | 不提供数据镜像，提供下载/预处理脚本 |
| 第三方依赖 | 许可审计进 CI | Lava/Loihi 相关组件注意 Intel 侧许可条款；eprop-PyTorch 引入前核查其许可证 |

> 现状备注：在许可证文件提交之前，仓库在法律意义上是"保留所有权利"状态——**这是开源的第一优先级**，不依赖任何研究进展。

### 12.2 分层开源路线

```text
第 1-2 个月   骨架库（biosnn-bus）：ModalityPlugin + 脉冲总线 + 注册机制
              特性：无 GPU 依赖、pip 可装、Colab 可跑、semver 稳定
第 4-8 个月   + 参考实现层：e-prop / IB-Hebbian / R-STDP 最小可跑示例
第 9 个月后   + 认知核心 + 持续学习基准（破坏性变更免责期内）
第 25 个月后 + MCP 服务封装 + Loihi 2 部署脚本
```

原则：**研究可以失败，骨架必须可用。** 骨架库不依赖任何学习规则研究先成功，却能在研究成功之前为项目积累用户、信誉与外部眼睛。

### 12.3 测试与 CI

| 检查项 | 工具 | 能拦截的问题 |
| :--- | :--- | :--- |
| 单元测试 | pytest（含 GPU 标记，CPU 可跑子集） | 实现错误 |
| 引用清单结构检查 | 自定义脚本（不联网） | 编号跳号 / 重复、URL 格式非法、核验状态写错字面量 |
| 引用链接存活检查 | lychee | 死链、错误 DOI（拦截"Khacef 署名"类错误的入口） |
| 文档代码块语法检查 | 自定义脚本 / mkdocs 插件 | `self.threshold` 这类未定义引用 |
| 版本号一致性检查 | 自定义脚本 | 文件名与内容版本号错位 |
| Markdown 表格结构检查 | 自定义脚本 | 表格里漏转义的竖线——该行超出的单元格在渲染时被静默丢弃，读原文却看不出问题 |
| 三语一致性检查 | 自定义脚本 | 改了一种语言忘了另两种——漂移的译文比没有译文更误导人 |
| 最小复现脚本 | 每篇核心文献对应一个 `examples/paper_<name>.py` | "论文说能跑但仓库跑不通" |
| 基线差距报告 | `scripts/gap_report.py`（CI artifact） | 文档引用的差距数字在记录里追溯不到，或三语各指一次不同的运行 |
| 依赖许可审计 | `pip-licenses` + 自定义脚本 | 引入强 copyleft 依赖；许可证不明的依赖未经登记就混进来 |
| wheel 干净环境安装 | `uv build` + 全新 venv | "本机跑得通、装到别人机器上就崩"——分发包本身不可用 |
| pre-commit | ruff / black / 上述检查钩子 | 提交即拦截 |

### 12.4 文档体系

- **README**：一段话说清项目是什么、不是什么（明确"研究原型，非生产框架"）；
- **快速开始**：5 分钟内注册一个自定义模态插件（Colab 链接）；
- **插件开发指南**：`encode` / `get_membrane` / `decode` 三方法教程 + 测试模板；
- **架构决策记录（ADR）**：每个重大取舍（为什么 e-prop 的二次存储成立却**不采纳** Trace Propagation、为什么 ES 仲裁、为什么 MCP）一篇，这是对抗 bus factor = 1 的核心手段；
- **复现清单**：每个阶段的精确环境（commit hash、依赖版本、随机种子）。

### 12.5 社区治理

- **CONTRIBUTING.md**：开发环境搭建、PR 流程、提交前核查清单（引用点验 + 代码块检查）；
- **CODE_OF_CONDUCT.md**：采用 Contributor Covenant；
- **Issue/PR 模板**：bug 报告强制要求环境信息与复现脚本；
- **破坏性变更免责期**：研究代码前 12 个月明确标注 API 不稳定，骨架库独立 semver 不受影响；
- **响应预期管理**：单人维护，issue 响应不设 SLA，在 README 明示——管理预期比假装有团队更利于社区信任。

### 12.6 开源成功指标（现实版）

| 时间 | 指标 | 性质 |
| :--- | :--- | :--- |
| 第 2 个月 | 骨架库 `pip install` 可用、CI 全绿、Colab 演示可跑 | 硬性 |
| 第 8 个月 | 外部 issue/PR ≥ 10；骨架库被非作者项目 import ≥ 1 | 硬性 |
| 第 15 个月 | e-prop 复现 + 基线差距报告被外部引用 ≥ 1（论文/issue/博客） | 硬性 |
| 第 24 个月 | 外部贡献的模态插件 ≥ 1 | 有机目标（不硬性验收） |
| 第 36 个月 | 第三方插件生态形成 | 愿景（不作为验收指标，避免冷启动悖论） |

核心成功定义：**被同行论文当作 baseline 或工具引用**——对这个体量的项目，这是比 star 数更真实的成功度量。

---

## 十三、总结

BioSNN-Plug v6.3 的核心创新在于：**完全放弃代理梯度，采用三种纯局部学习规则的有机组合，针对规则间的兼容性问题给出系统性解决方案，让 SNN 认知主体学会调度 LLM 作为可替换推理引擎，并以开源方式让整个验证过程公开可复现。**

- 核化 IB-Hebbian + 除法归一化驱动感知层特征提取 [Pogodin & Latham, 2020]；
- e-prop + ALIF 扩展驱动认知层在线时序学习 [Bellec et al., 2020]；
- TD-LTP Critic + R-STDP 驱动执行层强化学习决策 [Frémaux et al., 2013]；
- ES 元学习仲裁器动态调节三规则冲突与 LLM 调度决策 [Confavreux et al., 2025]；
- TAAF 双通道融合实现模态插件化 [Shen et al., 2025]；
- 神经发生 + 模式分离学习实现无损规模扩展 [Hu et al., 2026; Hajizada et al., 2026]；
- EMBER 式协同架构让 SNN 决定"何时行动"，LLM 负责"如何生成" [Savage, 2026]；
- **分层开源策略让骨架先行、研究随后、成果公开可验**（v6.2 新增）。

这条路径的技术风险确实更高，但它所探索的，正是类脑计算最核心的科学问题：**智能能否从局部规则中涌现，并学会借助外部推理扩展自身的边界。**

---

## 参考文献

[1] Bellec, G., Scherr, F., Subramoney, A., Hajek, E., Salaj, D., Legenstein, R., & Maass, W. (2020). A solution to the learning dilemma for recurrent networks of spiking neurons. *Nature Communications*, 11, 3625. https://www.nature.com/articles/s41467-020-17236-y

[2] Pes, L., Yin, B., Stuijk, S., & Corradi, F. (2025). Traces Propagation: Memory-Efficient and Scalable Forward-Only Learning in Spiking Neural Networks. *arXiv preprint arXiv:2509.13053*. https://arxiv.org/abs/2509.13053

[3] Frémaux, N., Sprekeler, H., & Gerstner, W. (2010). Functional Requirements for Reward-Modulated Spike-Timing-Dependent Plasticity. *Journal of Neuroscience*, 30(40), 13326-13337. https://www.jneurosci.org/content/30/40/13326

[4] Pogodin, R., & Latham, P. E. (2020). Kernelized information bottleneck leads to biologically plausible 3-factor Hebbian learning in deep networks. *Advances in Neural Information Processing Systems*, 33. https://proceedings.nips.cc/paper/2020/hash/517f24c02e620d5a4dac1db388664a63-Abstract.html

[5] Confavreux, B., Agnes, E. J., Zenke, F., Sprekeler, H., & Vogels, T. P. (2025). Balancing complexity, performance and plausibility to meta learn plasticity rules in recurrent spiking networks. *PLoS Computational Biology*, 21(4), e1012910. https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012910

[6] Shen, J., Xie, Y., Xu, Q., Pan, G., Tang, H., & Chen, B. (2025). Spiking Neural Networks with Temporal Attention-Guided Adaptive Fusion for imbalanced Multi-modal Learning. *Proceedings of the 33rd ACM International Conference on Multimedia*. https://dl.acm.org/doi/10.1145/3746027.3755622

[7] Hu, K., Wen, L., Zhang, T., & Zhang, H. (2026). PS-SNN: pattern separation learning for expandable spiking neural networks in class-incremental learning. *Scientific Reports*, 16, Article **12653**. https://www.nature.com/articles/s41598-026-42970-6 （v6.3 更正：原写 Article 42970——那是 DOI 尾段，不是文章号）

[8] Hajizada, E., Rager, D., Shea, T., Campos-Macias, L., Wild, A., Hüllermeier, E., Sandamirskaya, Y., & Davies, M. (**2025**). Online Continual Learning on Intel Loihi 2 via a Co-designed Spiking Neural Network. *arXiv preprint arXiv:2511.01553*. https://arxiv.org/abs/2511.01553 （v6.1 更正：原误署为 Khacef et al.；v6.3 更正：年份改为 2025——arXiv 编号 2511 即 2025 年 11 月，2026 只是 v2 修订年）

[9] Savage, W. (2026). EMBER: Autonomous Cognitive Behaviour from Learned Spiking Neural Network Dynamics in a Hybrid LLM Architecture. *arXiv preprint arXiv:2604.12167*. https://arxiv.org/abs/2604.12167

[10] tfatykhov. (2026). MEMBRAIN: Neuromorphic Memory Bridge for LLM Agents. *GitHub Repository*. https://github.com/tfatykhov/membrain （v6.3 更正：原把项目名当作作者；这是个人仓库，作者是 GitHub 账号 tfatykhov）

[11] christophejlegros-lgtm. (2026). ASTRA: Unified Research Lab + MCP Server. *GitHub Repository*. https://github.com/christophejlegros-lgtm/ASTRA-Unified-ResearchLab-MCP-v2.7 （v6.3 更正：同上，作者是 GitHub 账号而非项目名；另：该仓库**提供了**把 SNN 引擎经 MCP Server 暴露给 Claude Desktop 的接口，但「已验证可用」是比「提供了接口」更强的说法，正文已按此强度改写）

[12] Frémaux, N., Sprekeler, H., & Gerstner, W. (2013). Reinforcement Learning Using a Continuous Time Actor-Critic Framework with Spiking Neurons. *PLoS Computational Biology*, **9**(4), e1003024. https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003024 （**v6.3 新增**：TD-LTP 的命名出处。原文逐字："Because it has, roughly, the form of 'TD error signal × Hebbian LTP', we call this learning rule TD-LTP."（Critic learning 一节；对应 Eq. 17 与 Figure 2A 图注 "TD-LTP is the learning rule given in Eq. 17."）。该文开放获取，全文可核）

[13] Tihomirov, Y., Rybka, R., Serenko, A., & Sboev, A. (2025). Combination of reward-modulated spike-timing dependent plasticity and temporal difference long-term potentiation in actor-critic spiking neural network. *Cognitive Systems Research*, 90, 101334. https://doi.org/10.1016/j.cogsys.2025.101334 （**v6.3 新增**：TD-LTP 在脉冲 actor-critic 上的后续应用线，与本项目 §3.2 的架构选择同构，可作为该路线可行的外部证据）

[14] fangwei123456. (2026). SpikingJelly: An open-source deep learning framework for spiking neural networks based on PyTorch. *GitHub Repository*. https://github.com/fangwei123456/spikingjelly （**v6.3 新增**：许可证事实核验——SpikingJelly 用**启智开源许可证 1.0（OIOSL）**，**不是** Apache-2.0；商业使用或再发布前须在 AITISA 声明披露；2.0.0rc1 要求 Python ≥ 3.11。此前 §12.1 把它写成"与本项目同一许可证"，本版已更正。裁决见 [ADR-0008](docs/adr/ADR-0008-spikingjelly-license-and-python-floor.md)）

[15] Bellec, G., Scherr, F., Subramoney, A., Hajek, E., Salaj, D., Legenstein, R., & Maass, W. (2019–2020). eligibility_propagation: 论文 [1] 的官方实现。*GitHub Repository*. https://github.com/IGITUGraz/eligibility_propagation

[16] BEKO2210. (2026). Javis: An associative SNN memory co-processor for LLM agents. *GitHub Repository*. https://github.com/BEKO2210/Javis （**v6.3 新增**：与 EMBER 同属"SNN 当基质、LLM 当嘴"路线的同类先例，但分工相反——EMBER 决定**何时**行动，Javis 决定**送什么进上下文**。⚠️ 许可证是 **PolyForm Noncommercial 1.0.0**（非 OSI 认可），**代码不能并入本仓库**；⚠️ 同名混淆：`JuliaAnimators/Javis.jl` 是 MIT 的 Julia 动画库，勿张冠李戴。详见 [`docs/related_projects.md`](docs/related_projects.md)）

[17] Korcsák-Gorzo, A., Espinoza Valverde, J. A., Stapmanns, J., Plesser, H. E., Dahmen, D., Bolten, M., van Albada, S. J., & Diesmann, M. (2025). Event-driven eligibility propagation in large sparse networks: efficiency shaped by biological realism. *arXiv preprint* arXiv:2511.21674. https://arxiv.org/abs/2511.21674 （**v6.3 新增**：e-prop 的**事件驱动**实现，接进 NEST，弱/强 scaling 到 2 百万神经元，N-MNIST 上复现原版性能。为遵守严格局部性改掉了原版资格痕迹滤波器对输出神经元时间常数的依赖。注：arXiv 元数据把姓氏写作 `Korcsak-Gorzo`（去重音），文献通行拼法为 **Korcsák-Gorzo**；同名混淆风险：粒子物理界的 Katherine Korcsak-Gorzo 是另一个人。⚠️ 该文**没有**"事件优先级"这一机制或说法，勿混）

[18] Millidge, B. (2025). Generalizing E-prop to Deep Networks. *arXiv preprint* arXiv:2512.24506. https://arxiv.org/abs/2512.24506 （**v6.3 新增**：把 e-prop 的资格痕迹推广到任意深度网络/DAG 的纯数学笔记，**无实验**。⚠️ 题名拼写两版并存——论文 PDF 与 arXiv HTML 用 `Generalizing`，arXiv 元数据/OpenAlex 用 `Generalising`；本表按论文本体登记。它**不支持**任何性能数字，也不含"事件优先级"）

[19] Noè, D., Yamamoto, H., Katori, Y., & Sato, S. (2026). Predictive E-prop: A biologically inspired approach to train predictive coding-based recurrent spiking neural networks. *bioRxiv preprint* 2026.02.12.705507. https://doi.org/10.64898/2026.02.12.705507 （**v6.3 新增**：把 e-prop 的"第三因子"换成预测编码的**局部预测误差**。作者自定位为"a learning principle rather than a task specific model"——因此它属于 e-prop 变体，**不是**元认知/持续学习工作的出处（全文 0 次 metacognition / forgetting / working memory）。⚠️ 许可 `cc_no`（All rights reserved），**不能复用其图**；⚠️ 近名混淆：同组 2025 年另有一篇 e-prop 论文（*Neuromorphic Computing and Engineering* 5(4) 044002）内容是"连接度与内禀噪声分离"，勿互记结论）

[20] Graf, L., Su, Z., & Indiveri, G. (2024). EchoSpike Predictive Plasticity: An Online Local Learning Rule for Spiking Neural Networks. *arXiv preprint* arXiv:2405.13976. https://arxiv.org/abs/2405.13976 ；代码 https://github.com/Zhe-Su/ESPP （**v6.3 新增**：**"事件优先级"这个中文提法的原文出处**——§III-C 用输入活动阈值 + 损失阈值让规则自己决定哪些时间步才更新，原文 "ESPP intrinsically has the ability to selectively choose those time steps that matter the most"，实测 18%–27% 的时间步。SHD 上 84.32%。⚠️ 它是**另一条规则**（预测编码 + 对比编码的层间局部规则），不是 e-prop 的变体，别写进 §3.1 的 e-prop 链；⚠️ 论文脚注的 `largraf/EchoSpike` 已 404，权威仓库是 `Zhe-Su/ESPP`（Apache-2.0，LICENSE 的版权人一行仍是模板占位符）；⚠️ "ESPP" 是高频缩写（员工购股计划、欧洲粒子物理战略…），SNN 语境下才是这一个）

[21] Frenkel, C. (2022). eprop-PyTorch: PyTorch implementation of the eligibility propagation (e-prop) learning algorithm. *GitHub Repository*. https://github.com/ChFrenkel/eprop-PyTorch （**v6.3 新增**：计划书 §1.4/§八 点名的"e-prop 实现参考"。**Apache-2.0**（版权人 University of Zurich），许可合规；但**非官方**（官方是 [15]）、**非 PyPI、无 tag/release**（只能锚 commit `0f32a8f2`，2022-02-18 单一提交）、全仓只有 **LIF 且已显式移除 ALIF**、任务只有证据累积一种。⚠️ **ALIF 的对照物只能用 [15] 与 [1] 原文**——[ADR-0008](docs/adr/ADR-0008-spikingjelly-license-and-python-floor.md) 决策 3 里"拿它的源码核对 ALIF 资格痕迹"一句已在该 ADR 的后续更正注记中撤回） （**v6.3 新增**：LICENSE 正文是标准 **BSD-3-Clause**（GitHub API 标 `NOASSERTION` 只因版权头格式非标准，不是许可证不明），与 Apache-2.0 兼容。**仅作阅读参考与交叉验证，不进依赖**——与 [14] 同样处理）


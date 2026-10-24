# BioSNN-Plug

**面向高生物合理性的全模态脉冲神经网络认知原型**

> Spiking neural networks (SNN) · biologically plausible local learning rules · no surrogate gradients · e-prop / Hebbian / R-STDP · neuromorphic computing · modality plugins · LLM orchestration

[English](README.en.md) · [项目计划书 v6.2](BioSNN-Plug_项目计划书_v6.2.md) · [v6.3（引用修订版）](BioSNN-Plug_项目计划书_v6.3.md) · [计划书差距审计](BioSNN-Plug_计划书差距审计.md) · [插件开发指南](docs/plugin_guide.md) · [架构决策记录](docs/adr/README.md)

> **两个版本的关系**：**v6.2 是正式计划书**；**v6.3** 把第一批引用考证的更正并了进去（改了哪些见 [`docs/references.md`](docs/references.md) 的「计划书正文待更正项」）。引用计划书时以 v6.2 为准；需要用到更正后的表述（例如 Trace Propagation 的定位、TD-LTP 的出处）时用 v6.3。

[![CI](https://github.com/lzmd-arch/BioSNN-Plug/actions/workflows/ci.yml/badge.svg)](https://github.com/lzmd-arch/BioSNN-Plug/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lzmd-arch/BioSNN-Plug/blob/main/examples/quickstart_register_plugin.ipynb)

## 发表与存档

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23429379.svg)](https://doi.org/10.5281/zenodo.23429379)

| 产出 | 类型 / 许可 | 标识 | 状态 |
| :--- | :--- | :--- | :--- |
| **P2 论文**：W2 e-prop 与同预算 BPTT 的对照 | 预印本 / CC-BY-4.0 | [10.5281/zenodo.23429379](https://doi.org/10.5281/zenodo.23429379) | 已发表（v1.1，2026-10-17） |
| **P4 论文**：仲裁何时可被测量（协同基准 + 协议处决） | 预印本 / CC-BY-4.0 | [10.5281/zenodo.23613539](https://doi.org/10.5281/zenodo.23613539) | 已发表（v1.0，2026-10-24） |
| **记录 A**：W2 发现与协议快照 | 报告 / Apache-2.0 | [10.5281/zenodo.23428426](https://doi.org/10.5281/zenodo.23428426) | 已发布（v0.2，2026-10-17） |
| **记录 C**：P4 设计笔记（仲裁形式化、协同性基准判据、三条裁决） | 报告 / Apache-2.0 | [10.5281/zenodo.23603639](https://doi.org/10.5281/zenodo.23603639) | 已发布（v0.1，2026-10-24） |
| **记录 D**：架构与设计决策存证（计划书 v6.3、ADR 索引、四轴框架） | Other | [10.5281/zenodo.23603642](https://doi.org/10.5281/zenodo.23603642) | 已发布（v0.1，2026-10-24） |

⚠️ 存档是**不可变快照**，记的是存款当时的状态；当前数字以本 README 与[复现记录](docs/reproducibility.md)为准。

## 这是什么

研究原型，验证一个问题：**智能能否从纯局部学习规则中生长出来，并学会借助外部推理扩展自身边界？**

三条具体主张：

1. **跳过代理梯度**——突触更新全部由局部信号驱动（感知层核化 IB-Hebbian、认知层 e-prop、执行层 R-STDP）；
2. **模态插件化**——文本/图像/音频作为独立插件接入共享认知核心，新增模态不改已有架构；
3. **SNN 自主调用 LLM**——SNN 是认知主体，决定"何时行动、关联什么"；LLM 作为可替换的推理引擎只负责"生成什么"。

## 这不是什么

**不是生产框架。** 不追求 SOTA，不承诺可用性，单人维护，issue 不设 SLA。

**不是通用 SNN 库。** 性能不是目标，价值在于验证一条高风险路线是否成立。

**是证据，不是灵感。** 结论要能在本仓库复现；核验不到的引用如实标 `unverified`（见[引用清单](docs/references.md)）。

## 和同类项目的关系

同类路线一眼对照。**每一格的依据、以及许可边界（哪个能进依赖、哪个一个字都不能抄），
在 [`docs/related_projects.md`](docs/related_projects.md)。**

| | 纯局部学习 | 多模态插件 | 自主触发 | 持续学习 | 开源框架 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Javis** | ◐ 全是 STDP 家族、无反向传播（12 条可选）；⚠️ **但作者自建的 `iter-45` reward 基准两臂都没过随机水平**（README 逐字「neither arm reaches above-chance accuracy」） | ✗ | ✗ | ✗ | ◐ Rust，但**PolyForm 非商业** |
| **EMBER** | ◐ STDP + 资格痕迹 × 多巴胺门控 | ✗（仅文本） | ◐ 观察级，**N=1** | ✗ | ✗ 无公开仓库 |
| **MEMBRAIN** | ◐ Voja + PES | ✗ | ✗ | ◐ 自报与实现不符 | ✅ Nengo（MIT） |
| **eprop-PyTorch / ESPP** | ✅ e-prop；ESPP 是另一条并列规则 | ✗ | ✗ | ✗ | ◐ 代码片段，非框架 |
| **BioSNN-Plug**（本项目） | ✅ 三条线全部验收 | ✅ 四路插件（图像/MNIST/文本/音频） | ○ 第五阶段 | ○ 第三阶段 | ✅ `biosnn-bus` 已上 PyPI |

图例：**✅ 已实现并有验收数据** · **◐ 部分**（机制或证据不全） · **○ 只在路线图里（计划书有任务，未开始）** · **✗ 无**

⚠️ **本表最后一行有两格是 ○ 而不是 ✅**——「自主触发」排在第五阶段、「持续学习」排在第三阶段，
两者的实现都**尚未开始**。把符号分开写，就是为了不让这张表读起来像我们已经做到了。

## 研究定位

把"智能为什么涌现"拆成四轴：**基质**（规模、维度、突触数）、**动力学**（全局状态变量偏置学习——
激素、情绪、巩固、本能）、**组织**（基因预编码、脑区分化、新旧更替、自增殖）、**接地**（物理规则、
真实感知、模态拟真）。四者若真有关系，是**乘法**：任何一轴为零，整体就封顶。

**我们的押注**：退出基质轴的竞争——那里不缺参与者，也不缺算力；把资源押在
**动力学 × 组织能否替代基质**上，做一台能测这个问题的仪器。若答案是肯定的，规模就不是
涌现的原因，而是组织方式允许达到的尺度。

测量纪律不是宣言：预注册、平凡解排查、机器检查单与负向验证，都记录在[复现记录](docs/reproducibility.md)里。

## 当前状态

| 部分 | 状态 | 最近更新 |
| :--- | :--- | :--- |
| `biosnn-bus` 骨架库 | **0.3.0（含音频插件，已发布到 PyPI）**——插件接口、注册与发现、脉冲总线、**TAAF 时间注意力融合**（rate-based gating，权重由各路发放率现算，**无反向传播**）；`Used in:` [Axo](https://github.com/lzmd-arch/Axo)（生产终端）：其独立进程 `axo.busd` 常驻实例化 `SpikeBus` 并持续 `step()`（融合 + 稀疏随机投影），新记忆经自定义 `ModalityPlugin` 编码后进总线；UI 只读 stdlib 写的 JSONL 事件，不 import 本库 | 2026-10-18 |
| 模态插件 | **四路**：`image_diff`（DVS 事件编码）、`mnist_pulse`（发放率）、**`text_zscore`**（冻结本地 embedding + z-score top-k 群体编码）、**`audio_cochlea`**（对数频带分解 + 发放率；未发布到 PyPI） | 2026-10-18 |
| 研究代码：三条验证线 | **第一阶段实现完成**——W1 Hebbian 感知 MNIST **97.79%**（默认闭式解读出；SGD 读出 98.01%）✓；W2 e-prop 序列 sMNIST **0.7764**（30 epoch × 5 种子）→ **0.8889**（**官方收官**：**LIF** @ lr=1e-2 @ 250 epoch，已收敛；同档 ALIF 0.8831 为对照；**同预算 BPTT 对照 0.9630，领先 7.41 个百分点**，这条差距照实写：局部规则仍落后于反向传播；**配方待遇量化**：BPTT 的优化器形式在小学习率上值 **+8.53 pp**（1e-3），在最终选定的学习率上只剩 **+0.88 pp**（落在并列带内）——操作点上两臂的配方是可比的）；lr 交互扫描完成，**LIF 在四档 lr 上全网格领先** ✓；W3 R-STDP CartPole 中位数 **237.8 步**（阈值 ≥ 200）✓。数字与复现记录见[复现记录](docs/reproducibility.md) | 2026-10-18 |
| **网络规模** | **256 个脉冲神经元**（W2）；W1 另有 3,072 个**速率型**单元（不算脉冲神经元）。计划书 §2.2 的认知核心目标是约 5 万——**差约 195 倍**，详见 [research/README](research/README.md) | 2026-10-18 |
| 仲裁器 | **零件就位**（`research/arbiter/`）：冲突判定（余弦 > 90°）+ ES 元学习仲裁器，**已接进两条研究线**（共享输入突触）。⚠️ 但只共享输入→隐藏一处，且「减少冲突事件 50%」在合成数据上是**平凡成立**的——见该目录 README | 2026-10-24 |
| 执行层脉冲化（第三阶段） | **入口、线性 TD Critic、TD-LTP 群体 Critic 均已就位，但一个都没过 200。**⚠️ **2026-10-18 之前的数字全部作废**：训练回合漏了环境种子，同种子两次轨迹不同；修复后 **10 种子对照**：线性 TD 中位数 50.4、群体默认 55.0、工作点 37.1——**区间全部重叠、均 0/10**，三个主臂分不开。速率版中位数 237.8 仍是唯一通过验收的实现。见 [research/rstdp_spiking](research/rstdp_spiking/README.md) | 2026-10-18 |
| 跨模态检索 | **十轮检索实验**：原标准（绝对 R@1 50%）在两种数据上都无法区分「局部学习成立」，已重述为**恢复率**（合成集 75.2%、Flickr8k **78.7%**，完整网格）；Flickr8k 的**前端**另有验收口径——caption 内容词线性探针 AUC 中位数 **0.790**（文本侧自查 1.000） | 2026-10-17 |
| 认知核心 / LLM 协同 | **未开始**，见[计划书 §七](BioSNN-Plug_项目计划书_v6.2.md) | — |

**第二阶段的四条成功标准，实况**：

| 标准 | 状态 |
| :--- | :--- |
| 插件可独立替换 | ✅ 四路插件 + entry points，骨架库已上 PyPI |
| TAAF 微调样本量 < 从头训练 1/10 | ⚠️ **标准随 [ADR-0012](docs/adr/ADR-0012-taaf-locality-and-bus-output-contract.md) 消解，非达成**——融合权重按红线根本不训练，这条标准**失去了所指**（没有微调，就没有「微调样本量」）。见计划书修订说明 ㉗ |
| 跨模态检索 R@1 50%+ | ⚠️ **原标准（绝对 50%）未达成且被判无区分力**（合成集可被记忆化满足；Flickr8k 冻结前端上界 0.076）。**已重述为「恢复率」并通过**：局部规则拿回线性上界的 **75.2%（合成集）/ 78.7%（Flickr8k，完整网格）**——见 [research/retrieval](research/retrieval/README.md) |
| 仲裁器使双规则冲突事件减少 50%+ | ✅ 在**真实共享突触**上达标（10 → 0，`retention` 0.3） |

两处需要说清楚的边界：

- **TAAF 已实现，但默认策略仍是拼接**：`TemporalAttentionFusion` 是**显式选用**的融合策略（`SpikeBus(strategy=...)`），默认仍是平凡拼接。⚠️ 它的注意力权重**由各路发放率现算、不训练**（rate-based gating）——这条红线用 **AST 结构断言**钉死，见 [ADR-0012](docs/adr/ADR-0012-taaf-locality-and-bus-output-contract.md)。
- **感知层允许冻结预训练前端**（边界声明，见[计划书 §2.3](BioSNN-Plug_项目计划书_v6.3.md)）：「纯局部学习」的主张范围是**突触学习规则**，不是**输入编码器**。判据是「参数进不进梯度」，不是「模型大不大」——冻结的 2GB 预训练 embedding 与固定随机投影等价；反过来，100 行的本地编码器若用反向传播训自己参数，就**违反**这条边界。文本插件用的 `nomic-embed-text`（Ollama，768 维）就是冻结的。
- （骨架库为何独立成包、为何刻意不包含这些东西，见 [ADR-0001](docs/adr/ADR-0001-skeleton-as-separate-library.md)。）
- 骨架库**已发布到 PyPI**：`pip install biosnn-bus`（PyPI 上是 `0.3.0`，含 `audio_cochlea` 音频插件）。发布走 PyPI trusted publishing（OIDC，不存 token），流程见 [release.yml](.github/workflows/release.yml)。

## 探索项（不进当前排期）

以下三项已立项，但**不占用当前阶段的任何排期**。每条写明激活条件，条件不满足就不启动——
写在这里，是为了让"没做"和"没打算做"在门面上区分开。

| 项 | 归属 | 激活条件 |
| :--- | :--- | :--- |
| **世界模型组件** | 阶段四正式组件（承接 Predictive E-prop） | 阶段三验收通过，且组件化 ADR 立起来 |
| **唤醒通道** | 动力学轴的组件化 | 组件化 ADR 通过 |
| **PINN 物理先验** | 路线图 v7.0 候选（探索笔记） | 阶段三验收 + CL 探针落地；复查日期 2027 夏 |

## 架构

数据自下而上流动。✅ 是当前仓库里已经能跑的；◐ 是**该层的单条学习规则已在第一阶段验证**、模块整体仍未实现（多层网络、模态解码器、工作记忆回路等）；⬜ 是计划书里尚未开始实现的。画在同一张图上，因为这张图同时是路线图。

```mermaid
flowchart TB
    subgraph L5["LLM 调度与协同层"]
        MCP["MCP / API 接口<br/>可替换的推理引擎<br/>⬜ 第五阶段"]
    end

    subgraph L4["执行层"]
        ACT["动作生成 + 模态解码器<br/>R-STDP + 奖励预测 Critic<br/>◐ 第一阶段：速率版学习规则已验证（W3，中位数 237.8）<br/>◐ 第三阶段：脉冲版入口 + 两种 Critic 已实现<br/>&nbsp;&nbsp;&nbsp;⚠️ 10 种子下 0/10 过 200，且三主臂区间重叠<br/>⬜ 模态解码器"]
    end

    subgraph L3["认知层（认知核心）"]
        WM["工作记忆<br/>RSNN + LIF（默认；ALIF 待重评估）+ e-prop<br/>◐ 第一阶段：学习规则已验证（W2）<br/>⬜ 多层 RSNN 与工作记忆回路"]
        EM["情景记忆<br/>模式分离 + 神经发生<br/>⬜ 第四阶段"]
        MG["元认知门控<br/>不确定性监控<br/>⬜ 第四阶段"]
    end

    subgraph ARB["跨规则仲裁 research/arbiter"]
        CF["冲突判定：余弦 > 90°<br/>✅ 已实现"]
        ES["ES 元学习仲裁器<br/>✅ 已接进两条线（共享输入突触）<br/>⚠️「减少 50%」在合成数据上平凡成立"]
    end

    subgraph L2["骨架库 biosnn-bus"]
        BUS["SpikeBus<br/>按融合通道分组 · 时间网格对齐<br/>先投影后融合 · 稀疏随机投影<br/>TAAF 时间注意力融合（rate-based gating，不训练）<br/>✅ 已实现"]
    end

    subgraph L1["感知层（插件化）"]
        IMG["图像 image_diff<br/>DVS 事件编码<br/>✅ 已实现"]
        MN["MNIST mnist_pulse<br/>发放率编码<br/>✅ 已实现"]
        TXT["文本 text_zscore<br/>冻结 embedding + z-score top-k<br/>✅ 已实现"]
        AUD["音频 audio_cochlea<br/>对数频带分解 + 发放率<br/>✅ 已实现（未发布到 PyPI）"]
        THIRD["第三方插件<br/>entry points 接入<br/>✅ 机制已就绪"]
        HEB["学习规则：核化 IB-Hebbian + 除法归一化<br/>◐ 第一阶段已验证（W1，MNIST 97.79%）"]
    end

    IMG --> BUS
    MN --> BUS
    TXT --> BUS
    AUD -.-> BUS
    THIRD -.-> BUS

    BUS --> WM
    BUS -.->|接入脉冲总线| HEB
    HEB -.->|特征| WM
    WM --> EM
    EM --> MG
    MG --> ACT
    MG -.->|触发调用| MCP
    MCP -.->|结果编码回注| EM
    ARB -.->|调停层间冲突| WM
    ARB -.->|调停层间冲突| ACT

    subgraph W4["跨模态检索 research/retrieval"]
        FAC["联想层：外积 Hebb / delta / 岭回归上界 / kNN<br/>✅ 已实现（前两者是局部规则，后两者是上界探针）"]
        CRIT["两套口径<br/>✅ 绝对 R@1：未达成且被判无区分力<br/>✅ 恢复率：合成集 75.2%、Flickr8k 78.7%"]
        FE["前端验收：caption 内容词线性探针<br/>✅ 图像侧 AUC 0.790（文本侧自查 1.000）"]
    end

    BUS -.->|两路表示| FAC
    FAC --> CRIT
    FE -.->|说明上界为何低| CRIT
```

层间冲突由 **ES 元学习仲裁器**调节：冲突判定与仲裁器都已实现，并**已接进两条研究线**（`research/arbiter/shared_synapse.py`，两条规则共享输入→隐藏那一组突触）。⚠️ 但「冲突事件减少 50%」在合成数据上是**平凡成立**的——最优解是把一条规则关掉，所以降幅要与 `retention` 一起读；详见 [`research/arbiter/README`](research/arbiter/README.md)。

⚠️ **图里的 ✅ 是「已实现」，不等于「已接入路线图上的那个位置」**：例如 TAAF 已实现，但它是**显式选用**的融合策略（默认仍是拼接）；文本插件已实现，但它接的是**冻结的预训练前端**（边界声明见[计划书 §2.3](BioSNN-Plug_项目计划书_v6.3.md)——「纯局部学习」只管突触学习规则，不管输入编码器）。各层学习规则的完整说明见[计划书](BioSNN-Plug_项目计划书_v6.2.md) §2、§3。

## 快速开始

只依赖 numpy，无需 GPU：

```bash
pip install biosnn-bus
```

> ⚠️ **依赖安全提醒（2026-10）**：CA 厂商 eMudhra 的中间证书私钥被攻破（签发根为 emSign ECC Root CA - G3），`certifi < 2026.10.19` 仍信任这条链；本仓已锁到 `certifi 2026.10.19`，自己环境里的旧包请执行 `pip install -U "certifi>=2026.10.19"`（用 uv 的重新 `uv sync`）。

（要从 `main` 装**尚未发布**的改动，才走 Git：`pip install "biosnn-bus @ git+https://github.com/lzmd-arch/BioSNN-Plug.git#subdirectory=packages/biosnn-bus"`）

一个模态插件就是三个方法加两个属性：

```python
import numpy as np

from biosnn_bus import ModalityPlugin, PassThroughMembrane, SpikeBus, SpikeTrain


class LevelEncoder(ModalityPlugin):
    """把一维信号按水平编码成群体脉冲。"""

    def __init__(self, spike_dim: int = 16) -> None:
        self._spike_dim = spike_dim
        self.levels = np.linspace(0.0, 1.0, spike_dim)

    @property
    def modality_name(self) -> str:
        return "level"

    @property
    def spike_dim(self) -> int:
        return self._spike_dim

    def encode(self, raw_input) -> SpikeTrain:
        signal = np.asarray(raw_input, dtype=np.float64).reshape(-1)
        radius = 0.5 / (self._spike_dim - 1)
        return SpikeTrain(data=np.abs(signal[:, None] - self.levels) <= radius)

    def get_membrane(self):
        return PassThroughMembrane()

    def decode(self, spike_output: SpikeTrain):
        return spike_output.data.astype(float) @ self.levels


bus = SpikeBus(bus_dim=64, seed=0)
bus.register(LevelEncoder())
output = bus.step({"level": np.linspace(0, 1, 12)})
print(output)
```

[Colab 演示](https://colab.research.google.com/github/lzmd-arch/BioSNN-Plug/blob/main/examples/quickstart_register_plugin.ipynb)（5 分钟，无 GPU）会画出从插件到认知核心入口的完整脉冲栅格图。

想让第三方包提供插件？**不需要改本仓库任何一行代码**，见[插件开发指南](docs/plugin_guide.md#让第三方包提供插件)。

## 仓库结构

```text
packages/biosnn-bus/   骨架库：独立分发包，遵循 semver，不依赖任何研究代码
research/              研究代码：第一阶段起填充，12 个月破坏性变更免责期
examples/              可跑的示例；脚本是唯一真相源，notebook 由它生成
docs/                  插件指南、ADR、复现清单、引用清单
scripts/               CI 与 pre-commit 用的检查脚本
```

## 维护预期

单人维护，带完整复现步骤的报告优先处理。

- `biosnn-bus` 遵循 semver，当前 `0.x`——按约定次版本号允许破坏性变更，第一个被外部依赖的版本升到 `1.0.0`；
- `research/` 下的代码前 12 个月（至 2027-09）不做兼容性承诺；
- 第三方模态插件独立成包即可，不需要往本仓库提 PR（[ADR-0003](docs/adr/ADR-0003-entry-point-plugin-discovery.md)）。

## 致谢

本项目的开发过程使用了 **DeepSeek** 的大模型服务：文献与接口调研、代码草稿、调试、
文档润色等环节都有它参与。

边界按本仓库一贯的写法写清楚：**它没有产生任何一条未经复现的结论**。`research/` 与
`sweep_results/` 里的每个数字都来自可重跑的脚本与逐次记录，由测试与 CI 守住；主张与错误
由维护者负责。这与论文里的 AI 使用声明是同一口径，见
[paper/README.md](paper/README.md) 的 “Use of AI assistance” 一节。

## 许可证与引用

代码 [Apache-2.0](LICENSE)，文档与计划书 [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)。数据集遵循各数据源许可，本仓库不提供数据镜像；下载与预处理脚本是 [scripts/download_data.py](scripts/download_data.py)。

研究线依赖的 **SpikingJelly 用的是启智开源许可证 1.0（OIOSL），不是 Apache-2.0**。研究用途不触发它，但**商业使用或再发布须自行向 AITISA 声明披露**。取舍与它对 Python 下限的影响见 [ADR-0008](docs/adr/ADR-0008-spikingjelly-license-and-python-floor.md)。

引用请用 [CITATION.cff](CITATION.cff)。

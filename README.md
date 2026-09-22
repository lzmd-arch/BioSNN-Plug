# BioSNN-Plug

**面向高生物合理性的全模态脉冲神经网络认知原型**

[English](README.en.md) · [项目计划书](BioSNN-Plug_项目计划书_v6.2.md) · [插件开发指南](docs/plugin_guide.md) · [架构决策记录](docs/adr/README.md)

[![CI](https://github.com/lzmd-arch/BioSNN-Plug/actions/workflows/ci.yml/badge.svg)](https://github.com/lzmd-arch/BioSNN-Plug/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lzmd-arch/BioSNN-Plug/blob/main/examples/quickstart_register_plugin.ipynb)

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

## 当前状态

| 部分 | 状态 |
| :--- | :--- |
| `biosnn-bus` 骨架库 | **0.1.0 可用**——插件接口、注册与发现、脉冲总线骨架 |
| 研究代码：三条验证线 | **第一阶段实现完成**——W1 Hebbian 感知 MNIST **97.79%**（默认闭式解读出；SGD 读出 98.01%）✓；W2 e-prop 序列 sMNIST **77.48%**（活跃神经元 1.0000）✓；W3 R-STDP CartPole 中位数 **237.8 步**（阈值 ≥ 200）✓。数字与复现记录见[复现记录](docs/reproducibility.md) |
| 认知核心 / LLM 协同 | **未开始**，见[计划书 §七](BioSNN-Plug_项目计划书_v6.2.md) |

两处需要说清楚的边界：

- 脉冲总线是**骨架**：双通道路由已就位，但默认融合策略是平凡拼接，**不是**计划书 §2.2 描述的 TAAF 时间注意力引导融合——那是第二阶段的研究任务。（骨架库为何独立成包、
为何刻意不包含这些东西，见 [ADR-0001](docs/adr/ADR-0001-skeleton-as-separate-library.md)。）
- 骨架库**尚未发布到 PyPI**。`pip install biosnn-bus` 要等 `v0.1.0` tag 打上才可用（发布通道见 [release.yml](.github/workflows/release.yml)）。

## 架构

数据自下而上流动。✅ 是当前仓库里已经能跑的，⬜ 是计划书里尚未实现的——画在同一张图上，因为这张图同时是路线图。

```mermaid
flowchart TB
    subgraph L5["LLM 调度与协同层"]
        MCP["MCP / API 接口<br/>可替换的推理引擎<br/>⬜ 第五阶段"]
    end

    subgraph L4["执行层"]
        ACT["动作生成 + 模态解码器<br/>R-STDP + 奖励预测 Critic<br/>⬜ 第一至三阶段"]
    end

    subgraph L3["认知层（认知核心）"]
        WM["工作记忆<br/>RSNN + ALIF + e-prop<br/>⬜ 第一阶段"]
        EM["情景记忆<br/>模式分离 + 神经发生<br/>⬜ 第四阶段"]
        MG["元认知门控<br/>不确定性监控<br/>⬜ 第四阶段"]
    end

    subgraph L2["骨架库 biosnn-bus"]
        BUS["SpikeBus<br/>按融合通道分组 · 时间网格对齐<br/>融合 · 稀疏随机投影<br/>✅ 已实现"]
    end

    subgraph L1["感知层（插件化）"]
        IMG["图像插件<br/>差分编码 / DVS<br/>✅ 参考实现"]
        TXT["文本插件<br/>Token + 时间常数编码<br/>⬜ 第二阶段"]
        AUD["音频插件<br/>耳蜗模型频率分解<br/>⬜ 第三阶段"]
        THIRD["第三方插件<br/>entry points 接入<br/>✅ 机制已就绪"]
    end

    IMG --> BUS
    TXT -.-> BUS
    AUD -.-> BUS
    THIRD -.-> BUS

    BUS --> WM
    WM --> EM
    EM --> MG
    MG --> ACT
    MG -.->|触发调用| MCP
    MCP -.->|结果编码回注| EM
```

层间冲突由 **ES 元学习仲裁器**调节（⬜ 第二阶段）。各层学习规则的完整说明见[计划书](BioSNN-Plug_项目计划书_v6.2.md) §2、§3。

## 快速开始

只依赖 numpy，无需 GPU。尚未发布到 PyPI，当前从 Git 安装：

```bash
pip install "biosnn-bus @ git+https://github.com/lzmd-arch/BioSNN-Plug.git#subdirectory=packages/biosnn-bus"
```

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

## 许可证与引用

代码 [Apache-2.0](LICENSE)，文档与计划书 [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)。数据集遵循各数据源许可，本仓库不提供数据镜像；下载与预处理脚本将随第一阶段提供。

引用请用 [CITATION.cff](CITATION.cff)。

# biosnn-bus

[BioSNN-Plug](https://github.com/lzmd-arch/BioSNN-Plug) 的**模态插件骨架库**。

它只负责一件事：**让新增模态不需要修改已有架构**。

> **这是骨架，不是认知核心。** 本库不包含任何学习规则（e-prop / R-STDP /
> 核化 IB-Hebbian）、不包含脉冲神经元模型、不依赖 GPU、不依赖 PyTorch。它提供
> 的是插件的接口契约、注册与发现机制、以及脉冲总线的接缝。

## 安装

```bash
pip install biosnn-bus

# 需要 torch 桥接时（SpikeTrain.to_torch / from_torch）
pip install "biosnn-bus[torch]"
```

要从 `main` 分支装**尚未发布**的改动（例如正在评审的修复），可以走 Git：

```bash
pip install "biosnn-bus @ git+https://github.com/lzmd-arch/BioSNN-Plug.git#subdirectory=packages/biosnn-bus"
```

只依赖 numpy。

## 快速开始

```python
import numpy as np
from biosnn_bus import ModalityPlugin, PassThroughMembrane, SpikeBus, SpikeTrain


class MyPlugin(ModalityPlugin):
    @property
    def modality_name(self) -> str:
        return "my_modality"

    @property
    def spike_dim(self) -> int:
        return 64

    def encode(self, raw_input):
        return SpikeTrain(data=np.asarray(raw_input) > 0.5, channel=self.channel)

    def get_membrane(self):
        return PassThroughMembrane()

    def decode(self, spike_output):
        return spike_output.rates


bus = SpikeBus(bus_dim=128, seed=0)
bus.register(MyPlugin())
out = bus.step({"my_modality": np.random.default_rng(0).random((10, 64))})
print(out)
```

完整可跑的版本见 [`examples/quickstart_register_plugin.py`](https://github.com/lzmd-arch/BioSNN-Plug/blob/main/examples/quickstart_register_plugin.py)，
以及 [Colab notebook](https://colab.research.google.com/github/lzmd-arch/BioSNN-Plug/blob/main/examples/quickstart_register_plugin.ipynb)（无 GPU 可跑）。

## 核心概念

| 概念 | 说明 |
| :--- | :--- |
| `ModalityPlugin` | 模态插件的抽象基类：`encode` / `decode` / `get_membrane` 三方法，`modality_name` / `spike_dim` 两属性 |
| `SpikeTrain` | 形状 `(T, N)` 的脉冲序列容器，布尔（脉冲）或浮点（发放率）皆可 |
| `FusionChannel` | `temporal` / `semantic` 两条融合通道，对应计划书 §2.2 的"双通道融合" |
| `SpikeBus` | 脉冲总线：编码 → 时间对齐 → 按通道融合 → 稀疏随机投影到认知核心维度 |
| `register_plugin` | 进程内注册装饰器 |
| `discover_plugins` | 扫描第三方分发包声明的 entry points |

## 第三方插件怎么接入

在你的包里声明 entry point，**不需要修改 BioSNN-Plug 任何一行代码**：

```toml
# 你的包的 pyproject.toml
[project.entry-points."biosnn_bus.plugins"]
audio = "my_pkg.plugins.audio:AudioPlugin"
```

用户侧扫描并接入。下面这段可以直接跑——即使一个第三方插件都没装：

```python
from biosnn_bus import SpikeBus, discover_plugins, get_plugin, list_plugins
from biosnn_bus.plugins import DiffImagePlugin

discover_plugins()  # 扫描第三方分发包声明的插件；一个都没装也不会报错
print(list_plugins())  # 内置示例插件在 import 时已注册进注册表

bus = SpikeBus(bus_dim=128, seed=0)
bus.register(get_plugin("image_diff")())  # 等价于 bus.register(DiffImagePlugin())
print(bus)
```

## 骨架的边界

脉冲总线当前的默认融合策略是**平凡的拼接**（`ConcatenateFusion`），**不是**计划书
§2.2 描述的 TAAF 时间注意力引导融合——后者是第二阶段的研究任务。总线已经把
`FusionStrategy` 协议和双通道路由留好，替换策略不会改动调用方代码。

理由见 [ADR-0001](https://github.com/lzmd-arch/BioSNN-Plug/blob/main/docs/adr/ADR-0001-skeleton-as-separate-library.md)。

## 许可证

Apache-2.0。文档采用 CC-BY 4.0。

"""biosnn-bus —— BioSNN-Plug 的模态插件骨架库。

这个库负责一件事：**让新增模态不需要修改已有架构**（计划书 §1.2 第 2 条）。

它刻意不包含任何学习规则、认知核心或 GPU 依赖——那些属于 BioSNN-Plug 的研究
代码，见 ``docs/adr/ADR-0001`` 与 ``ADR-0002``。骨架库不依赖任何研究先成功，
却能在研究成功之前为项目积累用户、信誉与外部眼睛（计划书 §12.2）。

快速开始::

    import numpy as np
    from biosnn_bus import ModalityPlugin, SpikeBus, SpikeTrain


    class MyPlugin(ModalityPlugin):
        @property
        def modality_name(self) -> str:
            return "my_modality"

        @property
        def spike_dim(self) -> int:
            return 64

        def encode(self, raw_input):
            spikes = np.asarray(raw_input) > 0.5
            return SpikeTrain(data=spikes, channel=self.channel)

        def get_membrane(self):
            from biosnn_bus import PassThroughMembrane

            return PassThroughMembrane()

        def decode(self, spike_output):
            return spike_output.rates


    bus = SpikeBus(bus_dim=128, seed=0)
    bus.register(MyPlugin())
    out = bus.step({"my_modality": np.random.default_rng(0).random((10, 64))})

.. note::

   本项目是**研究原型，不是生产框架**（计划书 §12.4）。``biosnn-bus`` 遵循 semver，
   但当前是 0.x——按约定次版本号仍可能含破坏性变更，第一个被外部依赖的版本才升
   1.0.0。BioSNN-Plug 的研究代码在头 12 个月内不做任何兼容性承诺。
"""

from __future__ import annotations

from .bus import ConcatenateFusion, FusionStrategy, SparseRandomProjection, SpikeBus
from .errors import (
    BioSNNBusError,
    PluginContractError,
    PluginNotFoundError,
    PluginRegistrationError,
    UnknownModalityError,
)
from .membrane import PassThroughMembrane
from .plugin import ModalityPlugin
from .registry import (
    ENTRY_POINT_GROUP,
    discover_plugins,
    get_plugin,
    list_plugins,
    register_plugin,
    unregister_plugin,
)
from .types import FusionChannel, Membrane, SpikeTrain

__version__ = "0.1.0"

__all__ = [
    "ENTRY_POINT_GROUP",
    "BioSNNBusError",
    "ConcatenateFusion",
    "FusionChannel",
    "FusionStrategy",
    "Membrane",
    "ModalityPlugin",
    "PassThroughMembrane",
    "PluginContractError",
    "PluginNotFoundError",
    "PluginRegistrationError",
    "SparseRandomProjection",
    "SpikeBus",
    "SpikeTrain",
    "UnknownModalityError",
    "__version__",
    "discover_plugins",
    "get_plugin",
    "list_plugins",
    "register_plugin",
    "unregister_plugin",
]

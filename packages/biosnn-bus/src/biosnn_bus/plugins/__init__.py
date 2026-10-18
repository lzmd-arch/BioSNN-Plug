"""骨架库自带的示例插件。

这些插件的定位是**参考实现**与冒烟测试对象，不是完整的模态编码方案。

四个内置插件覆盖计划书 §2.3 模态表的四行（图像 / MNIST / 文本 / 音频）。其中音频的
:class:`~biosnn_bus.plugins.audio_cochlea.AudioCochleaPlugin` 只做**对数频带分解 + 发放率**
——它保留"频带按对数排布"这一条耳蜗性质，不实现基底膜动力学。更完整的编码器（时分编码、
主动放大）仍属于研究任务，届时以**独立分发包**形式经 entry points 接入，无需修改本项目代码
（见 ``docs/plugin_guide.md``）。
"""

from __future__ import annotations

from .audio_cochlea import AudioCochleaPlugin
from .image_diff import DiffImagePlugin
from .mnist_pulse import MnistPulsePlugin
from .text_zscore import OllamaEmbedding, TextZScorePlugin

__all__ = [
    "AudioCochleaPlugin",
    "DiffImagePlugin",
    "MnistPulsePlugin",
    "OllamaEmbedding",
    "TextZScorePlugin",
]

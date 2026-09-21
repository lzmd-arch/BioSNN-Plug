"""骨架库自带的示例插件。

这些插件的定位是**参考实现**与冒烟测试对象，不是完整的模态编码方案。真实模态
编码器（文本的时间常数编码、音频的耳蜗模型等，见计划书 §2.3 模态表）属于第一、
第二阶段的研究任务，届时会以**独立的分发包**形式提供，通过 entry points 接入，
无需修改本项目代码（见 ``docs/plugin_guide.md``）。
"""

from __future__ import annotations

from .image_diff import DiffImagePlugin

__all__ = ["DiffImagePlugin"]

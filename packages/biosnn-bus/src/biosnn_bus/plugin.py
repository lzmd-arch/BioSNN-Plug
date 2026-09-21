"""模态插件接口。

本模块的 :class:`ModalityPlugin` 逐条对应计划书 §2.3 的接口规范。唯一的偏离是
``get_membrane`` 的返回类型：计划书写的是 ``nn.Module``，这里改为
:class:`~biosnn_bus.types.Membrane` 结构化协议，以免骨架库被迫依赖 torch
（见 ``docs/adr/ADR-0002`` 与 ``ADR-0006``）。认知核心返回的 ``nn.Module``
结构上自动满足该协议，插件作者的写法不受影响。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .errors import PluginContractError
from .types import FusionChannel, Membrane, SpikeTrain

__all__ = ["ModalityPlugin"]


class ModalityPlugin(ABC):
    """一个模态的编码/解码插件。

    子类必须实现 :meth:`encode`、:meth:`get_membrane`、:meth:`decode` 三个方法，
    以及 :attr:`modality_name`、:attr:`spike_dim` 两个属性。:attr:`temporal_scale`
    与 :attr:`fusion_channel` 有默认值，按需覆盖即可。

    最小示例::

        class MyPlugin(ModalityPlugin):
            @property
            def modality_name(self) -> str:
                return "my_modality"

            @property
            def spike_dim(self) -> int:
                return 128

            def encode(self, raw_input):
                return SpikeTrain(data=..., channel=self.channel)

            def get_membrane(self):
                return PassThroughMembrane()

            def decode(self, spike_output):
                return ...

    ``SpikeBus`` 注册时会调用 :meth:`validate` 自检；``encode`` 的返回值类型与
    形状也会在总线入口处校验。
    """

    # ------------------------------------------------------------ 必须实现

    @abstractmethod
    def encode(self, raw_input: Any) -> SpikeTrain:
        """把原始模态输入编码为脉冲序列。

        Returns:
            形状为 ``(T, spike_dim)`` 的 :class:`~biosnn_bus.types.SpikeTrain`。
        """
        raise NotImplementedError

    @abstractmethod
    def get_membrane(self) -> Membrane:
        """返回本模态的膜电位模块。

        骨架库不定义学习规则；认知核心侧会在这里接上真实的脉冲神经元层。
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, spike_output: SpikeTrain) -> Any:
        """把脉冲序列解码回模态空间。

        与 :meth:`encode` 不必严格互逆——编码通常有损——但应当能反映原始结构。
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def modality_name(self) -> str:
        """模态标识符，在同一个 :class:`~biosnn_bus.bus.SpikeBus` 内必须唯一。"""
        raise NotImplementedError

    @property
    @abstractmethod
    def spike_dim(self) -> int:
        """:meth:`encode` 输出的神经元数 ``N``，必须为正整数。"""
        raise NotImplementedError

    # ------------------------------------------------------------ 带默认实现

    @property
    def temporal_scale(self) -> float:
        """该模态的典型时间尺度（毫秒），供 TAAF 模块初始化使用。"""
        return 10.0

    @property
    def fusion_channel(self) -> str:
        """返回 ``'temporal'`` 或 ``'semantic'``。"""
        return "temporal"

    # ------------------------------------------------------------ 派生与自检

    @property
    def channel(self) -> FusionChannel:
        """:attr:`fusion_channel` 的枚举形式，供总线内部使用。

        :attr:`fusion_channel` 非法时抛 :class:`PluginContractError`。
        """
        try:
            return FusionChannel(self.fusion_channel)
        except ValueError as exc:
            valid = ", ".join(repr(c.value) for c in FusionChannel)
            raise PluginContractError(
                f"插件 {type(self).__name__} 的 fusion_channel={self.fusion_channel!r} 非法，"
                f"只能是 {valid} 之一。"
            ) from exc

    def validate(self) -> None:
        """自检接口契约，不合法时抛 :class:`PluginContractError`。

        :meth:`~biosnn_bus.bus.SpikeBus.register` 会调用本方法。之所以不放在
        ``__init_subclass__``，是因为这些是实例属性，类定义期还拿不到值。
        """
        cls_name = type(self).__name__

        name = self.modality_name
        if not isinstance(name, str) or not name.strip():
            raise PluginContractError(
                f"插件 {cls_name} 的 modality_name 必须是非空字符串，收到 {name!r}。"
            )

        dim = self.spike_dim
        # bool 是 int 的子类，但显然不是合法的神经元数
        if isinstance(dim, bool) or not isinstance(dim, (int,)) or dim <= 0:
            raise PluginContractError(f"插件 {cls_name} 的 spike_dim 必须是正整数，收到 {dim!r}。")

        scale = self.temporal_scale
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or scale <= 0:
            raise PluginContractError(
                f"插件 {cls_name} 的 temporal_scale 必须是正数（毫秒），收到 {scale!r}。"
            )

        self.channel  # 触发 fusion_channel 校验  # noqa: B018

    def __repr__(self) -> str:
        try:
            name, dim, channel = self.modality_name, self.spike_dim, self.channel.value
        except Exception:
            return f"{type(self).__name__}(<契约未满足>)"
        return (
            f"{type(self).__name__}(modality_name={name!r}, spike_dim={dim}, channel={channel!r})"
        )

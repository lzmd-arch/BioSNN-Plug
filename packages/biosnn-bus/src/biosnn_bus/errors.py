"""骨架库的异常类型。

所有异常都继承 :class:`BioSNNBusError`，调用方可以只捕获基类。
"""

from __future__ import annotations

__all__ = [
    "BioSNNBusError",
    "PluginContractError",
    "PluginNotFoundError",
    "PluginRegistrationError",
    "UnknownModalityError",
]


class BioSNNBusError(Exception):
    """骨架库所有异常的基类。"""


class PluginContractError(BioSNNBusError, TypeError):
    """插件未满足 :class:`~biosnn_bus.plugin.ModalityPlugin` 的接口契约。

    同时继承 :class:`TypeError`，因为绝大多数违约都是类型/签名层面的问题。
    """


class PluginRegistrationError(BioSNNBusError, ValueError):
    """注册失败：名称非法、重名冲突、或注册对象不是插件子类。"""


class PluginNotFoundError(BioSNNBusError, KeyError):
    """按名称查找插件时未找到。"""

    def __str__(self) -> str:  # KeyError 的 __str__ 会加引号，这里改回可读信息
        return self.args[0] if self.args else ""


class UnknownModalityError(BioSNNBusError, KeyError):
    """``SpikeBus.step`` 收到了未注册模态的输入。"""

    def __str__(self) -> str:
        return self.args[0] if self.args else ""

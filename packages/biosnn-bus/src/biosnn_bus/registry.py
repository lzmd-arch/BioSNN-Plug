"""插件注册与发现。

两条通道并存（见 ``docs/adr/ADR-0003``）：

* **进程内注册**：:func:`register_plugin` 装饰器，适合同仓库内的插件与交互式
  探索；
* **entry points 发现**：:func:`discover_plugins` 扫描已安装的第三方分发包，
  第三方模态包只需在自己的 ``pyproject.toml`` 里声明 entry point，**不需要修改
  本项目任何一行代码**——这是计划书 §1.2"新增模态无需修改已有架构"与技术选型
  表中"插件化"的落地机制。
"""

from __future__ import annotations

import importlib.metadata
from typing import TYPE_CHECKING, TypeVar

from .errors import PluginNotFoundError, PluginRegistrationError
from .plugin import ModalityPlugin

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "ENTRY_POINT_GROUP",
    "discover_plugins",
    "get_plugin",
    "list_plugins",
    "register_plugin",
    "unregister_plugin",
]

#: 第三方插件包在 ``pyproject.toml`` 中声明的 entry point group。
ENTRY_POINT_GROUP = "biosnn_bus.plugins"

_REGISTRY: dict[str, type[ModalityPlugin]] = {}

_P = TypeVar("_P", bound=type[ModalityPlugin])


def register_plugin(name: str, *, override: bool = False) -> Callable[[_P], _P]:
    """把插件类注册到全局注册表，用作类装饰器。

    Args:
        name: 注册名。``SpikeBus.step`` 的输入字典以此作为键。
        override: 允许覆盖同名插件。默认 ``False``，重名直接报错，避免两个模态
            静默地互相顶掉。

    Returns:
        类装饰器。原类被原样返回，因此仍可直接实例化。

    Raises:
        PluginRegistrationError: ``name`` 非法、与已有注册重名（且未开
            ``override``）、或被装饰对象不是 :class:`ModalityPlugin` 的子类。
    """

    def decorator(cls: _P) -> _P:
        if not (isinstance(cls, type) and issubclass(cls, ModalityPlugin)):
            raise PluginRegistrationError(
                f"register_plugin 只能装饰 ModalityPlugin 的子类，收到 {cls!r}。"
            )
        _check_name(name)
        if not override and name in _REGISTRY and _REGISTRY[name] is not cls:
            existing = _REGISTRY[name]
            raise PluginRegistrationError(
                f"插件名 {name!r} 已被 {existing.__module__}.{existing.__qualname__} 占用；"
                f"若确实要替换，请传 override=True。"
            )
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_plugin(name: str) -> type[ModalityPlugin]:
    """按注册名取回插件类。

    Raises:
        PluginNotFoundError: 未注册该名称。错误信息里会列出当前已注册的名字。
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "（空）"
        raise PluginNotFoundError(
            f"未注册的插件名 {name!r}。当前已注册：{known}。"
            f"若是第三方包提供的插件，请先调用 discover_plugins()。"
        ) from None


def list_plugins() -> dict[str, type[ModalityPlugin]]:
    """返回当前注册表的浅拷贝。"""
    return dict(_REGISTRY)


def unregister_plugin(name: str) -> None:
    """移除一个注册项。主要给测试与交互式调试用。

    Raises:
        PluginNotFoundError: 未注册该名称。
    """
    if name not in _REGISTRY:
        raise PluginNotFoundError(f"未注册的插件名 {name!r}，无法移除。")
    del _REGISTRY[name]


def discover_plugins(group: str = ENTRY_POINT_GROUP) -> dict[str, type[ModalityPlugin]]:
    """扫描已安装分发包声明的插件，并注册到全局注册表。

    第三方包的 ``pyproject.toml`` 写法::

        [project.entry-points."biosnn_bus.plugins"]
        audio = "my_pkg.plugins.audio:AudioPlugin"

    Args:
        group: entry point group 名。默认 :data:`ENTRY_POINT_GROUP`。

    Returns:
        本次新发现的 ``{注册名: 插件类}``。已注册过的同名条目会被跳过，因此本函数
        可以重复调用。

    Raises:
        PluginRegistrationError: 某个 entry point 指向的对象不是
            :class:`ModalityPlugin` 子类。
    """
    found: dict[str, type[ModalityPlugin]] = {}
    for entry_point in importlib.metadata.entry_points(group=group):
        if entry_point.name in _REGISTRY:
            continue
        try:
            obj = entry_point.load()
        except Exception as exc:
            raise PluginRegistrationError(
                f"加载 entry point {entry_point.name!r}（{entry_point.value}）失败：{exc}"
            ) from exc
        if not (isinstance(obj, type) and issubclass(obj, ModalityPlugin)):
            raise PluginRegistrationError(
                f"entry point {entry_point.name!r} 指向 {obj!r}，它不是 ModalityPlugin 的子类。"
            )
        _REGISTRY[entry_point.name] = obj
        found[entry_point.name] = obj
    return found


def _check_name(name: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise PluginRegistrationError(f"插件名必须是非空字符串，收到 {name!r}。")
    if name != name.strip():
        raise PluginRegistrationError(f"插件名首尾不能有空白字符，收到 {name!r}。")
    if any(ch.isspace() for ch in name):
        raise PluginRegistrationError(f"插件名不能含空白字符，收到 {name!r}。")

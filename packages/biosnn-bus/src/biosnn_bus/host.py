"""宿主 SDK：给产品用的三个原语（SpikeBus 规范 v0.1 的宿主契约）。

产品**只**需要这三个入口：

* ``inject(modality, raw, t=None)`` —— 把原始输入推进总线；
* ``read_state()`` —— 读融合态的只读快照；
* ``wait_trigger(timeout_ms)`` —— 阻塞到 SNN 认为「值得行动」。

它不暴露学习规则、神经元模型或调度策略——那些是器官内部的事（ADR-0001/0006）。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .bus import SpikeBus

__all__ = ["BioSNNHost", "TriggerEvent"]


@dataclass(frozen=True)
class TriggerEvent:
    """一次「值得行动」的事件。

    Attributes:
        t: 触发时刻（毫秒，自宿主创建起）。
        salience: 显著性（调用方定义的标量，越大越值得行动）。
        associations: 触发时浮现的记忆/概念 id。
    """

    t: int
    salience: float
    associations: tuple[str, ...] = ()


@dataclass
class BioSNNHost:
    """把一个 :class:`SpikeBus` 包成宿主可用的三原语。

    Args:
        bus: 已注册插件的总线。
        trigger: 可选的触发回调；返回 :class:`TriggerEvent` 或 ``None``。
            ``None`` 表示总线未武装——``wait_trigger`` 立即返回 ``None``（与
            MCP 的 ``trigger.status`` 口径一致）。
        poll_interval_s: ``wait_trigger`` 的轮询间隔。
    """

    bus: SpikeBus
    trigger: Callable[[], TriggerEvent | None] | None = None
    poll_interval_s: float = 0.005
    _steps: int = field(default=0, init=False, repr=False)
    _last: dict[str, tuple[int, ...]] = field(default_factory=dict, init=False, repr=False)

    @property
    def armed(self) -> bool:
        """是否武装了触发回调。"""
        return self.trigger is not None

    def inject(self, modality: str, raw, t: int | None = None) -> None:
        """把一个模态的原始输入推进总线。

        Args:
            modality: 已注册的模态名。
            raw: 该插件的原始输入。
            t: 可选时间戳（毫秒）；只用于一致性检查，当前实现按调用顺序推进。

        Raises:
            PluginNotFoundError: 模态未注册。
            ValueError: ``t`` 为负数，或与当前步不一致。
        """
        if t is not None and t < 0:
            raise ValueError(f"t 必须非负，收到 {t!r}。")
        output = self.bus.step({modality: raw})
        self._steps += 1
        data = getattr(output, "data", None)
        if data is not None:
            self._last[modality] = tuple(int(value) for value in getattr(data, "shape", ()))

    def read_state(self) -> dict:
        """返回融合态的只读快照（调用方不得原地修改）。"""
        return {
            "dim": self.bus.bus_dim,
            "steps": self._steps,
            "armed": self.armed,
            "modalities": list(self.bus.registered_modalities),
            "last_shapes": dict(self._last),
        }

    def wait_trigger(self, timeout_ms: int = 0) -> TriggerEvent | None:
        """阻塞到触发或超时。

        Args:
            timeout_ms: 最长等待毫秒数（≥0）。未武装时立即返回 ``None``。

        Returns:
            触发事件，或 ``None``（未武装 / 超时）。
        """
        if timeout_ms < 0:
            raise ValueError(f"timeout_ms 必须非负，收到 {timeout_ms!r}。")
        if self.trigger is None:
            return None
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            event = self.trigger()
            if event is not None:
                return event
            if time.monotonic() >= deadline:
                return None
            time.sleep(self.poll_interval_s)

"""认知层组件（ADR-0015 / ADR-0016）。目前只有唤醒通道；世界模型尚在立项阶段。"""

from __future__ import annotations

from .arousal import ArousalChannel, ArousalConfig

__all__ = ["ArousalChannel", "ArousalConfig"]

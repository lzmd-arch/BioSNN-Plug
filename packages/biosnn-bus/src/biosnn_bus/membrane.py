"""膜电位模块的占位实现。

骨架库不包含任何学习规则——那是认知核心（第一阶段起）的职责，见计划书 §三。
这里只提供一个恒等膜模块，让示例插件有东西可返回，并作为插件作者的最小模板。
"""

from __future__ import annotations

from typing import Any

__all__ = ["PassThroughMembrane"]


class PassThroughMembrane:
    """恒等膜模块：``forward`` 原样返回输入。

    返回类型之所以满足 :class:`~biosnn_bus.types.Membrane` 协议，是因为协议只要求
    存在 ``forward`` 方法（结构化子类型）。
    """

    def forward(self, x: Any) -> Any:
        """原样返回 ``x``。"""
        return x

    def __call__(self, x: Any) -> Any:
        return self.forward(x)

    def __repr__(self) -> str:
        return "PassThroughMembrane()"

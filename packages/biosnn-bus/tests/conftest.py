"""测试夹具。

全局注册表是模块级状态，测试之间必须隔离，否则一个测试注册的插件会泄漏到下一个
测试，让"重名冲突"之类的断言变得依赖执行顺序。
"""

from __future__ import annotations

import pytest
from biosnn_bus.registry import _REGISTRY


@pytest.fixture(autouse=True)
def isolated_registry():
    """每个测试前后快照/还原全局注册表。"""
    snapshot = dict(_REGISTRY)
    try:
        yield
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(snapshot)

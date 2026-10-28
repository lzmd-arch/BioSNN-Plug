"""M1 mock LLM：可替换接口 + 确定性替身，不访问网络、不读 key。"""

from __future__ import annotations

from typing import Protocol


class LLMAdapter(Protocol):
    """外部推理引擎的最小接口；真实 adapter 以后实现同一协议。"""

    def generate(self, prompt: str, *, context: str = "") -> str: ...


class MockLLM:
    """确定性替身：把最后一条用户输入回显并标注 mock。

    它只用于 M1 的管道/trace 测试；**不能**用来报告任何模型能力。
    """

    name = "mock-llm"

    def __init__(self, prefix: str = "[mock]") -> None:
        self.prefix = prefix
        self.calls = 0

    def generate(self, prompt: str, *, context: str = "") -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt 必须是非空字符串。")
        self.calls += 1
        suffix = f" | context={len(context)} chars" if context else ""
        return f"{self.prefix} {prompt.strip()}{suffix}"

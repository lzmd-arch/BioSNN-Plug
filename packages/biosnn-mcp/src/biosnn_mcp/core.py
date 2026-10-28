"""阶段五 M0 的只读认知核接口与内存实现。

红线：M0 **不连接 LLM、不训练、不写记忆**；工具只读。真实认知核以后通过同样的
``CognitiveCore`` 协议接入，工具 schema 不变。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class ToolError(ValueError):
    """工具参数错误；MCP 层转成 -32602。"""


@dataclass(frozen=True)
class MemoryEntry:
    id: str
    text: str


class CognitiveCore(Protocol):
    """M0 工具需要的最小认知核接口。"""

    def state(self) -> dict[str, Any]: ...

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]: ...

    def trigger_status(self) -> dict[str, Any]: ...


class InMemoryCore:
    """M0 的替身：内存里的只读记忆条目 + 固定的未连接/未武装状态。"""

    def __init__(self, entries: list[MemoryEntry] | None = None) -> None:
        self._entries = tuple(entries or ())
        self._by_id = {entry.id: entry for entry in self._entries}
        if len(self._by_id) != len(self._entries):
            raise ValueError("记忆条目 id 必须唯一。")

    def state(self) -> dict[str, Any]:
        return {
            "phase": 5,
            "milestone": "M0",
            "read_only": True,
            "memory_entries": len(self._entries),
            "llm": "not_connected",
            "autonomy": "disabled",
        }

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ToolError("query 必须是非空字符串。")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 50:
            raise ToolError("top_k 必须是 1..50 的整数。")
        tokens = set(query.lower().split())
        scored: list[tuple[int, str, MemoryEntry]] = []
        for entry in self._entries:
            words = set(entry.text.lower().split())
            overlap = len(tokens & words)
            if overlap:
                scored.append((overlap, entry.id, entry))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            {
                "id": entry.id,
                "text": entry.text,
                "score": overlap / max(len(tokens), 1),
            }
            for overlap, _identifier, entry in scored[:top_k]
        ]

    def trigger_status(self) -> dict[str, Any]:
        return {
            "armed": False,
            "reason": "M0 is read-only; the autonomous trigger module is not implemented",
        }

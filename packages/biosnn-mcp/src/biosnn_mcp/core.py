"""阶段五的认知核接口与内存实现（M0 只读 → M1 显式可写）。

红线：**默认只读**。写工具（写记忆、巩固、武装触发）只有在构造时显式打开
``writes_enabled=True`` 才可用；服务器把这条开关暴露成 ``--allow-write``。
真实认知核以后通过同样的 ``CognitiveCore`` 协议接入，工具 schema 不变。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .consolidate import MemoryItem, SleepConsolidator
from .embeddings import HashEmbedding
from .trigger_spiking import SpikingTrigger


class ToolError(ValueError):
    """工具参数错误；MCP 层转成 -32602 之外的 ``isError`` 结果。"""


@dataclass(frozen=True)
class MemoryEntry:
    id: str
    text: str
    tags: tuple[str, ...] = ()


class CognitiveCore(Protocol):
    """工具需要的最小认知核接口。"""

    def state(self) -> dict[str, Any]: ...

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]: ...

    def trigger_status(self) -> dict[str, Any]: ...

    def write(self, text: str, identifier: str | None, tags: tuple[str, ...]) -> dict[str, Any]: ...

    def consolidate(self, **options: Any) -> dict[str, Any]: ...

    def arm(self, reason: str) -> dict[str, Any]: ...

    def disarm(self) -> dict[str, Any]: ...

    def feed(self, salience: float) -> dict[str, Any]: ...

    def poll(self) -> dict[str, Any]: ...


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"{name} 必须是非空字符串。")
    return value.strip()


class InMemoryCore:
    """内存实现：读永远可用；写、巩固与武装触发由 ``writes_enabled`` 把关。"""

    def __init__(
        self,
        entries: list[MemoryEntry] | None = None,
        *,
        writes_enabled: bool = False,
        trigger: SpikingTrigger | None = None,
        consolider: SleepConsolidator | None = None,
        embedder: HashEmbedding | None = None,
    ) -> None:
        self._entries = tuple(entries or ())
        self._by_id = {entry.id: entry for entry in self._entries}
        if len(self._by_id) != len(self._entries):
            raise ValueError("记忆条目 id 必须唯一。")
        self.writes_enabled = bool(writes_enabled)
        self._trigger = trigger or SpikingTrigger(seed=0)
        #: 巩固的默认器；测试可注入更小的 rounds/阈值。
        self._consolider = consolider or SleepConsolidator()
        #: 巩固要在没有 Ollama 的环境下也能跑，所以默认用确定性哈希嵌入。
        self._embedder = embedder or HashEmbedding(dimension=64)
        self._armed = False
        self._armed_reason = ""
        self._pending: list[dict[str, Any]] = []
        self._fed = 0
        self._writes = 0

    # ------------------------------------------------------------------ 只读
    def state(self) -> dict[str, Any]:
        return {
            "phase": 5,
            "milestone": "M1" if self.writes_enabled else "M0",
            "read_only": not self.writes_enabled,
            # 写工具的作用域：永远是内存替身核，不是真实记忆库（见 ADR-0027）。
            "write_scope": "stand-in",
            "memory_entries": len(self._entries),
            "llm": "not_connected",
            "autonomy": "armed" if self._armed else "disabled",
            "pending_triggers": len(self._pending),
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
                "tags": list(entry.tags),
                "score": overlap / max(len(tokens), 1),
            }
            for overlap, _identifier, entry in scored[:top_k]
        ]

    def trigger_status(self) -> dict[str, Any]:
        return {
            "armed": self._armed,
            "reason": self._armed_reason
            or (
                "read-only mode; arming requires --allow-write"
                if not self.writes_enabled
                else "not armed"
            ),
            "fed_samples": self._fed,
            "pending": len(self._pending),
            "scope": "stand-in",
        }

    def poll(self) -> dict[str, Any]:
        """取出并清空待处理的触发事件（只读模式也允许调用）。"""
        events = list(self._pending)
        self._pending.clear()
        return {"events": events, "count": len(events)}

    # ------------------------------------------------------------------ 写
    def _require_writable(self) -> None:
        if not self.writes_enabled:
            raise ToolError("只读模式：写操作需要服务器以 --allow-write 启动。")

    def write(self, text: str, identifier: str | None = None, tags: tuple[str, ...] = ()) -> dict:
        self._require_writable()
        body = _require_text(text, "text")
        if identifier is not None:
            identifier = _require_text(identifier, "id")
            if identifier in self._by_id:
                raise ToolError(f"id 已存在：{identifier}")
        if not all(isinstance(tag, str) and tag.strip() for tag in tags):
            raise ToolError("tags 必须是非空字符串的数组。")
        if identifier is None:
            self._writes += 1
            identifier = f"m{len(self._entries) + self._writes:04d}"
        entry = MemoryEntry(identifier, body, tuple(tag.strip() for tag in tags))
        self._entries = (*self._entries, entry)
        self._by_id[identifier] = entry
        # 措辞刻意用 accepted 而不是 stored：写的是替身核，不是真实库。
        return {
            "id": identifier,
            "entries": len(self._entries),
            "accepted": True,
            "scope": "stand-in",
        }

    def consolidate(
        self,
        *,
        threshold: float = 0.95,
        rounds: int = 3,
        noise_sigma: float = 0.05,
        seed: int = 0,
    ) -> dict[str, Any]:
        self._require_writable()
        if not -1.0 <= threshold <= 1.0:
            raise ToolError("threshold 必须在 [-1, 1]。")
        if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
            raise ToolError("rounds 必须是 ≥1 的整数。")
        if not 0.0 <= noise_sigma <= 1.0:
            raise ToolError("noise_sigma 必须在 [0, 1]。")
        if not self._entries:
            return {
                "merged_pairs": [],
                "rounds_used": 0,
                "original_count": 0,
                "final_count": 0,
                "mean_cosine_drift": 0.0,
                "scope": "stand-in",
            }
        embeddings = self._embedder.embed([entry.text for entry in self._entries])
        items = [
            MemoryItem(entry.id, entry.text, tuple(vector))
            for entry, vector in zip(self._entries, embeddings, strict=True)
        ]
        consolider = SleepConsolidator(
            noise_sigma=noise_sigma,
            merge_threshold=threshold,
            rounds=rounds,
            seed=seed,
        )
        survivors, report = consolider.consolidate(items)
        by_id = {entry.id: entry for entry in self._entries}
        self._entries = tuple(by_id[item.id] for item in survivors)
        self._by_id = {entry.id: entry for entry in self._entries}
        return {
            "merged_pairs": [list(pair) for pair in report.merged_pairs],
            "rounds_used": report.rounds_used,
            "original_count": report.original_count,
            "final_count": report.final_count,
            "mean_cosine_drift": report.mean_cosine_drift,
            "scope": "stand-in",
        }

    def arm(self, reason: str) -> dict[str, Any]:
        self._require_writable()
        self._armed_reason = _require_text(reason, "reason")
        self._armed = True
        return self.trigger_status()

    def disarm(self) -> dict[str, Any]:
        self._require_writable()
        self._armed = False
        self._armed_reason = "disarmed"
        self._pending.clear()
        return self.trigger_status()

    def feed(self, salience: float) -> dict[str, Any]:
        """喂一个时间步的显著性（[0, 1]），推进脉冲触发群体一步。"""
        self._require_writable()
        if not isinstance(salience, (int, float)) or isinstance(salience, bool):
            raise ToolError("salience 必须是数字。")
        value = float(salience)
        if not 0.0 <= value <= 1.0:
            raise ToolError("salience 必须在 [0, 1]。")
        self._fed += 1
        fired = bool(self._trigger.process(value))
        if fired and self._armed:
            self._pending.append({"t": self._fed, "salience": value})
        return {"fired": fired, "queued": len(self._pending), "armed": self._armed}

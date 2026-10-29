"""阶段五 M3e：SQLite 情景库（stdlib，存文本、嵌入与脉冲码）。

表结构::

    entries(id TEXT PRIMARY KEY, text TEXT NOT NULL,
            embedding TEXT NOT NULL,   -- JSON 数组
            code TEXT)                 -- 可选的脉冲码 JSON；NULL 表示未编码

只做「写入 / 余弦检索 / 全量替换」三件事；不做近似最近邻、不写事务日志策略。
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Iterable, Sequence

from .consolidate import MemoryItem

__all__ = ["EpisodicStore"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    embedding TEXT NOT NULL,
    code TEXT
)
"""


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


class EpisodicStore:
    """最小情景记忆库。

    Args:
        path: SQLite 文件路径；``":memory:"`` 为内存库（默认）。
    """

    def __init__(self, path: str = ":memory:") -> None:
        self._connection = sqlite3.connect(path)
        self._connection.execute(_SCHEMA)
        self._connection.commit()

    def __len__(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) FROM entries").fetchone()
        return int(row[0])

    def add(
        self,
        identifier: str,
        text: str,
        embedding: Sequence[float],
        code: Sequence[float] | None = None,
    ) -> None:
        """写入或覆盖一条记忆（同 id 覆盖）。"""
        if not identifier or not text:
            raise ValueError("id 与 text 都不能为空。")
        values = [float(value) for value in embedding]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("embedding 必须非空且为有限值。")
        payload = json.dumps(values)
        code_payload = json.dumps([float(value) for value in code]) if code is not None else None
        self._connection.execute(
            "INSERT OR REPLACE INTO entries(id, text, embedding, code) VALUES (?, ?, ?, ?)",
            (identifier, text, payload, code_payload),
        )
        self._connection.commit()

    def search(self, embedding: Sequence[float], top_k: int = 5) -> list[dict]:
        """按余弦相似度检索，返回 ``[{id, text, score, code}]``。"""
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError(f"top_k 必须是 ≥1 的整数，收到 {top_k!r}。")
        query = [float(value) for value in embedding]
        scored = []
        for identifier, text, payload, code_payload in self._connection.execute(
            "SELECT id, text, embedding, code FROM entries"
        ):
            vector = json.loads(payload)
            score = _cosine(query, vector)
            scored.append(
                {
                    "id": identifier,
                    "text": text,
                    "score": score,
                    "code": json.loads(code_payload) if code_payload else None,
                }
            )
        scored.sort(key=lambda row: (-row["score"], row["id"]))
        return scored[:top_k]

    def items(self) -> list[MemoryItem]:
        """导出为 :class:`MemoryItem` 列表（供巩固器使用）。"""
        return [
            MemoryItem(id=identifier, text=text, embedding=tuple(json.loads(payload)))
            for identifier, text, payload in self._connection.execute(
                "SELECT id, text, embedding FROM entries ORDER BY id"
            )
        ]

    def replace_all(self, items: Iterable[MemoryItem]) -> None:
        """全量替换（巩固之后写回）。"""
        self._connection.execute("DELETE FROM entries")
        for item in items:
            self.add(item.id, item.text, item.embedding)
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> EpisodicStore:
        return self

    def __exit__(self, *args) -> None:
        self.close()

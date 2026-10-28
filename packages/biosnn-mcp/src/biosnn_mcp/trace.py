"""M1 对话 trace：每轮记录状态、检索、触发决定、LLM 返回、巩固；可回放。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass
class TurnRecord:
    turn: int
    user: str
    state: dict[str, Any] = field(default_factory=dict)
    retrieved: list[str] = field(default_factory=list)
    trigger: str = "skipped"  # called / skipped / no_trigger
    llm_response: str | None = None
    consolidation: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.turn < 1:
            raise ValueError("turn 从 1 开始。")
        if self.trigger not in {"called", "skipped", "no_trigger"}:
            raise ValueError(f"未知 trigger：{self.trigger!r}。")


class DialogueTrace:
    """内存 + 可选 JSONL；写入是追加，读取按行回放。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._records: list[TurnRecord] = []
        if self.path is not None and self.path.exists():
            self._records = self.read(self.path)

    @property
    def records(self) -> list[TurnRecord]:
        return list(self._records)

    def append(self, record: TurnRecord) -> TurnRecord:
        if any(existing.turn == record.turn for existing in self._records):
            raise ValueError(f"turn {record.turn} 已存在。")
        self._records.append(record)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        return record

    @staticmethod
    def read(path: str | Path) -> list[TurnRecord]:
        records: list[TurnRecord] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(TurnRecord(**json.loads(line)))
        return records

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "turns": len(self._records),
            "llm_calls": sum(1 for record in self._records if record.trigger == "called"),
            "retrieved": sum(len(record.retrieved) for record in self._records),
            "consolidated": sum(1 for record in self._records if record.consolidation),
        }

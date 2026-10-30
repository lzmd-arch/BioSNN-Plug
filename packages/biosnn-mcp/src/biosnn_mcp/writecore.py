"""写版本的核心：提议进队列，门控说了算，它是唯一能落库的人（计划书 §3.2）。

三条不许破的规矩：

1. **提议不是写入**：``propose_*`` 只往队列里放一条，返回的是判定；
2. **门控是唯一写入口**：这个类里只有 ``_apply`` 会碰 ``InMemoryCore.write``；
   ``WriteCore.write`` 直接抛错——谁想绕队列都会被这条挡住（W5）；
3. **可回滚**：每条 accepted 记录留前像，``revert`` 把库恢复成落库前的样子（W4）。

存储是 biosnn 侧内存核的持久化文件（JSON），**不是** Axo 真库（决策 2）。
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from .core import InMemoryCore, MemoryEntry, ToolError, _require_text
from .embeddings import HashEmbedding
from .proposals import (
    Gate,
    ProposalQueue,
    merge_config,
    summarize,
)

STORE_VERSION = 1


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


class WriteCore:
    """队列 + 门控 + 可回滚存储；对 MCP 层暴露的只有提议类工具。"""

    def __init__(
        self,
        store_path: Path | str,
        queue_path: Path | str,
        *,
        embedder: HashEmbedding | None = None,
        audit: Any | None = None,
    ) -> None:
        self._store_path = Path(store_path)
        self._embedder = embedder or HashEmbedding(dimension=64)
        self._audit = audit
        self.queue = ProposalQueue(queue_path)
        self.gate = Gate(self.queue.load_gate_config())
        self._store = self._load_store()

    # ------------------------------------------------------------ 存储
    def _load_store(self) -> InMemoryCore:
        if not self._store_path.exists():
            return InMemoryCore(writes_enabled=True)
        data = json.loads(self._store_path.read_text(encoding="utf-8"))
        if data.get("version") != STORE_VERSION:
            raise ValueError(f"未知的存储版本：{data.get('version')!r}")
        entries = [
            MemoryEntry(item["id"], item["text"], tuple(item.get("tags", ())))
            for item in data.get("entries", [])
        ]
        return InMemoryCore(entries, writes_enabled=True)

    def snapshot(self) -> list[dict[str, Any]]:
        """库的完整前像/后像：按 id 排序，保证同一内容序列化后逐字节相同。"""
        entries = sorted(self._store._entries, key=lambda entry: entry.id)
        return [{"id": entry.id, "text": entry.text, "tags": list(entry.tags)} for entry in entries]

    def _persist(self) -> None:
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": STORE_VERSION, "entries": self.snapshot()}
        temp = self._store_path.with_suffix(self._store_path.suffix + ".tmp")
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temp, self._store_path)

    def _restore(self, image: list[dict[str, Any]]) -> None:
        entries = [
            MemoryEntry(item["id"], item["text"], tuple(item.get("tags", ()))) for item in image
        ]
        self._store = InMemoryCore(entries, writes_enabled=True)
        self._persist()

    # ------------------------------------------------------------ 只读面（转发）
    def state(self) -> dict[str, Any]:
        counts = self.queue.counts()
        hour, day = self.queue.quota_usage()
        base = self._store.state()
        base.update(
            {
                "milestone": "M2",
                "read_only": False,
                "write_scope": "gate",
                "queue": {
                    "pending": counts["pending"],
                    "accepted": counts["accepted"],
                    "rejected": counts["rejected"],
                    "reverted": counts["reverted"],
                },
                "quota": {
                    "hour": {"used": hour, "limit": self.gate.config.quota_hourly},
                    "day": {"used": day, "limit": self.gate.config.quota_daily},
                },
            }
        )
        return base

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self._store.search(query, top_k)

    def trigger_status(self) -> dict[str, Any]:
        payload = self._store.trigger_status()
        payload["scope"] = "gate"
        return payload

    def poll(self) -> dict[str, Any]:
        return self._store.poll()

    # ------------------------------------------------------------ 写面：只进队列
    def write(self, *_: Any, **__: Any) -> dict[str, Any]:
        """W5 的守卫：写工具不能直接落库，只能 propose。"""
        raise ToolError("写版本不接受直接写入：请用 propose.memory.write，由门控处置。")

    def consolidate(self, **_: Any) -> dict[str, Any]:
        raise ToolError("写版本不接受直接巩固：请用 propose.memory.consolidate。")

    def arm(self, reason: str) -> dict[str, Any]:
        return self._store.arm(reason)

    def disarm(self) -> dict[str, Any]:
        return self._store.disarm()

    def feed(self, salience: float) -> dict[str, Any]:
        return self._store.feed(salience)

    # ------------------------------------------------------------ 提议
    def propose_write(
        self,
        text: Any,
        identifier: Any = None,
        tags: Any = (),
        *,
        proposal_id: Any,
        source: str = "client",
    ) -> dict[str, Any]:
        body = _require_text(text, "text")
        if identifier is not None:
            identifier = _require_text(identifier, "id")
        if not isinstance(tags, (list, tuple)) or not all(
            isinstance(tag, str) and tag.strip() for tag in tags
        ):
            raise ToolError("tags 必须是非空字符串的数组。")
        clean_tags = [tag.strip() for tag in tags]
        key = _require_text(proposal_id, "proposal_id")
        if source not in ("user", "client", "llm"):
            raise ToolError("source 必须是 user / client / llm 之一。")

        seq, created = self.queue.enqueue(
            key,
            "memory.write",
            {"text": body, "id": identifier, "tags": clean_tags, "source": source},
        )
        if not created:
            return self._replay(seq)
        self._audit_event(
            "proposal.received",
            {
                "seq": seq,
                "proposal_id": key,
                "kind": "memory.write",
                "source": source,
                "text": body,
            },
        )
        return self._decide(seq, body, identifier, tuple(clean_tags), source=source)

    def propose_consolidate(
        self,
        *,
        proposal_id: Any,
        threshold: float = 0.95,
        rounds: int = 3,
        noise_sigma: float = 0.05,
        seed: int = 0,
    ) -> dict[str, Any]:
        key = _require_text(proposal_id, "proposal_id")
        payload = {
            "threshold": float(threshold),
            "rounds": int(rounds),
            "noise_sigma": float(noise_sigma),
            "seed": int(seed),
        }
        seq, created = self.queue.enqueue(key, "memory.consolidate", payload)
        if not created:
            return self._replay(seq)
        self._audit_event(
            "proposal.received",
            {"seq": seq, "proposal_id": key, "kind": "memory.consolidate"},
        )
        # 巩固是维护操作，不是"新记忆"：门控对它不做显著度判定，但仍留前后像可回滚。
        result = self._apply(seq, "memory.consolidate", payload)
        self.queue.decide(
            seq,
            "accepted",
            "maintenance: consolidation pass, not a new memory",
            salience=1.0,
            max_cosine=0.0,
        )
        self._audit_event("gate.accepted", {"seq": seq, "reason": "maintenance"})
        return {"seq": seq, "proposal_id": key, "status": "accepted", **result}

    def _decide(
        self,
        seq: int,
        body: str,
        identifier: str | None,
        tags: tuple[str, ...],
        *,
        source: str,
    ) -> dict[str, Any]:
        if identifier is not None and identifier in self._store._by_id:
            self.queue.decide(
                seq, "rejected", f"id already exists: {identifier}", salience=0.0, max_cosine=0.0
            )
            self._audit_event("gate.rejected", {"seq": seq, "reason": "id already exists"})
            return self._row(seq)

        max_cosine = self._max_cosine(body)
        hour, day = self.queue.quota_usage()
        decision = self.gate.evaluate(
            body,
            source=source,
            max_cosine=max_cosine,
            quota_used_hour=hour,
            quota_used_day=day,
        )
        if decision.status == "accepted":
            result = self._apply(
                seq, "memory.write", {"text": body, "id": identifier, "tags": tags}
            )
            self.queue.decide(
                seq,
                "accepted",
                decision.reason,
                salience=decision.salience,
                max_cosine=decision.max_cosine,
            )
            self._audit_event(
                "gate.accepted",
                {"seq": seq, "reason": decision.reason, "salience": decision.salience},
            )
            return {**self._row(seq), **result}
        self.queue.decide(
            seq,
            decision.status,
            decision.reason,
            salience=decision.salience,
            max_cosine=decision.max_cosine,
        )
        self._audit_event(
            f"gate.{decision.status}",
            {"seq": seq, "reason": decision.reason, "salience": decision.salience},
        )
        return self._row(seq)

    def _apply(self, seq: int, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        """**唯一**会改记忆的方法。谁要在别处调用 ``InMemoryCore.write``，W5 的守卫会红。"""
        before = self.snapshot()
        if kind == "memory.write":
            result = self._store.write(
                payload["text"], payload.get("id"), tuple(payload.get("tags", ()))
            )
        elif kind == "memory.consolidate":
            result = self._store.consolidate(
                threshold=payload["threshold"],
                rounds=payload["rounds"],
                noise_sigma=payload["noise_sigma"],
                seed=payload["seed"],
            )
        else:  # pragma: no cover - 枚举外的 kind 不该进队列
            raise ToolError(f"未知提议类型：{kind}")
        after = self.snapshot()
        self._persist()
        self.queue.record_apply(
            seq,
            entry_id=str(result.get("id", "")),
            before_image=before,
            after_image=after,
        )
        self._audit_event(
            "memory.applied",
            {"seq": seq, "kind": kind, "before": len(before), "after": len(after)},
        )
        return {"entries": len(after), "scope": "gate", **result}

    # ------------------------------------------------------------ 队列视图
    def queue_status(self) -> dict[str, Any]:
        counts = self.queue.counts()
        hour, day = self.queue.quota_usage()
        return {
            "counts": counts,
            "quota": {
                "hour": {"used": hour, "limit": self.gate.config.quota_hourly},
                "day": {"used": day, "limit": self.gate.config.quota_daily},
            },
            "gate": self.gate.config.as_dict(),
            "entries": len(self._store._entries),
            "recent": summarize(self.queue.list(limit=5), limit=5),
        }

    def queue_list(self, *, status: str | None = None, limit: int = 20) -> dict[str, Any]:
        if status is not None and status not in ("pending", "accepted", "rejected", "reverted"):
            raise ToolError("status 必须是 pending / accepted / rejected / reverted。")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ToolError("limit 必须是 1..100 的整数。")
        rows = summarize(self.queue.list(status=status, limit=limit), limit=limit)
        return {"proposals": rows, "count": len(rows)}

    def revert(self, seq: Any, reason: Any) -> dict[str, Any]:
        if not isinstance(seq, int) or isinstance(seq, bool):
            raise ToolError("seq 必须是整数。")
        why = _require_text(reason, "reason")
        row = self.queue.get(seq)
        if row["status"] == "reverted":
            raise ToolError(f"第 {seq} 条已经撤销过。")
        if row["status"] != "accepted":
            raise ToolError(f"第 {seq} 条不是 accepted，撤销无从谈起。")
        applied = self.queue.applied(seq)
        if applied is None:
            raise ToolError(f"第 {seq} 条没有落库前像，拒绝撤销（宁可报错也不猜）。")
        before = applied["before_image"]
        self._restore(before)
        self.queue.mark_reverted(seq)
        self._audit_event(
            "memory.revert",
            {"seq": seq, "reason": why, "restored_entries": len(before)},
        )
        return {
            "seq": seq,
            "reverted": True,
            "entries": len(before),
            "reason": why,
            "scope": "gate",
        }

    def configure_gate(self, reason: Any, changes: dict[str, Any]) -> dict[str, Any]:
        why = _require_text(reason, "reason")
        if not isinstance(changes, dict):
            raise ToolError("changes 必须是对象。")
        try:
            config = merge_config(self.gate.config, changes)
        except ValueError as error:
            raise ToolError(str(error)) from error
        before = self.gate.config.as_dict()
        self.queue.save_gate_config(config)
        self.gate = Gate(config)
        self._audit_event(
            "gate.configure",
            {"reason": why, "before": before, "after": config.as_dict()},
        )
        return {"gate": config.as_dict(), "reason": why}

    # ------------------------------------------------------------ 内部工具
    def _max_cosine(self, body: str) -> float:
        texts = [entry.text for entry in self._store._entries]
        if not texts:
            return 0.0
        vectors = self._embedder.embed([body, *texts])
        target = vectors[0]
        return max((_cosine(target, vector) for vector in vectors[1:]), default=0.0)

    def _replay(self, seq: int) -> dict[str, Any]:
        return {**self._row(seq), "replayed": True}

    def _row(self, seq: int) -> dict[str, Any]:
        row = self.queue.get(seq)
        return {
            "seq": row["seq"],
            "proposal_id": row["proposal_id"],
            "status": row["status"],
            "reason": row["reason"],
            "salience": row["salience"],
            "max_cosine": row["max_cosine"],
            "entries": len(self._store._entries),
            "scope": "gate",
        }

    def _audit_event(self, tool: str, arguments: dict[str, Any]) -> None:
        if self._audit is None:
            return
        self._audit.record(tool=tool, arguments=arguments, ok=True)

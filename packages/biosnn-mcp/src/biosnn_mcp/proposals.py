"""提议队列与巩固门控（实施计划 §3.2 / §9.1 的落地）。

分工写在类型上，别混：

* :class:`ProposalQueue` 只存**提议**与**判定**，自己从不碰记忆；
* :class:`Gate` 只做判定，不落库、不读库；
* 真正把 ``accepted`` 变成记忆条目的是 ``writecore.WriteCore`` 里**唯一**的 ``_apply``。

阈值与权重是**写死的常量**（计划书 §9.1：实现按此，改要 amend，不在实现过程中调参）。
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------- 冻结的常量
DEFAULT_DUPLICATE_COSINE = 0.90  # 与现有条目余弦 ≥ 此值 → 判重复
DEFAULT_ACCEPT_SALIENCE = 0.60  # 显著度 ≥ 此值且未超配额 → 自动接受
DEFAULT_FLOOR_SALIENCE = 0.30  # 显著度 < 此值 → 直接拒绝
DEFAULT_QUOTA_HOURLY = 20
DEFAULT_QUOTA_DAILY = 200

#: 显著度构成（权重之和为 1.0）。改这里等于改判据，必须 amend。
SOURCE_WEIGHTS = {"user": 0.45, "client": 0.25, "llm": 0.15}
LENGTH_WEIGHT = 0.20
NOVELTY_WEIGHT = 0.25
ENTITY_WEIGHT = 0.10
#: 合适的长度带：太短记不住，太长像粘贴。
LENGTH_MIN, LENGTH_MAX = 20, 400

#: 实测的"无关文本"相似度基线（真实库 193 条、随包 BGE、抽样 80，2026-10-31）。
#: 新颖度按它归一，否则假设相似度≈0 会把这一项压掉一半（见 amend 2）。
NOVELTY_BASELINE = 0.51

STATUSES = ("pending", "accepted", "rejected", "reverted")

_NUMBER = re.compile(r"\d")
_ENTITY = re.compile(r"\b[A-Z][A-Za-z0-9_-]{1,}\b")


def utc_now() -> str:
    """秒级 UTC 时间戳；审计与队列共用一个格式。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class GateConfig:
    """门控阈值与配额；``gate.configure`` 改的就是这几个数。"""

    duplicate_cosine: float = DEFAULT_DUPLICATE_COSINE
    accept_salience: float = DEFAULT_ACCEPT_SALIENCE
    floor_salience: float = DEFAULT_FLOOR_SALIENCE
    quota_hourly: int = DEFAULT_QUOTA_HOURLY
    quota_daily: int = DEFAULT_QUOTA_DAILY

    def validated(self) -> GateConfig:
        if not -1.0 <= self.duplicate_cosine <= 1.0:
            raise ValueError("duplicate_cosine 必须在 [-1, 1]。")
        if not 0.0 <= self.floor_salience <= self.accept_salience <= 1.0:
            raise ValueError("显著度阈值必须满足 0 ≤ floor ≤ accept ≤ 1。")
        if self.quota_hourly < 0 or self.quota_daily < 0:
            raise ValueError("配额不能为负。")
        return self

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Decision:
    """一次判定的结果；理由要能直接给人看。"""

    status: str
    reason: str
    salience: float
    max_cosine: float
    quota_used_hour: int = 0
    quota_used_day: int = 0


def has_entity_or_number(text: str) -> bool:
    """实体或数字：报告里的具体信息比空泛的句子更值得记。"""
    return bool(_NUMBER.search(text) or _ENTITY.search(text))


def salience_score(
    text: str,
    *,
    source: str = "client",
    max_cosine: float = 0.0,
) -> float:
    """显著度 ∈ [0, 1]；构成见模块顶部的常量，不许在调用处改权重。"""
    weight = SOURCE_WEIGHTS.get(source, SOURCE_WEIGHTS["client"])
    length = len(text.strip())
    if LENGTH_MIN <= length <= LENGTH_MAX:
        length_part = LENGTH_WEIGHT
    elif length > LENGTH_MAX:
        length_part = LENGTH_WEIGHT / 2
    else:
        length_part = 0.0
    span = 1.0 - NOVELTY_BASELINE
    novelty = NOVELTY_WEIGHT * min(1.0, max(0.0, (1.0 - max_cosine) / span))
    entity = ENTITY_WEIGHT if has_entity_or_number(text) else 0.0
    return round(min(1.0, weight + length_part + novelty + entity), 4)


class Gate:
    """把"是否该记"压成一个判定。它不落库，也不知道库长什么样。"""

    def __init__(self, config: GateConfig | None = None) -> None:
        self.config = (config or GateConfig()).validated()

    def evaluate(
        self,
        text: str,
        *,
        source: str,
        max_cosine: float,
        quota_used_hour: int,
        quota_used_day: int,
    ) -> Decision:
        cfg = self.config
        score = salience_score(text, source=source, max_cosine=max_cosine)
        if max_cosine >= cfg.duplicate_cosine:
            return Decision(
                "rejected",
                f"duplicate: cosine {max_cosine:.3f} ≥ {cfg.duplicate_cosine:.2f}",
                score,
                max_cosine,
                quota_used_hour,
                quota_used_day,
            )
        if score < cfg.floor_salience:
            return Decision(
                "rejected",
                f"not salient enough: {score:.3f} < {cfg.floor_salience:.2f}",
                score,
                max_cosine,
                quota_used_hour,
                quota_used_day,
            )
        over_quota = quota_used_hour >= cfg.quota_hourly or quota_used_day >= cfg.quota_daily
        if over_quota:
            return Decision(
                "pending",
                f"quota exhausted: {quota_used_hour}/{cfg.quota_hourly} per hour, "
                f"{quota_used_day}/{cfg.quota_daily} per day; held for review",
                score,
                max_cosine,
                quota_used_hour,
                quota_used_day,
            )
        if score >= cfg.accept_salience:
            return Decision(
                "accepted",
                f"salience {score:.3f} ≥ {cfg.accept_salience:.2f}, quota free",
                score,
                max_cosine,
                quota_used_hour,
                quota_used_day,
            )
        return Decision(
            "pending",
            f"salience {score:.3f} between {cfg.floor_salience:.2f} and "
            f"{cfg.accept_salience:.2f}; held for review",
            score,
            max_cosine,
            quota_used_hour,
            quota_used_day,
        )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS proposals (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    proposal_id  TEXT    NOT NULL UNIQUE,
    kind         TEXT    NOT NULL,
    payload      TEXT    NOT NULL,
    status       TEXT    NOT NULL,
    reason       TEXT    NOT NULL DEFAULT '',
    salience     REAL    NOT NULL DEFAULT 0,
    max_cosine   REAL    NOT NULL DEFAULT 0,
    created_at   TEXT    NOT NULL,
    decided_at   TEXT
);
CREATE TABLE IF NOT EXISTS applied (
    seq           INTEGER PRIMARY KEY,
    entry_id      TEXT    NOT NULL,
    before_image  TEXT    NOT NULL,
    after_image   TEXT    NOT NULL,
    reverted_at   TEXT
);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class ProposalQueue:
    """SQLite 队列：提议、判定、落库前后像、门控配置都在这里。"""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path))
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        self._db.commit()

    # ------------------------------------------------------------ 提议
    def enqueue(
        self,
        proposal_id: str,
        kind: str,
        payload: dict[str, Any],
        *,
        created_at: str | None = None,
    ) -> tuple[int, bool]:
        """返回 ``(seq, created)``；同一 ``proposal_id`` 再来只返回旧的那条（幂等）。"""
        existing = self.find(proposal_id)
        if existing is not None:
            return int(existing["seq"]), False
        cursor = self._db.execute(
            "INSERT INTO proposals (proposal_id, kind, payload, status, created_at) "
            "VALUES (?, ?, ?, 'pending', ?)",
            (proposal_id, kind, json.dumps(payload, ensure_ascii=False), created_at or utc_now()),
        )
        self._db.commit()
        return int(cursor.lastrowid), True

    def find(self, proposal_id: str) -> dict[str, Any] | None:
        row = self._db.execute(
            "SELECT * FROM proposals WHERE proposal_id = ?", (proposal_id,)
        ).fetchone()
        return self._row(row) if row else None

    def get(self, seq: int) -> dict[str, Any]:
        row = self._db.execute("SELECT * FROM proposals WHERE seq = ?", (seq,)).fetchone()
        if row is None:
            raise KeyError(seq)
        return self._row(row)

    def decide(
        self,
        seq: int,
        status: str,
        reason: str,
        *,
        salience: float = 0.0,
        max_cosine: float = 0.0,
        decided_at: str | None = None,
    ) -> None:
        if status not in STATUSES:
            raise ValueError(f"未知状态：{status}")
        self._db.execute(
            "UPDATE proposals SET status = ?, reason = ?, salience = ?, max_cosine = ?, "
            "decided_at = ? WHERE seq = ?",
            (status, reason, salience, max_cosine, decided_at or utc_now(), seq),
        )
        self._db.commit()

    def list(self, *, status: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        sql = "SELECT * FROM proposals"
        params: list[Any] = []
        if status is not None:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY seq DESC LIMIT ?"
        params.append(max(1, int(limit)))
        return [self._row(row) for row in self._db.execute(sql, params)]

    def counts(self) -> dict[str, int]:
        counts = dict.fromkeys(STATUSES, 0)
        for row in self._db.execute("SELECT status, COUNT(*) AS n FROM proposals GROUP BY status"):
            counts[str(row["status"])] = int(row["n"])
        return counts

    # ------------------------------------------------------------ 配额
    def accepted_since(self, since: str) -> int:
        row = self._db.execute(
            "SELECT COUNT(*) AS n FROM proposals WHERE status = 'accepted' AND decided_at >= ?",
            (since,),
        ).fetchone()
        return int(row["n"])

    def quota_usage(self, *, now: datetime | None = None) -> tuple[int, int]:
        moment = now or datetime.now(timezone.utc)
        hour = (moment - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        day = (moment - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        return self.accepted_since(hour), self.accepted_since(day)

    # ------------------------------------------------------------ 落库前后像
    def record_apply(
        self,
        seq: int,
        *,
        entry_id: str,
        before_image: list[dict[str, Any]],
        after_image: list[dict[str, Any]],
    ) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO applied (seq, entry_id, before_image, after_image) "
            "VALUES (?, ?, ?, ?)",
            (
                seq,
                entry_id,
                json.dumps(before_image, ensure_ascii=False),
                json.dumps(after_image, ensure_ascii=False),
            ),
        )
        self._db.commit()

    def applied(self, seq: int) -> dict[str, Any] | None:
        row = self._db.execute("SELECT * FROM applied WHERE seq = ?", (seq,)).fetchone()
        if row is None:
            return None
        return {
            "seq": int(row["seq"]),
            "entry_id": str(row["entry_id"]),
            "before_image": json.loads(row["before_image"]),
            "after_image": json.loads(row["after_image"]),
            "reverted_at": row["reverted_at"],
        }

    def mark_reverted(self, seq: int, *, at: str | None = None) -> None:
        stamp = at or utc_now()
        self._db.execute("UPDATE applied SET reverted_at = ? WHERE seq = ?", (stamp, seq))
        self._db.execute(
            "UPDATE proposals SET status = 'reverted', decided_at = ? WHERE seq = ?",
            (stamp, seq),
        )
        self._db.commit()

    # ------------------------------------------------------------ 门控配置
    def load_gate_config(self) -> GateConfig:
        row = self._db.execute("SELECT value FROM settings WHERE key = 'gate'").fetchone()
        if row is None:
            return GateConfig()
        return GateConfig(**json.loads(row["value"])).validated()

    def save_gate_config(self, config: GateConfig, *, at: str | None = None) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES ('gate', ?)",
            (json.dumps(config.validated().as_dict(), ensure_ascii=False),),
        )
        self._db.commit()

    # ------------------------------------------------------------ 杂项
    def close(self) -> None:
        self._db.close()

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "seq": int(row["seq"]),
            "proposal_id": str(row["proposal_id"]),
            "kind": str(row["kind"]),
            "payload": json.loads(row["payload"]),
            "status": str(row["status"]),
            "reason": str(row["reason"]),
            "salience": float(row["salience"]),
            "max_cosine": float(row["max_cosine"]),
            "created_at": str(row["created_at"]),
            "decided_at": row["decided_at"],
        }


def merge_config(config: GateConfig, changes: dict[str, Any]) -> GateConfig:
    """``gate.configure`` 用：只认 GateConfig 的字段，其余报错。"""
    unknown = set(changes) - set(config.as_dict())
    if unknown:
        raise ValueError(f"未知的门控参数：{sorted(unknown)}")
    cleaned: dict[str, Any] = {}
    for key, value in changes.items():
        if key.startswith("quota_"):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{key} 必须是整数。")
            cleaned[key] = value
        else:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{key} 必须是数字。")
            cleaned[key] = float(value)
    return replace(config, **cleaned).validated()


def summarize(rows: Iterable[dict[str, Any]], *, limit: int = 20) -> list[dict[str, Any]]:
    """``queue.list`` 的输出：把 payload 摊平，但保留原始字符串形态。"""
    out = []
    for row in rows[:limit]:
        out.append(
            {
                "seq": row["seq"],
                "proposal_id": row["proposal_id"],
                "kind": row["kind"],
                "status": row["status"],
                "reason": row["reason"],
                "salience": row["salience"],
                "max_cosine": row["max_cosine"],
                "created_at": row["created_at"],
                "decided_at": row["decided_at"],
                "text": row["payload"].get("text"),
            }
        )
    return out

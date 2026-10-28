"""M2 触发策略：意图 + 检索最高分的规则门（mock 协议，不是自主推断）。"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["TriggerDecision", "TriggerPolicy"]


@dataclass(frozen=True)
class TriggerDecision:
    call: bool
    reason: str
    score: float


class TriggerPolicy:
    """规则触发：knowledge+低分才调用；chat/memory 永远不调用。"""

    def __init__(
        self,
        *,
        knowledge_score_max: float = 0.25,
        unknown_score_max: float = 0.10,
    ) -> None:
        if not 0.0 <= knowledge_score_max <= 1.0:
            raise ValueError("knowledge_score_max 应在 [0,1]。")
        if not 0.0 <= unknown_score_max <= 1.0:
            raise ValueError("unknown_score_max 应在 [0,1]。")
        self.knowledge_score_max = float(knowledge_score_max)
        self.unknown_score_max = float(unknown_score_max)

    def decide(self, intent: str, score: float) -> TriggerDecision:
        if intent not in {"knowledge", "memory", "chat", "unknown"}:
            raise ValueError(f"未知 intent：{intent!r}")
        score = float(score)
        if intent == "chat":
            return TriggerDecision(False, "chat", score)
        if intent == "memory":
            return TriggerDecision(False, "memory", score)
        if intent == "knowledge" and score < self.knowledge_score_max:
            return TriggerDecision(True, "knowledge_low_retrieval", score)
        if intent == "unknown" and score < self.unknown_score_max:
            return TriggerDecision(True, "unknown_low_retrieval", score)
        return TriggerDecision(False, "retrieval_sufficient", score)

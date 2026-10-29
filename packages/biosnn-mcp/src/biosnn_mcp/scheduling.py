"""阶段五 M3g：可微调度的连续参数策略（ES 的第 4 类维度，协议层）。

策略只有两个参数，且都是连续的，便于 (1+λ)-ES 搜索：

* ``score_threshold``：检索分低于它才考虑调用；
* ``cooldown``：距上次调用至少隔多少轮（四舍五入，至少 1）。

它不包含意图判断——意图（knowledge/memory/chat）由调用方给出；与 ``TriggerPolicy``
的规则门在同一批合成会话上比较，用于验证 ES 优化器与调度接口的接线。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SchedulingPolicy"]


@dataclass(frozen=True)
class SchedulingPolicy:
    """两参数调度策略。

    Args:
        score_threshold: 检索分阈值，必须在 [0, 1]。
        cooldown: 冷却轮数（浮点，使用时四舍五入且至少 1），必须在 [0, 5]。

    Raises:
        ValueError: 参数越界。
    """

    score_threshold: float = 0.5
    cooldown: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError(f"score_threshold 应在 [0, 1]，收到 {self.score_threshold!r}。")
        if not 0.0 <= self.cooldown <= 5.0:
            raise ValueError(f"cooldown 应在 [0, 5]，收到 {self.cooldown!r}。")

    @property
    def cooldown_turns(self) -> int:
        """实际冷却轮数：四舍五入，至少 1。"""
        return max(1, round(self.cooldown))

    def decide(self, score: float, turns_since_call: int) -> bool:
        """是否调用：分数低于阈值且冷却已过。"""
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"score 应在 [0, 1]，收到 {score!r}。")
        if turns_since_call < 0:
            raise ValueError(f"turns_since_call 必须非负，收到 {turns_since_call!r}。")
        return score < self.score_threshold and turns_since_call >= self.cooldown_turns

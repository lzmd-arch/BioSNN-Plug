"""逐轮 ROC 助手：新颖度分数、秩和 AUC、命中窗口判定（阶段五 M3l 用）。

放在包里而不是脚本里，是因为它们是**判据本身**：AUC 的并列处理、命中窗口的边界
都得有守卫；脚本只负责取数据、跑触发、落盘。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = ["auc", "cosine", "normalise", "novelty_scores", "workpoint"]


def normalise(vector: Sequence[float]) -> list[float]:
    """L2 归一化；零向量原样返回（不抛错，调用方自己判断）。"""
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """余弦相似度（两边都应已归一化）。"""
    return sum(a * b for a, b in zip(left, right, strict=True))


def novelty_scores(vectors: Sequence[Sequence[float]], *, history: int = 5) -> list[float]:
    """`R_t = 1 − max_{1≤t−j≤history} cos(e_t, e_j)`，截断到 [0, 1]；首轮记 0。

    「相对最近 `history` 轮的意外度」——规则门的分数就是这个，脉冲触发把同一条
    分数当每步的发放概率喂进去。
    """
    if history <= 0:
        raise ValueError(f"history 必须为正，收到 {history!r}。")
    normalised = [normalise(vector) for vector in vectors]
    scores = [0.0]
    for index in range(1, len(normalised)):
        window = normalised[max(0, index - history) : index]
        best = max(cosine(normalised[index], other) for other in window)
        scores.append(min(1.0, max(0.0, 1.0 - best)))
    return scores


def auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """秩和 AUC（Mann-Whitney，含并列的平均秩）；两类有一类为空时返回 nan。"""
    if len(scores) != len(labels):
        raise ValueError("scores 与 labels 长度必须一致。")
    positives = sum(1 for label in labels if label)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[position]]:
            end += 1
        average = (position + end) / 2 + 1
        for offset in range(position, end + 1):
            ranks[order[offset]] = average
        position = end + 1
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def workpoint(
    fired: Sequence[bool],
    target_turns: Sequence[int],
    *,
    ms_per_turn: int,
    hit_window_ms: int,
) -> dict:
    """命中窗口判定：目标轮起点起 `hit_window_ms` 内算命中；其余触发算误触发。"""
    if ms_per_turn <= 0:
        raise ValueError("ms_per_turn 必须为正。")
    triggers = [index for index, value in enumerate(fired) if value]
    hits = 0
    for turn in target_turns:
        start = turn * ms_per_turn
        if any(start <= moment <= start + hit_window_ms for moment in triggers):
            hits += 1
    windows = [
        (turn * ms_per_turn, turn * ms_per_turn + hit_window_ms) for turn in target_turns
    ]
    false_triggers = sum(
        1 for moment in triggers if not any(low <= moment <= high for low, high in windows)
    )
    return {
        "triggers": triggers,
        "hits": hits,
        "false": false_triggers,
        "targets": len(target_turns),
    }

"""M3l 判据助手的守卫：AUC 的并列处理、新颖度截断、命中窗口边界。"""

from __future__ import annotations

import math

import pytest
from biosnn_mcp.roc import auc, cosine, normalise, novelty_scores, workpoint


def test_auc_perfect_and_reversed_and_ties():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [False, False, True, True]
    assert auc(scores, labels) == 1.0
    assert auc([-value for value in scores], labels) == 0.0
    assert auc([0.5, 0.5, 0.5, 0.5], labels) == 0.5


def test_auc_with_one_class_missing_is_nan():
    assert math.isnan(auc([0.1, 0.2], [False, False]))


def test_auc_length_mismatch_raises():
    with pytest.raises(ValueError):
        auc([0.1, 0.2], [True])


def test_novelty_first_turn_is_zero_and_identical_text_is_not_novel():
    vectors = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    scores = novelty_scores(vectors)
    assert scores[0] == 0.0
    assert all(value == pytest.approx(0.0) for value in scores[1:])


def test_novelty_of_an_orthogonal_turn_is_one():
    scores = novelty_scores([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert scores[2] == pytest.approx(1.0)


def test_novelty_is_clipped_and_uses_only_the_recent_window():
    """只和最近 `history` 轮比：第 4 轮与第 1 轮相同，但历史窗口里已经没有它。"""
    vectors = [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
    scores = novelty_scores(vectors, history=2)
    assert scores[-1] == pytest.approx(1.0)
    assert all(0.0 <= value <= 1.0 for value in scores)


def test_normalise_and_cosine_round_trip():
    left = normalise([3.0, 4.0])
    assert left == pytest.approx([0.6, 0.8])
    assert cosine(left, left) == pytest.approx(1.0)


def test_workpoint_counts_hits_and_false_triggers_with_inclusive_bounds():
    fired = [False] * 40
    # 窗口：第 0 轮 [0, 5]、第 1 轮 [10, 15]；0/4 命中第 0 轮，12 命中第 1 轮。
    for moment in (0, 4, 12, 20, 39):
        fired[moment] = True
    result = workpoint(fired, [0, 1], ms_per_turn=10, hit_window_ms=5)
    assert result["hits"] == 2
    assert result["false"] == 2  # 20 与 39 都不在任何命中窗口里
    assert result["targets"] == 2

"""持续学习指标的测试：用已知答案的矩阵钉住定义。"""

from __future__ import annotations

import numpy as np
import pytest

from research.continual.metrics import summarise


def _matrix(rows: list[list[float]]) -> np.ndarray:
    n = len(rows)
    m = np.full((n, n), np.nan)
    for t, row in enumerate(rows):
        m[t, : len(row)] = row
    return m


class TestACC:
    def test_no_forgetting_reduces_to_the_final_row(self):
        """每一步都不变时，ACC 应当等于最终准确率。"""
        m = _matrix([[0.9], [0.9, 0.9], [0.9, 0.9, 0.9]])
        assert summarise(m).acc == pytest.approx(0.9)

    def test_hand_computed(self):
        m = _matrix([[0.8], [0.6, 0.7]])
        # 步 0：0.8；步 1：(0.6+0.7)/2 = 0.65；ACC = (0.8+0.65)/2
        assert summarise(m).acc == pytest.approx((0.8 + 0.65) / 2)

    def test_acc_hides_forgetting(self):
        """ACC 会稀释旧任务的退化——这正是必须同时报 AF 的理由。"""
        forgetting = _matrix([[0.9], [0.0, 0.9], [0.0, 0.0, 0.9]])
        scores = summarise(forgetting)
        assert scores.acc == pytest.approx((0.9 + 0.45 + 0.3) / 3)
        assert scores.af == pytest.approx(0.9)


class TestAF:
    def test_zero_when_nothing_forgets(self):
        assert summarise(_matrix([[0.7], [0.7, 0.7]])).af == pytest.approx(0.0)

    def test_uses_the_best_ever_not_the_previous_step(self):
        """任务 0 从 0.9 掉到 0.4 再回到 0.6：回退是 0.9−0.6，不是 0.4−0.6。"""
        m = _matrix([[0.9], [0.4, 0.5], [0.6, 0.5, 0.5]])
        assert summarise(m).af == pytest.approx((0.9 - 0.6 + (0.5 - 0.5)) / 2)

    def test_single_task_has_no_af(self):
        assert np.isnan(summarise(_matrix([[0.8]])).af)


class TestValidation:
    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="方阵"):
            summarise(np.zeros((2, 3)))

    def test_rejects_seen_gap_in_triangular_part(self):
        m = _matrix([[0.8], [0.7, 0.8]])
        m[0, 0] = np.nan
        with pytest.raises(ValueError, match="nan"):
            summarise(m)

    def test_rejects_a_number_where_nothing_was_learned(self):
        m = _matrix([[0.8], [0.7, 0.8]])
        m[0, 1] = 0.5  # i > t，还没学过
        with pytest.raises(ValueError, match="必须是 nan"):
            summarise(m)

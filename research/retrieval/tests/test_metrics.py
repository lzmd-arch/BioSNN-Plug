"""Tests for cross-modal retrieval R@K.

The metric is the phase-two bar ("cross-modal retrieval R@1 50%+"), so it has to be
right before any dataset is downloaded. These use matrices with known answers -- no
images, no captions, no network.

The cases that matter are the ones where the expected answer is easy to get wrong:
a query whose correct item is *anti*-correlated ranks last, not second, because -1 is
below 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from research.retrieval.metrics import RetrievalScores, rank_of_correct, recall_at_k


class TestRankOfCorrect:
    def test_identical_is_rank_one(self):
        q = np.eye(4) * 2.0
        assert rank_of_correct(q[0], q, index=0) == 1

    def test_anti_correlated_is_rank_last(self):
        """-1 cosine is below every 0, so the correct item ranks last, not second."""
        q = np.eye(4) * 2.0
        bad = -np.eye(4) + 0.1
        assert rank_of_correct(q[0], bad, index=0) == 4

    def test_index_out_of_range(self):
        with pytest.raises(ValueError, match="index"):
            rank_of_correct(np.eye(2)[0], np.eye(2), index=5)

    def test_rejects_batched_query(self):
        """rank_of_correct takes ONE query; a batched array is a caller bug."""
        with pytest.raises(ValueError, match="一维"):
            rank_of_correct(np.eye(3), np.eye(3), index=0)


class TestRecallAtK:
    def test_perfect_pairing_is_one(self):
        q = np.eye(4) * 2.0
        assert recall_at_k(q, q, k=1).recall == 1.0

    def test_total_mismatch_is_zero(self):
        q = np.eye(4) * 2.0
        bad = -np.eye(4) + 0.1
        assert recall_at_k(q, bad, k=1).recall == 0.0

    def test_half_right(self):
        """Two queries hit at rank 1, two are anti-correlated (rank 4)."""
        q = np.eye(4) * 2.0
        gallery = np.array([[1.0, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
        scores = recall_at_k(q, gallery, k=1)
        assert scores.recall == 0.5
        assert scores.n_queries == 4
        assert isinstance(scores, RetrievalScores)

    def test_k_below_the_anti_correlated_rank_does_not_help(self):
        """R@2 stays 0.5 here: the wrong two sit at rank 4, beyond any small K."""
        q = np.eye(4) * 2.0
        gallery = np.array([[1.0, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
        assert recall_at_k(q, gallery, k=2).recall == 0.5
        assert recall_at_k(q, gallery, k=4).recall == 1.0

    def test_direction_is_recorded(self):
        q = np.eye(2)
        assert recall_at_k(q, q, k=1, direction="image2text").direction == "image2text"


class TestGuards:
    @pytest.mark.parametrize("k", [0, -1, 99])
    def test_rejects_bad_k(self, k):
        with pytest.raises(ValueError, match="k 必须"):
            recall_at_k(np.eye(3), np.eye(3), k=k)

    def test_rejects_dimension_mismatch(self):
        with pytest.raises(ValueError, match="维度"):
            recall_at_k(np.eye(4), np.eye(3), k=1)

    def test_rejects_fewer_gallery_than_queries(self):
        """Same dimension, fewer rows -- so it is the pairing guard that fires."""
        with pytest.raises(ValueError, match="候选数"):
            recall_at_k(np.ones((4, 3)), np.ones((2, 3)), k=1)

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            recall_at_k(np.zeros((0, 3)), np.eye(3), k=1)

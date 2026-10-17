"""前端探针的测试：秩统计 AUC、词表筛选、岭回归闭式解。"""

from __future__ import annotations

import numpy as np

from research.retrieval.probe import auc, content_words, probe_words, ridge_fit


class TestAUC:
    def test_perfect_separation(self):
        assert auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0])) == 1.0

    def test_reversed_is_zero(self):
        assert auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([1, 1, 0, 0])) == 0.0

    def test_all_ties_is_half(self):
        assert auc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([1, 1, 0, 0])) == 0.5

    def test_single_class_is_nan(self):
        assert np.isnan(auc(np.array([0.1, 0.2]), np.array([1, 1])))


class TestContentWords:
    def test_filters_stopwords(self):
        captions = ["a dog in the water"] * 5
        words = content_words(captions, min_df=1, top_k=10)
        assert "dog" in words and "water" in words
        assert "a" not in words and "the" not in words

    def test_min_df(self):
        captions = ["a dog"] * 3 + ["a cat"] * 1
        assert content_words(captions, min_df=2, top_k=10) == ["dog"]


class TestRidgeFit:
    def test_recovers_a_linear_relation(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(200, 4))
        true = np.array([1.5, -2.0, 0.0, 0.5])
        y = x @ true + 0.3
        weights = ridge_fit(x, y, lam=0.01)
        assert np.allclose(weights[1:], true, atol=0.05)
        assert abs(weights[0] - 0.3) < 0.05


class TestProbeWords:
    def test_a_perfectly_decodable_word_scores_one(self):
        rng = np.random.default_rng(1)
        n = 120
        labels = rng.integers(0, 2, size=n)
        features = rng.normal(size=(n, 6))
        features[:, 0] = labels * 5.0  # 第一维完全决定标签
        captions = ["a dog" if lab else "a cat" for lab in labels]
        results, dropped = probe_words(
            features, features, captions, captions, min_df=1, min_pos=10, lambdas=(0.1,), folds=2
        )
        by_word = {r.word: r.auc for r in results}
        assert dropped == []
        assert by_word["dog"] > 0.95

    def test_rare_words_are_dropped_not_scored(self):
        features = np.zeros((10, 3))
        captions = ["a dog"] * 9 + ["a cat"]
        _results, dropped = probe_words(
            features, features, captions, captions, min_df=1, min_pos=5, lambdas=(0.1,), folds=2
        )
        # "cat" 只出现 1 次 → 正例不足，应被剔除而不是给一个假的 AUC
        assert "cat" in dropped

"""特征可分性指标的测试：用已知可分性的合成数据。"""

from __future__ import annotations

import numpy as np
import pytest

from research.arbiter.separability import fisher_ratio, linear_probe_accuracy


def _blobs(
    separation: float, n_per_class: int = 50, n_classes: int = 3, dim: int = 4, seed: int = 0
):
    rng = np.random.default_rng(seed)
    centers = np.eye(n_classes, dim) * separation
    x = np.concatenate([centers[c] + rng.normal(size=(n_per_class, dim)) for c in range(n_classes)])
    y = np.repeat(np.arange(n_classes), n_per_class)
    return x, y


class TestFisher:
    def test_more_separation_gives_a_larger_ratio(self):
        low, _ = fisher_ratio(*_blobs(0.2)), None
        high, _ = fisher_ratio(*_blobs(5.0)), None
        assert high > low

    def test_uniform_shift_does_not_change_separability(self):
        x, y = _blobs(2.0)
        before = fisher_ratio(x, y)
        after = fisher_ratio(x + 100.0, y)
        assert after == pytest.approx(before, rel=1e-9)

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="不一致"):
            fisher_ratio(np.zeros((3, 2)), np.zeros(4, dtype=int))


class TestProbe:
    def test_perfectly_separable_features_score_one(self):
        x, y = _blobs(10.0)
        assert linear_probe_accuracy(x, y) == pytest.approx(1.0)

    def test_random_features_score_far_below_one(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(300, 8))
        y = rng.integers(0, 3, size=300)
        assert linear_probe_accuracy(x, y) < 0.85

    def test_uses_all_labels_including_a_class_absent_from_the_first_half(self):
        """探针是全监督的可分性度量——不因标签比例而变。"""
        x, y = _blobs(3.0, n_per_class=30, n_classes=3)
        assert linear_probe_accuracy(x, y) == linear_probe_accuracy(x, y)

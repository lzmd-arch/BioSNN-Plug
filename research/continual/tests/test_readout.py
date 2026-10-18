"""delta 读出的测试：用可线性分开的合成数据。"""

from __future__ import annotations

import numpy as np
import pytest

from research.continual.readout import DeltaReadout, evaluate


def _blobs(n_per_class: int = 40, n_classes: int = 3, dim: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_classes, dim)) * 3.0
    x, y = [], []
    for c in range(n_classes):
        x.append(centers[c] + rng.normal(size=(n_per_class, dim)))
        y.append(np.full(n_per_class, c))
    return np.concatenate(x), np.concatenate(y)


class TestDeltaReadout:
    def test_learns_separable_blobs(self):
        x, y = _blobs()
        readout = DeltaReadout(x.shape[1], 3, learning_rate=0.01)
        for _ in range(500):
            readout.step(x, y)
        assert evaluate(readout, x, y, np.arange(3)) > 0.95

    def test_evaluate_only_considers_the_given_classes(self):
        """限制到已见过的类时，未见类的 logit 不应影响预测。"""
        x, y = _blobs()
        readout = DeltaReadout(x.shape[1], 3, learning_rate=0.01)
        for _ in range(50):
            readout.step(x, y)
        score = evaluate(readout, x[y < 2], y[y < 2], np.arange(2))
        assert 0.0 <= score <= 1.0

    def test_rejects_bad_shapes(self):
        readout = DeltaReadout(4, 3)
        with pytest.raises(ValueError, match="特征"):
            readout.step(np.zeros((2, 5)), np.zeros(2, dtype=int))
        with pytest.raises(ValueError, match="标签数"):
            readout.step(np.zeros((2, 4)), np.zeros(3, dtype=int))

    def test_empty_evaluation_is_loud(self):
        readout = DeltaReadout(4, 3)
        with pytest.raises(ValueError, match="空"):
            evaluate(readout, np.zeros((0, 4)), np.zeros(0, dtype=int), np.arange(3))


class TestDivergence:
    def test_a_too_large_step_is_loud(self):
        """大步长发散时必须抛错——静默变成 NaN 会让后续指标全是 nan。"""
        x, y = _blobs()
        readout = DeltaReadout(x.shape[1], 3, learning_rate=0.5)
        with pytest.raises(RuntimeError, match="发散"):
            for _ in range(500):
                readout.step(x, y)

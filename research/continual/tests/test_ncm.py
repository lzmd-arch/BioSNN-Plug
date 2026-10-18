"""类原型读出：可分的合成数据、增量加入、白化路径、未见类不参与预测。"""

from __future__ import annotations

import numpy as np
import pytest

from research.continual.ncm import ClassPrototypes


def _blobs(n_classes=6, per_class=40, dim=8, separation=6.0, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_classes, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    x = np.concatenate(
        [separation * centers[c] + rng.normal(size=(per_class, dim)) for c in range(n_classes)]
    )
    y = np.repeat(np.arange(n_classes), per_class)
    return x, y


class TestPrototypes:
    def test_separable_blobs_are_recalled(self):
        x, y = _blobs()
        model = ClassPrototypes(6)
        model.add(x, y)
        assert model.accuracy(x, y) > 0.95

    def test_incremental_add_matches_one_shot(self):
        x, y = _blobs()
        one = ClassPrototypes(6)
        one.add(x, y)
        two = ClassPrototypes(6)
        two.add(x[y < 3], y[y < 3])
        two.add(x[y >= 3], y[y >= 3])
        assert np.allclose(one.prototypes(), two.prototypes())

    def test_unseen_classes_are_not_predicted(self):
        x, y = _blobs()
        model = ClassPrototypes(6)
        model.add(x[y < 2], y[y < 2])  # 只见过 0、1
        predicted = model.predict(x)
        assert set(np.unique(predicted)).issubset({0, 1})

    def test_old_classes_survive_new_tasks(self):
        """增量的核心：加入新类不影响旧类的原型。"""
        x, y = _blobs()
        model = ClassPrototypes(6)
        model.add(x[y < 3], y[y < 3])
        before = model.prototypes().copy()
        model.add(x[y >= 3], y[y >= 3])
        assert np.allclose(model.prototypes()[:3], before)

    def test_predict_before_any_class_is_loud(self):
        with pytest.raises(RuntimeError, match="还没有"):
            ClassPrototypes(6).predict(np.zeros((1, 8)))

    def test_whitening_path_runs_and_still_separates(self):
        x, y = _blobs()
        model = ClassPrototypes(6, whiten=True, shrinkage=0.5)
        model.add(x, y)
        assert model.accuracy(x, y) > 0.9

    def test_rejects_bad_shrinkage(self):
        with pytest.raises(ValueError, match="shrinkage"):
            ClassPrototypes(6, shrinkage=1.5)


class TestMemorySafeDistance:
    def test_expansion_matches_the_naive_difference(self):
        """回归：直接用 x[:,None,:]-mu[None,:,:] 会在 (n, C, d) 上 OOM（实测 5.36 GiB）。

        这条测试把两种算法对齐：只要结果一致，就说明展开式没有写错，
        而实现里不再实例化那个三维数组。
        """
        rng = np.random.default_rng(0)
        x = rng.normal(size=(60, 32))
        y = np.repeat(np.arange(6), 10)
        model = ClassPrototypes(6)
        model.add(x, y)

        predicted = model.predict(x)
        mu = model.prototypes()
        diff = x[:, None, :] - mu[None, :, :]
        naive = model.seen_classes[np.argmin((diff**2).sum(axis=2), axis=1)]
        assert np.array_equal(predicted, naive)

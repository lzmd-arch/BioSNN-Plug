"""协同性基准的数据侧：置换与分层标签。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from research.arbiter.synergy_task import label_mask, make_permutation, permute_inputs


class TestPermutation:
    def test_is_a_bijection(self):
        perm = make_permutation(28, seed=0)
        assert np.array_equal(np.sort(perm), np.arange(28))

    def test_is_deterministic(self):
        assert np.array_equal(make_permutation(28, seed=1), make_permutation(28, seed=1))

    def test_different_seeds_differ(self):
        assert not np.array_equal(make_permutation(28, seed=0), make_permutation(28, seed=1))

    def test_actually_reorders_features(self):
        inputs = torch.arange(6.0).reshape(1, 2, 3)
        perm = np.array([2, 0, 1])
        out = permute_inputs(inputs, perm)
        assert torch.equal(out[0, 0], torch.tensor([2.0, 0.0, 1.0]))

    def test_rejects_a_wrong_length(self):
        with pytest.raises(ValueError, match="置换长度"):
            permute_inputs(torch.zeros(1, 2, 3), np.array([0, 1]))

    def test_rejects_a_wrong_rank(self):
        with pytest.raises(ValueError, match=r"\(T, n, n_in\)"):
            permute_inputs(torch.zeros(2, 3), np.array([0, 1, 2]))


class TestLabelMask:
    def test_full_fraction_labels_everything(self):
        y = np.array([0, 1, 2, 3])
        assert label_mask(y, 1.0, seed=0).all()

    def test_stratified_counts(self):
        y = np.repeat(np.arange(10), 100)
        mask = label_mask(y, 0.05, seed=0)
        assert mask.sum() == 50
        # 每类 5 个
        assert np.all(np.bincount(y[mask], minlength=10) == 5)

    def test_small_classes_still_get_one(self):
        """1% 档下每类至少一个标签——否则"E 失败"会是算术必然，不是任务设计。"""
        y = np.repeat(np.arange(10), 10)
        mask = label_mask(y, 0.01, seed=0)
        assert np.all(np.bincount(y[mask], minlength=10) >= 1)

    def test_deterministic(self):
        y = np.repeat(np.arange(5), 20)
        assert np.array_equal(label_mask(y, 0.2, seed=3), label_mask(y, 0.2, seed=3))

    def test_rejects_bad_fraction(self):
        with pytest.raises(ValueError, match="fraction"):
            label_mask(np.array([0, 1]), 0.0, seed=0)

    def test_accepts_a_tensor(self):
        mask = label_mask(torch.tensor([0, 0, 1, 1]), 0.5, seed=0)
        assert mask.dtype == bool and mask.shape == (4,)

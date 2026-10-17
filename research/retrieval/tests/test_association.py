"""跨模态联想层的测试：用已知的线性关系，不碰真实数据。"""

from __future__ import annotations

import numpy as np
import pytest

from research.retrieval.association import (
    DeltaAssociator,
    HebbianAssociator,
    evaluate_direction,
    select_delta_hyperparameters,
    select_hyperparameters,
)


def _linear_problem(n=60, ds=8, dt=6, seed=0):
    """构造一个**真存在**线性映射的问题：target = source @ M。"""
    rng = np.random.default_rng(seed)
    source = rng.normal(size=(n, ds))
    mapping = rng.normal(size=(ds, dt))
    target = source @ mapping
    return source, target


class TestHebbianAssociator:
    def test_recovers_a_linear_map(self):
        source, target = _linear_problem()
        assoc = HebbianAssociator(eta=1.0, decay=0.0).fit(source, target)
        scores = evaluate_direction(assoc, source, target, direction="t2i")
        # 外积和**不是**最小二乘解（那是 W = Σ v t^T 再乘伪逆），所以拿不到 1.0。
        # 这条断言问的是"它学到了结构没有"：远高于随机（1/60 ≈ 0.017）即可。
        assert scores["r1"] > 0.5

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            HebbianAssociator().transform(np.zeros((2, 3)))

    def test_sample_count_mismatch(self):
        with pytest.raises(ValueError, match="样本数"):
            HebbianAssociator().fit(np.zeros((3, 2)), np.zeros((4, 2)))

    def test_decay_shrinks_but_does_not_reorder(self):
        """λ 是全局缩放，余弦对它不敏感——这是预注册里"η 单独没有意义"的同一件事。"""
        source, target = _linear_problem()
        a = HebbianAssociator(eta=1.0, decay=0.0).fit(source, target)
        b = HebbianAssociator(eta=1.0, decay=2.0).fit(source, target)
        assert evaluate_direction(a, source, target, direction="t2i") == evaluate_direction(
            b, source, target, direction="t2i"
        )


class TestSelectHyperparameters:
    def test_returns_a_grid_point(self):
        source, target = _linear_problem()
        eta, decay, grid = select_hyperparameters(
            source, target, eta_grid=(0.1, 1.0), decay_grid=(0.0, 0.5)
        )
        assert (eta, decay) in grid
        assert set(grid) == {(0.1, 0.0), (0.1, 0.5), (1.0, 0.0), (1.0, 0.5)}

    def test_leave_one_out_does_not_reward_memorising(self):
        """随机标签下，留一分数应当接近 1/n，而不是 1（那是"记住了"）。"""
        rng = np.random.default_rng(1)
        source = rng.normal(size=(40, 5))
        target = rng.normal(size=(40, 5))
        _, _, grid = select_hyperparameters(source, target, eta_grid=(1.0,), decay_grid=(0.0,))
        assert grid[(1.0, 0.0)] < 0.2


class TestDeltaAssociator:
    def test_learns_a_linear_map_better_than_the_outer_product(self):
        source, target = _linear_problem()
        hebb = HebbianAssociator(eta=1.0, decay=0.0).fit(source, target)
        delta = DeltaAssociator(eta=0.01, epochs=20, seed=0).fit(source, target)
        hebb_r1 = evaluate_direction(hebb, source, target, direction="t2i")["r1"]
        delta_r1 = evaluate_direction(delta, source, target, direction="t2i")["r1"]
        assert delta_r1 >= hebb_r1

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            DeltaAssociator().transform(np.zeros((2, 3)))

    def test_rejects_zero_epochs(self):
        with pytest.raises(ValueError, match="epochs"):
            DeltaAssociator(epochs=0).fit(np.zeros((2, 2)), np.zeros((2, 2)))

    def test_cv_selection_returns_a_grid_point(self):
        source, target = _linear_problem()
        eta, epochs, grid = select_delta_hyperparameters(
            source, target, eta_grid=(0.01,), epoch_grid=(1, 3), folds=3
        )
        assert (eta, epochs) in grid
        assert set(grid) == {(0.01, 1), (0.01, 3)}


class TestDeltaDivergence:
    def test_a_huge_step_is_flagged_not_silently_nan(self):
        """大步长会发散；发散必须被标出来，否则 NaN 会顺着选参污染平均值。"""
        source, target = _linear_problem()
        model = DeltaAssociator(eta=1e6, epochs=5, seed=0).fit(source, target)
        assert model.diverged
        assert not np.all(np.isfinite(model.weight))

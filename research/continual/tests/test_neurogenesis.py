"""神经发生控制器/增长层的最小测试。"""

from __future__ import annotations

import numpy as np

from research.continual.neurogenesis import GrowableFeatureReadout, GrowthController


def test_controller_grows_only_below_threshold_and_caps():
    controller = GrowthController(n0=8, growth=8, max_ratio=2.0, min_accuracy=0.9)
    assert controller.decide(0.95, 8).grow is False
    assert controller.decide(0.5, 8).next_size == 16
    assert controller.decide(0.5, 16).grow is False


def test_growth_pads_old_readout_rows_with_zeros():
    model = GrowableFeatureReadout(4, 3, n0=8, growth=4, seed=0)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 4))
    y = rng.integers(0, 3, size=20)
    model.fit_task(x, y, classes=np.arange(3))
    before = model.weights.copy()
    model.grow()
    assert model.n_features == 12
    assert np.allclose(model.weights[:, :8], before)
    assert np.all(model.weights[:, 8:] == 0.0)


def test_decisions_are_recorded():
    model = GrowableFeatureReadout(4, 2, n0=8, growth=8, max_ratio=2.0, seed=0)
    model.maybe_grow(0.1)
    assert len(model.decisions) == 1
    assert model.n_features == 16

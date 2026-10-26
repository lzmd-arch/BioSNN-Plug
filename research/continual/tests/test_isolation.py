"""预注册 v4 的守卫：只更新激活行、其余行逐位冻结、随机控制臂行数相同。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from research.continual import run_continual
from research.continual.data import CifarSplit
from research.continual.readout import DeltaReadout


def _readout() -> DeltaReadout:
    rng = np.random.default_rng(0)
    readout = DeltaReadout(4, 6, learning_rate=0.1)
    readout.weights[:] = rng.normal(size=(6, 4))
    readout.bias[:] = rng.normal(size=6)
    return readout


class TestReadoutMask:
    def test_only_active_rows_change(self):
        readout = _readout()
        before_w = readout.weights.copy()
        before_b = readout.bias.copy()
        features = np.random.default_rng(1).normal(size=(8, 4))
        labels = np.array([1, 1, 2, 2, 3, 3, 1, 2])

        readout.step(features, labels, active_classes=np.array([1, 3]))

        changed = np.any(readout.weights != before_w, axis=1)
        assert changed.tolist() == [False, True, False, True, False, False]
        changed_b = readout.bias != before_b
        assert changed_b.tolist() == [False, True, False, True, False, False]

    def test_inactive_rows_stay_bitwise_frozen_over_steps(self):
        readout = _readout()
        before = readout.weights[[0, 2, 4, 5]].copy()
        features = np.random.default_rng(2).normal(size=(8, 4))
        labels = np.array([1, 1, 3, 3, 1, 3, 1, 3])

        for _ in range(5):
            readout.step(features, labels, active_classes=np.array([1, 3]))

        assert np.array_equal(readout.weights[[0, 2, 4, 5]], before)

    def test_none_updates_every_row(self):
        readout = _readout()
        before = readout.weights.copy()
        features = np.random.default_rng(3).normal(size=(8, 4))
        labels = np.array([0, 1, 2, 3, 4, 5, 0, 5])

        readout.step(features, labels)

        assert np.any(readout.weights != before)
        assert np.all(np.any(readout.weights != before, axis=1))

    def test_inactive_scale_zero_matches_the_frozen_path(self):
        first = _readout()
        second = _readout()
        features = np.random.default_rng(4).normal(size=(8, 4))
        labels = np.array([0, 1, 2, 3, 4, 5, 0, 5])

        first.step(features, labels, active_classes=np.array([1, 3]))
        second.step(features, labels, active_classes=np.array([1, 3]), inactive_scale=0.0)

        assert np.array_equal(first.weights, second.weights)
        assert np.array_equal(first.bias, second.bias)

    def test_inactive_scale_one_matches_the_full_update(self):
        first = _readout()
        second = _readout()
        features = np.random.default_rng(5).normal(size=(8, 4))
        labels = np.array([0, 1, 2, 3, 4, 5, 0, 5])

        first.step(features, labels)
        second.step(features, labels, active_classes=np.array([1, 3]), inactive_scale=1.0)

        assert np.array_equal(first.weights, second.weights)
        assert np.array_equal(first.bias, second.bias)

    def test_inactive_rows_move_by_the_scaled_amount(self):
        full = _readout()
        partial = _readout()
        features = np.random.default_rng(6).normal(size=(8, 4))
        labels = np.array([0, 1, 2, 3, 4, 5, 0, 5])
        before = full.weights.copy()

        full.step(features, labels)
        partial.step(features, labels, active_classes=np.array([1, 3]), inactive_scale=0.25)

        full_delta = full.weights - before
        partial_delta = partial.weights - before
        assert np.allclose(partial_delta[1], full_delta[1])
        assert np.allclose(partial_delta[3], full_delta[3])
        assert np.allclose(partial_delta[[0, 2, 4, 5]], 0.25 * full_delta[[0, 2, 4, 5]])

    def test_inactive_scale_needs_active_classes(self):
        readout = _readout()
        with pytest.raises(ValueError):
            readout.step(np.zeros((2, 4)), np.array([0, 1]), inactive_scale=0.5)

    def test_inactive_scale_must_be_in_range(self):
        readout = _readout()
        with pytest.raises(ValueError):
            readout.step(
                np.zeros((2, 4)), np.array([0, 1]), active_classes=np.array([0]), inactive_scale=1.5
            )

    def test_invalid_active_classes_are_rejected(self):
        readout = _readout()
        features = np.zeros((2, 4))
        labels = np.array([0, 1])
        with pytest.raises(ValueError):
            readout.step(features, labels, active_classes=np.array([], dtype=int))
        with pytest.raises(ValueError):
            readout.step(features, labels, active_classes=np.array([6]))


class TestActiveRowSelection:
    def test_none_and_current(self):
        classes = np.arange(10, 20)
        rng = np.random.default_rng(0)
        assert run_continual._active_rows_for_task("none", classes, rng) is None
        assert np.array_equal(
            run_continual._active_rows_for_task("current-rows", classes, rng), classes
        )

    def test_random_has_the_same_size_and_is_deterministic(self):
        classes = np.arange(10, 20)
        first = run_continual._active_rows_for_task(
            "random-rows", classes, np.random.default_rng(0)
        )
        second = run_continual._active_rows_for_task(
            "random-rows", classes, np.random.default_rng(0)
        )

        assert len(first) == len(classes)
        assert np.array_equal(first, second)
        assert first.min() >= 0 and first.max() < 100
        assert np.all(np.diff(first) > 0)

    def test_unknown_isolation_is_rejected(self):
        with pytest.raises(ValueError):
            run_continual._active_rows_for_task(
                "per-class", np.arange(10), np.random.default_rng(0)
            )


class TestRunnerWiring:
    """runner 必须把每任务的激活行传下去，并把它记进 summary。"""

    def _run(self, monkeypatch, isolation: str):
        rows, cols = 100, 8
        labels = np.arange(100)
        train_x = np.zeros((rows, 32, 32, 3), dtype=np.uint8)
        test_x = np.zeros((100, 32, 32, 3), dtype=np.uint8)
        train = CifarSplit(x=train_x, y=labels)
        test = CifarSplit(x=test_x, y=labels.copy())
        train_features = np.random.default_rng(0).normal(size=(rows, cols))
        test_features = np.random.default_rng(1).normal(size=(100, cols))

        monkeypatch.setattr(run_continual, "load_cifar100", lambda root: (train, test))
        monkeypatch.setattr(
            run_continual,
            "image_features",
            lambda images, **kwargs: train_features if images is train_x else test_features,
        )
        return run_continual.run_with_frozen_frontend_delta(
            seed=0,
            n_tasks=2,
            limit_per_class=None,
            data_root=Path("unused"),
            device=torch.device("cpu"),
            readout_epochs=1,
            readout_batch=8,
            isolation=isolation,
        )

    def test_current_rows_are_recorded(self, monkeypatch):
        result = self._run(monkeypatch, "current-rows")

        assert result["isolation"] == "current-rows"
        assert result["active_rows"][0] == list(range(50))
        assert result["active_rows"][1] == list(range(50, 100))

    def test_random_rows_have_the_same_size(self, monkeypatch):
        result = self._run(monkeypatch, "random-rows")

        assert [len(row) for row in result["active_rows"]] == [50, 50]
        assert result["active_rows"][0] != list(range(50))

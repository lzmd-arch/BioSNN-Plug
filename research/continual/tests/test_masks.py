"""特征级掩码的守卫：Fisher 只由本任务决定、随机臂同尺寸、读出按类屏蔽。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from research.continual import run_continual
from research.continual.data import CifarSplit
from research.continual.masks import fisher_task_mask, jaccard, random_task_mask
from research.continual.readout import DeltaReadout


class TestMaskBuilders:
    def test_fisher_selects_k_and_is_deterministic(self):
        rng = np.random.default_rng(0)
        features = rng.normal(size=(40, 8))
        labels = np.repeat(np.arange(4), 10)
        first = fisher_task_mask(features, labels, 3)
        second = fisher_task_mask(features, labels, 3)

        assert first.sum() == 3
        assert np.array_equal(first, second)
        assert set(np.unique(first)) <= {0.0, 1.0}

    def test_fisher_picks_the_class_separating_feature(self):
        rng = np.random.default_rng(1)
        labels = np.repeat(np.arange(4), 10)
        first = rng.normal(size=(40, 8))
        first[:, 0] += labels * 10.0
        second = rng.normal(size=(40, 8))
        second[:, 3] += labels * 10.0

        assert fisher_task_mask(first, labels, 1)[0] == 1.0
        assert fisher_task_mask(second, labels, 1)[3] == 1.0

    def test_random_has_the_same_size_and_seed(self):
        first = random_task_mask(8, 3, np.random.default_rng(0))
        second = random_task_mask(8, 3, np.random.default_rng(0))
        assert first.sum() == 3
        assert np.array_equal(first, second)

    def test_jaccard(self):
        a = np.array([1, 1, 0, 0], dtype=float)
        b = np.array([1, 0, 1, 0], dtype=float)
        assert jaccard(a, b) == 1 / 3
        assert jaccard(np.zeros(4), np.zeros(4)) == 0.0

    def test_invalid_k(self):
        features = np.zeros((4, 8))
        labels = np.array([0, 0, 1, 1])
        with np.testing.assert_raises(ValueError):
            fisher_task_mask(features, labels, 0)
        with np.testing.assert_raises(ValueError):
            fisher_task_mask(features, labels, 9)


class TestReadoutMask:
    def test_predict_uses_the_class_mask(self):
        readout = DeltaReadout(2, 2, learning_rate=0.1)
        readout.weights[:] = 1.0
        readout.set_masks_for_classes(np.array([0]), np.array([1.0, 0.0]))

        logits = readout.predict(np.array([[1.0, 1.0]]))

        assert np.allclose(logits, [[1.0, 0.0]])

    def test_step_only_updates_masked_features(self):
        readout = DeltaReadout(2, 2, learning_rate=0.5)
        readout.set_masks_for_classes(np.array([0]), np.array([1.0, 0.0]))
        before = readout.weights.copy()

        readout.step(np.array([[1.0, 1.0]]), np.array([0]))

        delta = readout.weights - before
        assert delta[0, 0] != 0.0
        assert delta[0, 1] == 0.0
        assert np.all(delta[1] == 0.0)

    def test_unmasked_path_still_matches_the_original_formula(self):
        readout = DeltaReadout(3, 2, learning_rate=0.1)
        rng = np.random.default_rng(2)
        readout.weights[:] = rng.normal(size=(2, 3))
        readout.bias[:] = rng.normal(size=2)
        x = rng.normal(size=(4, 3))
        predicted = readout.predict(x)
        assert np.allclose(predicted, x @ readout.weights.T + readout.bias)
        assert predicted.shape == (4, 2)


class TestRunnerWiring:
    def _run(self, monkeypatch, feature_mask: str):
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
            feature_mask=feature_mask,
            mask_k=4,
        )

    def test_fisher_masks_are_recorded_per_task(self, monkeypatch):
        result = self._run(monkeypatch, "fisher")

        assert result["feature_mask"] == "fisher"
        assert [len(indices) for indices in result["mask_indices"]] == [4, 4]
        assert 0.0 <= result["mask_jaccard"] <= 1.0

    def test_random_masks_are_recorded_and_differ(self, monkeypatch):
        result = self._run(monkeypatch, "random")

        assert [len(indices) for indices in result["mask_indices"]] == [4, 4]
        assert result["mask_indices"][0] != result["mask_indices"][1]

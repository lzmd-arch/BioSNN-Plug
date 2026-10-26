"""任务推断臂的守卫：原型只由本任务决定、路由可分、per-task 头与 oracle 只差路由。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from research.continual import run_continual, task_inference
from research.continual.data import CifarSplit
from research.continual.task_inference import _route, _task_stats, run_with_task_inference


def _features(labels: np.ndarray, dims: int = 100) -> np.ndarray:
    """全局类 one-hot：任务内 10 个类可分，任务原型也天然分开（各自占一个块）。"""
    features = np.zeros((labels.shape[0], dims))
    features[np.arange(labels.shape[0]), labels] = 1.0
    return features


class TestPrototypeAndRouting:
    def test_stats_are_the_mean_of_class_means_and_within_variance(self):
        features = np.array([[1.0, 0.0], [3.0, 0.0], [10.0, 2.0], [14.0, 2.0]])
        labels = np.array([0, 0, 1, 1])
        mean, variance = _task_stats(features, labels, np.array([0, 1]))

        assert np.allclose(mean, [7.0, 1.0])
        assert np.allclose(variance, [2.5, 0.0])

    def test_route_picks_the_nearest_prototype(self):
        features = np.array([[1.0, 0.0], [9.0, 0.0]])
        prototypes = [np.array([0.0, 0.0]), np.array([10.0, 0.0])]

        assert _route(features, prototypes).tolist() == [0, 1]

    def test_mahalanobis_uses_the_within_task_variance(self):
        features = np.array([[5.0, 0.0]])
        prototypes = [np.array([0.0, 0.0]), np.array([10.0, 0.0])]
        variances = [np.array([1.0, 1.0]), np.array([100.0, 1.0])]

        assert _route(features, prototypes, variances).tolist() == [1]


class TestRunnerWiring:
    def _run(self, monkeypatch, routing: str):
        rows, cols = 2000, 100
        labels = np.repeat(np.arange(100), 20)
        features = _features(labels, cols)
        train_x = np.zeros((rows, 32, 32, 3), dtype=np.uint8)
        test_x = np.zeros((rows, 32, 32, 3), dtype=np.uint8)
        train = CifarSplit(x=train_x, y=labels)
        test = CifarSplit(x=test_x, y=labels.copy())

        monkeypatch.setattr(task_inference, "load_cifar100", lambda root: (train, test))
        monkeypatch.setattr(
            task_inference,
            "image_features",
            lambda images, **kwargs: features,
        )
        return run_with_task_inference(
            seed=0,
            n_tasks=10,
            limit_per_class=None,
            data_root=Path("unused"),
            device=torch.device("cpu"),
            readout_epochs=10,
            readout_batch=32,
            routing=routing,
        )

    def test_routed_separates_the_two_tasks(self, monkeypatch):
        result = self._run(monkeypatch, "routed")

        assert result["routing"] == "routed"
        assert result["routing_accuracy"] == 1.0
        assert result["acc"] > 0.8
        assert result["valid"] is True
        assert all(accuracy >= 0.5 for accuracy in result["head_train_accuracy"])
        assert all(norm > 0 for norm in result["head_weight_norm"])
        assert np.array(result["accuracy_matrix"]).shape == (10, 10)

    def test_oracle_uses_the_true_task_head(self, monkeypatch):
        result = self._run(monkeypatch, "oracle")

        assert result["routing_accuracy"] == 1.0
        assert result["acc"] > 0.8
        matrix = np.array(result["accuracy_matrix"])
        assert np.all(matrix[np.tril_indices(10)] > 0.8)

    def test_cli_rejects_combinations(self):
        with np.testing.assert_raises(SystemExit):
            run_continual.main(
                ["--task-inference", "routed", "--features", "w1", "--readout", "delta"]
            )
        with np.testing.assert_raises(SystemExit):
            run_continual.main(["--task-inference", "routed", "--replay-rho", "0.3"])

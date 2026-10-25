"""预注册 v3 的归一化守卫：统计量只来自任务 1、冻结、可复核、不改变默认路径。

这些测试盯的是**行为**，不是"变量设了没"：统计量是否只由任务 1 决定、应用是否逐位一致、
常数维会不会变成 NaN。runner 层的接线另用 monkeypatch 直接量 digest。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from research.continual import run_continual
from research.continual.data import CifarSplit
from research.continual.normalization import (
    IdentityTransform,
    Task1Standardizer,
    build_transform,
)


def _features(seed: int = 0, rows: int = 64, cols: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(rows, cols)) * 3.0 + 1.5


class TestTask1Standardizer:
    def test_outputs_are_standardized(self):
        features = _features()
        transform = Task1Standardizer().fit(features)
        out = transform.transform(features)

        assert np.allclose(out.mean(axis=0), 0.0, atol=1e-10)
        assert np.allclose(out.std(axis=0), 1.0, atol=1e-10)

    def test_match_scale_keeps_the_row_norm(self):
        features = _features()
        before = np.linalg.norm(features, axis=1).mean()
        out = Task1Standardizer(scale="match-task1").fit(features).transform(features)

        assert np.isclose(np.linalg.norm(out, axis=1).mean(), before, rtol=1e-12)
        # 逐维尺度仍然被拉平——否则这一档就退化成"只乘一个标量"。
        assert np.allclose(out.std(axis=0), out.std(axis=0)[0], rtol=1e-10)

    def test_fit_is_frozen(self):
        """拟合后再喂新数据，变换不能改——统计量冻结是预注册写死的那条。"""
        transform = Task1Standardizer().fit(_features(seed=0))
        first = transform.transform(_features(seed=1, rows=8))
        _ = transform.transform(_features(seed=2, rows=32))
        second = transform.transform(_features(seed=1, rows=8))

        assert np.array_equal(first, second)
        assert np.array_equal(transform.mu, Task1Standardizer().fit(_features(seed=0)).mu)

    def test_constant_dimension_stays_finite(self):
        features = _features()
        features[:, 3] = 7.0
        out = Task1Standardizer().fit(features).transform(features)

        assert np.all(np.isfinite(out))
        assert np.allclose(out[:, 3], 0.0)

    def test_digest_is_reproducible_and_data_dependent(self):
        first = Task1Standardizer().fit(_features(seed=0)).digest()
        second = Task1Standardizer().fit(_features(seed=0)).digest()
        other = Task1Standardizer().fit(_features(seed=1)).digest()

        assert first == second
        assert first["mu_sha256"] != other["mu_sha256"]
        assert first["kind"] == "task1-standardize"

    def test_unknown_scale_is_rejected(self):
        with pytest.raises(ValueError):
            Task1Standardizer(scale="per-task")
        with pytest.raises(ValueError):
            Task1Standardizer(eps=0.0)


class TestIdentity:
    def test_identity_is_bitwise_unchanged(self):
        features = _features(seed=3).astype(np.float32)
        out = IdentityTransform().fit(features).transform(features)

        assert out is features
        assert np.array_equal(out, features)

    def test_build_transform_rejects_silent_noops(self):
        with pytest.raises(ValueError):
            build_transform("none", scale="match-task1")
        with pytest.raises(ValueError):
            build_transform("whiten")


class TestRunnerWiring:
    """runner 必须在任务 1 上拟合并冻结；不能偷看任务 2 的特征。"""

    def _run(self, monkeypatch, *, task_one_shift: float, task_two_shift: float):
        rows, cols = 100, 8
        # task_classes(2, 100) 是 50 类 + 50 类；每个类一条样本就够量 digest 了。
        labels = np.arange(100)
        train_x = np.zeros((rows, 32, 32, 3), dtype=np.uint8)
        test_x = np.zeros((100, 32, 32, 3), dtype=np.uint8)
        train = CifarSplit(x=train_x, y=labels)
        test = CifarSplit(x=test_x, y=labels.copy())
        train_features = np.random.default_rng(0).normal(size=(rows, cols))
        train_features[labels < 50] += task_one_shift
        train_features[labels >= 50] += task_two_shift
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
            normalize="task1-standardize",
        )

    def test_digest_ignores_later_tasks(self, monkeypatch):
        base = self._run(monkeypatch, task_one_shift=0.0, task_two_shift=0.0)
        shifted = self._run(monkeypatch, task_one_shift=0.0, task_two_shift=5.0)

        assert base["normalizer"] == shifted["normalizer"]
        assert base["normalize"] == "task1-standardize"
        assert base["normalizer"]["scale"] == "native"

    def test_digest_tracks_the_first_task(self, monkeypatch):
        base = self._run(monkeypatch, task_one_shift=0.0, task_two_shift=0.0)
        shifted = self._run(monkeypatch, task_one_shift=5.0, task_two_shift=0.0)

        assert base["normalizer"] != shifted["normalizer"]

    def test_cli_rejects_conflicting_flags(self):
        with pytest.raises(SystemExit):
            run_continual.main(["--normalize", "none", "--normalize-scale", "match-task1"])
        with pytest.raises(SystemExit):
            run_continual.main(["--readout", "ncm", "--normalize", "task1-standardize"])

"""预注册 CIL 口径的守卫：样本按任务切，argmax 在**全部已见类**里。

同一份 logits，两种口径可以差很多——这正是连续出错两次的地方：

- 累计类集口径（原 `formal/`）：a[t][i] 变成"前 i+1 个任务的联合准确率"；
- oracle 口径（`formal_corrected/`）：argmax 只在本任务 10 类里，那是**上界**；
- 预注册口径：样本按任务切，分类器面对任务 t 时的全部已见类，**不提供任务 id**。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from research.continual.readout import DeltaReadout, evaluate
from research.continual.run_continual import (
    _sample_replay,
    _train_delta_readout,
    evaluate_task,
)


class _IdentityFeatures:
    def features(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _PrefersClassTwo:
    """无论输入是什么，都把 class 2 的 logit 打到最高。"""

    def predict(self, features: np.ndarray) -> np.ndarray:
        out = np.zeros((features.shape[0], 4), dtype=np.float64)
        out[:, 2] = 1.0
        return out


def test_cil_metric_keeps_all_seen_classes_in_the_argmax() -> None:
    x = torch.eye(4)
    y = torch.tensor([0, 1, 2, 3])
    task_a = np.array([0, 1])
    seen = np.array([0, 1, 2, 3])

    cil = evaluate_task(_IdentityFeatures(), _PrefersClassTwo(), x, y, task_a, seen)
    oracle = evaluate(_PrefersClassTwo(), x.numpy()[:2], y.numpy()[:2], task_a)

    assert cil == 0.0, "全部已见类的 argmax 选了 class 2，任务 A 的两个样本都错"
    assert oracle == 0.5, "oracle 只在本任务两类里选，样本 0 对、样本 1 错"


def test_cil_metric_returns_nan_when_the_task_has_no_samples() -> None:
    x = torch.eye(4)
    y = torch.tensor([0, 1, 2, 3])

    assert np.isnan(
        evaluate_task(
            _IdentityFeatures(),
            _PrefersClassTwo(),
            x,
            y,
            np.array([9]),
            np.array([0, 1, 2, 3]),
        )
    )


class TestReplayHelpers:
    def test_sample_replay_takes_a_fixed_number_per_class(self) -> None:
        features = np.arange(40, dtype=np.float64).reshape(20, 2)
        labels = np.repeat([0, 1], 10)
        rng = np.random.default_rng(0)

        sampled_features, sampled_labels = _sample_replay(
            features, labels, np.array([0, 1]), 3, rng
        )

        assert sampled_features.shape == (6, 2)
        assert sorted(sampled_labels.tolist()) == [0, 0, 0, 1, 1, 1]
        assert set(np.unique(sampled_features[:, 0])).issubset(set(features[:, 0]))

    def test_replay_requires_a_minibatch_size(self) -> None:
        readout = DeltaReadout(2, 2)
        with pytest.raises(ValueError, match="readout-batch"):
            _train_delta_readout(
                readout,
                np.zeros((4, 2)),
                np.zeros(4, dtype=int),
                np.zeros((2, 2)),
                np.zeros(2, dtype=int),
                epochs=1,
                batch=0,
                rho=0.1,
                rng=np.random.default_rng(0),
            )

    def test_minibatch_training_mixes_replay_and_changes_the_weights(self) -> None:
        rng = np.random.default_rng(0)
        features = rng.normal(size=(32, 4))
        labels = rng.integers(0, 2, size=32)
        replay = rng.normal(size=(8, 4))
        replay_labels = rng.integers(0, 2, size=8)
        readout = DeltaReadout(4, 2, learning_rate=0.1)
        before = readout.weights.copy()

        _train_delta_readout(
            readout,
            features,
            labels,
            replay,
            replay_labels,
            epochs=1,
            batch=8,
            rho=0.5,
            rng=np.random.default_rng(1),
        )

        assert not np.allclose(readout.weights, before)
        assert np.all(np.isfinite(readout.weights))

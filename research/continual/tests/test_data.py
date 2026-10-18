"""任务划分与子集切分：不下载，用合成的数组。"""

from __future__ import annotations

import numpy as np
import pytest

from research.continual.data import CifarSplit, task_classes


class TestTaskClasses:
    def test_balanced_split_is_disjoint_and_complete(self):
        tasks = task_classes(10, 100)
        assert len(tasks) == 10
        assert all(len(t) == 10 for t in tasks)
        flat = np.concatenate(tasks)
        assert np.array_equal(np.sort(flat), np.arange(100))

    def test_order_is_ascending(self):
        tasks = task_classes(10, 100)
        assert np.array_equal(tasks[0], np.arange(0, 10))
        assert np.array_equal(tasks[-1], np.arange(90, 100))

    def test_rejects_uneven_split(self):
        with pytest.raises(ValueError, match="整除"):
            task_classes(7, 100)


class TestSubset:
    def _split(self) -> CifarSplit:
        x = np.arange(20, dtype=np.uint8).reshape(20, 1, 1, 1)
        y = np.array([0, 0, 1, 1] * 5)
        return CifarSplit(x=x, y=y)

    def test_keeps_only_the_given_classes(self):
        split = self._split().subset(np.array([1]))
        assert set(split.y.tolist()) == {1}
        assert split.x.shape[0] == 10

    def test_limit_caps_each_class(self):
        split = self._split().subset(np.array([0, 1]), limit=2)
        assert split.y.tolist().count(0) == 2
        assert split.y.tolist().count(1) == 2
        assert split.x.shape[0] == 4

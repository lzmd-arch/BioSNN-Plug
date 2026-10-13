"""Tests for conflict detection (plan section 3.4's criterion).

The criterion is "cosine similarity of the two rules' update directions on the same
synapse; angle > 90 degrees is a conflict". These tests pin that and nothing else.

The per-synapse choice matters and is asserted: comparing flattened tensors would
answer a different (weaker) question and would hide a region of total disagreement.
"""

from __future__ import annotations

import numpy as np
import pytest

from research.arbiter.conflict import (
    ConflictReport,
    cosine_similarity,
    is_conflict,
    measure_conflict,
)


class TestCosine:
    """Two readings of "update direction on the same synapse"; both are asserted."""

    def test_identical_updates_have_cosine_one(self):
        a = np.array([[1.0, -2.0], [0.5, 3.0]])
        assert np.allclose(cosine_similarity(a, a), 1.0)

    def test_opposite_updates_have_cosine_minus_one(self):
        a = np.array([[1.0, 2.0]])
        assert np.allclose(cosine_similarity(a, -a), -1.0)

    def test_per_synapse_cosine_is_degenerate(self):
        """Per-synapse cosine is always +-1 -- "angle > 90 degrees" collapses to sign."""
        a = np.array([[1.0, -3.0, 2.5]])
        b = np.array([[4.0, 0.1, -0.2]])
        per_synapse = cosine_similarity(a, b, axis=None)
        assert set(np.round(per_synapse.ravel(), 6)) <= {-1.0, 1.0}

    def test_per_row_cosine_carries_a_real_angle(self):
        """Per-row cosine takes intermediate values -- a threshold can express grading."""
        a = np.array([[1.0, 1.0]])
        b = np.array([[1.0, 0.5]])
        value = float(cosine_similarity(a, b, axis=1)[0, 0])
        assert 0.0 < value < 1.0

    def test_is_per_group_not_global(self):
        """A big agreeing block must not hide a region of total disagreement.

        This is the failure mode the criterion has to survive: rule A and rule B agree
        on most synapses and are exactly opposite on one. A cosine computed on the
        flattened tensor says "they agree"; per-synapse says the fourth synapse is a
        total disagreement. Neither reading is wrong -- they answer different questions,
        and the flattened one is the dangerous one to reach for.
        """
        # 三个大正向 + 一个小反向：整体余弦仍然是正的（反向那一项被压过）。
        a = np.array([[10.0, 10.0, 10.0, -1.0]])
        b = np.array([[10.0, 10.0, 10.0, 1.0]])
        flat_cosine = float((a.ravel() @ b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b)))
        assert flat_cosine > 0, "flattened cosine reports agreement"

        per_synapse = cosine_similarity(a, b, axis=None)
        assert per_synapse[0, 3] < 0, "the fourth synapse is a total disagreement"

        # 逐行口径下整行仍以正向为主，余弦为正——所以逐行也看不见那个反向。
        # 要看见它就得逐突触（axis=None）。**这正是两种口径必须都提供、且不能混用的理由。**
        assert float(cosine_similarity(a, b, axis=1)[0, 0]) > 0

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="形状"):
            cosine_similarity(np.zeros((2, 2)), np.zeros((3, 3)))


class TestConflict:
    def test_angle_above_90_is_conflict(self):
        """Per-synapse: opposite sign is the conflict. `axis=None` for that reading."""
        a = np.array([[1.0, 1.0]])
        b = np.array([[-1.0, 1.0]])
        mask = is_conflict(a, b, axis=None)
        assert mask.tolist() == [[True, False]]

    def test_threshold_controls_strictness(self):
        """A graded threshold only makes sense per row, where the angle is real."""
        # 每行一个"方向向量"，三行的余弦分别是 1.0 / 0.949 / −1.0。
        a = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        b = np.array([[1.0, 1.0], [1.0, 0.5], [-1.0, -1.0]])
        cosine = cosine_similarity(a, b, axis=1).ravel()
        assert np.allclose(cosine, [1.0, 0.9487, -1.0], atol=1e-3)

        # 严格（阈值 0.8）：只有余弦 < 0.8 的那行冲突——第二行 0.949 不算。
        assert is_conflict(a, b, threshold=0.8, axis=1).ravel().tolist() == [False, False, True]
        # 宽松（阈值 0.0，即"夹角 > 90°"的原文口径）：只有完全反向那行。
        assert is_conflict(a, b, threshold=0.0, axis=1).ravel().tolist() == [False, False, True]
        # 放宽到 −0.5：只有余弦 < −0.5 的那行冲突——第三行的 −1.0 仍算，前两行不算。
        assert is_conflict(a, b, threshold=-0.5, axis=1).ravel().tolist() == [False, False, True]
        # 放宽到 −1.0：没有余弦严格小于 −1，所以一行都不冲突。
        assert is_conflict(a, b, threshold=-1.0, axis=1).ravel().tolist() == [False, False, False]

    def test_event_is_one_when_any_group_conflicts(self):
        """One step with any conflicting group counts as one conflict event.

        Per-row reading (the default): one row wholly reversed is a conflict.
        """
        a = np.ones((4, 4))
        b = np.ones((4, 4))
        b[2, :] = -1.0
        report = measure_conflict(a, b)
        assert report.events == 1
        assert report.n_conflicting == 1
        assert report.n_synapses == 4  # 四行

    def test_per_synapse_reading_counts_each_synapse(self):
        """axis=None counts synapses, not rows -- the two readings really differ."""
        a = np.ones((4, 4))
        b = np.ones((4, 4))
        b[2, 3] = -1.0
        report = measure_conflict(a, b, axis=None)
        assert report.n_synapses == 16
        assert report.n_conflicting == 1
        assert report.events == 1

    def test_no_event_when_all_agree(self):
        report = measure_conflict(np.ones((3, 3)), np.ones((3, 3)))
        assert report.events == 0
        assert report.n_conflicting == 0
        assert report.conflict_fraction == 0.0

    def test_report_is_immutable(self):
        report = measure_conflict(np.ones((2, 2)), -np.ones((2, 2)))
        assert isinstance(report, ConflictReport)
        with pytest.raises(AttributeError):
            report.events = 5

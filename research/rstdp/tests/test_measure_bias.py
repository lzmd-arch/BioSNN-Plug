"""``research.rstdp.measure_bias`` 的测试。

这是计划书 §七 的验收指标（成功偏移 < 10%σR），所以定义必须写死并可查：偏移、σR、
比值各自怎么算、边界怎么处理。**口径写错不会报错，只会让验收结论失真。**
"""

from __future__ import annotations

import math

import pytest

from research.rstdp.measure_bias import OFFSET_THRESHOLD, BiasReport, BiasTracker


class TestBiasReport:
    def test_offset_is_the_mean(self):
        report = BiasReport(n=3, offset=0.5, sigma=1.0)
        assert report.offset == pytest.approx(0.5)

    def test_ratio_is_absolute_offset_over_sigma(self):
        assert BiasReport(n=10, offset=0.2, sigma=2.0).ratio == pytest.approx(0.1)
        assert BiasReport(n=10, offset=-0.2, sigma=2.0).ratio == pytest.approx(0.1)

    def test_ratio_is_unsigned_so_a_negative_offset_does_not_hide(self):
        """偏移 < −0.4σR 是那条"性能低于学习前"的退化路径，符号不能被丢掉。"""
        negative = BiasReport(n=10, offset=-0.5, sigma=1.0)
        assert negative.ratio == pytest.approx(0.5)
        assert not negative.passed

    def test_threshold_boundary(self):
        assert BiasReport(n=10, offset=0.099, sigma=1.0).passed
        assert not BiasReport(n=10, offset=0.101, sigma=1.0).passed

    def test_threshold_constant_matches_the_project_plan(self):
        assert OFFSET_THRESHOLD == 0.10

    def test_zero_sigma_with_a_nonzero_offset_fails(self):
        """σR = 0 且偏移非零时比值无定义——必须判失败，不能蒙混成达标。"""
        report = BiasReport(n=5, offset=0.3, sigma=0.0)
        assert math.isinf(report.ratio)
        assert not report.passed

    def test_zero_sigma_with_a_zero_offset_passes(self):
        report = BiasReport(n=5, offset=0.0, sigma=0.0)
        assert report.ratio == 0.0
        assert report.passed

    def test_render_names_the_verdict(self):
        assert "达到" in BiasReport(n=5, offset=0.0, sigma=1.0).render()
        assert "未达到" in BiasReport(n=5, offset=1.0, sigma=1.0).render()


class TestBiasTracker:
    def test_offset_and_sigma_match_a_hand_computed_case(self):
        """用一组能手算的数验证，并确认 σ 用的是**总体**标准差（ddof = 0）。

        序列 [1, 2, 3, 4]：均值 2.5，总体标准差 = sqrt(1.25) ≈ 1.1180，
        样本标准差（ddof=1）≈ 1.2910。用后者会把比值压小，等于偷偷放宽判据。
        """
        tracker = BiasTracker(tail_window=None)
        tracker.extend([1.0, 2.0, 3.0, 4.0])

        report = tracker.overall
        assert report.n == 4
        assert report.offset == pytest.approx(2.5)
        assert report.sigma == pytest.approx(math.sqrt(1.25))
        assert report.sigma == pytest.approx(1.118033988749895, rel=1e-12)

    def test_ratio_for_the_hand_computed_case(self):
        tracker = BiasTracker(tail_window=None)
        tracker.extend([1.0, 2.0, 3.0, 4.0])
        assert tracker.overall.ratio == pytest.approx(2.5 / math.sqrt(1.25))

    def test_a_centred_signal_has_no_offset(self):
        tracker = BiasTracker(tail_window=None)
        tracker.extend([-1.0, 1.0, -1.0, 1.0])
        assert tracker.overall.offset == pytest.approx(0.0)
        assert tracker.overall.passed

    def test_a_constant_positive_signal_fails(self):
        """恒定的成功信号意味着 Critic 完全没在工作——不该算达标。"""
        tracker = BiasTracker(tail_window=None)
        tracker.extend([1.0] * 20)
        assert not tracker.overall.passed

    def test_record_and_len(self):
        tracker = BiasTracker()
        tracker.record(0.5)
        tracker.extend([1.0, 2.0])
        assert len(tracker) == 3

    def test_tail_uses_the_last_window_only(self):
        tracker = BiasTracker(tail_window=2)
        tracker.extend([100.0, 100.0, 1.0, -1.0])
        # 全程的偏移很大，末段的偏移是 0
        assert tracker.overall.offset > 10.0
        assert tracker.tail.offset == pytest.approx(0.0)
        assert tracker.tail.passed

    def test_tail_falls_back_to_overall_when_short(self):
        tracker = BiasTracker(tail_window=100)
        tracker.extend([1.0, 2.0])
        assert tracker.tail.n == tracker.overall.n

    def test_tail_window_none_disables_the_split(self):
        tracker = BiasTracker(tail_window=None)
        tracker.extend([1.0, 2.0, 3.0])
        assert tracker.tail.n == 3

    def test_empty_tracker_is_well_defined(self):
        tracker = BiasTracker()
        assert tracker.overall.n == 0
        assert tracker.overall.passed, "空序列不该被判失败——它只是还没测量"

    def test_render_mentions_both_windows(self):
        tracker = BiasTracker(tail_window=2)
        tracker.extend([1.0, 2.0, 3.0, 4.0])
        rendered = tracker.render()
        assert "全程" in rendered
        assert "末段" in rendered

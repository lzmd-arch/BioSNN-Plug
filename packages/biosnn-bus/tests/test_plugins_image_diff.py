"""示例插件 ``DiffImagePlugin`` 的测试。

这个插件同时是插件作者的参考实现，因此它的测试也充当**测试模板**：形状契约、
参数校验、编解码往返、以及在总线里的实际接入。
"""

from __future__ import annotations

import numpy as np
import pytest
from biosnn_bus import PassThroughMembrane, SpikeBus, SpikeTrain
from biosnn_bus.plugins import DiffImagePlugin

HEIGHT, WIDTH, THRESHOLD = 4, 5, 0.25


@pytest.fixture
def plugin() -> DiffImagePlugin:
    return DiffImagePlugin(height=HEIGHT, width=WIDTH, threshold=THRESHOLD)


def frames_from_steps(steps: np.ndarray) -> np.ndarray:
    """把逐步跳变序列累加成帧序列。"""
    return np.cumsum(steps, axis=0)


class TestParameters:
    @pytest.mark.parametrize(("height", "width"), [(0, 4), (4, 0), (-1, 4), (4, 1.5), (True, 4)])
    def test_rejects_bad_dimensions(self, height, width):
        with pytest.raises(ValueError):
            DiffImagePlugin(height=height, width=width)

    @pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5])
    def test_rejects_bad_threshold(self, threshold):
        with pytest.raises(ValueError, match="threshold"):
            DiffImagePlugin(height=4, width=4, threshold=threshold)

    def test_spike_dim_counts_on_and_off_channels(self, plugin):
        assert plugin.spike_dim == HEIGHT * WIDTH * 2

    def test_declares_the_semantic_channel(self, plugin):
        # 计划书 §2.3 模态表：图像走语义融合通道
        assert plugin.fusion_channel == "semantic"

    def test_repr(self, plugin):
        assert "DiffImagePlugin" in repr(plugin) and "4" in repr(plugin)


class TestEncode:
    def test_shape(self, plugin):
        train = plugin.encode(np.zeros((6, HEIGHT, WIDTH)))
        assert train.data.shape == (6, plugin.spike_dim)
        assert train.is_binary
        assert train.channel is plugin.channel

    def test_blank_sequence_is_silent(self, plugin):
        train = plugin.encode(np.zeros((6, HEIGHT, WIDTH)))
        assert train.spike_count == 0

    def test_detects_rising_edge_on_the_on_channel(self, plugin):
        frames = np.zeros((2, HEIGHT, WIDTH))
        frames[1, 1, 2] = THRESHOLD
        train = plugin.encode(frames)
        assert train.data[1, 1 * WIDTH + 2], "亮度上升应触发 ON 通道"

    def test_detects_falling_edge_on_the_off_channel(self, plugin):
        frames = np.zeros((3, HEIGHT, WIDTH))
        frames[1, 1, 2] = THRESHOLD
        frames[2, 1, 2] = 0.0
        train = plugin.encode(frames)
        off_offset = HEIGHT * WIDTH
        assert train.data[2, off_offset + 1 * WIDTH + 2], "亮度下降应触发 OFF 通道"

    def test_change_below_threshold_is_ignored(self, plugin):
        frames = np.zeros((2, HEIGHT, WIDTH))
        frames[1] = THRESHOLD * 0.5
        assert plugin.encode(frames).spike_count == 0

    def test_change_exactly_at_threshold_fires(self, plugin):
        """阈值语义是"达到即触发"，与"超过才触发"差一个边界，值得钉死。"""
        frames = np.zeros((2, HEIGHT, WIDTH))
        frames[1] = THRESHOLD
        assert plugin.encode(frames).spike_count == HEIGHT * WIDTH

    def test_rejects_wrong_shape(self, plugin):
        with pytest.raises(ValueError, match="期望形状"):
            plugin.encode(np.zeros((6, HEIGHT, WIDTH + 1)))

    def test_rejects_non_finite(self, plugin):
        frames = np.zeros((2, HEIGHT, WIDTH))
        frames[1, 0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            plugin.encode(frames)


class TestDecode:
    def test_shape(self, plugin):
        train = plugin.encode(np.zeros((6, HEIGHT, WIDTH)))
        assert plugin.decode(train).shape == (6, HEIGHT, WIDTH)

    def test_exact_roundtrip_on_threshold_sized_jumps(self, plugin):
        """每步跳变恰为 ±threshold 时，编解码应当精确互逆。"""
        rng = np.random.default_rng(0)
        steps = rng.integers(-1, 2, size=(7, HEIGHT, WIDTH)) * THRESHOLD
        original = frames_from_steps(steps)

        restored = plugin.decode(plugin.encode(original))
        np.testing.assert_allclose(restored, original, atol=1e-12)

    def test_per_step_error_is_bounded_for_larger_jumps(self, plugin):
        """跳变幅度超出阈值时幅度信息丢失，但**每个时间步**的误差有明确上界。

        注意比较的是逐步跳变（一阶差分），不是累积帧值——累积误差会漂移，见下一条
        测试。把两者混为一谈会写出看起来严格、实则不成立的断言。
        """
        multipliers = np.array([-3.0, -2.0, -1.5, 0.0, 1.5, 2.0, 3.0])
        rng = np.random.default_rng(1)
        steps = rng.choice(multipliers, size=(8, HEIGHT, WIDTH)) * THRESHOLD

        original = frames_from_steps(steps)
        restored = plugin.decode(plugin.encode(original))

        def deltas(sequence: np.ndarray) -> np.ndarray:
            return np.diff(sequence, axis=0, prepend=np.zeros_like(sequence[:1]))

        max_jump = np.abs(steps).max()
        bound = max(THRESHOLD, max_jump - THRESHOLD)
        assert np.abs(deltas(restored) - deltas(original)).max() <= bound + 1e-12

    def test_cumulative_drift_on_consecutive_same_sign_jumps(self, plugin):
        """钉住已知限制：误差会沿时间轴累积。

        一个 ``(时间步, 像素)`` 最多产生一个脉冲，所以连续同向跳变时每一步都少记
        ``|Δ| - threshold``，漂移不断变大。这不是缺陷，是事件编码的固有性质——
        写清楚它，比假装 decode 是无损逆变换要诚实。
        """
        jump = 2 * THRESHOLD
        steps = np.full((5, HEIGHT, WIDTH), jump)
        original = frames_from_steps(steps)
        restored = plugin.decode(plugin.encode(original))

        drift = (restored - original)[:, 0, 0]
        # 每一步都少记 (jump - threshold)，第 t 步累计少记 (t+1) 份
        expected = -(np.arange(5) + 1) * (jump - THRESHOLD)
        np.testing.assert_allclose(drift, expected)
        assert abs(drift[-1]) > abs(drift[0]), "漂移应当单调增大"

    def test_rejects_wrong_container(self, plugin):
        with pytest.raises(TypeError, match="SpikeTrain"):
            plugin.decode(np.zeros((4, plugin.spike_dim)))

    def test_rejects_wrong_dimension(self, plugin):
        with pytest.raises(ValueError, match="spike_dim"):
            plugin.decode(SpikeTrain(data=np.zeros((4, plugin.spike_dim + 1), dtype=bool)))


class TestBusIntegration:
    def test_plugs_into_the_bus(self, plugin):
        bus = SpikeBus(bus_dim=64, seed=0)
        bus.register(plugin)

        frames = np.zeros((5, HEIGHT, WIDTH))
        frames[1, 0, 0] = THRESHOLD
        frames[3, 2, 3] = -THRESHOLD

        out = bus.step({"image_diff": frames})
        assert out.data.shape == (5, 64)
        assert out.spike_count > 0

    def test_routes_to_the_semantic_channel(self, plugin):
        bus = SpikeBus(bus_dim=64, seed=0)
        bus.register(plugin)
        assert bus.describe()["modalities_by_channel"]["semantic"] == ["image_diff"]

    def test_membrane_is_usable(self, plugin):
        membrane = plugin.get_membrane()
        assert isinstance(membrane, PassThroughMembrane)
        payload = np.zeros((2, 3))
        assert membrane.forward(payload) is payload


class TestRegistration:
    def test_example_plugin_is_registered_on_import(self):
        from biosnn_bus import get_plugin

        assert get_plugin("image_diff") is DiffImagePlugin

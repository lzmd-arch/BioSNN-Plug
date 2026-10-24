"""``TemporalAttentionFusion`` 的测试：权重的局部性、加权求和的正确性、接口契约。

⚠️ **第一节是本模块的存在理由**：TAAF 的注意力权重**必须由局部可计算的信号驱动**，
不允许反向传播训练。否则"纯局部学习"的声明在融合层就被戳穿。所以这里有一条
**结构断言**（不是运行时断言）：源码里不能出现任何训练机制。
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest
from biosnn_bus import BioSNNBusError, SpikeBus, SpikeTrain, TemporalAttentionFusion
from biosnn_bus.plugins import DiffImagePlugin, MnistPulsePlugin


def trains(values: list[np.ndarray], dt: float = 1.0) -> list[SpikeTrain]:
    return [SpikeTrain(data=v, dt=dt, channel="semantic") for v in values]


class TestLocalityRedLine:
    """融合层的红线：无反向传播、无可训练参数。"""

    def test_no_training_machinery_in_source(self):
        """源码里不许出现任何训练机制——**结构性断言**，不靠肉眼。"""
        source = inspect.getsource(__import__("biosnn_bus.taaf", fromlist=["*"]))
        tree = ast.parse(source)

        imported = set()
        calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
            elif isinstance(node, ast.Call):
                calls.add(ast.unparse(node.func))

        assert not any(name.startswith("torch") for name in imported), imported
        assert "torch" not in imported
        forbidden = {"backward", "autograd", "optim", "Parameter"}
        hits = {c for c in calls if any(f in c for f in forbidden)}
        assert not hits, f"发现训练相关调用：{hits}"

    def test_weights_depend_only_on_local_rates(self):
        """权重只依赖各路的发放率——换一路的绝对尺度，权重随之变。"""
        fus = TemporalAttentionFusion()
        quiet = trains([np.zeros((4, 10))])
        loud = trains([np.ones((4, 10))])
        w1 = fus.attention_weights([quiet[0], loud[0]])
        w2 = fus.attention_weights([loud[0], quiet[0]])
        # 对调输入 → 权重对调
        assert w1[0] == pytest.approx(w2[1])
        assert w1[1] == pytest.approx(w2[0])
        assert w1[1] > w1[0], "发放率更高的一路应拿到更大权重"

    def test_no_trainable_state(self):
        """TAAF 没有任何可训练的参数——只有两个常量。"""
        fus = TemporalAttentionFusion(temperature=2.0, eps=1e-9)
        assert fus.temperature == 2.0
        assert fus.eps == 1e-9
        assert list(vars(fus)) == ["temperature", "eps"]


class TestWeights:
    def test_sums_to_one(self):
        fus = TemporalAttentionFusion()
        w = fus.attention_weights(trains([np.random.rand(3, 5) for _ in range(4)]))
        assert w.shape == (4,)
        assert w.sum() == pytest.approx(1.0)
        assert (w > 0).all()

    def test_uniform_when_all_zero(self):
        fus = TemporalAttentionFusion()
        w = fus.attention_weights(trains([np.zeros((3, 5)) for _ in range(3)]))
        assert w == pytest.approx(np.full(3, 1 / 3))

    def test_temperature_controls_sharpness(self):
        arr = trains([np.full((3, 5), 0.1), np.full((3, 5), 0.9)])
        flat = TemporalAttentionFusion(temperature=100.0).attention_weights(arr)
        sharp = TemporalAttentionFusion(temperature=0.01).attention_weights(arr)
        assert abs(sharp[1] - 0.5) > abs(flat[1] - 0.5)

    @pytest.mark.parametrize("temperature", [0.0, -1.0])
    def test_rejects_bad_temperature(self, temperature):
        with pytest.raises(ValueError, match="temperature"):
            TemporalAttentionFusion(temperature=temperature)

    def test_rejects_empty(self):
        with pytest.raises(BioSNNBusError, match="至少需要一路"):
            TemporalAttentionFusion().attention_weights([])


class TestFuseProjected:
    def test_weighted_sum(self):
        """手算一遍：权重 × 各路之和。"""
        fus = TemporalAttentionFusion(temperature=1.0)
        a = np.full((2, 3), 1.0)
        b = np.full((2, 3), 0.0)
        out = fus.fuse_projected(trains([a, b]))
        w = fus.attention_weights(trains([a, b]))
        expected = w[0] * a + w[1] * b
        assert np.allclose(out.data, expected)

    def test_rejects_mixed_channels(self):
        fus = TemporalAttentionFusion()
        a = SpikeTrain(data=np.zeros((2, 3)), channel="semantic")
        b = SpikeTrain(data=np.zeros((2, 3)), channel="temporal")
        with pytest.raises(BioSNNBusError, match="通道"):
            fus.fuse_projected([a, b])

    def test_rejects_mismatched_shapes(self):
        fus = TemporalAttentionFusion()
        a = SpikeTrain(data=np.zeros((2, 3)), channel="semantic")
        b = SpikeTrain(data=np.zeros((2, 5)), channel="semantic")
        with pytest.raises(BioSNNBusError, match="同一维度"):
            fus.fuse_projected([a, b])

    def test_rejects_empty(self):
        with pytest.raises(BioSNNBusError, match="至少需要一路"):
            TemporalAttentionFusion().fuse_projected([])


class TestInBus:
    def test_output_dim_is_bus_dim_even_with_mixed_spikedom(self):
        """C3 的核心：各路 spike_dim 不同也能加权求和，输出第二维恒为 bus_dim。"""
        bus = SpikeBus(bus_dim=128, seed=0, strategy=TemporalAttentionFusion())
        bus.register(DiffImagePlugin(height=8, width=8))  # spike_dim = 128
        bus.register(MnistPulsePlugin(height=28, width=28, n_steps=8, seed=0))  # 784

        frames = np.zeros((4, 8, 8))
        frames[1, 2:6, 2:6] = 0.8
        mnist = np.random.default_rng(0).random((28, 28))

        out = bus.step({"image_diff": frames, "mnist_pulse": mnist})
        assert out.n_neurons == 128 == bus.bus_dim
        # ADR-0012 的 dtype 契约是 float32，两种融合工作点都一样。先投影后融合的
        # 加权和内部是 float64，总线在出口收口——这条断言原先钉的是 float64，
        # 与契约表的旧错法一致。
        assert out.data.dtype == np.float32

    def test_is_deterministic(self):
        """同一 seed 两次必须给出同一输出——TAAF 不引入随机性。"""

        def run():
            bus = SpikeBus(bus_dim=64, seed=0, strategy=TemporalAttentionFusion())
            bus.register(MnistPulsePlugin(height=28, width=28, n_steps=4, seed=7))
            return bus.step({"mnist_pulse": np.random.default_rng(1).random((28, 28))}).data

        assert np.array_equal(run(), run())

    def test_describe_reports_strategy(self):
        bus = SpikeBus(bus_dim=16, seed=0, strategy=TemporalAttentionFusion())
        assert bus.describe()["strategy"] == "TemporalAttentionFusion"

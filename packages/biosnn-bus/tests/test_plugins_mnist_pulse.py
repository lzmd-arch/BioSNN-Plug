"""``MnistPulsePlugin`` 的测试：形状契约、参数校验、编解码的误差性质、总线接入。

⚠️ 这里钉住的是**发放率编码**的数学性质，不是"准确率"：

* ``decode(encode(x))`` 的**期望**等于 ``x``（无偏），但单次结果偏离 ``O(1/sqrt(T))``；
* ``T`` 越大越准——所以断言的是**误差随 T 下降**，而不是"无损"。

这两条是本插件的存在理由；断言写松了等于没测。
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from biosnn_bus import PassThroughMembrane, SpikeBus, SpikeTrain
from biosnn_bus.plugins import MnistPulsePlugin

HEIGHT, WIDTH = 28, 28
N_PIXELS = HEIGHT * WIDTH


@pytest.fixture
def plugin() -> MnistPulsePlugin:
    return MnistPulsePlugin(height=HEIGHT, width=WIDTH, n_steps=16, seed=0)


class TestParameters:
    @pytest.mark.parametrize(
        ("height", "width", "n_steps"),
        [(0, 28, 16), (28, 0, 16), (28, 28, 0), (-1, 28, 16), (28, 28, 1.5), (True, 28, 16)],
    )
    def test_rejects_bad_dimensions(self, height, width, n_steps):
        with pytest.raises(ValueError):
            MnistPulsePlugin(height=height, width=width, n_steps=n_steps)

    def test_spike_dim_matches_pixels(self, plugin):
        assert plugin.spike_dim == N_PIXELS


class TestEncoding:
    def test_single_frame_expands_to_n_steps(self, plugin):
        """单帧必须按 ``n_steps`` 展开——否则 ``T=1``，``decode`` 无从平均。"""
        train = plugin.encode(np.full((HEIGHT, WIDTH), 0.5))
        assert train.n_steps == plugin.n_steps
        assert train.n_neurons == N_PIXELS

    def test_batch_frames_keep_their_own_length(self, plugin):
        """多帧输入时 ``T`` 就是帧数，``n_steps`` 不参与。"""
        frames = np.zeros((7, HEIGHT, WIDTH))
        assert plugin.encode(frames).n_steps == 7

    def test_zero_pixel_never_fires(self, plugin):
        train = plugin.encode(np.zeros((HEIGHT, WIDTH)))
        assert not np.asarray(train.data).any()

    def test_one_pixel_always_fires(self, plugin):
        train = plugin.encode(np.ones((HEIGHT, WIDTH)))
        assert np.asarray(train.data).all()

    def test_rejects_values_out_of_unit_range(self, plugin):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            plugin.encode(np.full((HEIGHT, WIDTH), 1.5))
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            plugin.encode(np.full((HEIGHT, WIDTH), -0.1))

    def test_rejects_wrong_shape(self, plugin):
        with pytest.raises(ValueError, match="形状"):
            plugin.encode(np.zeros((10, 10)))
        with pytest.raises(ValueError, match="2 维或 3 维"):
            plugin.encode(np.zeros((2, 3, HEIGHT, WIDTH)))

    def test_fixed_seed_is_reproducible(self, plugin):
        img = np.random.default_rng(1).random((HEIGHT, WIDTH))
        first = np.asarray(plugin.encode(img).data)
        second = np.asarray(plugin.encode(img).data)
        assert np.array_equal(first, second)


class TestDecoding:
    def test_decode_returns_image_shaped(self, plugin):
        train = plugin.encode(np.zeros((HEIGHT, WIDTH)))
        assert plugin.decode(train).shape == (HEIGHT, WIDTH)

    def test_decode_is_in_unit_range(self, plugin):
        img = np.random.default_rng(2).random((HEIGHT, WIDTH))
        recon = plugin.decode(plugin.encode(img))
        assert recon.min() >= 0.0
        assert recon.max() <= 1.0

    def test_decode_is_unbiased(self):
        """``decode(encode(x))`` 的**期望**等于 ``x``——本插件的核心性质。

        用 30 个不同种子各采一次，把 30 次解码结果平均；偏差应显著小于单次的
        采样误差。断言的是**无偏**，不是"单次就准"——后者由 ``n_steps`` 控制。
        """
        img = np.random.default_rng(7).random((HEIGHT, WIDTH))
        n_steps = 64
        rates = np.stack(
            [
                MnistPulsePlugin(HEIGHT, WIDTH, n_steps, seed=s).decode(
                    MnistPulsePlugin(HEIGHT, WIDTH, n_steps, seed=s).encode(img)
                )
                for s in range(30)
            ]
        )
        assert rates.shape == (30, HEIGHT, WIDTH)

        single_err = np.abs(rates[0] - img).mean()
        mean_err = np.abs(rates.mean(0) - img).mean()
        assert mean_err < single_err / 2, f"均值 {mean_err:.4f} 未明显小于单次 {single_err:.4f}"

    def test_error_shrinks_as_steps_grow(self):
        """误差随 ``T`` 下降——这是"用时间换精度"的证据。"""
        img = np.random.default_rng(3).random((HEIGHT, WIDTH))
        errors = []
        for n_steps in (4, 16, 64, 256):
            recon = MnistPulsePlugin(HEIGHT, WIDTH, n_steps, seed=0).decode(
                MnistPulsePlugin(HEIGHT, WIDTH, n_steps, seed=0).encode(img)
            )
            errors.append(np.sqrt(((recon - img) ** 2).mean()))
        # 严格递减
        assert all(a > b for a, b in itertools.pairwise(errors)), errors

    def test_decode_rejects_wrong_neuron_count(self, plugin):
        bad = SpikeTrain(data=np.zeros((2, 5)), channel="semantic")
        with pytest.raises(ValueError, match="spike_dim"):
            plugin.decode(bad)

    def test_decode_rejects_non_spike_train(self, plugin):
        with pytest.raises(TypeError, match="SpikeTrain"):
            plugin.decode(np.zeros((1, N_PIXELS)))


class TestContract:
    def test_membrane_is_passthrough(self, plugin):
        assert isinstance(plugin.get_membrane(), PassThroughMembrane)

    def test_channel_is_semantic(self, plugin):
        assert plugin.channel == "semantic"

    def test_registers_under_stable_name(self, plugin):
        bus = SpikeBus(bus_dim=N_PIXELS, seed=0)
        assert bus.register(plugin) == "mnist_pulse"

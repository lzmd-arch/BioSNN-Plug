"""``SpikeTrain`` 与 ``FusionChannel`` 的行为测试。"""

from __future__ import annotations

import importlib.util
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from biosnn_bus import FusionChannel, Membrane, PassThroughMembrane, SpikeTrain


class TestConstruction:
    def test_accepts_binary_and_float(self):
        binary = SpikeTrain(data=np.array([[True, False], [False, True]]))
        assert binary.is_binary
        assert binary.spike_count == 2

        rate = SpikeTrain(data=np.array([[0.5, 0.0], [0.25, 1.0]]))
        assert not rate.is_binary
        np.testing.assert_allclose(rate.rates, [0.375, 0.5])

    def test_coerces_plain_lists(self):
        train = SpikeTrain(data=[[1, 0], [0, 1]])
        assert train.data.shape == (2, 2)

    def test_coerces_string_channel(self):
        assert SpikeTrain(data=np.zeros((2, 2)), channel="semantic").channel is (
            FusionChannel.SEMANTIC
        )

    def test_does_not_alias_caller_array(self):
        source = np.zeros((2, 2), dtype=bool)
        train = SpikeTrain(data=source)
        assert train.data is source  # 只做校验，不复制——零拷贝是刻意的

    @pytest.mark.parametrize(
        ("data", "match"),
        [
            (np.zeros(4), "二维"),
            (np.zeros((0, 4)), r"T 与 N"),
            (np.zeros((4, 0)), r"T 与 N"),
            (np.array([["a", "b"]]), "dtype"),
            (np.array([[np.nan, 0.0]]), "NaN"),
            (np.array([[np.inf, 0.0]]), "NaN"),
        ],
    )
    def test_rejects_bad_data(self, data, match):
        with pytest.raises(ValueError, match=match):
            SpikeTrain(data=data)

    @pytest.mark.parametrize("dt", [0.0, -1.0])
    def test_rejects_nonpositive_dt(self, dt):
        with pytest.raises(ValueError, match="dt 必须为正"):
            SpikeTrain(data=np.zeros((2, 2)), dt=dt)

    @pytest.mark.parametrize("scale", [0.0, -3.0])
    def test_rejects_nonpositive_temporal_scale(self, scale):
        with pytest.raises(ValueError, match="temporal_scale 必须为正"):
            SpikeTrain(data=np.zeros((2, 2)), temporal_scale=scale)

    def test_rejects_unknown_channel(self):
        with pytest.raises(ValueError):
            SpikeTrain(data=np.zeros((2, 2)), channel="auditory")


class TestProperties:
    def test_shape_and_duration(self):
        train = SpikeTrain(data=np.zeros((7, 3)), dt=2.0)
        assert (train.n_steps, train.n_neurons) == (7, 3)
        assert train.duration == 14.0

    def test_density(self):
        train = SpikeTrain(data=np.array([[1, 0], [0, 0]]))
        assert train.density == pytest.approx(0.25)

    def test_frozen(self):
        train = SpikeTrain(data=np.zeros((2, 2)))
        with pytest.raises(FrozenInstanceError):
            train.dt = 5.0  # type: ignore[misc]


class TestEquality:
    def test_value_equality(self):
        a = SpikeTrain(data=np.array([[1, 0]]), dt=2.0)
        b = SpikeTrain(data=np.array([[1, 0]]), dt=2.0)
        assert a == b

    def test_detects_difference(self):
        a = SpikeTrain(data=np.array([[1, 0]]))
        assert a != SpikeTrain(data=np.array([[1, 1]]))
        assert a != SpikeTrain(data=np.array([[1, 0]]), dt=2.0)
        assert a != SpikeTrain(data=np.array([[1, 0]]), channel="semantic")

    def test_comparison_against_other_types_is_not_an_error(self):
        # 回归测试：numpy 数组字段曾让 a == 42 抛 ValueError
        assert SpikeTrain(data=np.zeros((2, 2))) != 42

    def test_repr_is_informative(self):
        text = repr(SpikeTrain(data=np.zeros((3, 5)), dt=2.0))
        assert "(3, 5)" in text and "binary" not in text
        assert "binary" in repr(SpikeTrain(data=np.zeros((3, 5), dtype=bool)))


class TestBinary:
    def test_thresholds_floats(self):
        train = SpikeTrain(data=np.array([[0.9, 0.1]]))
        assert train.binary(threshold=0.5).data.tolist() == [[True, False]]

    def test_identity_on_binary_input(self):
        train = SpikeTrain(data=np.array([[True, False]]))
        assert train.binary() is train


class TestReplace:
    def test_replaces_fields(self):
        train = SpikeTrain(data=np.zeros((2, 2)))
        updated = train.replace(dt=5.0, channel="semantic")
        assert updated.dt == 5.0
        assert updated.channel is FusionChannel.SEMANTIC
        assert train.dt == 1.0  # 原对象不变

    def test_rejects_unknown_field(self):
        with pytest.raises(TypeError, match="没有这些字段"):
            SpikeTrain(data=np.zeros((2, 2))).replace(dt=1.0, n_neurons=9)


class TestResample:
    def test_same_length_is_identity(self):
        train = SpikeTrain(data=np.zeros((4, 2)))
        assert train.resample(4) is train

    @pytest.mark.parametrize("target", [1, 3, 7, 16, 64])
    def test_preserves_duration(self, target):
        train = SpikeTrain(data=np.zeros((8, 2)), dt=2.5)
        assert train.resample(target).duration == pytest.approx(train.duration)

    def test_downsample_keeps_every_spike(self):
        train = SpikeTrain(data=np.ones((8, 2), dtype=bool))
        result = train.resample(3)
        assert result.n_steps == 3
        assert np.all(result.data), "降采样用均值会稀释稀疏脉冲，用最大值才不会丢"

    def test_upsample_does_not_invent_spikes(self):
        rng = np.random.default_rng(0)
        train = SpikeTrain(data=rng.random((5, 3)) > 0.7)
        result = train.resample(20)
        assert result.spike_count == train.spike_count

    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError, match="n_steps 必须"):
            SpikeTrain(data=np.zeros((4, 2))).resample(0)


class TestMembraneProtocol:
    def test_passthrough_satisfies_protocol(self):
        assert isinstance(PassThroughMembrane(), Membrane)

    def test_object_without_forward_does_not(self):
        assert not isinstance(object(), Membrane)


class TestTorchBridge:
    def test_helpful_error_without_torch(self):
        if importlib.util.find_spec("torch") is not None:
            pytest.skip("当前环境已安装 torch")
        with pytest.raises(ImportError, match=r"biosnn-bus\[torch\]"):
            SpikeTrain(data=np.zeros((2, 2))).to_torch()

    def test_roundtrip_when_torch_available(self):
        torch = pytest.importorskip("torch")
        source = np.random.default_rng(0).random((4, 3)) > 0.5
        train = SpikeTrain(data=source, dt=2.0, temporal_scale=5.0, channel="semantic")

        tensor = train.to_torch()
        assert tuple(tensor.shape) == (4, 3)

        restored = SpikeTrain.from_torch(tensor, dt=2.0, temporal_scale=5.0, channel="semantic")
        assert restored == train
        assert isinstance(tensor, torch.Tensor)

    def test_from_torch_detaches_gradients(self):
        torch = pytest.importorskip("torch")
        leaf = torch.zeros((3, 2), dtype=torch.float32, requires_grad=True)
        restored = SpikeTrain.from_torch(leaf * 1.0)
        assert not restored.data.flags.writeable or restored.data.base is None

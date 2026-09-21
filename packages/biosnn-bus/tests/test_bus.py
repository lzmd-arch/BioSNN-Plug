"""``SpikeBus`` 行为测试。

总线是骨架库与认知核心之间的唯一接口，因此这里覆盖的重点是：契约违规能否在入口
处被拦下、双通道路由是否正确、以及**同一份配置是否一定产生同一份输出**（可复现性，
见 ``docs/reproducibility.md``）。
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from biosnn_bus import (
    BioSNNBusError,
    ConcatenateFusion,
    ModalityPlugin,
    PassThroughMembrane,
    PluginContractError,
    SparseRandomProjection,
    SpikeBus,
    SpikeTrain,
    UnknownModalityError,
)
from helpers import ConstantPlugin, make_pattern


class TestConstruction:
    @pytest.mark.parametrize("dim", [0, -5, 2.5, "128", None, True])
    def test_rejects_bad_bus_dim(self, dim):
        with pytest.raises(ValueError, match="bus_dim"):
            SpikeBus(bus_dim=dim)

    @pytest.mark.parametrize("dt", [0, -1.0])
    def test_rejects_bad_dt(self, dt):
        with pytest.raises(ValueError, match="dt 必须为正"):
            SpikeBus(bus_dim=64, dt=dt)

    def test_defaults(self):
        bus = SpikeBus(bus_dim=64)
        assert isinstance(bus.strategy, ConcatenateFusion)
        assert bus.registered_modalities == ()
        assert "SpikeBus(bus_dim=64" in repr(bus)


class TestRegistration:
    def test_register_returns_the_name(self):
        bus = SpikeBus(bus_dim=32)
        assert bus.register(ConstantPlugin(modality="a")) == "a"
        assert bus.registered_modalities == ("a",)

    def test_register_validates_the_contract(self):
        bus = SpikeBus(bus_dim=32)

        class BadDim(ConstantPlugin):
            @property
            def spike_dim(self) -> int:
                return 0

        with pytest.raises(PluginContractError, match="spike_dim"):
            bus.register(BadDim(modality="bad"))

    def test_rejects_non_plugin(self):
        with pytest.raises(PluginContractError, match="ModalityPlugin 的实例"):
            SpikeBus(bus_dim=32).register(object())

    def test_rejects_duplicate_name(self):
        bus = SpikeBus(bus_dim=32)
        bus.register(ConstantPlugin(modality="a"))
        with pytest.raises(PluginContractError, match="已被占用"):
            bus.register(ConstantPlugin(modality="a"))

    def test_explicit_name_allows_two_configs_of_one_modality(self):
        bus = SpikeBus(bus_dim=32)
        bus.register(ConstantPlugin(modality="img", dim=4), name="img_low")
        bus.register(ConstantPlugin(modality="img", dim=16), name="img_high")
        assert bus.registered_modalities == ("img_low", "img_high")

    @pytest.mark.parametrize("name", ["", "  "])
    def test_rejects_bad_explicit_name(self, name):
        bus = SpikeBus(bus_dim=32)
        with pytest.raises(PluginContractError, match="注册名"):
            bus.register(ConstantPlugin(), name=name)

    def test_none_name_falls_back_to_the_plugin(self):
        bus = SpikeBus(bus_dim=32)
        assert bus.register(ConstantPlugin(modality="auto"), name=None) == "auto"

    def test_unregister(self):
        bus = SpikeBus(bus_dim=32)
        bus.register(ConstantPlugin(modality="a"))
        bus.unregister("a")
        assert bus.registered_modalities == ()
        with pytest.raises(UnknownModalityError):
            bus.unregister("a")


class TestStepContract:
    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="至少需要一路输入"):
            SpikeBus(bus_dim=32).step({})

    def test_unknown_modality_names_what_is_registered(self):
        bus = SpikeBus(bus_dim=32)
        bus.register(ConstantPlugin(modality="known"))
        with pytest.raises(UnknownModalityError) as excinfo:
            bus.step({"unknown": make_pattern()})
        message = str(excinfo.value)
        assert "unknown" in message and "known" in message
        assert not message.startswith("'"), "KeyError.__str__ 会加引号，这里要可读信息"

    def test_encode_returning_wrong_type_is_caught(self):
        class BadReturn(ConstantPlugin):
            def encode(self, raw_input):
                return np.zeros((4, 4))  # 忘了包成 SpikeTrain

        bus = SpikeBus(bus_dim=32)
        bus.register(BadReturn(modality="bad"))
        with pytest.raises(PluginContractError, match="必须是 SpikeTrain"):
            bus.step({"bad": make_pattern()})

    def test_encode_returning_wrong_dim_is_caught(self):
        class WrongDim(ConstantPlugin):
            def encode(self, raw_input):
                return SpikeTrain(data=np.zeros((4, 99)))

        bus = SpikeBus(bus_dim=32)
        bus.register(WrongDim(modality="bad", dim=4))
        with pytest.raises(PluginContractError, match="声明 spike_dim=4"):
            bus.step({"bad": make_pattern()})


class TestFusion:
    def test_output_shape(self):
        bus = SpikeBus(bus_dim=64)
        bus.register(ConstantPlugin(modality="a", dim=4, steps=8))
        out = bus.step({"a": make_pattern(8, 4)})
        assert out.data.shape == (8, 64)

    def test_output_is_float_current_not_binary(self):
        bus = SpikeBus(bus_dim=64)
        bus.register(ConstantPlugin(modality="a", dim=4))
        out = bus.step({"a": make_pattern()})
        assert not out.is_binary, "融合后的输出是输入电流，不是 0/1 脉冲"
        assert out.binary().is_binary  # 需要脉冲时显式二值化

    def test_both_channels_are_routed(self):
        bus = SpikeBus(bus_dim=64)
        bus.register(ConstantPlugin(modality="sem", channel="semantic"))
        bus.register(ConstantPlugin(modality="tmp", channel="temporal"))
        by_channel = bus.describe()["modalities_by_channel"]
        assert by_channel == {"temporal": ["tmp"], "semantic": ["sem"]}

        both = bus.step({"sem": make_pattern(), "tmp": make_pattern()})
        assert both.data.shape == (8, 64)

    def test_single_channel_output_differs_from_dual_channel(self):
        pattern = make_pattern()
        single = SpikeBus(bus_dim=64, seed=0)
        single.register(ConstantPlugin(modality="sem", channel="semantic"))

        dual = SpikeBus(bus_dim=64, seed=0)
        dual.register(ConstantPlugin(modality="sem", channel="semantic"))
        dual.register(ConstantPlugin(modality="tmp", channel="temporal"))

        only_sem = single.step({"sem": pattern})
        sem_and_tmp = dual.step({"sem": pattern, "tmp": pattern})
        assert not np.allclose(only_sem.data, sem_and_tmp.data)

    def test_aligns_modalities_of_different_duration(self):
        bus = SpikeBus(bus_dim=32, dt=1.0)
        bus.register(ConstantPlugin(modality="short", steps=4, dt=1.0))  # 4ms
        bus.register(ConstantPlugin(modality="long", steps=9, dt=2.0))  # 18ms
        out = bus.step({"short": make_pattern(4, 4), "long": make_pattern(9, 4)})
        assert out.n_steps == 18
        assert out.dt == 1.0
        assert out.duration == pytest.approx(18.0)

    def test_bus_dt_controls_output_resolution(self):
        bus = SpikeBus(bus_dim=32, dt=3.0)
        bus.register(ConstantPlugin(modality="a", steps=9, dt=1.0))  # 9ms
        out = bus.step({"a": make_pattern(9, 4)})
        assert out.n_steps == 3
        assert out.duration == pytest.approx(9.0)

    def test_temporal_scale_of_slowest_modality_wins(self):
        bus = SpikeBus(bus_dim=32)
        bus.register(ConstantPlugin(modality="fast", temporal_scale=1.0))
        bus.register(ConstantPlugin(modality="slow", temporal_scale=100.0))
        out = bus.step({"fast": make_pattern(), "slow": make_pattern()})
        assert out.temporal_scale == 100.0

    def test_uses_a_custom_fusion_strategy(self):
        class SumFusion:
            def __init__(self):
                self.calls = 0

            def fuse(self, trains):
                self.calls += 1
                data = np.sum([t.data.astype(np.float64) for t in trains], axis=0)
                return trains[0].replace(data=data)

        strategy = SumFusion()
        bus = SpikeBus(bus_dim=32, strategy=strategy)
        bus.register(ConstantPlugin(modality="a"))
        bus.step({"a": make_pattern()})
        assert strategy.calls == 1, "TAAF 将来就是从这个位置插进来"

    def test_fusion_rejects_mixed_channels(self):
        a = SpikeTrain(data=np.zeros((2, 2)), channel="temporal")
        b = SpikeTrain(data=np.zeros((2, 2)), channel="semantic")
        with pytest.raises(BioSNNBusError, match="按通道分组"):
            ConcatenateFusion().fuse([a, b])

    def test_fusion_rejects_empty(self):
        with pytest.raises(BioSNNBusError, match="至少需要一路"):
            ConcatenateFusion().fuse([])


class TestReproducibility:
    def _run(self, seed: int, order: tuple[str, ...]) -> np.ndarray:
        bus = SpikeBus(bus_dim=64, seed=seed)
        bus.register(ConstantPlugin(modality="a", dim=4))
        bus.register(ConstantPlugin(modality="b", dim=4))
        inputs = {"a": make_pattern(8, 4, seed=1), "b": make_pattern(8, 4, seed=2)}
        return bus.step({name: inputs[name] for name in order}).data

    def test_same_seed_same_output(self):
        np.testing.assert_array_equal(self._run(0, ("a", "b")), self._run(0, ("a", "b")))

    def test_different_seed_different_output(self):
        assert not np.allclose(self._run(0, ("a", "b")), self._run(1, ("a", "b")))

    def test_output_is_independent_of_input_dict_order(self):
        """回归测试：融合按神经元维拼接，列序决定投影矩阵的含义。

        如果分组顺序跟着输入字典走，同一组输入换个传入顺序就会得到不同的总线输出，
        "固定种子即可复现"这句话就不成立了。
        """
        np.testing.assert_array_equal(self._run(0, ("a", "b")), self._run(0, ("b", "a")))

    def test_repeated_steps_reuse_the_projection(self):
        bus = SpikeBus(bus_dim=64, seed=0)
        bus.register(ConstantPlugin(modality="a"))
        pattern = make_pattern()
        np.testing.assert_array_equal(bus.step({"a": pattern}).data, bus.step({"a": pattern}).data)

    def test_projection_seed_survives_a_process_boundary(self):
        """回归测试：内置 ``hash()`` 对 str 按进程加盐（PYTHONHASHSEED）。

        如果投影矩阵的种子是从字符串 ``hash()`` 派生的，"固定种子即可复现"就只在
        单进程内成立——同一份配置换台机器、换个进程就会得到不同的总线输出。这里
        用两个不同 ``PYTHONHASHSEED`` 的子进程直接验证跨进程一致性。
        """
        script = textwrap.dedent(
            """
            import numpy as np
            from biosnn_bus import SpikeBus
            from helpers import ConstantPlugin, make_pattern

            bus = SpikeBus(bus_dim=64, seed=0)
            bus.register(ConstantPlugin(modality="a", dim=4))
            bus.register(ConstantPlugin(modality="b", dim=4))
            out = bus.step({
                "a": make_pattern(8, 4, seed=1),
                "b": make_pattern(8, 4, seed=2),
            })
            weights = np.arange(out.data.size, dtype=np.float64).reshape(out.data.shape)
            print(repr(float(np.sum(out.data * weights))))
            """
        )

        def run(hash_seed: str) -> str:
            env = {**os.environ, "PYTHONHASHSEED": hash_seed}
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=True,
                env=env,
                cwd=Path(__file__).parent,
            )
            return result.stdout.strip()

        assert run("0") == run("31337")
        assert run("0") != "0.0", "输出不应恒为零——那样这条测试就失去了意义"


class TestProjection:
    def test_shape_and_rejects_bad_input(self):
        projection = SparseRandomProjection(10, 4, density=0.5, seed=0)
        assert projection(np.zeros((3, 10))).shape == (3, 4)
        with pytest.raises(ValueError, match="投影期望形状"):
            projection(np.zeros((3, 11)))

    @pytest.mark.parametrize(
        ("in_dim", "out_dim", "density"), [(0, 4, 0.1), (4, 0, 0.1), (4, 4, 0.0), (4, 4, 1.5)]
    )
    def test_rejects_bad_configuration(self, in_dim, out_dim, density):
        with pytest.raises(ValueError):
            SparseRandomProjection(in_dim, out_dim, density=density)

    def test_is_deterministic(self):
        a = SparseRandomProjection(20, 8, seed=7)
        b = SparseRandomProjection(20, 8, seed=7)
        data = np.random.default_rng(0).random((5, 20))
        np.testing.assert_array_equal(a(data), b(data))

    def test_approximately_preserves_pairwise_distances(self):
        """±1 稀疏随机投影的用处就在于此：能被区分开的模式，投影后仍能被区分开。"""
        rng = np.random.default_rng(0)
        x = rng.normal(size=(30, 512))
        projection = SparseRandomProjection(512, 512, density=0.1, seed=0)
        projected = projection(x)

        def pairwise(a: np.ndarray) -> np.ndarray:
            return np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1).ravel()

        correlation = np.corrcoef(pairwise(x), pairwise(projected))[0, 1]
        assert correlation > 0.9, f"保距离性质太差（相关系数 {correlation:.3f}）"

    def test_every_input_dimension_gets_connections(self):
        """回归测试：在整个矩阵上均匀撒点会随机丢空部分输入维度。

        被丢空的输入神经元对认知核心完全不可见，而哪个神经元中招取决于种子——
        这类 bug 靠肉眼看权重矩阵是发现不了的。
        """
        projection = SparseRandomProjection(500, 64, density=0.05, seed=0)
        assert np.unique(projection._cols).size == 500
        assert projection.fan_out_per_input >= 1

    def test_sparsity_is_actually_sparse(self):
        projection = SparseRandomProjection(1024, 256, density=0.1, seed=0)
        dense_equivalent = 1024 * 256
        assert projection.n_connections < 0.2 * dense_equivalent


class TestIntrospection:
    def test_describe_reports_the_full_configuration(self):
        bus = SpikeBus(bus_dim=64, seed=3, dt=2.0, projection_density=0.2)
        bus.register(ConstantPlugin(modality="a", dim=4, channel="semantic", temporal_scale=5.0))
        info = bus.describe()

        assert info["bus_dim"] == 64
        assert info["dt"] == 2.0
        assert info["seed"] == 3
        assert info["projection_density"] == 0.2
        assert info["strategy"] == "ConcatenateFusion"
        assert info["modalities"]["a"] == {
            "plugin": "ConstantPlugin",
            "spike_dim": 4,
            "channel": "semantic",
            "temporal_scale": 5.0,
        }
        assert info["modalities_by_channel"] == {"temporal": [], "semantic": ["a"]}


class TestMembraneAccess:
    def test_plugin_membrane_is_reachable_through_the_bus(self):
        bus = SpikeBus(bus_dim=32)
        plugin = ConstantPlugin(modality="a")
        bus.register(plugin)
        assert isinstance(plugin.get_membrane(), PassThroughMembrane)
        assert isinstance(plugin, ModalityPlugin)

"""``ModalityPlugin`` 接口契约测试。

计划书 §2.3 规定了插件的接口。这里逐条验证：漏实现方法、声明非法维度、返回非法
通道等错误都能在**注册时**就被拦下，而不是等到训练跑了一半才炸。
"""

from __future__ import annotations

import numpy as np
import pytest
from biosnn_bus import (
    FusionChannel,
    ModalityPlugin,
    PassThroughMembrane,
    PluginContractError,
    SpikeTrain,
)


class BasePlugin(ModalityPlugin):
    """测试用的基类：契约完全合法，子类按需覆盖单个字段。"""

    @property
    def modality_name(self) -> str:
        return "base"

    @property
    def spike_dim(self) -> int:
        return 4

    def encode(self, raw_input) -> SpikeTrain:
        return SpikeTrain(data=np.asarray(raw_input), channel=self.channel)

    def get_membrane(self) -> PassThroughMembrane:
        return PassThroughMembrane()

    def decode(self, spike_output: SpikeTrain):
        return spike_output.rates


def make_plugin(**overrides) -> ModalityPlugin:
    """新建一个 ``BasePlugin`` 子类实例，属性按 ``overrides`` 覆盖。

    每个测试都用全新的子类，避免用 ``type(plugin).field = property(...)`` 直接改写
    共享类——那会把抽象属性从基类上删掉，污染同一进程里后续所有测试。
    """
    body = {name: property(lambda self, value=value: value) for name, value in overrides.items()}
    return type("Sub", (BasePlugin,), body)()


class TestAbstractness:
    @pytest.mark.parametrize("missing", ["encode", "get_membrane", "decode"])
    def test_cannot_instantiate_without_methods(self, missing):
        body = {
            "modality_name": property(lambda self: "x"),
            "spike_dim": property(lambda self: 4),
            "encode": lambda self, raw: SpikeTrain(data=np.zeros((1, 4))),
            "get_membrane": lambda self: PassThroughMembrane(),
            "decode": lambda self, out: None,
        }
        del body[missing]
        incomplete = type(f"Incomplete_{missing}", (ModalityPlugin,), body)
        with pytest.raises(TypeError, match="abstract"):
            incomplete()

    def test_cannot_instantiate_without_abstract_properties(self):
        class NoName(ModalityPlugin):
            @property
            def spike_dim(self) -> int:
                return 1

            def encode(self, raw_input):
                return SpikeTrain(data=np.zeros((1, 1)))

            def get_membrane(self):
                return PassThroughMembrane()

            def decode(self, spike_output):
                return None

        with pytest.raises(TypeError, match="modality_name"):
            NoName()


class TestDefaults:
    def test_temporal_scale_default(self):
        assert BasePlugin().temporal_scale == 10.0

    def test_fusion_channel_default(self):
        assert BasePlugin().fusion_channel == "temporal"

    def test_channel_property_coerces(self):
        assert BasePlugin().channel is FusionChannel.TEMPORAL


class TestValidate:
    def test_accepts_good_plugin(self):
        BasePlugin().validate()

    @pytest.mark.parametrize("name", ["", "   ", None, 42])
    def test_rejects_bad_modality_name(self, name):
        with pytest.raises(PluginContractError, match="modality_name"):
            make_plugin(modality_name=name).validate()

    @pytest.mark.parametrize("dim", [0, -1, 1.5, "4", None, True])
    def test_rejects_bad_spike_dim(self, dim):
        with pytest.raises(PluginContractError, match="spike_dim"):
            make_plugin(spike_dim=dim).validate()

    @pytest.mark.parametrize("scale", [0, -1.0, "fast", None])
    def test_rejects_bad_temporal_scale(self, scale):
        with pytest.raises(PluginContractError, match="temporal_scale"):
            make_plugin(temporal_scale=scale).validate()

    def test_rejects_unknown_fusion_channel(self):
        with pytest.raises(PluginContractError, match="fusion_channel"):
            make_plugin(fusion_channel="auditory").validate()

    def test_error_message_names_the_offending_class(self):
        with pytest.raises(PluginContractError, match="Sub"):
            make_plugin(spike_dim=-3).validate()


class TestRepr:
    def test_shows_contract(self):
        text = repr(make_plugin(modality_name="good", spike_dim=4))
        assert "Sub" in text and "good" in text and "4" in text

    def test_survives_broken_contract(self):
        class Broken(ModalityPlugin):
            @property
            def modality_name(self):
                raise RuntimeError("boom")

            @property
            def spike_dim(self):
                raise RuntimeError("boom")

            def encode(self, raw_input):
                raise NotImplementedError

            def get_membrane(self):
                raise NotImplementedError

            def decode(self, spike_output):
                raise NotImplementedError

        # __repr__ 不应该再抛异常——最需要它的时刻正是对象已经坏掉的时候
        assert "契约未满足" in repr(Broken())

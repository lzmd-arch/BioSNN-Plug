"""注册与发现机制测试。

计划书 §1.2 把"新增模态无需修改已有架构"列为核心能力之一。它能否成立，全看这里的
两条通道：进程内装饰器注册，以及第三方分发包的 entry points 发现。
"""

from __future__ import annotations

import numpy as np
import pytest
from biosnn_bus import (
    ModalityPlugin,
    PassThroughMembrane,
    PluginNotFoundError,
    PluginRegistrationError,
    SpikeTrain,
    discover_plugins,
    get_plugin,
    list_plugins,
    register_plugin,
    unregister_plugin,
)


def _make_plugin_class(name: str) -> type[ModalityPlugin]:
    class Anonymous(ModalityPlugin):
        @property
        def modality_name(self) -> str:
            return name

        @property
        def spike_dim(self) -> int:
            return 2

        def encode(self, raw_input) -> SpikeTrain:
            return SpikeTrain(data=np.zeros((1, 2)))

        def get_membrane(self) -> PassThroughMembrane:
            return PassThroughMembrane()

        def decode(self, spike_output: SpikeTrain):
            return None

    return Anonymous


class TestDecoratorRegistration:
    def test_registers_and_returns_the_class_unchanged(self):
        @register_plugin("alpha")
        class Alpha(ModalityPlugin):
            @property
            def modality_name(self) -> str:
                return "alpha"

            @property
            def spike_dim(self) -> int:
                return 2

            def encode(self, raw_input) -> SpikeTrain:
                return SpikeTrain(data=np.zeros((1, 2)))

            def get_membrane(self):
                return PassThroughMembrane()

            def decode(self, spike_output):
                return None

        assert get_plugin("alpha") is Alpha
        assert Alpha().modality_name == "alpha"  # 原类可直接实例化

    def test_rejects_non_plugin(self):
        with pytest.raises(PluginRegistrationError, match="ModalityPlugin 的子类"):

            @register_plugin("nope")
            class NotAPlugin:
                pass

    def test_rejects_instance(self):
        with pytest.raises(PluginRegistrationError, match="ModalityPlugin 的子类"):
            register_plugin("nope")(object())

    @pytest.mark.parametrize("name", ["", "   ", "has space", " padded", None, 7])
    def test_rejects_bad_names(self, name):
        with pytest.raises(PluginRegistrationError):
            register_plugin(name)(_make_plugin_class("x"))

    def test_duplicate_name_conflicts(self):
        register_plugin("dup")(_make_plugin_class("one"))
        with pytest.raises(PluginRegistrationError, match="已被"):
            register_plugin("dup")(_make_plugin_class("two"))

    def test_override_allows_replacement(self):
        first = register_plugin("dup")(_make_plugin_class("one"))
        second = register_plugin("dup", override=True)(_make_plugin_class("two"))
        assert get_plugin("dup") is second is not first

    def test_reregistering_the_same_class_is_idempotent(self):
        cls = _make_plugin_class("same")
        register_plugin("same")(cls)
        register_plugin("same")(cls)  # 不应抛错
        assert get_plugin("same") is cls


class TestLookup:
    def test_unknown_name_lists_what_is_registered(self):
        register_plugin("known")(_make_plugin_class("known"))
        with pytest.raises(PluginNotFoundError) as excinfo:
            get_plugin("missing")
        message = str(excinfo.value)
        assert "missing" in message and "known" in message
        # KeyError.__str__ 会加引号，这条断言防止回归
        assert not message.startswith("'")

    def test_list_returns_a_copy(self):
        register_plugin("only")(_make_plugin_class("only"))
        listing = list_plugins()
        listing.clear()
        assert "only" in list_plugins(), "list_plugins 必须是浅拷贝"

    def test_unregister(self):
        register_plugin("temp")(_make_plugin_class("temp"))
        unregister_plugin("temp")
        assert "temp" not in list_plugins()
        with pytest.raises(PluginNotFoundError):
            unregister_plugin("temp")


class TestEntryPointDiscovery:
    """用伪造的 entry point 验证第三方发现路径，无需真的安装一个包。"""

    @staticmethod
    def _fake_entry_point(name: str, value: str, obj):
        class FakeEntryPoint:
            def __init__(self):
                self.name = name
                self.value = value

            def load(self):
                return obj

        return FakeEntryPoint()

    def test_registers_valid_plugin(self, monkeypatch):
        plugin_cls = _make_plugin_class("third_party")

        def fake_entry_points(*, group):
            assert group == "biosnn_bus.plugins"
            return [self._fake_entry_point("audio", "pkg:AudioPlugin", plugin_cls)]

        monkeypatch.setattr("importlib.metadata.entry_points", fake_entry_points)

        found = discover_plugins()
        assert found == {"audio": plugin_cls}
        assert get_plugin("audio") is plugin_cls

    def test_is_idempotent(self, monkeypatch):
        plugin_cls = _make_plugin_class("third_party")
        monkeypatch.setattr(
            "importlib.metadata.entry_points",
            lambda *, group: [self._fake_entry_point("audio", "pkg:AudioPlugin", plugin_cls)],
        )
        assert discover_plugins() == {"audio": plugin_cls}
        assert discover_plugins() == {}, "第二次调用不应重复注册"

    def test_rejects_entry_point_that_is_not_a_plugin(self, monkeypatch):
        monkeypatch.setattr(
            "importlib.metadata.entry_points",
            lambda *, group: [self._fake_entry_point("bad", "pkg:NotAPlugin", object)],
        )
        with pytest.raises(PluginRegistrationError, match="不是 ModalityPlugin 的子类"):
            discover_plugins()

    def test_reports_import_failures_with_context(self, monkeypatch):
        class Exploding:
            def load(self):
                raise ImportError("missing native library")

            name = "broken"
            value = "pkg:Broken"

        monkeypatch.setattr("importlib.metadata.entry_points", lambda *, group: [Exploding()])
        with pytest.raises(PluginRegistrationError, match=r"broken.*missing native library"):
            discover_plugins()

    def test_does_not_clobber_existing_registration(self, monkeypatch):
        mine = register_plugin("audio")(_make_plugin_class("mine"))
        monkeypatch.setattr(
            "importlib.metadata.entry_points",
            lambda *, group: [
                self._fake_entry_point("audio", "pkg:Audio", _make_plugin_class("theirs"))
            ],
        )
        assert discover_plugins() == {}
        assert get_plugin("audio") is mine

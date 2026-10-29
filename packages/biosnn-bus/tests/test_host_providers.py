"""宿主三原语与 provider 三后端的守卫测试（不联网）。"""

from __future__ import annotations

import json

import numpy as np
import pytest
from biosnn_bus import (
    BioSNNHost,
    GeminiProvider,
    ModalityPlugin,
    OpenAIProvider,
    PassThroughMembrane,
    ProviderError,
    SpikeBus,
    SpikeTrain,
    TriggerEvent,
    VLLMProvider,
)


class _TinyPlugin(ModalityPlugin):
    @property
    def modality_name(self) -> str:
        return "tiny"

    @property
    def spike_dim(self) -> int:
        return 8

    def encode(self, raw_input):
        return SpikeTrain(data=np.asarray(raw_input) > 0.5, channel=self.channel)

    def get_membrane(self):
        return PassThroughMembrane()

    def decode(self, spike_output):
        return spike_output.rates


def _host(trigger=None) -> BioSNNHost:
    bus = SpikeBus(bus_dim=16, seed=0)
    bus.register(_TinyPlugin())
    return BioSNNHost(bus, trigger=trigger)


def test_inject_and_read_state():
    host = _host()
    host.inject("tiny", np.zeros((4, 8), dtype=np.float32))
    state = host.read_state()
    assert state["dim"] == 16
    assert state["steps"] == 1
    assert state["armed"] is False
    assert state["modalities"] == ["tiny"]


def test_unarmed_wait_trigger_returns_none():
    host = _host()
    assert host.wait_trigger(timeout_ms=0) is None


def test_armed_wait_trigger_returns_event():
    host = _host(trigger=lambda: TriggerEvent(t=3, salience=0.9, associations=("a",)))
    event = host.wait_trigger(timeout_ms=10)
    assert event is not None and event.t == 3


def test_wait_trigger_timeout():
    host = _host(trigger=lambda: None)
    assert host.wait_trigger(timeout_ms=1) is None


def test_rejects_bad_inputs():
    host = _host()
    with pytest.raises(ValueError):
        host.inject("tiny", np.zeros((4, 8)), t=-1)
    with pytest.raises(ValueError):
        host.wait_trigger(timeout_ms=-1)


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _opener(payload: dict, seen: dict):
    def opener(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data.decode("utf-8"))
        seen["headers"] = dict(request.headers)
        return _Response(payload)

    return opener


def test_openai_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen = {}
    provider = OpenAIProvider(opener=_opener({"choices": [{"message": {"content": "hi"}}]}, seen))
    assert provider.complete("hello", context="ctx") == "hi"
    assert seen["url"].endswith("/chat/completions")
    assert seen["body"]["messages"][0]["role"] == "system"
    assert seen["headers"]["Authorization"] == "Bearer sk-test"


def test_gemini_provider(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    seen = {}
    payload = {"candidates": [{"content": {"parts": [{"text": "yo"}]}}]}
    provider = GeminiProvider(opener=_opener(payload, seen))
    assert provider.complete("hello") == "yo"
    assert ":generateContent" in seen["url"]


def test_vllm_provider_needs_no_key():
    seen = {}
    provider = VLLMProvider(
        "local-model", opener=_opener({"choices": [{"message": {"content": "x"}}]}, seen)
    )
    assert provider.complete("hi") == "x"


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderError):
        OpenAIProvider(opener=_opener({}, {})).complete("hi")


def test_bad_payload_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with pytest.raises(ProviderError):
        OpenAIProvider(opener=_opener({}, {})).complete("hi")

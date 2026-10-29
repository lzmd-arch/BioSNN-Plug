"""Ollama adapter 的无网络测试。"""

from __future__ import annotations

import json

from biosnn_mcp import OllamaLLM


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_posts_prompt_and_returns_response():
    seen = {}

    def opener(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data.decode("utf-8"))
        return _Response({"response": "hello from ollama"})

    llm = OllamaLLM(opener=opener)
    assert llm.generate("hi", context="ctx") == "hello from ollama"
    assert seen["url"].endswith("/api/generate")
    assert seen["body"]["stream"] is False
    assert "ctx" in seen["body"]["prompt"]


def test_missing_response_is_error():
    llm = OllamaLLM(opener=lambda request, timeout: _Response({}))
    try:
        llm.generate("hi")
    except RuntimeError:
        return
    raise AssertionError("缺少 response 应报错")


def test_think_false_is_sent_only_when_requested():
    seen = {}

    def opener(request, timeout):
        seen["body"] = json.loads(request.data.decode("utf-8"))
        return _Response({"response": "ok"})

    OllamaLLM(opener=opener).generate("hi")
    assert "think" not in seen["body"]
    OllamaLLM(think=False, opener=opener).generate("hi")
    assert seen["body"]["think"] is False


def test_thinking_block_is_stripped():
    llm = OllamaLLM(
        opener=lambda request, timeout: _Response({"response": "<think>hmm</think>paris"})
    )
    assert llm.generate("hi") == "paris"

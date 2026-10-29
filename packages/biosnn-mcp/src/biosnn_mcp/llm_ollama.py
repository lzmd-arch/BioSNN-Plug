"""本地 Ollama LLM adapter：stdlib HTTP，无 key；只作外部推理引擎。"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable

__all__ = ["OllamaLLM"]


def _strip_thinking(text: str) -> str:
    """去掉 ``<think>…</think>`` 块：思考文字会让关键短语判定误命中，也占上下文。"""
    while "<think>" in text and "</think>" in text:
        start = text.index("<think>")
        end = text.index("</think>", start) + len("</think>")
        text = text[:start] + text[end:]
    return text.strip()


class OllamaLLM:
    """调用本地 Ollama `/api/generate`；不进入任何学习路径。"""

    name = "ollama"

    def __init__(
        self,
        model: str = "qwen3:4b",
        *,
        host: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
        think: bool | None = None,
        opener: Callable | None = None,
    ) -> None:
        self.model = model
        # think=False 让 qwen3 这类思考模型不再把 ``<think>`` 写进 response：既快，也避免
        # 判定时在推理文字里误命中关键短语。None = 不发这个字段（老服务端的兼容默认）。
        self.think = think
        self.host = host.rstrip("/")
        self.timeout = float(timeout)
        self._open = opener or urllib.request.urlopen

    def generate(self, prompt: str, *, context: str = "") -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt 必须是非空字符串。")
        payload = {
            "model": self.model,
            "prompt": prompt if not context else f"{context}\n\n{prompt}",
            "stream": False,
        }
        if self.think is not None:
            payload["think"] = bool(self.think)
        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._open(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        answer = body.get("response")
        if not isinstance(answer, str):
            raise RuntimeError(f"Ollama 返回缺少 response：{body!r}")
        return _strip_thinking(answer)

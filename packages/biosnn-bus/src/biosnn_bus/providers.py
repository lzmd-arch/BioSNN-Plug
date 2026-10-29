"""LLM provider 抽象：OpenAI / Gemini / vLLM 三个后端（宿主可选）。

宿主把 LLM 当**外部推理引擎**：本模块只做 HTTP 适配，不参与任何学习路径，
不把 key 写进日志或仓库。所有后端都接受可注入的 ``opener``，便于无网络测试。

约定：
* key 从环境变量读（``OPENAI_API_KEY`` / ``GEMINI_API_KEY`` / ``VLLM_API_KEY``）；
* vLLM 是本机服务，默认走**无代理** opener（系统代理会把 loopback 送进代理）；
* 返回纯文本；HTTP/解析错误抛 :class:`ProviderError`。
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Protocol

__all__ = [
    "GeminiProvider",
    "LLMProvider",
    "OpenAIProvider",
    "ProviderError",
    "VLLMProvider",
]


class ProviderError(RuntimeError):
    """provider 调用失败（网络、鉴权或响应格式）。"""


class LLMProvider(Protocol):
    """一个只做「文本进、文本出」的推理引擎。"""

    name: str

    def complete(self, prompt: str, *, context: str = "") -> str: ...


def _default_opener(disable_proxy: bool = False) -> Callable:
    if disable_proxy:
        return urllib.request.build_opener(urllib.request.ProxyHandler({})).open
    return urllib.request.urlopen


def _post_json(opener: Callable, url: str, payload: dict, headers: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with opener(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except OSError as error:
        raise ProviderError(f"请求失败：{error}") from error
    except json.JSONDecodeError as error:
        raise ProviderError(f"响应不是 JSON：{error}") from error


class OpenAIProvider:
    """OpenAI Chat Completions 后端（也兼容任何 OpenAI-API 服务）。"""

    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 60.0,
        opener: Callable | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout = float(timeout)
        self._opener = opener or _default_opener()

    def _key(self) -> str:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise ProviderError(f"缺少环境变量 {self.api_key_env}。")
        return key

    def complete(self, prompt: str, *, context: str = "") -> str:
        if not prompt.strip():
            raise ValueError("prompt 不能为空。")
        messages = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": prompt})
        body = _post_json(
            self._opener,
            f"{self.base_url}/chat/completions",
            {"model": self.model, "messages": messages, "temperature": 0},
            {"Authorization": f"Bearer {self._key()}"},
            self.timeout,
        )
        try:
            return str(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as error:
            raise ProviderError(f"OpenAI 响应缺少 choices[0].message.content：{body!r}") from error


class GeminiProvider:
    """Google Gemini generateContent 后端。"""

    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        *,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        api_key_env: str = "GEMINI_API_KEY",
        timeout: float = 60.0,
        opener: Callable | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout = float(timeout)
        self._opener = opener or _default_opener()

    def _key(self) -> str:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise ProviderError(f"缺少环境变量 {self.api_key_env}。")
        return key

    def complete(self, prompt: str, *, context: str = "") -> str:
        if not prompt.strip():
            raise ValueError("prompt 不能为空。")
        text = f"{context}\n\n{prompt}" if context else prompt
        body = _post_json(
            self._opener,
            f"{self.base_url}/models/{self.model}:generateContent?key={urllib.parse.quote(self._key())}",
            {"contents": [{"parts": [{"text": text}]}], "generationConfig": {"temperature": 0}},
            {},
            self.timeout,
        )
        try:
            return str(body["candidates"][0]["content"]["parts"][0]["text"])
        except (KeyError, IndexError, TypeError) as error:
            raise ProviderError(
                f"Gemini 响应缺少 candidates[0].content.parts[0].text：{body!r}"
            ) from error


class VLLMProvider(OpenAIProvider):
    """本机 vLLM（OpenAI 兼容 API）；默认无代理、key 可选。"""

    name = "vllm"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key_env: str = "VLLM_API_KEY",
        timeout: float = 60.0,
        opener: Callable | None = None,
    ) -> None:
        super().__init__(
            model,
            base_url=base_url,
            api_key_env=api_key_env,
            timeout=timeout,
            opener=opener or _default_opener(disable_proxy=True),
        )

    def _key(self) -> str:
        # vLLM 默认不鉴权：没有 key 时给一个占位，避免本地部署被强制配环境变量。
        return os.environ.get(self.api_key_env, "").strip() or "local"

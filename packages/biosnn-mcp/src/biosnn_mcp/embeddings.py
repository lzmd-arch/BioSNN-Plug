"""阶段五 M3e：本地嵌入适配器（stdlib HTTP，只连本机 Ollama）。

只做一件事：把文本批量送进本机 Ollama 的 `/api/embed`，拿回稠密向量。
不做缓存、不落盘原始嵌入、不引入第三方依赖。

⚠️ 系统代理会把 loopback 送进代理，所以默认 ``opener`` 用空 ``ProxyHandler``。
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Sequence

__all__ = ["HashEmbedding", "OllamaEmbedding"]


class OllamaEmbedding:
    """调用本机 Ollama 的嵌入接口。

    Args:
        model: 模型名（默认 ``nomic-embed-text``）。
        host: 服务地址；只应是 loopback。
        timeout: 单次请求超时（秒）。
        opener: 可注入的 opener（测试用），签名 ``(request, timeout) -> response``。
    """

    name = "ollama-embed"

    def __init__(
        self,
        model: str = "nomic-embed-text",
        *,
        host: str = "http://127.0.0.1:11434",
        timeout: float = 60.0,
        opener: Callable | None = None,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = float(timeout)
        self._open = opener or urllib.request.build_opener(
            urllib.request.ProxyHandler({})
        ).open

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """批量嵌入；返回与输入等长的向量列表。"""
        items = [str(text) for text in texts]
        if not items or not all(item.strip() for item in items):
            raise ValueError("texts 必须是非空、非空白的字符串序列。")
        payload = {"model": self.model, "input": items}
        request = urllib.request.Request(
            f"{self.host}/api/embed",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._open(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(items):
            raise RuntimeError(f"Ollama 返回的 embeddings 不合法：{body!r}")
        return [[float(value) for value in vector] for vector in embeddings]


class HashEmbedding:
    """确定性的哈希嵌入，仅用于单元测试与无 Ollama 的冒烟。"""

    name = "hash"

    def __init__(self, dimension: int = 32) -> None:
        if dimension < 1:
            raise ValueError(f"dimension 必须为正，收到 {dimension!r}。")
        self.dimension = dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimension
            for index, char in enumerate(str(text)):
                vector[(ord(char) + index) % self.dimension] += 1.0
            norm = sum(value * value for value in vector) ** 0.5 or 1.0
            vectors.append([value / norm for value in vector])
        return vectors

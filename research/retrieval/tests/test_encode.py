"""编码与缓存的测试：假后端，不联网、不依赖 Ollama。"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from biosnn_bus.plugins.text_zscore import EmbeddingBackend

from research.retrieval.encode import CachedEmbedding, build_bus, encode_items
from research.retrieval.procedural import generate, split


class FakeEmbedding(EmbeddingBackend):
    """确定性的假 embedding：同一个词永远同一个向量。"""

    def __init__(self, dim: int = 32) -> None:
        self._dim = dim
        self.calls = 0

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        self.calls += 1
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        return np.random.default_rng(seed).normal(size=self._dim)


class TestCachedEmbedding:
    def test_second_call_is_cached(self, tmp_path):
        fake = FakeEmbedding()
        cached = CachedEmbedding(fake, tmp_path / "c.json", model="fake")
        first = cached.embed("hello")
        second = cached.embed("hello")
        assert fake.calls == 1
        assert np.allclose(first, second)

    def test_round_trip(self, tmp_path):
        path = tmp_path / "c.json"
        first = CachedEmbedding(FakeEmbedding(), path, model="fake")
        first.embed("hello")
        first.save()
        cached = CachedEmbedding(FakeEmbedding(), path, model="fake")
        cached.embed("hello")
        assert cached.inner.calls == 0

    def test_model_mismatch_is_loud(self, tmp_path):
        path = tmp_path / "c.json"
        c = CachedEmbedding(FakeEmbedding(), path, model="fake")
        c.embed("hello")
        c.save()
        with pytest.raises(ValueError, match="模型"):
            CachedEmbedding(FakeEmbedding(), path, model="another-model")


class TestEncodeItems:
    def test_shapes_and_separation(self):
        items = split(generate(), test_fraction=0.2, seed=0)[1][:6]
        bus = build_bus(bus_dim=32, seed=0, backend=FakeEmbedding())
        reprs = encode_items(bus, items)
        assert reprs.n == 6
        assert reprs.text_bus.shape == (6, 32)
        assert reprs.image_bus.shape == (6, 32)
        assert reprs.text_plugin.shape[0] == 6
        assert reprs.image_plugin.shape[0] == 6

    def test_the_two_bus_sides_live_in_different_subspaces(self):
        """只有文本 / 只有图像两条路径经过**不同的**随机投影矩阵——预注册的前提。"""
        items = split(generate(), test_fraction=0.2, seed=0)[1]
        bus = build_bus(bus_dim=32, seed=0, backend=FakeEmbedding())
        reprs = encode_items(bus, items)
        cos = (reprs.text_bus * reprs.image_bus).sum(axis=1) / (
            np.linalg.norm(reprs.text_bus, axis=1) * np.linalg.norm(reprs.image_bus, axis=1) + 1e-12
        )
        assert abs(float(np.mean(cos))) < 0.5

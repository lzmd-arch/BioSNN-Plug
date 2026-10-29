"""M3e 适配器与情景库的守卫测试（不联网）。"""

from __future__ import annotations

import json

import pytest

from biosnn_mcp import EpisodicStore, HashEmbedding, MemoryItem, OllamaEmbedding


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_ollama_embedding_payload_and_parse():
    seen = {}

    def opener(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data.decode("utf-8"))
        return _Response({"embeddings": [[1.0, 0.0], [0.0, 2.0]]})

    vectors = OllamaEmbedding(model="nomic-embed-text", opener=opener).embed(["a", "b"])
    assert vectors == [[1.0, 0.0], [0.0, 2.0]]
    assert seen["url"].endswith("/api/embed")
    assert seen["body"]["model"] == "nomic-embed-text"
    assert seen["body"]["input"] == ["a", "b"]


def test_ollama_embedding_rejects_bad_shape():
    llm = OllamaEmbedding(opener=lambda request, timeout: _Response({"embeddings": [[1.0]]}))
    with pytest.raises(RuntimeError):
        llm.embed(["a", "b"])


def test_ollama_embedding_rejects_empty():
    llm = OllamaEmbedding(opener=lambda request, timeout: _Response({"embeddings": []}))
    with pytest.raises(ValueError):
        llm.embed([])


def test_hash_embedding_is_deterministic_and_normalized():
    encoder = HashEmbedding(dimension=8)
    first = encoder.embed(["hello"])[0]
    second = encoder.embed(["hello"])[0]
    assert first == second
    assert sum(value * value for value in first) == pytest.approx(1.0)


def test_store_search_orders_by_cosine():
    with EpisodicStore() as store:
        store.add("a", "alpha", [1.0, 0.0])
        store.add("b", "beta", [0.0, 1.0])
        hits = store.search([0.9, 0.1], top_k=2)
        assert [hit["id"] for hit in hits] == ["a", "b"]
        assert hits[0]["score"] > hits[1]["score"]


def test_store_items_and_replace_all():
    with EpisodicStore() as store:
        store.add("a", "alpha", [1.0, 0.0])
        store.add("b", "beta", [0.0, 1.0])
        items = store.items()
        assert [item.id for item in items] == ["a", "b"]
        store.replace_all([MemoryItem(id="c", text="gamma", embedding=(0.5, 0.5))])
        assert len(store) == 1
        assert store.items()[0].id == "c"


def test_store_persists_to_file(tmp_path):
    path = str(tmp_path / "episodic.db")
    with EpisodicStore(path) as store:
        store.add("a", "alpha", [1.0, 0.0])
    with EpisodicStore(path) as reopened:
        assert len(reopened) == 1
        assert reopened.items()[0].text == "alpha"


def test_store_rejects_bad_inputs():
    with EpisodicStore() as store:
        with pytest.raises(ValueError):
            store.add("", "text", [1.0])
        with pytest.raises(ValueError):
            store.add("a", "text", [])
        with pytest.raises(ValueError):
            store.search([1.0], top_k=0)

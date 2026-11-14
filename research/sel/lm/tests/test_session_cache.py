"""会话级状态缓存的守卫：**前缀判断错一格，分布就完全不同，而且不会报错。**

真正的等价性（增量 vs 重算逐位相同）由 `serve_snn` 的实测覆盖；
这里钉的是"什么情况下允许复用、什么情况下必须重算"。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest

from research.sel.lm import serve_snn


@dataclass
class FakeTensor:
    value: list

    def clone(self) -> FakeTensor:
        return FakeTensor(list(self.value))


@dataclass
class FakeTrace:
    v: FakeTensor
    z: FakeTensor | None = None


@pytest.fixture(autouse=True)
def _clean() -> None:
    serve_snn._SESSIONS.clear()
    yield
    serve_snn._SESSIONS.clear()


def _store(session: str, ids: tuple, consumed: int, stamp: float | None = None) -> None:
    trace = FakeTrace(FakeTensor([consumed]))
    serve_snn._SESSIONS[session] = (ids, trace, consumed, time.time() if stamp is None else stamp)


def test_reuses_on_extension() -> None:
    ids = tuple(range(10))
    _store("a", ids, consumed=9)
    trace, done = serve_snn._reuse_trace("a", ids + (10,), max_len=256)
    assert done == 9
    assert trace.v.value == [9]


def test_reuse_returns_a_clone_not_the_cached_object() -> None:
    """**必须拷贝**：`step` 会原地改 trace，直接复用会把缓存写脏。"""
    ids = tuple(range(10))
    _store("a", ids, consumed=9)
    trace, _ = serve_snn._reuse_trace("a", ids + (10,), max_len=256)
    trace.v.value.append(999)
    assert serve_snn._SESSIONS["a"][1].v.value == [9]


def test_refuses_after_divergence() -> None:
    """分叉必须发生在**已消费的那段里**才该拒绝。

    第 9 位（还没消费到）变了不算分叉——前 9 步的递推结果照样有效。
    """
    _store("a", tuple(range(10)), consumed=9)
    trace, done = serve_snn._reuse_trace("a", tuple(range(8)) + (99, 10, 11), max_len=256)
    assert (trace, done) == (None, 0)


def test_reuses_when_divergence_is_beyond_the_consumed_prefix() -> None:
    _store("a", tuple(range(10)), consumed=9)
    trace, done = serve_snn._reuse_trace("a", tuple(range(9)) + (99, 11), max_len=256)
    assert done == 9 and trace is not None


def test_refuses_when_context_shrank() -> None:
    _store("a", tuple(range(10)), consumed=9)
    trace, done = serve_snn._reuse_trace("a", tuple(range(5)), max_len=256)
    assert (trace, done) == (None, 0)


def test_refuses_under_truncation() -> None:
    """窗口一滑动，id 序列就不是前缀关系了；宁可重算。"""
    _store("a", tuple(range(10)), consumed=9)
    trace, done = serve_snn._reuse_trace("a", tuple(range(400)), max_len=256)
    assert (trace, done) == (None, 0)


def test_refuses_without_session() -> None:
    _store("a", tuple(range(10)), consumed=9)
    assert serve_snn._reuse_trace(None, tuple(range(11)), max_len=256) == (None, 0)


def test_expires() -> None:
    _store("a", tuple(range(10)), consumed=9, stamp=time.time() - serve_snn._SESSION_TTL - 1)
    trace, _ = serve_snn._reuse_trace("a", tuple(range(11)), max_len=256)
    assert trace is None
    assert "a" not in serve_snn._SESSIONS


def test_capacity_evicts_oldest() -> None:
    for index in range(serve_snn._SESSION_CAP):
        _store(f"s{index}", tuple(range(10)), consumed=9, stamp=1000.0 + index)
    serve_snn._store_trace("new", tuple(range(10)), FakeTrace(FakeTensor([9])), 9)
    assert len(serve_snn._SESSIONS) <= serve_snn._SESSION_CAP
    assert "s0" not in serve_snn._SESSIONS
    assert "new" in serve_snn._SESSIONS


def test_store_keeps_none_fields_none() -> None:
    serve_snn._store_trace("a", tuple(range(10)), FakeTrace(FakeTensor([9]), None), 9)
    assert serve_snn._SESSIONS["a"][1].z is None

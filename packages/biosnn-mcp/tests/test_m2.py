"""M2 触发策略的规则测试。"""

from __future__ import annotations

from biosnn_mcp import TriggerPolicy


def test_knowledge_low_retrieval_calls():
    decision = TriggerPolicy().decide("knowledge", 0.05)
    assert decision.call is True
    assert decision.reason == "knowledge_low_retrieval"


def test_knowledge_with_retrieval_skips():
    assert TriggerPolicy().decide("knowledge", 0.8).call is False


def test_memory_and_chat_never_call():
    policy = TriggerPolicy()
    assert policy.decide("memory", 0.0).call is False
    assert policy.decide("chat", 0.0).call is False


def test_unknown_low_retrieval_calls():
    assert TriggerPolicy().decide("unknown", 0.05).call is True

"""M1 mock LLM 与对话 trace 测试。"""

from __future__ import annotations

import json

import pytest
from biosnn_mcp import DialogueTrace, MockLLM, TurnRecord


class TestMockLLM:
    def test_is_deterministic_and_records_calls(self):
        llm = MockLLM()
        first = llm.generate("hello")
        second = llm.generate("hello")
        assert first == second == "[mock] hello"
        assert llm.calls == 2

    def test_includes_context_length_only(self):
        llm = MockLLM()
        assert llm.generate("q", context="abc") == "[mock] q | context=3 chars"

    def test_rejects_empty_prompt(self):
        with pytest.raises(ValueError):
            MockLLM().generate("  ")


class TestDialogueTrace:
    def test_append_and_summary(self):
        trace = DialogueTrace()
        trace.append(TurnRecord(1, "hi", retrieved=["m1"], trigger="skipped"))
        trace.append(
            TurnRecord(
                2,
                "explain",
                retrieved=["m1", "m2"],
                trigger="called",
                llm_response="[mock] explain",
                consolidation={"stored": True},
            )
        )
        assert trace.summary() == {
            "schema_version": 1,
            "turns": 2,
            "llm_calls": 1,
            "retrieved": 3,
            "consolidated": 1,
        }

    def test_jsonl_round_trip_and_replay(self, tmp_path):
        path = tmp_path / "trace.jsonl"
        trace = DialogueTrace(path)
        trace.append(TurnRecord(1, "a", trigger="no_trigger"))
        trace.append(TurnRecord(2, "b", trigger="called", llm_response="[mock] b"))
        restored = DialogueTrace(path)
        assert [record.user for record in restored.records] == ["a", "b"]
        lines = path.read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["turn"] for line in lines] == [1, 2]

    def test_duplicate_turn_is_rejected(self):
        trace = DialogueTrace()
        trace.append(TurnRecord(1, "a"))
        with pytest.raises(ValueError):
            trace.append(TurnRecord(1, "b"))

    def test_invalid_trigger_is_rejected(self):
        with pytest.raises(ValueError):
            TurnRecord(1, "a", trigger="maybe")

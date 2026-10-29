"""M3b 对话评测装置的守卫测试：用替身 LLM，不联网、不依赖 Ollama。"""

from __future__ import annotations

from biosnn_mcp import DialogueTrace, TriggerPolicy
from biosnn_mcp.dialogue import (
    KNOWLEDGE,
    WritableMemory,
    build_scenarios,
    run_session,
    seeded_entries,
)


class GoldLLM:
    """按题库返回标准答案；用来验证「答对时完成率上升」这条通路。"""

    name = "gold"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, *, context: str = "") -> str:
        self.calls += 1
        for question, gold in KNOWLEDGE:
            if question in prompt.lower():
                return gold
        return "i don't know"


class WrongLLM:
    name = "wrong"

    def generate(self, prompt: str, *, context: str = "") -> str:
        return "no idea"


def test_scenarios_have_memory_turn_and_short_length():
    scenarios = build_scenarios()
    assert len(scenarios) == 20
    for scenario in scenarios:
        assert len(scenario.turns) <= 10
        memory_turns = [turn for turn in scenario.turns if turn.intent == "memory"]
        assert memory_turns and memory_turns[0].memory_id


def test_seeded_fraction_is_one_third():
    seeded = seeded_entries()
    assert len(seeded) == len([1 for i in range(len(KNOWLEDGE)) if i % 3 == 0])


def test_memory_search_requires_all_content_tokens():
    memory = WritableMemory([("what is the chemical symbol for sodium", "na")])
    assert memory.search("what is the chemical symbol for sodium") is not None
    assert memory.search("what is the chemical symbol for gold") is None


def test_knowledge_turn_triggers_and_consolidates():
    scenario = build_scenarios()[1]  # 该会话的知识题不在预置里
    llm = GoldLLM()
    trace = DialogueTrace(None)
    row = run_session(scenario, llm=llm, policy=TriggerPolicy(), use_llm=True, trace=trace)
    assert llm.calls >= 1
    assert row["first_call_turn"] is not None
    assert row["consolidation_rate"] == 1.0


def test_wrong_answers_lower_completion():
    scenario = build_scenarios()[1]
    gold_row = run_session(
        scenario, llm=GoldLLM(), policy=TriggerPolicy(), use_llm=True, trace=DialogueTrace(None)
    )
    wrong_row = run_session(
        scenario, llm=WrongLLM(), policy=TriggerPolicy(), use_llm=True, trace=DialogueTrace(None)
    )
    assert gold_row["completion_rate"] > wrong_row["completion_rate"]


def test_baseline_never_calls_llm():
    scenario = build_scenarios()[1]
    llm = GoldLLM()
    row = run_session(
        scenario, llm=llm, policy=TriggerPolicy(), use_llm=False, trace=DialogueTrace(None)
    )
    assert llm.calls == 0
    assert row["first_call_turn"] is None


def test_memory_turns_never_trigger():
    scenario = build_scenarios()[1]
    trace = DialogueTrace(None)
    run_session(
        scenario, llm=GoldLLM(), policy=TriggerPolicy(), use_llm=True, trace=trace
    )
    memory_turns = [turn for turn in scenario.turns if turn.intent == "memory"]
    records = {record.turn: record for record in trace.records}
    for index, turn in enumerate(scenario.turns, start=1):
        if turn in memory_turns:
            assert records[index].trigger == "skipped"

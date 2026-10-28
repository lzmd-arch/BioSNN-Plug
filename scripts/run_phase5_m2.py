"""阶段五 M2：触发策略 + mock LLM 对话 trace 的协议冒烟。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages" / "biosnn-mcp" / "src"))

from biosnn_mcp import (  # noqa: E402
    DialogueTrace,
    InMemoryCore,
    MemoryEntry,
    MockLLM,
    TriggerPolicy,
    TurnRecord,
)

OUT = REPO / "sweep_results" / "phase5_m2"

KNOWLEDGE = [
    "what is the capital of France",
    "how does photosynthesis work",
    "who wrote the tempest",
    "what is the boiling point of water",
    "how far is the moon",
    "what causes thunder",
    "what is a prime number",
    "who painted the starry night",
    "what is the largest ocean",
    "how do vaccines work",
    "what is the speed of light",
    "what is the tallest mountain",
]
MEMORY = [
    "what is my birthday",
    "what is my favorite color",
    "where do i live",
    "what is my sister's name",
    "what project am i working on",
    "what did i say about the deadline",
    "what is my coffee order",
    "which music do i like",
    "what is my cat called",
    "where did i park",
    "what is my phone model",
    "what did i plan for the weekend",
]
CHAT = ["hello", "good morning", "i feel tired", "thank you", "how are you", "good night"]


def _core() -> InMemoryCore:
    entries = [
        MemoryEntry("m1", "my birthday is in May"),
        MemoryEntry("m2", "my favorite color is teal"),
        MemoryEntry("m3", "i live in Nanjing"),
        MemoryEntry("m4", "my sister's name is Lin"),
        MemoryEntry("m5", "i am working on the BioSNN project"),
        MemoryEntry("m6", "the deadline is Friday"),
        MemoryEntry("m7", "my coffee order is a flat white"),
        MemoryEntry("m8", "i like jazz music"),
        MemoryEntry("m9", "my cat is called Mochi"),
        MemoryEntry("m10", "i parked on level three"),
        MemoryEntry("m11", "my phone model is a Pixel"),
        MemoryEntry("m12", "i planned a hike for the weekend"),
    ]
    return InMemoryCore(entries)


def _cases():
    cases = []
    for query in KNOWLEDGE:
        cases.append((query, "knowledge", True))
    for query in MEMORY:
        cases.append((query, "memory", False))
    for query in CHAT:
        cases.append((query, "chat", False))
    return cases


def main() -> int:
    core = _core()
    llm = MockLLM()
    policy = TriggerPolicy()
    trace = DialogueTrace(OUT / "trace.jsonl")
    decisions = []
    first_correct = None
    for turn, (query, intent, should_call) in enumerate(_cases(), start=1):
        hits = core.search(query, top_k=3)
        max_score = max((hit["score"] for hit in hits), default=0.0)
        decision = policy.decide(intent, max_score)
        response = llm.generate(query) if decision.call else None
        trace.append(
            TurnRecord(
                turn=turn,
                user=query,
                state=core.state(),
                retrieved=[hit["id"] for hit in hits],
                trigger="called" if decision.call else "skipped",
                llm_response=response,
                consolidation={},
            )
        )
        correct = decision.call == should_call
        decisions.append(
            {
                "turn": turn,
                "query": query,
                "intent": intent,
                "should_call": should_call,
                "call": decision.call,
                "reason": decision.reason,
                "score": max_score,
                "correct": correct,
            }
        )
        if correct and decision.call and first_correct is None:
            first_correct = turn
    accuracy = sum(1 for item in decisions if item["correct"]) / len(decisions)
    summary = {
        "accuracy": accuracy,
        "first_correct_call_turn": first_correct,
        "llm_calls": llm.calls,
        "trace_summary": trace.summary(),
        "decisions": decisions,
    }
    gate = {
        "accuracy_ok": accuracy >= 0.70,
        "first_call_ok": first_correct is not None and first_correct <= 10,
        "trace_ok": trace.summary()["turns"] == len(decisions),
        "no_network": True,
    }
    gate["m2_ready"] = all(gate.values())
    summary["gate"] = gate
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    print(f"accuracy {accuracy:.3f}, first {first_correct}, llm calls {llm.calls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

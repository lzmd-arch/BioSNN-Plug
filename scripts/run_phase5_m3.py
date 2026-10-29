"""阶段五 M3a：mock LLM 结果回注与下一轮检索的协议冒烟。"""

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

OUT = REPO / "sweep_results" / "phase5_m3"
QUESTIONS = [
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


def main() -> int:
    core = InMemoryCore([MemoryEntry("m0", "unrelated note")])
    llm = MockLLM()
    policy = TriggerPolicy()
    trace = DialogueTrace(OUT / "trace.jsonl")
    memory: dict[str, str] = {}
    successes = 0
    decisions = []
    for turn, question in enumerate(QUESTIONS, start=1):
        hits = core.search(question, top_k=1)
        score = hits[0]["score"] if hits else 0.0
        decision = policy.decide("knowledge", score)
        response = llm.generate(question) if decision.call else None
        memory_id = f"answer:{turn}"
        stored = False
        retrieved = None
        if response is not None:
            memory[memory_id] = response
            stored = True
            retrieved = memory.get(memory_id)
        success = bool(stored and retrieved == response)
        successes += int(success)
        trace.append(
            TurnRecord(
                turn=turn,
                user=question,
                state=core.state(),
                retrieved=[memory_id] if success else [],
                trigger="called" if decision.call else "skipped",
                llm_response=response,
                consolidation={"memory_id": memory_id, "stored": stored, "retrieved": success},
            )
        )
        decisions.append(
            {
                "turn": turn,
                "question": question,
                "call": decision.call,
                "stored": stored,
                "retrieved": success,
            }
        )
    rate = successes / len(QUESTIONS)
    gate = {
        "consolidation_rate": rate,
        "consolidation_ok": rate >= 0.80,
        "trace_ok": trace.summary()["turns"] == len(QUESTIONS),
        "no_network": True,
    }
    gate["m3a_ready"] = all(gate.values())
    summary = {
        "kind": "mock consolidation protocol; not a real-LLM capability result",
        "consolidation_rate": rate,
        "llm_calls": llm.calls,
        "trace_summary": trace.summary(),
        "gate": gate,
        "decisions": decisions,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

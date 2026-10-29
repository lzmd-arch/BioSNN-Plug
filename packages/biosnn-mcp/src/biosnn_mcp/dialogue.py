"""阶段五 M3b：合成对话评测的会话集与单会话执行器。

**这是评测装置，不是能力本身**：知识题是固定题库，判定用预设关键短语，记忆替身用词重叠
检索（没有向量召回）。边界写在 ``sweep_results/phase5_m3b_prereg.md`` 与结果 README，
不得把这里的数字外推到开放域对话。

两个臂共用同一会话、同一触发策略、同一预置记忆：

* ``use_llm=True``：触发策略决定是否调用 LLM，回答写入记忆；
* ``use_llm=False``：纯 SNN 基线，只用记忆检索，检索不到答 ``i don't know``。
"""

from __future__ import annotations

from dataclasses import dataclass

from .trace import DialogueTrace, TurnRecord
from .trigger import TriggerPolicy

__all__ = [
    "CHAT",
    "KNOWLEDGE",
    "Scenario",
    "Turn",
    "WritableMemory",
    "build_scenarios",
    "content_tokens",
    "run_session",
]

KNOWLEDGE: list[tuple[str, str]] = [
    ("what is the capital of france", "paris"),
    ("at what temperature celsius does water boil at sea level", "100"),
    ("who wrote hamlet", "shakespeare"),
    ("what is the largest planet in the solar system", "jupiter"),
    ("what is the chemical symbol for gold", "au"),
    ("what is the largest ocean on earth", "pacific"),
    ("who invented the telephone", "bell"),
    ("what is the capital of japan", "tokyo"),
    ("at what temperature celsius does water freeze", "0"),
    ("how many continents are there", "7"),
    ("who wrote pride and prejudice", "austen"),
    ("what is the tallest mountain on earth", "everest"),
    ("what is the currency of japan", "yen"),
    ("which planet is closest to the sun", "mercury"),
    ("which organ pumps blood through the body", "heart"),
    ("what is the largest mammal", "whale"),
    ("who wrote the novel 1984", "orwell"),
    ("what is the capital of australia", "canberra"),
    ("how many sides does a hexagon have", "6"),
    ("who painted the mona lisa", "vinci"),
    ("which is the largest desert on earth", "antarctic"),
    ("what is the hardest natural substance", "diamond"),
    ("what is the capital of canada", "ottawa"),
    ("who discovered penicillin", "fleming"),
    ("which country is largest by area", "russia"),
    ("what is the main gas in earth's atmosphere", "nitrogen"),
    ("who wrote romeo and juliet", "shakespeare"),
    ("what is the capital of italy", "rome"),
    ("how many strings does a violin have", "4"),
    ("who developed the theory of relativity", "einstein"),
    ("what is the largest bone in the human body", "femur"),
    ("what is the capital of egypt", "cairo"),
    ("who wrote the origin of species", "darwin"),
    ("what is the chemical symbol for sodium", "na"),
    ("what is the largest island in the world", "greenland"),
    ("what is the capital of brazil", "brasilia"),
    ("who composed the ninth symphony", "beethoven"),
    ("what is the smallest prime number", "2"),
]

CHAT: list[str] = [
    "hello there",
    "how are you today",
    "tell me a joke",
    "good morning",
    "thanks for your help",
    "what can you do",
]

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "at",
        "be",
        "by",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "its",
        "many",
        "of",
        "on",
        "or",
        "that",
        "the",
        "there",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    ]
)


def content_tokens(text: str) -> set[str]:
    """去掉停用词后的词集合；记忆替身的检索与判分都用它。"""
    return {token for token in text.lower().split() if token not in STOPWORDS}


@dataclass(frozen=True)
class Turn:
    intent: str  # knowledge / chat / memory
    text: str
    gold: str | None = None
    memory_id: str | None = None


@dataclass(frozen=True)
class Scenario:
    name: str
    turns: tuple[Turn, ...]


def build_scenarios() -> list[Scenario]:
    """20 个会话：knowledge 与 chat 交错，随后一个 memory 轮问刚巩固的事实。"""
    scenarios = []
    for index in range(20):
        turns: list[Turn] = []
        questions = []
        for offset in range(2):
            question, gold = KNOWLEDGE[(index * 2 + offset) % len(KNOWLEDGE)]
            questions.append(question)
            turns.append(Turn("knowledge", question, gold=gold))
            turns.append(Turn("chat", CHAT[(index + offset) % len(CHAT)]))
        turns.append(Turn("memory", questions[0], memory_id=f"answer:{questions[0]}"))
        scenarios.append(Scenario(f"session{index:02d}", tuple(turns)))
    return scenarios


def seeded_entries() -> list[tuple[str, str]]:
    """两个臂都预置的知识：每 3 条里的第 1 条。没有它，基线在 knowledge 轮上结构性全错。"""
    return [(question, gold) for index, (question, gold) in enumerate(KNOWLEDGE) if index % 3 == 0]


class WritableMemory:
    """可写的记忆替身：条目文本是「问句 + 回答」，检索按内容词重叠。

    只认「问句的内容词全部出现在条目里」：这个替身没有语义检索，放宽阈值会让
    ``chemical symbol for gold`` 命中 ``chemical symbol for sodium``，测到的就不再是触发
    策略，而是词重叠的混淆。真实记忆库用向量召回，不靠这条规则。

    Args:
        seeded: 预置的 (问句, 回答) 对，两个臂一致。
    """

    def __init__(self, seeded: list[tuple[str, str]] | None = None) -> None:
        self.entries: dict[str, tuple[str, str]] = {}
        for question, answer in seeded or []:
            self.entries[f"answer:{question}"] = (question, answer)

    def add(self, identifier: str, question: str, answer: str) -> None:
        self.entries[identifier] = (question, answer)

    def search(self, query: str) -> tuple[str, str, float] | None:
        tokens = content_tokens(query)
        if not tokens:
            return None
        best: tuple[str, str, float] | None = None
        for identifier, (question, answer) in self.entries.items():
            words = content_tokens(question + " " + answer)
            score = len(tokens & words) / len(tokens)
            if score >= 0.999 and (best is None or score > best[2]):
                best = (identifier, answer, score)
        return best


def run_session(
    scenario: Scenario,
    *,
    llm,
    policy: TriggerPolicy,
    use_llm: bool,
    trace: DialogueTrace,
    turn_offset: int = 0,
) -> dict:
    """跑一个会话，返回逐会话指标。

    Args:
        scenario: 会话。
        llm: LLM 适配器（只在 ``use_llm`` 且触发时调用）。
        policy: 触发策略。
        use_llm: False 即纯 SNN 基线。
        trace: 记录器；``turn_offset`` 用来避免多会话 / 多臂的轮次号冲突。
        turn_offset: 轮次号偏移。
    """
    memory = WritableMemory(seeded_entries())
    call_turns: list[int] = []
    triggers_correct = 0
    total_turns = 0
    completion: list[bool] = []
    consolidation_success = 0
    consolidation_total = 0
    answers: dict[str, str] = {}

    for turn_index, turn in enumerate(scenario.turns, start=1):
        hit = memory.search(turn.text)
        score = hit[2] if hit else 0.0
        decision = policy.decide(turn.intent, score)
        answer = None
        if turn.intent == "knowledge" and use_llm and decision.call:
            answer = llm.generate(
                f"Answer with a single word or number, no explanation: {turn.text}"
            )
            memory.add(f"answer:{turn.text}", turn.text, answer)
            answers[f"answer:{turn.text}"] = answer
            call_turns.append(turn_index)
        elif turn.intent == "knowledge":
            answer = hit[1] if hit and score >= policy.knowledge_score_max else "i don't know"
        elif turn.intent == "chat":
            answer = "hello"
        elif turn.intent == "memory":
            consolidated = turn.memory_id in answers
            if consolidated:
                consolidation_total += 1
            if (
                hit
                and hit[0] == turn.memory_id
                and (not consolidated or hit[1] == answers[turn.memory_id])
            ):
                answer = hit[1]
                if consolidated:
                    consolidation_success += 1
            else:
                answer = "i don't know"

        expected_call = (
            turn.intent == "knowledge" and use_llm and score < policy.knowledge_score_max
        )
        total_turns += 1
        triggers_correct += int(decision.call == expected_call)
        if turn.intent == "knowledge":
            lowered = (answer or "").lower()
            completion.append(bool(turn.gold and turn.gold in lowered))
        trace.append(
            TurnRecord(
                turn=turn_offset + turn_index,
                user=turn.text,
                state={"intent": turn.intent, "score": score},
                retrieved=[hit[0]] if hit else [],
                trigger="called" if decision.call else "skipped",
                llm_response=answer if decision.call else None,
                consolidation={},
            )
        )

    return {
        "scenario": scenario.name,
        "turns": total_turns,
        "first_call_turn": call_turns[0] if call_turns else None,
        "trigger_accuracy": triggers_correct / total_turns,
        "completion_rate": sum(completion) / len(completion) if completion else 0.0,
        "consolidation_rate": (
            consolidation_success / consolidation_total if consolidation_total else None
        ),
        "answers": answers,
    }

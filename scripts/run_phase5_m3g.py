"""阶段五 M3g：ES 调度维度（判据见 sweep_results/phase5_m3g_prereg.md）。

(1+λ)-ES 优化两参数调度策略，与规则门 TriggerPolicy 在同一批合成会话上比较。
只比较调度决策；MockLLM 的回答质量不参与。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages" / "biosnn-mcp" / "src"))

from biosnn_mcp import SchedulingPolicy, TriggerPolicy  # noqa: E402
from biosnn_mcp.dialogue import CHAT, KNOWLEDGE  # noqa: E402

OUT = REPO / "sweep_results" / "phase5_m3g"
CALL_COST = 0.05


def build_sessions(
    seed: int, count: int = 20, *, graded: bool = False
) -> list[list[tuple[str, str, float]]]:
    """生成会话：``(intent, text, retrieval_score)``。

    默认（M3g）：三分之一的 knowledge 题已预置（score=1.0），其余未预置（0.0）。
    ``graded=True``（M3h）：已预置的题分三档 {0.20, 0.40, 0.60}，未预置 0.0；
    这样阈值是可辨识的。chat/memory 轮 score=0.0 且不参与调用决策。
    """
    rng = random.Random(seed)
    sessions = []
    for _ in range(count):
        turns = []
        for _ in range(2):
            question, _gold = KNOWLEDGE[rng.randrange(len(KNOWLEDGE))]
            seeded = rng.random() < (1.0 / 3.0)
            if graded:
                score = rng.choice([0.2, 0.4, 0.6]) if seeded else 0.0
            else:
                score = 1.0 if seeded else 0.0
            turns.append(("knowledge", question, score))
            turns.append(("chat", CHAT[rng.randrange(len(CHAT))], 0.0))
        turns.append(("memory", "what did you say", 0.0))
        sessions.append(turns)
    return sessions


def run_policy(policy, sessions, *, is_rule: bool, graded: bool = False) -> dict:
    """正确版本：逐轮维护 since_call。"""
    completed = 0
    total_knowledge = 0
    calls = 0
    for turns in sessions:
        since_call = 99
        for intent, _text, score in turns:
            call = False
            if intent == "knowledge":
                total_knowledge += 1
                if is_rule:
                    call = policy.decide(intent, score).call
                else:
                    call = policy.decide(score, since_call)
                if call:
                    calls += 1
                if graded:
                    # M3h：已预置（score>0）无论是否调用都算完成；未预置只有调用才算。
                    if score > 0.0 or call:
                        completed += 1
                elif call and score < 0.5:
                    # M3g 的旧口径（已发布结果按此口径，不重跑）。
                    completed += 1
            since_call = 0 if call else since_call + 1
    completion = completed / total_knowledge if total_knowledge else 0.0
    utility = completion - CALL_COST * calls / len(sessions)
    return {
        "completion": completion,
        "calls": calls,
        "utility": utility,
    }


def perturb(policy: SchedulingPolicy, rng: random.Random, sigma: float) -> SchedulingPolicy:
    threshold = min(1.0, max(0.0, policy.score_threshold + rng.gauss(0.0, sigma)))
    cooldown = min(5.0, max(0.0, policy.cooldown + rng.gauss(0.0, sigma)))
    return SchedulingPolicy(score_threshold=threshold, cooldown=cooldown)


def evolve(
    train_sessions,
    *,
    seed: int,
    generations: int = 20,
    lam: int = 8,
    sigma: float = 0.1,
    graded: bool = False,
):
    rng = random.Random(seed)
    current = SchedulingPolicy()
    best_score = run_policy(current, train_sessions, is_rule=False, graded=graded)["utility"]
    history = []
    for _ in range(generations):
        candidates = [perturb(current, rng, sigma) for _ in range(lam)]
        scored = [
            (
                run_policy(candidate, train_sessions, is_rule=False, graded=graded)["utility"],
                candidate,
            )
            for candidate in candidates
        ]
        top_score, top_policy = max(scored, key=lambda pair: pair[0])
        if top_score > best_score:
            best_score, current = top_score, top_policy
        history.append(
            {"best_utility": best_score, "theta": [current.score_threshold, current.cooldown]}
        )
    return current, best_score, history


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M3g ES 调度维度")
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--lam", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument(
        "--graded", action="store_true", help="M3h：已预置题目用 {0.2,0.4,0.6} 三档连续分"
    )
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)
    seeds = _parse_seeds(args.seeds)

    rows = []
    for seed in seeds:
        train_sessions = build_sessions(0, graded=args.graded)
        held_out = build_sessions(1000 + seed, graded=args.graded)
        policy, train_utility, _history = evolve(
            train_sessions,
            seed=seed,
            generations=args.generations,
            lam=args.lam,
            sigma=args.sigma,
            graded=args.graded,
        )
        es_metrics = run_policy(policy, held_out, is_rule=False, graded=args.graded)
        rule_metrics = run_policy(TriggerPolicy(), held_out, is_rule=True, graded=args.graded)
        # 确定性：同种子再进化一次
        policy_again, _, _ = evolve(
            train_sessions,
            seed=seed,
            generations=args.generations,
            lam=args.lam,
            sigma=args.sigma,
            graded=args.graded,
        )
        deterministic = policy == policy_again
        distance = ((policy.score_threshold - 0.5) ** 2 + (policy.cooldown - 1.0) ** 2) ** 0.5
        rows.append(
            {
                "seed": seed,
                "theta": [policy.score_threshold, policy.cooldown],
                "distance_from_init": distance,
                "train_utility": train_utility,
                "es": es_metrics,
                "rule": rule_metrics,
                "deterministic": deterministic,
            }
        )
        print(
            f"seed {seed}: theta=({policy.score_threshold:.3f}, {policy.cooldown:.3f}) "
            f"es utility {es_metrics['utility']:.3f} vs rule {rule_metrics['utility']:.3f}",
            flush=True,
        )

    better = sum(1 for row in rows if row["es"]["utility"] >= row["rule"]["utility"])
    not_costlier = sum(1 for row in rows if row["es"]["calls"] <= row["rule"]["calls"] + 1)
    deterministic = all(row["deterministic"] for row in rows)
    moved = sum(1 for row in rows if row["distance_from_init"] >= 0.1)
    gate = {
        "utility_not_worse": better >= 7,
        "calls_not_higher": not_costlier >= 7,
        "deterministic": deterministic,
    }
    if args.graded:
        gate["moved_from_init"] = moved >= 7
    gate["ready"] = all(gate.values())
    summary = {
        "kind": "protocol: (1+lambda)-ES scheduling dimension vs rule gate (MockLLM)",
        "config": {"generations": args.generations, "lam": args.lam, "sigma": args.sigma},
        "seeds": rows,
        "counts": {"utility_not_worse": better, "calls_not_higher": not_costlier},
        "gate": gate,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return 0


def _parse_seeds(text: str) -> list[int]:
    text = text.strip()
    if "-" in text and "," not in text:
        low, high = text.split("-")
        return list(range(int(low), int(high) + 1))
    return [int(part) for part in text.split(",") if part]


if __name__ == "__main__":
    raise SystemExit(main())

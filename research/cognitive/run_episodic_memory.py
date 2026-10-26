"""情景记忆 v0 的执行入口：稀疏键 + Hebbian 外积 + 容量曲线。

预注册与两条修订见 ``sweep_results/episodic_memory_v0_prereg*.md``。事件池 200 回合，
终止事件分层采样；主指标 N=500、σ=0.1。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.cognitive.episodic_memory import (
    HebbianEpisodicMemory,
    RawNearestNeighbor,
    SparseEncoder,
    build_values,
)
from research.cognitive.run_world_model_closed_loop import collect_episodes
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "episodic_memory_v0"

CAPACITIES = (100, 300, 500, 1000, 2000)
NOISE = 0.1
ACTION_MIN_500 = 0.70
TERM_BALANCED_MIN = 0.70
TERM_TPR_MIN = 0.70
ORDER_MIN = 0.75
ACTION_MIN_1000 = 0.65
MIN_TERMINALS = 5


def _pool(episodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = collect_episodes(episodes, seed=0, noise=NOISE)
    states = np.concatenate([episode.states for episode in data])
    actions = np.concatenate([episode.actions for episode in data])
    terminated = np.concatenate([episode.terminated for episode in data])
    timestamps = np.arange(states.shape[0], dtype=np.float64) / states.shape[0]
    return states, actions, terminated, timestamps


def _balanced_termination(
    labels: np.ndarray, predicted: np.ndarray, rng: np.random.Generator
) -> tuple[float | None, float | None]:
    positives = np.flatnonzero(labels)
    negatives = np.flatnonzero(~labels)
    if positives.size < MIN_TERMINALS or negatives.size == 0:
        return None, None
    chosen_negative = rng.choice(negatives, size=positives.size, replace=False)
    index = np.concatenate([positives, chosen_negative])
    truth = labels[index]
    guess = predicted[index]
    tpr = float(np.mean(guess[truth]))
    tnr = float(np.mean(~guess[~truth]))
    return (tpr + tnr) / 2.0, tpr


def _temporal_order(
    times: np.ndarray, recalled: np.ndarray, rng: np.random.Generator, pairs: int = 200
) -> float:
    if times.size < 2:
        return float("nan")
    first = rng.integers(0, times.size, size=pairs)
    second = rng.integers(0, times.size, size=pairs)
    keep = np.abs(times[first] - times[second]) > 0.05
    if not np.any(keep):
        return float("nan")
    truth = np.sign(times[first[keep]] - times[second[keep]])
    guess = np.sign(recalled[first[keep]] - recalled[second[keep]])
    return float(np.mean(truth == guess))


def _decode(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions = np.argmax(values[:, :2], axis=1)
    terminated = np.argmax(values[:, 2:4], axis=1).astype(bool)
    times = values[:, 4]
    return actions, terminated, times


def run(
    seeds: list[int],
    *,
    episodes: int,
    out: Path,
    capacities: tuple[int, ...] = CAPACITIES,
) -> dict[str, Any]:
    started = time.perf_counter()
    states, actions, terminated, timestamps = _pool(episodes)
    terminal_indices = np.flatnonzero(terminated)
    print(f"池：{states.shape[0]} 事件，{len(terminal_indices)} 个终止事件", flush=True)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        encoder = SparseEncoder(seed=seed)
        for capacity in capacities:
            rng = np.random.default_rng(seed * 100_000 + capacity)
            n_terminal = min(len(terminal_indices), math.ceil(0.1 * capacity))
            chosen_terminals = rng.choice(terminal_indices, size=n_terminal, replace=False)
            nonterminal = np.setdiff1d(np.arange(states.shape[0]), terminal_indices)
            chosen_nonterminals = rng.choice(nonterminal, size=capacity - n_terminal, replace=False)
            index = np.concatenate([chosen_terminals, chosen_nonterminals])
            rng.shuffle(index)

            clean_states = states[index]
            clean_keys = encoder.encode(clean_states)
            values = build_values(actions[index], terminated[index], timestamps[index])
            memory = HebbianEpisodicMemory()
            memory.store(clean_keys, values)

            noisy_states = clean_states + rng.normal(scale=NOISE, size=clean_states.shape)
            recalled = memory.recall(encoder.encode(noisy_states))
            pred_action, pred_term, pred_time = _decode(recalled)

            nearest = RawNearestNeighbor()
            nearest.store(clean_states, values)
            raw_recalled = nearest.recall(noisy_states)
            raw_action, raw_term, raw_time = _decode(raw_recalled)

            term_balanced, term_tpr = _balanced_termination(terminated[index], pred_term, rng)
            raw_term_balanced, raw_term_tpr = _balanced_termination(
                terminated[index], raw_term, rng
            )
            rows.append(
                {
                    "seed": seed,
                    "capacity": capacity,
                    "terminals_stored": int(n_terminal),
                    "action_accuracy": float(np.mean(pred_action == actions[index])),
                    "raw_action_accuracy": float(np.mean(raw_action == actions[index])),
                    "termination_balanced": term_balanced,
                    "termination_tpr": term_tpr,
                    "raw_termination_balanced": raw_term_balanced,
                    "raw_termination_tpr": raw_term_tpr,
                    "temporal_order": _temporal_order(timestamps[index], pred_time, rng),
                    "raw_temporal_order": _temporal_order(timestamps[index], raw_time, rng),
                }
            )
            print(
                f"seed {seed} N={capacity}: action {rows[-1]['action_accuracy']:.3f} "
                f"(raw {rows[-1]['raw_action_accuracy']:.3f}) | term {term_balanced} "
                f"| order {rows[-1]['temporal_order']}",
                flush=True,
            )

    def at(capacity: int, key: str) -> list[float]:
        return [row[key] for row in rows if row["capacity"] == capacity and row[key] is not None]

    def mean_at(capacity: int, key: str) -> float | None:
        values = at(capacity, key)
        return float(np.mean(values)) if values else None

    gate = {
        "action_500": mean_at(500, "action_accuracy"),
        "termination_balanced_500": mean_at(500, "termination_balanced"),
        "termination_tpr_500": mean_at(500, "termination_tpr"),
        "temporal_order_500": mean_at(500, "temporal_order"),
        "action_1000": mean_at(1000, "action_accuracy"),
    }
    gate["v0_ready"] = bool(
        gate["action_500"] is not None
        and gate["action_500"] >= ACTION_MIN_500
        and gate["termination_balanced_500"] is not None
        and gate["termination_balanced_500"] >= TERM_BALANCED_MIN
        and gate["termination_tpr_500"] is not None
        and gate["termination_tpr_500"] >= TERM_TPR_MIN
        and gate["temporal_order_500"] is not None
        and gate["temporal_order_500"] >= ORDER_MIN
        and gate["action_1000"] is not None
        and gate["action_1000"] >= ACTION_MIN_1000
    )
    summary = {
        "experiment": "情景记忆 v0：稀疏键 + Hebbian 外积",
        "config": {
            "policy_noise": NOISE,
            "episodes": episodes,
            "capacities": list(capacities),
            "seeds": seeds,
            "noise": NOISE,
        },
        "per_seed": rows,
        "gate": gate,
        "repro": collect(
            "情景记忆 v0",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "事件池：平衡启发式 + ε=0.1；终止事件分层采样",
                "键：固定随机投影 + top-32 符号；值：动作/终止 one-hot + 时间戳",
                "终止回忆用平衡准确率与 TPR；原始最近邻是基线，不是联想记忆成绩",
            ],
            device=None,
        ).render_block(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="情景记忆 v0")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    capacities = (20, 40) if args.smoke else CAPACITIES
    episodes = 10 if args.smoke else args.episodes
    run(parse_seeds(args.seeds), episodes=episodes, out=args.out, capacities=capacities)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

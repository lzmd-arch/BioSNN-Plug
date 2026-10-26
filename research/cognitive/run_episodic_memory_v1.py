"""情景记忆 v1：局部吸引子 + 独立时间戳 + 伪逆上限（预注册见 sweep_results/episodic_memory_v1_prereg.md）。"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.cognitive.episodic_memory import (
    EpisodicMemoryV1,
    PseudoInverseMemory,
    RawNearestNeighbor,
    SparseEncoder,
    build_action_term,
    build_values,
)
from research.cognitive.run_episodic_memory import (
    _balanced_termination,
    _pool,
    _temporal_order,
)
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "episodic_memory_v1"

CAPACITIES = (100, 300, 500, 1000, 2000)
NOISE = 0.1
ATTRACTOR_STEPS = 3
ACTION_MIN_500 = 0.70
TERM_MIN = 0.70
ORDER_MIN = 0.75
ACTION_MIN_1000 = 0.65


def _decode_v1(values: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions = np.argmax(values[:, :2], axis=1)
    terminated = np.argmax(values[:, 2:4], axis=1).astype(bool)
    return actions, terminated, times


def _metrics(
    actions: np.ndarray,
    terminated: np.ndarray,
    timestamps: np.ndarray,
    predicted_actions: np.ndarray,
    predicted_terminated: np.ndarray,
    predicted_times: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    term_balanced, term_tpr = _balanced_termination(terminated, predicted_terminated, rng)
    return {
        "action_accuracy": float(np.mean(predicted_actions == actions)),
        "termination_balanced": term_balanced,
        "termination_tpr": term_tpr,
        "temporal_order": _temporal_order(timestamps, predicted_times, rng),
    }


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
            noisy_states = clean_states + rng.normal(scale=NOISE, size=clean_states.shape)
            noisy_keys = encoder.encode(noisy_states)
            action_term = build_action_term(actions[index], terminated[index])
            times = timestamps[index]

            memory = EpisodicMemoryV1(sparsity=encoder.k)
            memory.store(clean_keys, action_term, times)
            values_oneshot, times_oneshot = memory.recall(noisy_keys, attractor_steps=0)
            values_attractor, times_attractor = memory.recall(
                noisy_keys, attractor_steps=ATTRACTOR_STEPS
            )
            oneshot = _decode_v1(values_oneshot, times_oneshot)
            attractor = _decode_v1(values_attractor, times_attractor)

            pseudo = PseudoInverseMemory()
            pseudo.store(clean_keys, build_values(actions[index], terminated[index], times))
            pseudo_values = pseudo.recall(noisy_keys)
            pseudo_decoded = (
                np.argmax(pseudo_values[:, :2], axis=1),
                np.argmax(pseudo_values[:, 2:4], axis=1).astype(bool),
                pseudo_values[:, 4],
            )

            nearest = RawNearestNeighbor()
            nearest.store(clean_states, build_values(actions[index], terminated[index], times))
            raw_values = nearest.recall(noisy_states)
            raw_decoded = (
                np.argmax(raw_values[:, :2], axis=1),
                np.argmax(raw_values[:, 2:4], axis=1).astype(bool),
                raw_values[:, 4],
            )

            arms = {
                "one_shot": oneshot,
                "attractor": attractor,
                "pseudo_inverse": pseudo_decoded,
                "raw_nn": raw_decoded,
            }
            for arm, (pred_a, pred_t, pred_time) in arms.items():
                metrics = _metrics(
                    actions[index],
                    terminated[index],
                    times,
                    pred_a,
                    pred_t,
                    pred_time,
                    rng,
                )
                rows.append(
                    {
                        "seed": seed,
                        "capacity": capacity,
                        "terminals_stored": int(n_terminal),
                        "arm": arm,
                        **metrics,
                    }
                )
            print(
                f"seed {seed} N={capacity}: "
                + " | ".join(
                    f"{arm} a={rows[-len(arms) + i]['action_accuracy']:.3f} "
                    f"t={rows[-len(arms) + i]['termination_balanced']} "
                    f"o={rows[-len(arms) + i]['temporal_order']}"
                    for i, arm in enumerate(arms)
                ),
                flush=True,
            )

    def mean_at(capacity: int, arm: str, key: str) -> float | None:
        values = [
            row[key]
            for row in rows
            if row["capacity"] == capacity and row["arm"] == arm and row[key] is not None
        ]
        return float(np.mean(values)) if values else None

    gate: dict[str, Any] = {}
    for arm in ("one_shot", "attractor"):
        action_500 = mean_at(500, arm, "action_accuracy")
        term_500 = mean_at(500, arm, "termination_balanced")
        tpr_500 = mean_at(500, arm, "termination_tpr")
        order_500 = mean_at(500, arm, "temporal_order")
        action_1000 = mean_at(1000, arm, "action_accuracy")
        ready = bool(
            action_500 is not None
            and action_500 >= ACTION_MIN_500
            and term_500 is not None
            and term_500 >= TERM_MIN
            and tpr_500 is not None
            and tpr_500 >= TERM_MIN
            and order_500 is not None
            and order_500 >= ORDER_MIN
            and action_1000 is not None
            and action_1000 >= ACTION_MIN_1000
        )
        gate[arm] = {
            "action_500": action_500,
            "termination_balanced_500": term_500,
            "termination_tpr_500": tpr_500,
            "temporal_order_500": order_500,
            "action_1000": action_1000,
            "ready": ready,
        }
    gate["pseudo_inverse_ceiling"] = {
        "action_500": mean_at(500, "pseudo_inverse", "action_accuracy"),
        "termination_balanced_500": mean_at(500, "pseudo_inverse", "termination_balanced"),
        "temporal_order_500": mean_at(500, "pseudo_inverse", "temporal_order"),
        "action_1000": mean_at(1000, "pseudo_inverse", "action_accuracy"),
    }
    gate["raw_nn_baseline"] = {
        "action_500": mean_at(500, "raw_nn", "action_accuracy"),
        "termination_balanced_500": mean_at(500, "raw_nn", "termination_balanced"),
        "temporal_order_500": mean_at(500, "raw_nn", "temporal_order"),
    }
    gate["v1_ready"] = gate["one_shot"]["ready"] or gate["attractor"]["ready"]
    summary = {
        "experiment": "情景记忆 v1：局部吸引子 + 独立时间戳",
        "config": {
            "policy_noise": NOISE,
            "episodes": episodes,
            "capacities": list(capacities),
            "seeds": seeds,
            "attractor_steps": ATTRACTOR_STEPS,
        },
        "per_seed": rows,
        "gate": gate,
        "repro": collect(
            "情景记忆 v1",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "动作/终止 4 维值 + 独立时间标量记忆；吸引子 3 步 top-k",
                "伪逆是上限对照，不是局部候选；原始 NN 单列",
                "数据/编码器/噪声/容量与 v0 相同",
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
    parser = argparse.ArgumentParser(description="情景记忆 v1")
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

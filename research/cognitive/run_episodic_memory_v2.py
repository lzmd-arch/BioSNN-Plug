"""情景记忆 v2：编码器对照（预注册见 sweep_results/episodic_memory_v2_prereg.md）。"""

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
OUT = REPO_ROOT / "sweep_results" / "episodic_memory_v2"

CAPACITIES = (100, 300, 500, 1000, 2000)
NOISE = 0.1
ACTION_MIN_500 = 0.70
TERM_MIN = 0.70
ORDER_MIN = 0.75
ACTION_MIN_1000 = 0.65

ENCODERS: dict[str, dict[str, Any]] = {
    "E0_sparse": {"dim": 512, "k": 32, "mode": "sparse"},
    "E1_dense_sign": {"dim": 512, "k": 512, "mode": "dense_sign"},
    "E2_wide_sparse": {"dim": 2048, "k": 128, "mode": "sparse"},
    "E3_dense_cont": {"dim": 512, "k": 512, "mode": "dense"},
}


def _decode(values: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.argmax(values[:, :2], axis=1),
        np.argmax(values[:, 2:4], axis=1).astype(bool),
        times,
    )


def _row(
    actions: np.ndarray,
    terminated: np.ndarray,
    times: np.ndarray,
    decoded: tuple[np.ndarray, np.ndarray, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, Any]:
    pred_action, pred_term, pred_time = decoded
    term_balanced, term_tpr = _balanced_termination(terminated, pred_term, rng)
    return {
        "action_accuracy": float(np.mean(pred_action == actions)),
        "termination_balanced": term_balanced,
        "termination_tpr": term_tpr,
        "temporal_order": _temporal_order(times, pred_time, rng),
    }


def run(
    seeds: list[int], *, episodes: int, out: Path, capacities: tuple[int, ...] = CAPACITIES
) -> dict:
    started = time.perf_counter()
    states, actions, terminated, timestamps = _pool(episodes)
    terminal_indices = np.flatnonzero(terminated)
    print(f"池：{states.shape[0]} 事件，{len(terminal_indices)} 个终止事件", flush=True)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for capacity in capacities:
            rng = np.random.default_rng(seed * 100_000 + capacity)
            n_terminal = min(len(terminal_indices), math.ceil(0.1 * capacity))
            chosen_terminals = rng.choice(terminal_indices, size=n_terminal, replace=False)
            nonterminal = np.setdiff1d(np.arange(states.shape[0]), terminal_indices)
            chosen_nonterminals = rng.choice(
                nonterminal, size=capacity - n_terminal, replace=False
            )
            index = np.concatenate([chosen_terminals, chosen_nonterminals])
            rng.shuffle(index)
            clean_states = states[index]
            noisy_states = clean_states + rng.normal(scale=NOISE, size=clean_states.shape)
            action_term = build_action_term(actions[index], terminated[index])
            times = timestamps[index]

            for arm, config in ENCODERS.items():
                encoder = SparseEncoder(seed=seed, **config)
                memory = EpisodicMemoryV1(sparsity=encoder.k)
                memory.store(encoder.encode(clean_states), action_term, times)
                values, recalled_times = memory.recall(
                    encoder.encode(noisy_states), attractor_steps=0
                )
                rows.append(
                    {
                        "seed": seed,
                        "capacity": capacity,
                        "arm": arm,
                        **_row(
                            actions[index],
                            terminated[index],
                            times,
                            _decode(values, recalled_times),
                            rng,
                        ),
                    }
                )

            e0 = SparseEncoder(seed=seed, **ENCODERS["E0_sparse"])
            pseudo = PseudoInverseMemory()
            pseudo.store(
                e0.encode(clean_states),
                build_values(actions[index], terminated[index], times),
            )
            pseudo_values = pseudo.recall(e0.encode(noisy_states))
            nearest = RawNearestNeighbor()
            nearest.store(
                clean_states, build_values(actions[index], terminated[index], times)
            )
            raw_values = nearest.recall(noisy_states)
            for arm, decoded in (
                (
                    "pseudo_inverse",
                    (
                        np.argmax(pseudo_values[:, :2], axis=1),
                        np.argmax(pseudo_values[:, 2:4], axis=1).astype(bool),
                        pseudo_values[:, 4],
                    ),
                ),
                (
                    "raw_nn",
                    (
                        np.argmax(raw_values[:, :2], axis=1),
                        np.argmax(raw_values[:, 2:4], axis=1).astype(bool),
                        raw_values[:, 4],
                    ),
                ),
            ):
                rows.append(
                    {
                        "seed": seed,
                        "capacity": capacity,
                        "arm": arm,
                        **_row(
                            actions[index], terminated[index], times, decoded, rng
                        ),
                    }
                )
            print(f"seed {seed} N={capacity} done", flush=True)

    def mean_at(capacity: int, arm: str, key: str) -> float | None:
        values = [
            row[key]
            for row in rows
            if row["capacity"] == capacity and row["arm"] == arm and row[key] is not None
        ]
        return float(np.mean(values)) if values else None

    gate: dict[str, Any] = {}
    for arm in ("E1_dense_sign", "E2_wide_sparse", "E3_dense_cont"):
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
    gate["reference_E0"] = {
        "action_500": mean_at(500, "E0_sparse", "action_accuracy"),
        "termination_balanced_500": mean_at(500, "E0_sparse", "termination_balanced"),
        "temporal_order_500": mean_at(500, "E0_sparse", "temporal_order"),
    }
    gate["pseudo_inverse_ceiling"] = {
        "action_500": mean_at(500, "pseudo_inverse", "action_accuracy"),
        "termination_balanced_500": mean_at(500, "pseudo_inverse", "termination_balanced"),
        "temporal_order_500": mean_at(500, "pseudo_inverse", "temporal_order"),
    }
    gate["raw_nn_baseline"] = {
        "action_500": mean_at(500, "raw_nn", "action_accuracy"),
        "termination_balanced_500": mean_at(500, "raw_nn", "termination_balanced"),
        "temporal_order_500": mean_at(500, "raw_nn", "temporal_order"),
    }
    gate["v2_ready"] = any(
        gate[arm]["ready"] for arm in ("E1_dense_sign", "E2_wide_sparse", "E3_dense_cont")
    )
    summary = {
        "experiment": "情景记忆 v2：编码器对照",
        "config": {
            "policy_noise": NOISE,
            "episodes": episodes,
            "capacities": list(capacities),
            "seeds": seeds,
            "encoders": ENCODERS,
        },
        "per_seed": rows,
        "gate": gate,
        "repro": collect(
            "情景记忆 v2",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "四编码：E0 参考、E1 dense-sign、E2 wide-sparse、E3 dense-cont",
                "读出固定 one-shot；伪逆/字面 NN 单列，不参与 ready",
                "数据/噪声/容量/判据与 v0/v1 相同",
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
    parser = argparse.ArgumentParser(description="情景记忆 v2")
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

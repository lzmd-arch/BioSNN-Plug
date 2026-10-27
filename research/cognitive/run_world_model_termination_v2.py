"""P3 终止头 v2：多步想象状态训练（预注册见 sweep_results/world_model_termination_v2_prereg.md）。"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np

from research.cognitive.run_world_model_closed_loop import (
    DIVERGENCE_NMSE,
    HORIZON,
    MIN_DIVERGENCE_STEP,
    NMSE_RATIO_MAX,
    POLICY_NOISE,
    TERMINATION_AUROC_MIN,
    TERMINATION_RATE_MIN,
    _at,
    _flatten,
    closed_loop,
    collect_episodes,
    policy_action,
)
from research.cognitive.termination_head import TerminationHead
from research.cognitive.world_model import CopyLastModel, RidgeActionModel, auroc
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "world_model_termination_v2"

IMAGINED_HORIZON = 200
BOUNDS = np.array([2.4, 3.0, 0.2095, 3.0], dtype=np.float64)


def _imagined_data(
    model: RidgeActionModel,
    episodes: list,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Model rollout states; labels are the real env's terminated on the same actions."""
    rng = np.random.default_rng(seed)
    states, actions, labels = [], [], []
    env = gymnasium.make("CartPole-v1")
    try:
        for episode in episodes:
            env.reset(seed=episode.seed)
            predicted = episode.states[0].copy()
            for _ in range(IMAGINED_HORIZON):
                action = policy_action(predicted, rng, POLICY_NOISE)
                _next, _reward, term, trunc, _ = env.step(action)
                prediction = model.predict(predicted[None, :], np.array([action]))
                predicted = predicted + prediction.delta[0]
                if not np.all(np.isfinite(predicted)):
                    break
                states.append(predicted.copy())
                actions.append(action)
                labels.append(bool(term))
                if term or trunc:
                    break
    finally:
        env.close()
    return np.stack(states), np.asarray(actions, dtype=np.int64), np.asarray(labels, dtype=bool)


def run(
    seeds: list[int],
    *,
    episodes: int,
    test_episodes: int,
    horizon: int,
    out: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        train = collect_episodes(episodes, seed=seed)
        test = collect_episodes(test_episodes, seed=seed + 10_000)
        states, actions, next_states, terminated = _flatten(train)
        scale = np.var(np.concatenate([states, next_states]), axis=0)

        model = RidgeActionModel(quadratic=True)
        model.fit(states, actions, next_states, terminated)
        baseline = CopyLastModel()
        baseline.fit(states, actions, next_states, terminated)

        imagined_states, imagined_actions, imagined_labels = _imagined_data(
            model, train, seed=seed + 300_000
        )
        head = TerminationHead()
        head.fit(imagined_states, imagined_actions, imagined_labels)

        loops = [
            closed_loop(
                model,
                ep,
                scale=scale,
                horizon=horizon,
                policy_seed=seed * 1000 + index,
                termination_head=head,
            )
            for index, ep in enumerate(test)
        ]
        nmse = [_at(loop.nmse, 10) for loop in loops]
        nmse = [value for value in nmse if value is not None]
        scores = [score for loop in loops for score in loop.termination_scores]
        labels = [label for loop in loops for label in loop.termination_labels]
        episode_rate = float(np.mean([any(loop.termination_labels) for loop in loops]))
        step_rate = float(np.mean(labels)) if labels else 0.0
        term_auroc = (
            auroc(np.asarray(scores), np.asarray(labels)) if labels and any(labels) else None
        )
        baseline_loops = [
            closed_loop(
                baseline,
                ep,
                scale=scale,
                horizon=horizon,
                policy_seed=seed * 1000 + index,
            )
            for index, ep in enumerate(test)
        ]
        copy_nmse = [_at(loop.nmse, 10) for loop in baseline_loops]
        copy_nmse = [value for value in copy_nmse if value is not None]
        divergence = statistics.median(
            [
                next(
                    (index for index, value in enumerate(loop.nmse) if value > DIVERGENCE_NMSE),
                    horizon,
                )
                for loop in loops
            ]
        )
        ratio = float(np.median(nmse) / np.median(copy_nmse)) if nmse and copy_nmse else None
        rows.append(
            {
                "seed": seed,
                "nmse_h10": float(np.median(nmse)) if nmse else None,
                "copy_last_h10": float(np.median(copy_nmse)) if copy_nmse else None,
                "ratio": ratio,
                "termination_episode_rate": episode_rate,
                "termination_step_rate": step_rate,
                "termination_auroc": term_auroc,
                "divergence_step": float(divergence),
                "imagined_states": int(imagined_states.shape[0]),
                "imagined_positives": int(np.sum(imagined_labels)),
            }
        )
        print(
            f"seed {seed}: NMSE {rows[-1]['nmse_h10']} ratio {ratio} "
            f"term {episode_rate:.3f}/{step_rate:.4f} AUROC {term_auroc} "
            f"drift {divergence} imagined {rows[-1]['imagined_states']} "
            f"pos {rows[-1]['imagined_positives']}",
            flush=True,
        )

    def mean(key: str) -> float | None:
        values = [row[key] for row in rows if row[key] is not None]
        return float(np.mean(values)) if values else None

    ratio = mean("ratio")
    term_auroc = mean("termination_auroc")
    term_rate = mean("termination_episode_rate")
    divergence = mean("divergence_step")
    gate = {
        "ratio": ratio,
        "termination_episode_rate": term_rate,
        "termination_auroc": term_auroc,
        "divergence_step": divergence,
        "ratio_ok": ratio is not None and ratio <= NMSE_RATIO_MAX,
        "termination_ok": (
            term_auroc is not None
            and term_auroc >= TERMINATION_AUROC_MIN
            and term_rate is not None
            and term_rate >= TERMINATION_RATE_MIN
        ),
        "divergence_ok": divergence is not None and divergence >= MIN_DIVERGENCE_STEP,
    }
    gate["p3_v2_ready"] = gate["ratio_ok"] and gate["termination_ok"] and gate["divergence_ok"]
    summary = {
        "experiment": "P3 终止头 v2：多步想象状态训练",
        "config": {
            "seeds": seeds,
            "episodes": episodes,
            "test_episodes": test_episodes,
            "horizon": horizon,
            "imagined_horizon": IMAGINED_HORIZON,
            "bounds": BOUNDS.tolist(),
        },
        "per_seed": rows,
        "gate": gate,
        "repro": collect(
            "P3 终止头 v2",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "头训练于模型多步 rollout 状态；标签为真实环境在同一动作序列上的 terminated",
                "闭环 H=500；不调状态模型/阈值",
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
    parser = argparse.ArgumentParser(description="P3 终止头 v2")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    run(
        parse_seeds(args.seeds),
        episodes=args.episodes,
        test_episodes=args.test_episodes,
        horizon=args.horizon,
        out=args.out,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

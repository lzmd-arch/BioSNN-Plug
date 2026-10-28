"""P3 终止头 v3：固定动作序列的开环想象（预注册见 sweep_results/world_model_termination_v3_prereg.md）。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np

from research.cognitive.run_world_model_closed_loop import (
    POLICY_NOISE,
    _flatten,
    collect_episodes,
    policy_action,
)
from research.cognitive.run_world_model_termination_v2 import BOUNDS
from research.cognitive.world_model import RidgeActionModel, auroc
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "world_model_termination_v3"

HORIZON = 200
TERMINATION_AUROC_MIN = 0.75
TERMINATION_RATE_MIN = 0.01
TERMINAL_ERROR_MAX = 0.01


def _margin(next_state: np.ndarray) -> float:
    return float(max(abs(next_state[0]) / BOUNDS[0], abs(next_state[2]) / BOUNDS[2]))


def _open_loop(
    model: RidgeActionModel,
    episode,
    *,
    sequence: np.ndarray,
    horizon: int,
) -> tuple[list[float], list[bool], list[float]]:
    """固定动作序列：模型与真实环境吃同一串动作，动作不随想象状态反馈。"""
    env = gymnasium.make("CartPole-v1")
    env.reset(seed=episode.seed)
    predicted = episode.states[0].copy()
    margins: list[float] = []
    labels: list[bool] = []
    errors: list[float] = []
    try:
        for action in sequence[:horizon]:
            action = int(action)
            nxt, _reward, term, trunc, _ = env.step(action)
            prediction = model.predict(predicted[None, :], np.array([action]))
            predicted_next = predicted + prediction.delta[0]
            margins.append(_margin(predicted_next))
            labels.append(bool(term))
            errors.append(float(np.max(np.abs(predicted_next - nxt))))
            if term or trunc:
                break
            predicted = predicted_next
    finally:
        env.close()
    return margins, labels, errors


def _closed_loop(
    model: RidgeActionModel,
    episode,
    *,
    policy_seed: int,
    horizon: int,
) -> tuple[list[float], list[bool]]:
    """v2 口径的动作反馈闭环，只用于边界对照。"""
    env = gymnasium.make("CartPole-v1")
    env.reset(seed=episode.seed)
    rng = np.random.default_rng(policy_seed)
    predicted = episode.states[0].copy()
    margins: list[float] = []
    labels: list[bool] = []
    try:
        for _ in range(horizon):
            action = policy_action(predicted, rng, POLICY_NOISE)
            _nxt, _reward, term, trunc, _ = env.step(action)
            prediction = model.predict(predicted[None, :], np.array([action]))
            predicted_next = predicted + prediction.delta[0]
            margins.append(_margin(predicted_next))
            labels.append(bool(term))
            if term or trunc:
                break
            predicted = predicted_next
    finally:
        env.close()
    return margins, labels


def run(
    seeds: list[int],
    *,
    episodes: int,
    test_episodes: int,
    out: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        train = collect_episodes(episodes, seed=seed)
        test = collect_episodes(test_episodes, seed=seed + 10_000)
        states, actions, next_states, terminated = _flatten(train)
        model = RidgeActionModel(quadratic=True)
        model.fit(states, actions, next_states, terminated)

        open_margins: list[float] = []
        open_labels: list[bool] = []
        terminal_errors: list[float] = []
        episode_terms = 0
        for index, episode in enumerate(test):
            rng = np.random.default_rng(seed + 900_000 + index)
            sequence = rng.integers(0, 2, size=HORIZON)
            margins, labels, errors = _open_loop(model, episode, sequence=sequence, horizon=HORIZON)
            open_margins.extend(margins)
            open_labels.extend(labels)
            if any(labels):
                episode_terms += 1
                terminal_errors.append(errors[int(np.argmax(labels))])

        closed_margins: list[float] = []
        closed_labels: list[bool] = []
        for index, episode in enumerate(test):
            margins, labels = _closed_loop(
                model, episode, policy_seed=seed * 1000 + index, horizon=500
            )
            closed_margins.extend(margins)
            closed_labels.extend(labels)

        open_labels_array = np.asarray(open_labels)
        margins_array = np.asarray(open_margins)
        row = {
            "seed": seed,
            "open_auroc": float(auroc(margins_array, open_labels_array))
            if open_labels_array.any()
            else None,
            "open_episode_rate": episode_terms / len(test),
            "open_step_rate": float(np.mean(open_labels_array)),
            "terminal_error_median": float(np.median(terminal_errors)) if terminal_errors else None,
            "margin_positive_median": float(np.median(margins_array[open_labels_array]))
            if open_labels_array.any()
            else None,
            "margin_negative_max": float(np.max(margins_array[~open_labels_array])),
            "closed_auroc": float(auroc(np.asarray(closed_margins), np.asarray(closed_labels)))
            if any(closed_labels)
            else None,
        }
        rows.append(row)
        print(
            f"seed {seed}: open AUROC {row['open_auroc']} rate {row['open_episode_rate']:.3f} "
            f"term-err {row['terminal_error_median']} | margin +{row['margin_positive_median']} "
            f"/ -{row['margin_negative_max']} | closed {row['closed_auroc']}",
            flush=True,
        )

    def mean(key: str) -> float | None:
        values = [row[key] for row in rows if row[key] is not None]
        return float(np.mean(values)) if values else None

    values = {
        "open_auroc": mean("open_auroc"),
        "open_episode_rate": mean("open_episode_rate"),
        "terminal_error_median": mean("terminal_error_median"),
        "margin_positive_median": mean("margin_positive_median"),
        "margin_negative_max": mean("margin_negative_max"),
        "closed_auroc": mean("closed_auroc"),
    }
    gate = {
        **values,
        "auroc_ok": values["open_auroc"] is not None
        and values["open_auroc"] >= TERMINATION_AUROC_MIN,
        "rate_ok": values["open_episode_rate"] is not None
        and values["open_episode_rate"] >= TERMINATION_RATE_MIN,
        "terminal_error_ok": values["terminal_error_median"] is not None
        and values["terminal_error_median"] <= TERMINAL_ERROR_MAX,
        "margin_separation_ok": values["margin_positive_median"] is not None
        and values["margin_positive_median"] > 1.0
        and values["margin_negative_max"] < 1.0,
    }
    gate["p3_v3_ready"] = all(
        gate[key] for key in ("auroc_ok", "rate_ok", "terminal_error_ok", "margin_separation_ok")
    )
    summary = {
        "experiment": "P3 终止头 v3：固定动作序列的开环想象",
        "config": {
            "seeds": seeds,
            "episodes": episodes,
            "test_episodes": test_episodes,
            "horizon": HORIZON,
            "bounds": BOUNDS.tolist(),
        },
        "per_seed": rows,
        "gate": gate,
        "repro": collect(
            "P3 终止头 v3",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "开环：每回合固定随机动作序列，模型/环境共用；分数=预测下一状态的边界余量",
                "闭环：v2 口径动作反馈，仅作边界对照，不参与 p3_v3_ready",
                "终止界用 CartPole 已知界；不是时间/步数特征",
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
    parser = argparse.ArgumentParser(description="P3 终止头 v3：开环想象")
    parser.add_argument("--seeds", default="5-9")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    run(
        parse_seeds(args.seeds),
        episodes=20 if args.smoke else args.episodes,
        test_episodes=5 if args.smoke else args.test_episodes,
        out=args.out,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

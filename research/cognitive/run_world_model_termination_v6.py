"""P3 终止头 v6：滚动时域闭环（预注册见 sweep_results/world_model_termination_v6_prereg.md）。"""

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
from research.cognitive.run_world_model_termination_v3 import _closed_loop
from research.cognitive.run_world_model_termination_v5 import (
    _choose_threshold,
    _operating_point,
)
from research.cognitive.world_model import RidgeActionModel, auroc
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "world_model_termination_v6"

HORIZON = 500
BOUNDS = np.array([2.4, 3.0, 0.2095, 3.0])
AUROC_MIN = 0.75
RATE_MIN = 0.01
ERROR_MAX = 0.01
FALSE_POSITIVE_MAX = 1e-3
HIT_RATE_MIN = 0.99


def _margin(next_state: np.ndarray) -> float:
    return float(max(abs(next_state[0]) / BOUNDS[0], abs(next_state[2]) / BOUNDS[2]))


def _receding(
    model: RidgeActionModel,
    episodes,
    *,
    seed_offset: int,
) -> tuple[np.ndarray, np.ndarray, list[float], int]:
    """每步用真实观测出动作，模型只预测下一状态的风险。"""
    margins: list[float] = []
    labels: list[bool] = []
    terminal_errors: list[float] = []
    episode_terms = 0
    for index, episode in enumerate(episodes):
        env = gymnasium.make("CartPole-v1")
        env.reset(seed=episode.seed)
        rng = np.random.default_rng(seed_offset + index)
        current_terms = False
        try:
            for _ in range(HORIZON):
                real = np.asarray(env.unwrapped.state, dtype=np.float64)
                action = policy_action(real, rng, POLICY_NOISE)
                nxt, _reward, term, trunc, _ = env.step(action)
                prediction = model.predict(real[None, :], np.array([action]))
                predicted_next = real + prediction.delta[0]
                margins.append(_margin(predicted_next))
                labels.append(bool(term))
                if term:
                    current_terms = True
                    terminal_errors.append(float(np.max(np.abs(predicted_next - nxt))))
                if term or trunc:
                    break
        finally:
            env.close()
        if current_terms:
            episode_terms += 1
    return np.asarray(margins), np.asarray(labels), terminal_errors, episode_terms


def run(
    seeds: list[int],
    *,
    episodes: int,
    calibration_episodes: int,
    test_episodes: int,
    out: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        train = collect_episodes(episodes, seed=seed)
        calibration = collect_episodes(calibration_episodes, seed=seed + 30_000)
        test = collect_episodes(test_episodes, seed=seed + 10_000)
        states, actions, next_states, terminated = _flatten(train)
        model = RidgeActionModel(quadratic=True)
        model.fit(states, actions, next_states, terminated)

        cal_margins, cal_labels, _, _ = _receding(model, calibration, seed_offset=seed + 800_000)
        threshold = _choose_threshold(cal_margins, cal_labels)
        cal_point = _operating_point(cal_margins, cal_labels, threshold)
        margins, labels, terminal_errors, episode_terms = _receding(
            model, test, seed_offset=seed + 900_000
        )
        point = _operating_point(margins, labels, threshold)

        imagined_margins: list[float] = []
        imagined_labels: list[bool] = []
        for index, episode in enumerate(test):
            current_margins, current_labels = _closed_loop(
                model, episode, policy_seed=seed * 1000 + index, horizon=HORIZON
            )
            imagined_margins.extend(current_margins)
            imagined_labels.extend(current_labels)

        row = {
            "seed": seed,
            "threshold": threshold,
            "calibration_false_positive": cal_point["false_positive_rate"],
            "calibration_hit": cal_point["hit_rate"],
            "auroc": float(auroc(margins, labels)) if labels.any() else None,
            "episode_rate": episode_terms / len(test),
            "terminal_error_median": float(np.median(terminal_errors)) if terminal_errors else None,
            "false_positive_rate": point["false_positive_rate"],
            "hit_rate": point["hit_rate"],
            "imagined_closed_loop_auroc": float(
                auroc(np.asarray(imagined_margins), np.asarray(imagined_labels))
            )
            if any(imagined_labels)
            else None,
        }
        rows.append(row)
        print(
            f"seed {seed}: tau {threshold:.4f} | AUROC {row['auroc']} "
            f"rate {row['episode_rate']:.3f} err {row['terminal_error_median']} "
            f"FP {row['false_positive_rate']:.2e} hit {row['hit_rate']:.3f} "
            f"| imagined {row['imagined_closed_loop_auroc']}",
            flush=True,
        )

    def mean(key: str) -> float | None:
        values = [row[key] for row in rows if row[key] is not None]
        return float(np.mean(values)) if values else None

    values = {
        "auroc": mean("auroc"),
        "episode_rate": mean("episode_rate"),
        "terminal_error_median": mean("terminal_error_median"),
        "false_positive_rate": mean("false_positive_rate"),
        "hit_rate": mean("hit_rate"),
        "imagined_closed_loop_auroc": mean("imagined_closed_loop_auroc"),
    }
    gate = {
        **values,
        "auroc_ok": values["auroc"] is not None and values["auroc"] >= AUROC_MIN,
        "rate_ok": values["episode_rate"] is not None and values["episode_rate"] >= RATE_MIN,
        "terminal_error_ok": values["terminal_error_median"] is not None
        and values["terminal_error_median"] <= ERROR_MAX,
        "false_positive_ok": values["false_positive_rate"] is not None
        and values["false_positive_rate"] <= FALSE_POSITIVE_MAX,
        "hit_rate_ok": values["hit_rate"] is not None and values["hit_rate"] >= HIT_RATE_MIN,
    }
    gate["p3_v6_ready"] = all(
        gate[key]
        for key in ("auroc_ok", "rate_ok", "terminal_error_ok", "false_positive_ok", "hit_rate_ok")
    )
    summary = {
        "experiment": "P3 终止头 v6：滚动时域闭环",
        "config": {
            "seeds": seeds,
            "episodes": episodes,
            "calibration_episodes": calibration_episodes,
            "test_episodes": test_episodes,
            "horizon": HORIZON,
        },
        "per_seed": rows,
        "gate": gate,
        "repro": collect(
            "P3 终止头 v6",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "滚动时域：动作来自真实观测，模型只预测一步；不是 500 步纯想象",
                "阈值只在 calibration（seed+30000）上选；测试用种子 20-24",
                "纯想象闭环 AUROC 仅作边界对照，不参与 p3_v6_ready",
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
    parser = argparse.ArgumentParser(description="P3 终止头 v6：滚动时域闭环")
    parser.add_argument("--seeds", default="20-24")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--calibration-episodes", type=int, default=100)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    run(
        parse_seeds(args.seeds),
        episodes=20 if args.smoke else args.episodes,
        calibration_episodes=5 if args.smoke else args.calibration_episodes,
        test_episodes=5 if args.smoke else args.test_episodes,
        out=args.out,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

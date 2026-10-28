"""P3 终止头 v5：校准集阈值（预注册见 sweep_results/world_model_termination_v5_prereg.md）。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.cognitive.run_world_model_closed_loop import _flatten, collect_episodes
from research.cognitive.run_world_model_termination_v3 import (
    HORIZON,
    _closed_loop,
    _open_loop,
)
from research.cognitive.world_model import RidgeActionModel, auroc
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "world_model_termination_v5"

AUROC_MIN = 0.75
RATE_MIN = 0.01
ERROR_MAX = 0.01
FALSE_POSITIVE_MAX = 1e-3
HIT_RATE_MIN = 0.99


def _rollout(
    model: RidgeActionModel,
    episodes,
    *,
    seed_offset: int,
) -> tuple[np.ndarray, np.ndarray, list[float], int]:
    margins: list[float] = []
    labels: list[bool] = []
    terminal_errors: list[float] = []
    episode_terms = 0
    for index, episode in enumerate(episodes):
        rng = np.random.default_rng(seed_offset + index)
        sequence = rng.integers(0, 2, size=HORIZON)
        current, current_labels, errors = _open_loop(
            model, episode, sequence=sequence, horizon=HORIZON
        )
        margins.extend(current)
        labels.extend(current_labels)
        if any(current_labels):
            episode_terms += 1
            terminal_errors.append(errors[int(np.argmax(current_labels))])
    return (
        np.asarray(margins),
        np.asarray(labels),
        terminal_errors,
        episode_terms,
    )


def _choose_threshold(margins: np.ndarray, labels: np.ndarray) -> float:
    """最小 τ 使校准 FP≤1e-3；命中随 τ 递减，所以它同时最大化可行命中。"""
    if not labels.any() or labels.all():
        return float(margins.max() + 1e-6)
    negative = ~labels
    for threshold in np.unique(margins):
        if float(np.mean(margins[negative] >= threshold)) <= FALSE_POSITIVE_MAX:
            return float(threshold)
    return float(margins.max() + 1e-6)


def _operating_point(margins: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = margins >= threshold
    negative = ~labels
    positive = labels
    return {
        "false_positive_rate": float(np.mean(predicted[negative])) if np.any(negative) else 0.0,
        "hit_rate": float(np.mean(predicted[positive])) if np.any(positive) else 0.0,
    }


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

        cal_margins, cal_labels, _, _ = _rollout(model, calibration, seed_offset=seed + 700_000)
        threshold = _choose_threshold(cal_margins, cal_labels)
        cal_point = _operating_point(cal_margins, cal_labels, threshold)

        margins, labels, terminal_errors, episode_terms = _rollout(
            model, test, seed_offset=seed + 900_000
        )
        point = _operating_point(margins, labels, threshold)

        closed_margins: list[float] = []
        closed_labels: list[bool] = []
        for index, episode in enumerate(test):
            current_margins, current_labels = _closed_loop(
                model, episode, policy_seed=seed * 1000 + index, horizon=500
            )
            closed_margins.extend(current_margins)
            closed_labels.extend(current_labels)

        row = {
            "seed": seed,
            "threshold": threshold,
            "calibration_false_positive": cal_point["false_positive_rate"],
            "calibration_hit": cal_point["hit_rate"],
            "open_auroc": float(auroc(margins, labels)) if labels.any() else None,
            "open_episode_rate": episode_terms / len(test),
            "terminal_error_median": float(np.median(terminal_errors)) if terminal_errors else None,
            "false_positive_rate": point["false_positive_rate"],
            "hit_rate": point["hit_rate"],
            "closed_auroc": float(auroc(np.asarray(closed_margins), np.asarray(closed_labels)))
            if any(closed_labels)
            else None,
        }
        rows.append(row)
        print(
            f"seed {seed}: tau {threshold:.4f} (cal FP {row['calibration_false_positive']:.2e} "
            f"hit {row['calibration_hit']:.3f}) | test AUROC {row['open_auroc']} "
            f"FP {row['false_positive_rate']:.2e} hit {row['hit_rate']:.3f} "
            f"| closed {row['closed_auroc']}",
            flush=True,
        )

    def mean(key: str) -> float | None:
        values = [row[key] for row in rows if row[key] is not None]
        return float(np.mean(values)) if values else None

    values = {
        "open_auroc": mean("open_auroc"),
        "open_episode_rate": mean("open_episode_rate"),
        "terminal_error_median": mean("terminal_error_median"),
        "false_positive_rate": mean("false_positive_rate"),
        "hit_rate": mean("hit_rate"),
        "closed_auroc": mean("closed_auroc"),
    }
    gate = {
        **values,
        "auroc_ok": values["open_auroc"] is not None and values["open_auroc"] >= AUROC_MIN,
        "rate_ok": values["open_episode_rate"] is not None
        and values["open_episode_rate"] >= RATE_MIN,
        "terminal_error_ok": values["terminal_error_median"] is not None
        and values["terminal_error_median"] <= ERROR_MAX,
        "false_positive_ok": values["false_positive_rate"] is not None
        and values["false_positive_rate"] <= FALSE_POSITIVE_MAX,
        "hit_rate_ok": values["hit_rate"] is not None and values["hit_rate"] >= HIT_RATE_MIN,
    }
    gate["p3_v5_ready"] = all(
        gate[key]
        for key in ("auroc_ok", "rate_ok", "terminal_error_ok", "false_positive_ok", "hit_rate_ok")
    )
    summary = {
        "experiment": "P3 终止头 v5：校准集阈值",
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
            "P3 终止头 v5",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "开环固定动作序列；阈值只在 calibration（seed+30000）上选，测试不参与",
                "目标：校准 FP<=1e-3 的最小 tau；测试用同一 tau",
                "闭环 v2 口径仅作边界对照，不参与 p3_v5_ready",
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
    parser = argparse.ArgumentParser(description="P3 终止头 v5：校准集阈值")
    parser.add_argument("--seeds", default="15-19")
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

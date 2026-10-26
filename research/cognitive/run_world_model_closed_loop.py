"""世界模型 P3：CartPole 闭环想象（预注册见 ``sweep_results/world_model_closed_loop_prereg.md``）。

行为策略是**平衡启发式 + 动作噪声**，只用于采集与驱动闭环，不代表项目的学习主张。
主指标：闭环 H=10 的 NMSE 相对 `copy_last` 闭环、终止 AUROC、首次发散步。
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np

from research.cognitive.termination_head import TerminationHead
from research.cognitive.world_model import CopyLastModel, RidgeActionModel, auroc
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "world_model_closed_loop"

POLICY_NOISE = 0.1
HORIZON = 500
NMSE_RATIO_MAX = 0.90
TERMINATION_AUROC_MIN = 0.75
TERMINATION_RATE_MIN = 0.01
MIN_DIVERGENCE_STEP = 10
DIVERGENCE_NMSE = 0.5


@dataclass(frozen=True)
class Episode:
    seed: int
    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray
    terminated: np.ndarray


def policy_action(state: np.ndarray, rng: np.random.Generator, noise: float) -> int:
    """平衡启发式；以 ``noise`` 概率换成随机动作。"""
    action = 1 if float(state[2]) + 0.5 * float(state[3]) > 0 else 0
    if rng.random() < noise:
        action = int(rng.integers(2))
    return int(action)


def collect_episodes(
    episodes: int, *, seed: int, noise: float = POLICY_NOISE, env_name: str = "CartPole-v1"
) -> list[Episode]:
    rng = np.random.default_rng(seed)
    env = gymnasium.make(env_name)
    collected: list[Episode] = []
    try:
        for _ in range(episodes):
            env_seed = int(rng.integers(0, 2**31 - 1))
            state, _ = env.reset(seed=env_seed)
            states: list[np.ndarray] = []
            actions: list[int] = []
            next_states: list[np.ndarray] = []
            terminated: list[bool] = []
            done = False
            while not done:
                action = policy_action(np.asarray(state), rng, noise)
                nxt, _reward, term, trunc, _ = env.step(action)
                states.append(np.asarray(state, dtype=np.float64))
                actions.append(action)
                next_states.append(np.asarray(nxt, dtype=np.float64))
                terminated.append(bool(term))
                state = nxt
                done = bool(term or trunc)
            collected.append(
                Episode(
                    seed=env_seed,
                    states=np.stack(states),
                    actions=np.asarray(actions, dtype=np.int64),
                    next_states=np.stack(next_states),
                    terminated=np.asarray(terminated, dtype=bool),
                )
            )
    finally:
        env.close()
    return collected


def _flatten(episodes: list[Episode]) -> tuple[np.ndarray, ...]:
    states = np.concatenate([ep.states for ep in episodes])
    actions = np.concatenate([ep.actions for ep in episodes])
    next_states = np.concatenate([ep.next_states for ep in episodes])
    terminated = np.concatenate([ep.terminated for ep in episodes])
    return states, actions, next_states, terminated


def _step_nmse(predicted: np.ndarray, true: np.ndarray, scale: np.ndarray) -> float:
    return float(np.mean((predicted - true) ** 2 / np.maximum(scale, 1e-12)))


@dataclass
class ClosedLoop:
    nmse: list[float]
    termination_scores: list[float]
    termination_labels: list[bool]


def closed_loop(
    model,
    episode: Episode,
    *,
    scale: np.ndarray,
    horizon: int,
    policy_seed: int,
    termination_head: TerminationHead | None = None,
) -> ClosedLoop:
    """从测试回合的初始状态开始；策略在**模型预测的状态**上选动作，动作同时喂给两边。"""
    env = gymnasium.make("CartPole-v1")
    env.reset(seed=episode.seed)
    rng = np.random.default_rng(policy_seed)
    predicted = episode.states[0].copy()
    nmse: list[float] = []
    scores: list[float] = []
    labels: list[bool] = []
    try:
        for _ in range(horizon):
            action = policy_action(predicted, rng, POLICY_NOISE)
            nxt, _reward, term, trunc, _ = env.step(action)
            true_next = np.asarray(nxt, dtype=np.float64)
            prediction = model.predict(predicted[None, :], np.array([action]))
            predicted_next = predicted + prediction.delta[0]
            nmse.append(_step_nmse(predicted_next, true_next, scale))
            if termination_head is None:
                scores.append(float(prediction.termination[0]))
            else:
                scores.append(
                    float(termination_head.predict_proba(predicted[None, :], np.array([action]))[0])
                )
            labels.append(bool(term))
            if term or trunc:
                break
            predicted = predicted_next
    finally:
        env.close()
    return ClosedLoop(nmse=nmse, termination_scores=scores, termination_labels=labels)


def open_loop(model, episode: Episode, *, scale: np.ndarray, horizon: int) -> list[float]:
    """教师强制：每一步都从真实状态出发；只量一步误差随回合内位置的变化。"""
    count = min(horizon, episode.states.shape[0])
    nmse: list[float] = []
    for step in range(count):
        state = episode.states[step][None, :]
        action = np.array([episode.actions[step]])
        prediction = model.predict(state, action)
        predicted = state[0] + prediction.delta[0]
        nmse.append(_step_nmse(predicted, episode.next_states[step], scale))
    return nmse


def _at(values: list[float], index: int) -> float | None:
    return values[index] if len(values) > index else None


def run(
    seeds: list[int],
    *,
    episodes: int,
    test_episodes: int,
    horizon: int,
    out: Path,
    termination_model: str = "ridge",
) -> dict:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        train = collect_episodes(episodes, seed=seed)
        test = collect_episodes(test_episodes, seed=seed + 10_000)
        states, actions, next_states, terminated = _flatten(train)
        scale = np.var(np.concatenate([states, next_states]), axis=0)

        model = RidgeActionModel(quadratic=True)
        model.fit(states, actions, next_states, terminated)
        termination_head = None
        if termination_model == "dedicated":
            termination_head = TerminationHead()
            termination_head.fit(states, actions, terminated)
        elif termination_model != "ridge":
            raise ValueError(
                f"termination_model 只能是 'ridge' 或 'dedicated'，收到 {termination_model!r}。"
            )
        baseline = CopyLastModel()
        baseline.fit(states, actions, next_states, terminated)

        loops = [
            closed_loop(
                model,
                ep,
                scale=scale,
                horizon=horizon,
                policy_seed=seed * 1000 + i,
                termination_head=termination_head,
            )
            for i, ep in enumerate(test)
        ]
        base_loops = [
            closed_loop(baseline, ep, scale=scale, horizon=horizon, policy_seed=seed * 1000 + i)
            for i, ep in enumerate(test)
        ]
        open_10 = [_at(open_loop(model, ep, scale=scale, horizon=10), 9) for ep in test]
        closed_10 = [_at(loop.nmse, 9) for loop in loops]
        base_10 = [_at(loop.nmse, 9) for loop in base_loops]
        valid = [
            (c, b, o)
            for c, b, o in zip(closed_10, base_10, open_10, strict=True)
            if c is not None and b is not None and o is not None
        ]
        closed_mean = float(np.mean([c for c, _, _ in valid])) if valid else float("nan")
        base_mean = float(np.mean([b for _, b, _ in valid])) if valid else float("nan")
        open_mean = float(np.mean([o for _, _, o in valid])) if valid else float("nan")
        scores = [s for loop in loops for s in loop.termination_scores]
        labels = [lab for loop in loops for lab in loop.termination_labels]
        step_rate = float(np.mean(labels)) if labels else 0.0
        episode_rate = float(np.mean([any(loop.termination_labels) for loop in loops]))
        term_auroc = (
            auroc(np.asarray(scores), np.asarray(labels, dtype=int))
            if episode_rate >= TERMINATION_RATE_MIN and len(set(labels)) > 1
            else None
        )
        divergences = []
        for loop in loops:
            hits = [i + 1 for i, value in enumerate(loop.nmse) if value > DIVERGENCE_NMSE]
            divergences.append(hits[0] if hits else len(loop.nmse) + 1)
        rows.append(
            {
                "seed": seed,
                "closed_nmse_h10": closed_mean,
                "copy_last_nmse_h10": base_mean,
                "open_loop_nmse_h10": open_mean,
                "ratio_vs_copy": closed_mean / base_mean if base_mean else None,
                "termination_rate": episode_rate,
                "termination_step_rate": step_rate,
                "termination_auroc": term_auroc,
                "divergence_step_median": statistics.median(divergences),
                "valid_episodes": len(valid),
                "test_episodes": len(test),
            }
        )
        print(
            f"seed {seed}: closed {closed_mean:.6g} / copy {base_mean:.6g} / open {open_mean:.6g} | "
            f"term ep {episode_rate:.3f} / step {step_rate:.4f} AUROC {term_auroc} | "
            f"diverge {rows[-1]['divergence_step_median']}",
            flush=True,
        )

    ratio = float(np.nanmean([row["ratio_vs_copy"] for row in rows]))
    aurocs = [row["termination_auroc"] for row in rows if row["termination_auroc"] is not None]
    rates = [row["termination_rate"] for row in rows]
    divergences = [row["divergence_step_median"] for row in rows]
    gate = {
        "ratio_vs_copy": ratio,
        "ratio_ok": ratio <= NMSE_RATIO_MAX,
        "termination_rate": float(np.mean(rates)),
        "termination_auroc": float(np.mean(aurocs)) if aurocs else None,
        "termination_ok": bool(aurocs)
        and float(np.mean(aurocs)) >= TERMINATION_AUROC_MIN
        and float(np.mean(rates)) >= TERMINATION_RATE_MIN,
        "divergence_median": float(np.median(divergences)),
        "divergence_ok": float(np.median(divergences)) >= MIN_DIVERGENCE_STEP,
    }
    gate["p3_ready"] = gate["ratio_ok"] and gate["termination_ok"] and gate["divergence_ok"]
    summary = {
        "experiment": "世界模型 P3：CartPole 闭环想象",
        "config": {
            "policy_noise": POLICY_NOISE,
            "horizon": horizon,
            "train_episodes": episodes,
            "test_episodes": test_episodes,
            "seeds": seeds,
            "termination_model": termination_model,
        },
        "per_seed": rows,
        "gate": gate,
        "repro": collect(
            "世界模型 P3：CartPole 闭环想象",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "行为策略：θ+0.5θ̇ 平衡启发式 + ε=0.1 动作噪声",
                "闭环：策略在模型预测状态上选动作，动作同时喂给模型与真实环境",
                "归一化用训练集逐维方差",
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
    parser = argparse.ArgumentParser(description="世界模型 P3：CartPole 闭环想象")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--termination", choices=("ridge", "dedicated"), default="ridge")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    run(
        parse_seeds(args.seeds),
        episodes=args.episodes,
        test_episodes=args.test_episodes,
        horizon=args.horizon,
        out=args.out,
        termination_model=args.termination,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

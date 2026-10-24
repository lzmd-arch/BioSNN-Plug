"""世界模型打底：采集 CartPole 一步转移，跑四个基线，按预注册判据报数。

**打底，不是阶段四。** 预注册见
``sweep_results/world_model_groundwork_prereg.md``：不训练神经世界模型、不做 rollout、
不做 PINN，只回答「一步动力学是否可测、动作条件化是否成立」。

跑法::

    uv run python -m research.cognitive.run_world_model --smoke
    uv run python -m research.cognitive.run_world_model --seeds 0-4

产物：``sweep_results/world_model_groundwork/summary.json``，含自证块。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from research.cognitive.world_model import (
    CopyLastModel,
    MeanDeltaModel,
    PerActionMeanModel,
    Prediction,
    RidgeActionModel,
    action_sensitivity,
    auroc,
    build_rollout_windows,
    per_dim_nmse,
    rollout,
    rollout_nmse,
    rollout_teacher_forcing,
)
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "world_model_groundwork"

#: 预注册的三条判据。
NMSE_RATIO_MAX = 0.90
ACTION_SENSITIVITY_MIN = 0.01
TERMINATION_AUROC_MIN = 0.75

MODELS = {
    "copy_last": CopyLastModel,
    "mean_delta": MeanDeltaModel,
    "per_action_mean": PerActionMeanModel,
    "ridge_action": RidgeActionModel,
    "ridge_quadratic": lambda: RidgeActionModel(quadratic=True),
}

__all__ = ["Transitions", "collect_transitions", "evaluate", "main"]


@dataclass(frozen=True)
class Transitions:
    """一段按回合采集的转移。``episode`` 用来按回合切分，防止相邻步泄漏。"""

    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray
    terminated: np.ndarray
    episode: np.ndarray

    def subset(self, mask: np.ndarray) -> Transitions:
        return Transitions(
            states=self.states[mask],
            actions=self.actions[mask],
            next_states=self.next_states[mask],
            terminated=self.terminated[mask],
            episode=self.episode[mask],
        )


def collect_transitions(episodes: int, *, seed: int, env_name: str = "CartPole-v1") -> Transitions:
    """用均匀随机动作采集 ``episodes`` 个回合。

    **按回合记 ``episode``**：训练/测试必须按回合切，按步切会让同一回合的相邻步
    同时出现在两边——那会把"记忆"当成"预测"。
    """
    if episodes <= 0:
        raise ValueError(f"episodes 必须为正，收到 {episodes!r}。")
    import gymnasium

    rng = np.random.default_rng(seed)
    env = gymnasium.make(env_name)
    states: list[np.ndarray] = []
    actions: list[int] = []
    next_states: list[np.ndarray] = []
    terminated: list[bool] = []
    episode_ids: list[int] = []
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            done = False
            while not done:
                action = int(rng.integers(env.action_space.n))
                nxt, _reward, term, trunc, _ = env.step(action)
                states.append(np.asarray(state, dtype=np.float64))
                actions.append(action)
                next_states.append(np.asarray(nxt, dtype=np.float64))
                terminated.append(bool(term))
                episode_ids.append(episode)
                state = nxt
                done = bool(term or trunc)
    finally:
        env.close()
    return Transitions(
        states=np.stack(states),
        actions=np.asarray(actions, dtype=np.int64),
        next_states=np.stack(next_states),
        terminated=np.asarray(terminated, dtype=bool),
        episode=np.asarray(episode_ids, dtype=np.int64),
    )


def evaluate(model: Any, train: Transitions, test: Transitions) -> dict[str, Any]:
    """在一个切分上拟合并评估。**分母只用训练集的方差。**"""
    model.fit(train.states, train.actions, train.next_states, train.terminated)
    prediction: Prediction = model.predict(test.states, test.actions)
    true_delta = test.next_states - test.states
    scale = train.next_states.var(axis=0)
    nmse_dims = per_dim_nmse(prediction.delta, true_delta, scale)
    return {
        "nmse": float(nmse_dims.mean()),
        "nmse_per_dim": [float(value) for value in nmse_dims],
        "termination_auroc": auroc(prediction.termination, test.terminated),
        "action_sensitivity": action_sensitivity(model, test.states, scale),
    }


def _rollout_curves(
    train: Transitions,
    test: Transitions,
    *,
    horizon: int,
    models: dict[str, Any],
) -> dict[str, Any]:
    """在测试回合上切窗口，报每个 horizon 的自由 rollout 与教师强制误差。"""
    start, actions, targets = build_rollout_windows(
        test.states, test.actions, test.next_states, test.episode, horizon
    )
    if start.shape[0] == 0:
        return {"horizon": horizon, "windows": 0, "curves": {}}
    scale = train.next_states.var(axis=0)
    curves: dict[str, Any] = {}
    for name, model in models.items():
        free = rollout(model, start, actions)
        teacher = rollout_teacher_forcing(model, start, actions, targets)
        curves[name] = {
            "free_nmse": rollout_nmse(free, targets, scale).tolist(),
            "teacher_nmse": rollout_nmse(teacher, targets, scale).tolist(),
        }
    return {"horizon": horizon, "windows": int(start.shape[0]), "curves": curves}


def _run_one_seed(
    seed: int,
    *,
    train_episodes: int,
    test_episodes: int,
    env_name: str,
    primary_model: str,
    rollout_horizon: int,
) -> dict[str, Any]:
    data = collect_transitions(train_episodes + test_episodes, seed=seed, env_name=env_name)
    train = data.subset(data.episode < train_episodes)
    test = data.subset(data.episode >= train_episodes)
    results = {name: evaluate(factory(), train, test) for name, factory in MODELS.items()}
    ridge = results[primary_model]
    baseline = results["copy_last"]
    ratio = ridge["nmse"] / baseline["nmse"] if baseline["nmse"] > 0 else float("inf")
    checks = {
        "nmse_ratio": bool(ratio <= NMSE_RATIO_MAX),
        "action_sensitivity": bool(ridge["action_sensitivity"] >= ACTION_SENSITIVITY_MIN),
        "termination_auroc": bool(
            np.isfinite(ridge["termination_auroc"])
            and ridge["termination_auroc"] >= TERMINATION_AUROC_MIN
        ),
    }
    rollout_info = None
    if rollout_horizon > 0:
        fitted = {}
        for name in dict.fromkeys((primary_model, "copy_last")):
            model = MODELS[name]()
            model.fit(train.states, train.actions, train.next_states, train.terminated)
            fitted[name] = model
        rollout_info = _rollout_curves(train, test, horizon=rollout_horizon, models=fitted)

    return {
        "seed": seed,
        "n_train": int(train.states.shape[0]),
        "n_test": int(test.states.shape[0]),
        "test_termination_rate": float(test.terminated.mean()),
        "models": results,
        "nmse_ratio_vs_copy_last": float(ratio),
        "checks": checks,
        "passed": all(checks.values()),
        "rollout": rollout_info,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="世界模型打底：一步动力学基线")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--train-episodes", type=int, default=200)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument(
        "--rollout-horizon",
        type=int,
        default=0,
        help=">0 时额外报 k 步自由 rollout / 教师强制的误差曲线；0=不算。",
    )
    parser.add_argument(
        "--primary-model",
        choices=tuple(MODELS),
        default="ridge_action",
        help="判据看哪个模型；amendment 1 起用 ridge_quadratic。",
    )
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--smoke", action="store_true", help="小规模冒烟：5/3 回合")
    args = parser.parse_args(argv)

    if args.smoke:
        args.seeds = "0"
        args.train_episodes = 5
        args.test_episodes = 3
        args.out = REPO_ROOT / "data" / "_smoke_world_model"

    seeds = parse_seeds(args.seeds)
    started = time.perf_counter()
    per_seed = [
        _run_one_seed(
            seed,
            train_episodes=args.train_episodes,
            test_episodes=args.test_episodes,
            env_name=args.env,
            primary_model=args.primary_model,
            rollout_horizon=args.rollout_horizon,
        )
        for seed in seeds
    ]
    elapsed = time.perf_counter() - started

    def aggregate(key: str) -> dict[str, float]:
        values = [row["models"][args.primary_model][key] for row in per_seed]
        finite = [value for value in values if np.isfinite(value)]
        return {
            "mean": float(np.mean(finite)) if finite else float("nan"),
            "min": float(np.min(finite)) if finite else float("nan"),
            "max": float(np.max(finite)) if finite else float("nan"),
        }

    summary = {
        "experiment": "世界模型打底：CartPole 一步动力学基线",
        "config": {
            "env": args.env,
            "train_episodes": args.train_episodes,
            "test_episodes": args.test_episodes,
            "behaviour_policy": "uniform random",
            "ridge_lambda": 1.0,
            "primary_model": args.primary_model,
            "rollout_horizon": args.rollout_horizon,
        },
        "per_seed": per_seed,
        "aggregate": {
            "nmse_ratio_vs_copy_last": aggregate("nmse"),
            "action_sensitivity": aggregate("action_sensitivity"),
            "termination_auroc": aggregate("termination_auroc"),
        },
        "seeds_passed": sum(1 for row in per_seed if row["passed"]),
        "verdict": (
            "task_ready"
            if per_seed and all(row["passed"] for row in per_seed)
            else "task_not_ready"
        ),
        "repro": collect(
            "世界模型打底：CartPole 一步动力学基线",
            seeds=",".join(str(seed) for seed in seeds),
            elapsed_s=elapsed,
            notes=[
                "按回合切分；分母用训练集 Var(s')；奖励恒为 1，不评。",
                "ridge_action 是测量标尺，不是局部学习规则。",
            ],
        ).render_block(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"{'模型':<18}{'NMSE':>8}{'终止AUROC':>12}{'动作敏感度':>12}")
    for name in MODELS:
        row = per_seed[0]["models"][name]
        print(
            f"{name:<18}{row['nmse']:>8.4f}"
            f"{row['termination_auroc']:>12.4f}{row['action_sensitivity']:>12.4f}"
        )
    print(
        f"\n判据（{len(seeds)} 种子）：通过 {summary['seeds_passed']}/{len(seeds)}"
        f" → {summary['verdict']}"
    )
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""世界模型在 Acrobot 上的**控制**实验：基线 → on-policy 补数据（DAgger）。

为什么要 Acrobot：CartPole 上 3 步前瞻已经顶到 500 的回合上限，**再优化也测不出来**。
Acrobot 上同一个 `ridge_quadratic` 的一步 NMSE 是 `copy_last` 的 0.36 倍（远不如 CartPole
的近乎完美），有真实的提升空间。

优化的核心假设（模型式控制最经典的失效原因）：
**训练数据来自随机策略，而控制器访问的状态与它不同。** 所以第二步是在**控制器经过的状态**
上补采数据、重新拟合，看回报是否改善。

指标：Acrobot 每步奖励 −1、到 500 截断 ⇒ **回报 = −回合长度**，越接近 −100 越好。
"成功"指摆到目标高度而终止。

用法::

    uv run python -m research.cognitive.run_wm_acrobot --episodes 10 --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.cognitive.world_model import RidgeActionModel

OUT = Path("sweep_results/world_model_acrobot_control")
N_ACTIONS = 3
MAX_STEPS = 500
#: 速度项的惩罚：只用来在同样高度下偏好"稳"的动作，量级刻意很小。
VELOCITY_PENALTY = 0.01


def height(states: np.ndarray) -> np.ndarray:
    """Acrobot 的目标量：``−cos θ1 − cos(θ1+θ2)``，>1.0 即终止。"""
    s = np.atleast_2d(states)
    cos1, sin1, cos2, sin2 = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    cos12 = cos1 * cos2 - sin1 * sin2
    return -cos1 - cos12


def project(states: np.ndarray) -> np.ndarray:
    """把预测状态投影回物理可行流形。

    Acrobot 的状态是 ``[cos θ1, sin θ1, cos θ2, sin θ2, ω1, ω2]``，
    **前两对必须满足 cos²+sin²=1**。模型把它们当四个自由变量预测，从不受约束；
    三次项把外推放大之后，最先坏掉的就是这个不变量（rollout 里先溢出再报 invalid）。

    这里只投影**真不变量**，不裁剪观测范围——后者已被证否（会把目标区域一起裁掉）。
    """
    out = np.array(states, dtype=np.float64, copy=True)
    for lo in (0, 2):
        pair = out[:, lo : lo + 2]
        norm = np.linalg.norm(pair, axis=1, keepdims=True)
        out[:, lo : lo + 2] = pair / np.maximum(norm, 1e-12)
    return out


def score(states: np.ndarray) -> np.ndarray:
    s = np.atleast_2d(states)
    return height(s) - VELOCITY_PENALTY * (s[:, 4] ** 2 + s[:, 5] ** 2)


def collect_random(episodes: int, seed: int, cap: int = 300) -> dict[str, np.ndarray]:
    import gymnasium as gym

    env = gym.make("Acrobot-v1")
    rng = np.random.default_rng(seed)
    states, actions, nexts, terminals = [], [], [], []
    for _ in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        for _ in range(cap):
            action = int(rng.integers(0, N_ACTIONS))
            nxt, _, term, trunc, _ = env.step(action)
            states.append(obs)
            actions.append(action)
            nexts.append(nxt)
            terminals.append(float(term))
            obs = nxt
            if term or trunc:
                break
    env.close()
    return {
        "states": np.asarray(states, dtype=np.float64),
        "actions": np.asarray(actions, dtype=np.int64),
        "next_states": np.asarray(nexts, dtype=np.float64),
        "terminated": np.asarray(terminals, dtype=np.float64),
    }


def fit(data: dict[str, np.ndarray]) -> RidgeActionModel:
    model = RidgeActionModel(ridge=1.0, quadratic=True)
    model.fit(data["states"], data["actions"], data["next_states"], data["terminated"])
    return model


def play(
    model: RidgeActionModel,
    episodes: int,
    seed: int,
    *,
    horizon: int = 3,
    record: dict[str, list] | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    random_prob: float = 0.0,
    project_states: bool = False,
    physics_on: bool = False,
    physics_adaptive: bool = False,
    envelope: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, float]:
    """跑回合。给 ``record`` 时把控制器的轨迹也收集下来（DAgger 用）。"""
    import gymnasium as gym

    env = gym.make("Acrobot-v1")
    rng = np.random.default_rng(seed + 20_000)
    lengths: list[int] = []
    successes = 0
    for _ in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        steps = 0
        done = False
        while not done and steps < MAX_STEPS:
            rolled = np.tile(obs, (N_ACTIONS, 1))
            for _ in range(max(horizon, 1)):
                previous_rolled = rolled
                pred = model.predict(rolled, np.arange(N_ACTIONS))
                rolled = rolled + pred.delta
                if physics_adaptive and envelope is not None:
                    from research.cognitive.physics import apply_adaptive

                    rolled = apply_adaptive(previous_rolled, rolled, envelope)
                elif physics_on:
                    from research.cognitive.physics import apply_all

                    rolled = apply_all(previous_rolled, rolled)
                if project_states:
                    rolled = project(rolled)
                if bounds is not None:
                    # **把预测状态裁到训练数据覆盖的范围内**。二次特征在数据外会
                    # 指数式发散（实测 rollout 溢出成 inf），而 inf 之后的打分全是垃圾。
                    # 这是模型式控制的常见护栏，不是为了调分。
                    low, high = bounds
                    rolled = np.clip(rolled, low, high)
            action = int(np.argmax(score(rolled)))
            # β-混合：按比例掺随机动作，避免采到的轨迹过度集中在策略熟悉的状态上。
            if random_prob > 0.0 and rng.random() < random_prob:
                action = int(rng.integers(0, N_ACTIONS))
            nxt, _, term, trunc, _ = env.step(action)
            if record is not None:
                record["states"].append(obs)
                record["actions"].append(action)
                record["next_states"].append(nxt)
                record["terminated"].append(float(term))
            obs = nxt
            done = bool(term or trunc)
            successes += int(term)
            steps += 1
        lengths.append(steps)
    env.close()
    return {
        "mean_return": float(-np.mean(lengths)),
        "median_steps": float(np.median(lengths)),
        "success_rate": successes / max(episodes, 1),
    }


def one_step_nmse(model: RidgeActionModel, data: dict[str, np.ndarray]) -> float:
    """留出集上的一步 NMSE（相对方差归一）。**验证门用它**。"""
    truth = data["next_states"] - data["states"]
    pred = model.predict(data["states"], data["actions"])
    return float(np.mean((pred.delta - truth) ** 2) / max(np.var(truth), 1e-12))


def concat(a: dict[str, np.ndarray], b: dict[str, list]) -> dict[str, np.ndarray]:
    return {
        key: np.concatenate([np.asarray(a[key]), np.asarray(b[key], dtype=np.float64)])
        for key in ("states", "actions", "next_states", "terminated")
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Acrobot 上的世界模型控制：基线 vs on-policy 补数据")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--train-episodes", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=2, help="DAgger 轮数")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--val-episodes", type=int, default=20)
    ap.add_argument("--beta-mix", type=float, default=0.3)
    ap.add_argument("--no-gate", action="store_true", help="关掉验证门（复现朴素 DAgger）")
    ap.add_argument("--select-by", choices=("nmse", "control"), default="nmse")
    ap.add_argument("--dev-seed", type=int, default=7, help="控制准则用的开发种子（不参与报告）")
    ap.add_argument(
        "--val-mode",
        choices=("random", "mixed"),
        default="mixed",
        help="验证集口径。random 会让门偏向基座模型（它就是用随机数据训的）",
    )
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument(
        "--clip",
        action="store_true",
        help="把预测状态裁到训练范围。**实测是负结果**（h=10 中位 −178 → −287），默认关",
    )
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows = []
    for seed in seeds:
        base_data = collect_random(args.train_episodes, seed)
        # **验证门的数据是独立的**：留出的随机策略回合，只用来判"新模型是不是真的更好"。
        val_random = collect_random(args.val_episodes, seed + 5000)
        bounds = (
            (base_data["states"].min(axis=0) - 1e-9, base_data["states"].max(axis=0) + 1e-9)
            if args.clip
            else None
        )
        model = fit(base_data)
        # **验证集口径决定这道门偏向谁**：只用随机策略数据，基座模型必胜（它就是那么训的），
        # 于是任何 on-policy 数据都会被否——那不是"数据没用"，是门选错了。
        # 混合口径把控制器实际访问的状态也纳入，才对应"用它做决策"这个目标。
        val_data = val_random
        if args.val_mode == "mixed":
            val_record: dict[str, list] = {
                "states": [],
                "actions": [],
                "next_states": [],
                "terminated": [],
            }
            play(
                model,
                args.val_episodes,
                seed,
                horizon=args.horizon,
                record=val_record,
                random_prob=args.beta_mix,
            )
            val_data = concat(
                {
                    "states": val_random["states"],
                    "actions": val_random["actions"],
                    "next_states": val_random["next_states"],
                    "terminated": val_random["terminated"],
                },
                val_record,
            )
        best_nmse = one_step_nmse(model, val_data)
        best_score = (
            play(model, args.episodes, args.dev_seed, horizon=args.horizon)["mean_return"]
            if args.select_by == "control"
            else float("-inf")
        )
        baseline = play(model, args.episodes, seed, horizon=args.horizon, bounds=bounds)

        history = [{"round": 0, **baseline}]
        data = base_data
        for round_index in range(1, args.rounds + 1):
            record: dict[str, list] = {
                "states": [],
                "actions": [],
                "next_states": [],
                "terminated": [],
            }
            play(
                model,
                args.episodes,
                seed,
                horizon=args.horizon,
                record=record,
                bounds=bounds,
                random_prob=args.beta_mix,
            )
            candidate_data = concat(data, record)
            candidate = fit(candidate_data)
            candidate_nmse = one_step_nmse(candidate, val_data)
            previous_nmse = best_nmse
            if args.select_by == "control":
                # **用控制目标本身就当采纳准则。** 前 10 轮的教训：一步 NMSE 降 15%
                # 的同时回合长度从 125 掉到 0——预测更准不等于决策更好。
                # 这里在**开发种子**（默认 7，不参与报告）上跑真实控制。
                candidate_score = play(
                    candidate, args.episodes, args.dev_seed, horizon=args.horizon
                )["mean_return"]
                adopted = args.no_gate or candidate_score > best_score
                if adopted:
                    data, model, best_score = candidate_data, candidate, candidate_score
                    best_nmse = candidate_nmse
            else:
                adopted = args.no_gate or candidate_nmse < best_nmse
                if adopted:
                    data, model, best_nmse = candidate_data, candidate, candidate_nmse
            stats = play(model, args.episodes, seed, horizon=args.horizon, bounds=bounds)
            history.append(
                {"round": round_index, "val_nmse": round(best_nmse, 4), "adopted": adopted, **stats}
            )
            print(
                f"  种子 {seed} 第 {round_index} 轮：候选 NMSE {candidate_nmse:.4f}"
                f" vs 现有 {previous_nmse:.4f}"
                f" ⇒ {'采纳' if adopted else '否决'}；回报 {stats['mean_return']:.1f}",
                flush=True,
            )

        rows.append({"seed": seed, "history": history})
        print(
            f"种子 {seed}: 基线 回报 {baseline['mean_return']:.1f}"
            f"（{baseline['median_steps']:.0f} 步）→ 末轮 {history[-1]['mean_return']:.1f}",
            flush=True,
        )

    first = float(np.mean([r["history"][0]["mean_return"] for r in rows]))
    last = float(np.mean([r["history"][-1]["mean_return"] for r in rows]))
    print(
        f"\n三种子平均回报：基线 {first:.1f} → 末轮 {last:.1f}（相对改善 {abs(first) - abs(last):.1f}）"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "episodes": args.episodes,
                "train_episodes": args.train_episodes,
                "rounds": args.rounds,
                "horizon": args.horizon,
                "rows": rows,
                "baseline_return": first,
                "final_return": last,
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"写入 {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

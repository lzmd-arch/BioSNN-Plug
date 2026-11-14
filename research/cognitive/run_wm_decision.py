"""世界模型**决策**实验：1 步前瞻控制（预注册见 `prereg-world-model-decision.md`）。

四臂的设计要点只有一个：**搜索完全相同、每步的模型评估次数完全相同，只换预测器。**

    搜索：对 2 个候选动作各预测一次下一步，按同一个"站得稳"打分选动作；平局随机打破
    R    随机动作（下限）
    C    copy_last：预测器不含任何动作信息
    B    动作标签打乱后训练的同一个模型族：状态动力学在，学不到"这个动作导致什么"
    W    ridge_quadratic：学出来的动作条件化世界模型

**W − B 才是"动作条件化"本身的净贡献。** 只跟 R 或 C 比，赢了也只能说明
"搜索有用"或"模型里有动力学信息"，不能说明世界模型的动作敏感度值钱。

用法::

    uv run python -m research.cognitive.run_wm_decision --episodes 100 --seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.cognitive.world_model import CopyLastModel, RidgeActionModel

OUT = Path("sweep_results/world_model_decision")

#: CartPole 的"站得稳"打分：**与预测器无关**，四臂共用。
#: 位置与角度都要小，角度的权重更高（倒下比走开更容易终止）。
#: 默认权重；`--weights` 可在**训练种子**上调，再在**留出种子**上评估。
DEFAULT_WEIGHTS = np.array([1.0, 0.5, 2.0, 0.5])


def score_states(states: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    w = DEFAULT_WEIGHTS if weights is None else np.asarray(weights, dtype=np.float64)
    return -(np.abs(states) * w).sum(axis=1)


def collect(episodes: int, seed: int, env_steps: int = 200) -> dict[str, np.ndarray]:
    """均匀随机动作采集。"""
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    states, actions, nexts, terminated = [], [], [], []
    rng = np.random.default_rng(seed)
    for _ in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        for _ in range(env_steps):
            action = int(rng.integers(0, env.action_space.n))
            nxt, _, term, trunc, _ = env.step(action)
            states.append(obs)
            actions.append(action)
            nexts.append(nxt)
            terminated.append(float(term))
            obs = nxt
            if term or trunc:
                break
    env.close()
    return {
        "states": np.asarray(states, dtype=np.float64),
        "actions": np.asarray(actions, dtype=np.int64),
        "next_states": np.asarray(nexts, dtype=np.float64),
        "terminated": np.asarray(terminated, dtype=np.float64),
    }


def play(
    policy: str,
    model,
    episodes: int,
    seed: int,
    max_steps: int = 500,
    *,
    weights: np.ndarray | None = None,
    horizon: int = 1,
) -> list[int]:
    """跑回合，返回每回合长度。``policy="random"`` 时不看模型。"""
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed + 10_000)
    lengths: list[int] = []
    for _ in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        steps = 0
        done = False
        while not done and steps < max_steps:
            if policy == "random":
                action = int(rng.integers(0, 2))
            else:
                batch = np.tile(obs, (2, 1))
                rolled = batch.copy()
                for _ in range(max(horizon, 1)):
                    pred = model.predict(rolled, np.arange(2))
                    rolled = rolled + pred.delta
                scores = score_states(rolled, weights)
                # **平局必须随机打破**：动作不敏感的预测器（copy_last、动作打乱后的模型）
                # 会让两个候选打平，`argmax` 就退化成"永远选动作 0"——那不是搜索，
                # 是常数策略。第一版没有这一步，控制臂因此只有 9 步，被当成"模型很强"。
                candidates = np.flatnonzero(np.isclose(scores, scores.max()))
                action = int(rng.choice(candidates))
            obs, _, term, trunc, _ = env.step(action)
            done = bool(term or trunc)
            steps += 1
        lengths.append(steps)
    env.close()
    return lengths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="世界模型 1 步前瞻控制（四臂）")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--train-episodes", type=int, default=200)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows = []
    for seed in seeds:
        train = collect(args.train_episodes, seed)
        learned = RidgeActionModel(ridge=1.0, quadratic=True)
        learned.fit(train["states"], train["actions"], train["next_states"], train["terminated"])

        # B 臂：把动作标签打乱后训练同一个模型族。状态动力学还在，但学不到动作的作用。
        shuffled = np.random.default_rng(seed + 777).permutation(train["actions"])
        blind = RidgeActionModel(ridge=1.0, quadratic=True)
        blind.fit(train["states"], shuffled, train["next_states"], train["terminated"])

        trivial = CopyLastModel()
        trivial.fit(train["states"], train["actions"], train["next_states"], train["terminated"])

        row = {
            "seed": seed,
            "random_median": float(np.median(play("random", None, args.episodes, seed))),
            "copy_last_median": float(np.median(play("lookahead", trivial, args.episodes, seed))),
            "action_shuffled_median": float(
                np.median(play("lookahead", blind, args.episodes, seed))
            ),
            "world_model_median": float(np.median(play("lookahead", learned, args.episodes, seed))),
        }
        rows.append(row)
        print(
            f"种子 {seed}: 随机 {row['random_median']:.0f} | "
            f"copy_last {row['copy_last_median']:.0f} | "
            f"动作打乱 {row['action_shuffled_median']:.0f} | "
            f"世界模型 {row['world_model_median']:.0f}",
            flush=True,
        )

    def median_of(key: str) -> float:
        return float(np.median([r[key] for r in rows]))

    random_m = median_of("random_median")
    control_m = median_of("copy_last_median")
    blind_m = median_of("action_shuffled_median")
    model_m = median_of("world_model_median")
    ratio = model_m / max(control_m, 1e-9)
    action_gain = model_m / max(blind_m, 1e-9)

    print(
        f"\n中位：随机 {random_m:.0f} / copy_last {control_m:.0f} / "
        f"动作打乱 {blind_m:.0f} / 世界模型 {model_m:.0f}"
    )
    print(f"WM-1a（世界模型 ≥ 1.2 × copy_last）：{ratio:.2f}× ⇒ {'过' if ratio >= 1.2 else '不过'}")
    print(f"动作条件化的净贡献（世界模型 vs 动作打乱）：{action_gain:.2f}×")
    print(
        f"WM-0a（copy_last 不得优于随机）：{control_m:.0f} vs {random_m:.0f}"
        f" ⇒ {'过' if control_m <= random_m * 1.05 else '不过'}"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "episodes": args.episodes,
        "train_episodes": args.train_episodes,
        "rows": rows,
        "random_median": random_m,
        "copy_last_median": control_m,
        "action_shuffled_median": blind_m,
        "world_model_median": model_m,
        "ratio_model_over_control": round(ratio, 3),
        "ratio_model_over_action_shuffled": round(action_gain, 3),
        "wm1a": ratio >= 1.2,
        "wm0a": control_m <= random_m * 1.05,
    }
    (args.out / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"写入 {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

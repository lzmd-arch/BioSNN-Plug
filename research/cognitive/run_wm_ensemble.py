"""世界模型：用**集成分歧度**代替已被证否的状态裁剪，看能不能撑住更深的前瞻。

背景（见 `README.md` 结果二）：h=20 崩溃，原因是二次特征在数据外发散。
裁剪到训练范围是错的（把目标区域也裁掉了，实测更差）。
这里的假设是：**让集成中各模型的分歧度参与打分**，规划器就会自己避开"大家都说不准"的地方。

    score(a) = mean_k[ score(s'_k) ] − λ · std_k[ score(s'_k) ]

λ=0 即只用均值（与单模型同口径），λ>0 才启用惩罚。

用法::

    uv run python -m research.cognitive.run_wm_ensemble --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.cognitive.run_wm_acrobot import (
    MAX_STEPS,
    N_ACTIONS,
    collect_random,
    score,
)
from research.cognitive.world_model import RidgeActionModel

OUT = Path("sweep_results/world_model_ensemble")


def fit_ensemble(data: dict[str, np.ndarray], size: int, seed: int) -> list[RidgeActionModel]:
    """自助采样训练 ``size`` 个模型——分歧度来自数据扰动，不是超参扰动。"""
    rng = np.random.default_rng(seed)
    n = len(data["states"])
    models = []
    for _ in range(size):
        index = rng.integers(0, n, n)
        model = RidgeActionModel(ridge=1.0, quadratic=True)
        model.fit(
            data["states"][index],
            data["actions"][index],
            data["next_states"][index],
            data["terminated"][index],
        )
        models.append(model)
    return models


def play_ensemble(
    models: list[RidgeActionModel],
    episodes: int,
    seed: int,
    *,
    horizon: int,
    lam: float,
) -> dict[str, float]:
    import gymnasium as gym

    env = gym.make("Acrobot-v1")
    rng = np.random.default_rng(seed + 30_000)
    lengths: list[int] = []
    successes = 0
    for _ in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        steps = 0
        done = False
        while not done and steps < MAX_STEPS:
            per_model = []
            for model in models:
                rolled = np.tile(obs, (N_ACTIONS, 1))
                for _ in range(max(horizon, 1)):
                    pred = model.predict(rolled, np.arange(N_ACTIONS))
                    rolled = rolled + pred.delta
                per_model.append(score(rolled))
            stacked = np.stack(per_model, axis=0)  # (K, N_ACTIONS)
            value = stacked.mean(axis=0) - lam * stacked.std(axis=0)
            action = int(np.argmax(value))
            obs, _, term, trunc, _ = env.step(action)
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="集成分歧度惩罚 vs 单模型")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--train-episodes", type=int, default=60)
    ap.add_argument("--ensemble", type=int, default=5)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows = []
    for seed in seeds:
        data = collect_random(args.train_episodes, seed)
        models = fit_ensemble(data, args.ensemble, seed)
        record = {"seed": seed, "arms": {}}
        for horizon in (10, 20):
            for lam in (0.0, 1.0, 4.0):
                stats = play_ensemble(models, args.episodes, seed, horizon=horizon, lam=lam)
                record["arms"][f"h{horizon}_lam{lam}"] = stats
                print(
                    f"种子 {seed} h={horizon:2d} λ={lam:.0f}: 回报 {stats['mean_return']:.0f}"
                    f" 成功 {stats['success_rate']:.2f}",
                    flush=True,
                )
        rows.append(record)

    print("\n三种子平均回报：")
    for key in rows[0]["arms"]:
        mean = float(np.mean([r["arms"][key]["mean_return"] for r in rows]))
        print(f"  {key:>12}: {mean:.0f}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps({"rows": rows, "ensemble": args.ensemble}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"写入 {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

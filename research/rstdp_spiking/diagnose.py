"""W3 脉冲化的判别实验：**诊断，不是验收**。

预注册阈值见 [`diagnosis-plan.md`](diagnosis-plan.md)；本轮先做 ③（评测噪声），
因为它在三条里最便宜、也最先该排除——在噪声里比较算法等于浪费 GPU。

跑法::

    uv run python -m research.rstdp_spiking.diagnose --experiment noise --device cuda
    uv run python -m research.rstdp_spiking.diagnose --experiment noise --smoke

产物：``sweep_results/rstdp_spiking_diagnosis/noise/summary.json`` + 自证块。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.common.provenance import collect
from research.components.cartpole_encoding import encode_state
from research.rstdp_spiking.cartpole import train
from research.rstdp_spiking.critic import LinearCritic
from research.rstdp_spiking.pop_critic import PopulationCritic

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "rstdp_spiking_diagnosis" / "noise"

#: 预注册阈值（diagnosis-plan.md ③）：CI 半宽 ≤ 中位数的 20% 才算可用。
CI_HALF_WIDTH_RATIO_MAX = 0.20

#: 与 `cartpole.py` 的验收配置一致（线性 TD 那条臂）。
BASE_CONFIG: dict[str, Any] = {
    "n_features": 64,
    "sigma": 1.0,
    "steps": 25,
    # **与 `sweep_results/rstdp_spiking/td_10seeds/summary.json` 的发布配置一致**。
    # 写成 1e-2 会把输出层推到几乎沉默，greedy 永远选动作 0——评测值会因此逐位相同，
    # 看起来像“评测很稳”，其实是策略退化（本诊断第一版踩过）。
    "learning_rate": 1e-4,
    "baseline_decay": 0.05,
    "critic_kind": "td",
    "critic_rate": 1e-2,
    "gamma": 0.99,
    "warmup": 10,
}

__all__ = ["bootstrap_ci", "main", "mc_returns"]


def bootstrap_ci(
    values: np.ndarray, *, seed: int = 0, resamples: int = 1000
) -> tuple[float, float]:
    """均值的 95% bootstrap 置信区间（固定 RNG，可复现）。"""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("bootstrap_ci 需要至少一个值。")
    rng = np.random.default_rng(seed)
    means = np.array(
        [rng.choice(array, size=array.size, replace=True).mean() for _ in range(resamples)]
    )
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def mc_returns(rewards: list[float], gamma: float) -> list[float]:
    """从 t 起的折扣蒙特卡洛回报 ``G_t = Σ γ^k r_{t+k}``。"""
    values: list[float] = []
    running = 0.0
    for reward in reversed(rewards):
        running = float(reward) + gamma * running
        values.append(running)
    values.reverse()
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="W3 脉冲化诊断（③ 评测噪声）")
    parser.add_argument("--experiment", choices=("noise", "entropy", "critic"), default="noise")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-noise-seeds", type=int, default=20)
    parser.add_argument("--critic", choices=("td", "pop"), default="td")
    parser.add_argument("--train-episodes", type=int, default=300)
    parser.add_argument("--critic-episodes", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    if args.smoke:
        args.episodes = 5
        args.eval_episodes = 1
        args.eval_noise_seeds = 3
        args.train_episodes = 5
        args.critic_episodes = 3
        args.out = REPO_ROOT / "data" / f"_smoke_w3_{args.experiment}"

    if args.out == OUT:
        name = args.experiment if args.experiment != "critic" else f"critic_{args.critic}"
        args.out = OUT.parent / name

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()

    if args.experiment == "critic":
        train_kwargs = {
            **BASE_CONFIG,
            "critic_kind": args.critic,
        }
        if args.critic == "pop":
            # 与 10 种子对照的默认群体臂一致：gradient 后因子 + 均匀读出 + 归一化。
            train_kwargs["pop_post"] = "gradient"
        result = train(
            seed=args.seed,
            episodes=args.train_episodes,
            eval_every=25,
            eval_episodes=args.eval_episodes,
            warmup=0 if args.smoke else BASE_CONFIG["warmup"],
            device=device,
            keep_actor=True,
            **{key: value for key, value in train_kwargs.items() if key != "warmup"},
        )
        if result.actor is None or result.centers is None:  # pragma: no cover - 防御
            raise RuntimeError("train 没有返回 actor/centers。")
        actor = result.actor
        centers = result.centers
        if args.critic == "td":
            critic = LinearCritic(
                BASE_CONFIG["n_features"],
                gamma=BASE_CONFIG["gamma"],
                learning_rate=1e-2,
            )
        else:
            critic = PopulationCritic(
                BASE_CONFIG["n_features"],
                n_units=64,
                gain=8.0,
                bias=0.4,
                gamma=BASE_CONFIG["gamma"],
                learning_rate=1e-2,
                trace_post_factor="gradient",
                readout="uniform",
                normalize=True,
                generator=torch.Generator(device="cpu").manual_seed(50_000 + args.seed),
                device=device,
            )
        import gymnasium

        env = gymnasium.make("CartPole-v1")
        episodes: list[tuple[list[Any], list[float]]] = []
        try:
            for episode in range(args.critic_episodes):
                generator = torch.Generator(device="cpu").manual_seed(30_000 + episode)
                raw, _ = env.reset(seed=40_000 + episode)
                features = encode_state(
                    torch.tensor(raw, dtype=torch.float32, device=device),
                    centers,
                    BASE_CONFIG["sigma"],
                )
                states: list[Any] = []
                rewards: list[float] = []
                for _ in range(500):
                    action = actor.step(
                        features, generator=generator, greedy=False, learn_trace=False
                    )
                    raw_next, reward, terminated, truncated, _ = env.step(action)
                    next_features = encode_state(
                        torch.tensor(raw_next, dtype=torch.float32, device=device),
                        centers,
                        BASE_CONFIG["sigma"],
                    )
                    critic.update(
                        features, float(reward), next_features, terminated=bool(terminated)
                    )
                    states.append(features.detach())
                    rewards.append(float(reward))
                    features = next_features
                    if terminated or truncated:
                        break
                episodes.append((states, rewards))
        finally:
            env.close()

        last = episodes[-20:] if len(episodes) >= 20 else episodes
        returns: list[float] = []
        predictions: list[float] = []
        for states, rewards in last:
            for state, target in zip(
                states, mc_returns(rewards, BASE_CONFIG["gamma"]), strict=True
            ):
                returns.append(target)
                predictions.append(critic.value(state))
        targets = np.asarray(returns, dtype=np.float64)
        values = np.asarray(predictions, dtype=np.float64)
        mean_target = float(targets.mean()) if targets.size else float("nan")
        var_critic = float(((targets - values) ** 2).mean()) if targets.size else float("nan")
        var_constant = (
            float(((targets - mean_target) ** 2).mean()) if targets.size else float("nan")
        )
        reduction = 1.0 - var_critic / var_constant if var_constant > 1e-12 else float("nan")
        verdict = "signal" if np.isfinite(reduction) and reduction >= 0.30 else "no_signal"
        summary = {
            "experiment": "W3 诊断 ①：Critic 信号",
            "config": {
                **BASE_CONFIG,
                "critic_kind": args.critic,
                "train_episodes": args.train_episodes,
                "critic_episodes": args.critic_episodes,
                "eval_episodes": args.eval_episodes,
                "device": str(device),
            },
            "seed": args.seed,
            "train_best_steps": result.best_steps,
            "n_episodes_evaluated": len(last),
            "var_critic": var_critic,
            "var_constant": var_constant,
            "reduction": reduction,
            "threshold": 0.30,
            "verdict": verdict,
            "repro": collect(
                "W3 诊断 ①：Critic 信号",
                seeds=str(args.seed),
                elapsed_s=time.perf_counter() - started,
                notes=[
                    "诊断，不是验收；操作化见 research/rstdp_spiking/diagnosis-plan.md ①。",
                    "固定策略 = 最好权重快照；目标是折扣蒙特卡洛回报。",
                ],
                device=device,
            ).render_block(),
        }
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(
            f"Critic 误差方差 {var_critic:.3f} vs 常数基线 {var_constant:.3f}，"
            f"下降 {reduction:.1%} → {verdict}"
        )
        print(f"written to {args.out}")
        return 0

    if args.experiment == "entropy":
        result = train(
            seed=args.seed,
            episodes=args.episodes,
            eval_every=25,
            eval_episodes=args.eval_episodes,
            warmup=0 if args.smoke else BASE_CONFIG["warmup"],
            device=device,
            entropy_log=True,
            **{key: value for key, value in BASE_CONFIG.items() if key != "warmup"},
        )
        entropies = np.asarray(result.entropies, dtype=np.float64)
        upto = entropies[:100] if entropies.size >= 100 else entropies
        window = upto[-10:] if upto.size >= 10 else upto
        early_mean = float(window.mean()) if window.size else float("nan")
        if early_mean < 0.1:
            verdict = "collapse"
        elif early_mean >= 0.3:
            verdict = "no_collapse"
        else:
            verdict = "uncertain"
        summary = {
            "experiment": "W3 诊断 ②：策略熵崩塌",
            "config": {
                **BASE_CONFIG,
                "episodes": args.episodes,
                "eval_episodes": args.eval_episodes,
                "device": str(device),
            },
            "seed": args.seed,
            "entropy_window_mean": early_mean,
            "thresholds": {"collapse": 0.1, "no_collapse": 0.3},
            "verdict": verdict,
            "entropies": [float(value) for value in entropies],
            "repro": collect(
                "W3 诊断 ②：策略熵崩塌",
                seeds=str(args.seed),
                elapsed_s=time.perf_counter() - started,
                notes=[
                    "诊断，不是验收；阈值见 research/rstdp_spiking/diagnosis-plan.md。",
                    "取第 100 回合前最后 10 回合的 Boltzmann 熵均值（nats）。",
                ],
                device=device,
            ).render_block(),
        }
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"第 100 回合前最后 10 回合平均熵 {early_mean:.3f} nats → {verdict}")
        print(f"written to {args.out}")
        return 0
    result = train(
        seed=args.seed,
        episodes=args.episodes,
        eval_every=25,
        eval_episodes=args.eval_episodes,
        warmup=0 if args.smoke else BASE_CONFIG["warmup"],
        device=device,
        eval_noise_seeds=args.eval_noise_seeds,
        **{key: value for key, value in BASE_CONFIG.items() if key != "warmup"},
    )
    elapsed = time.perf_counter() - started

    values = np.asarray(result.eval_noise, dtype=np.float64)
    low, high = bootstrap_ci(values)
    half = (high - low) / 2.0
    median = float(np.median(values))
    ratio = half / median if median > 0 else float("inf")
    verdict = "usable" if ratio <= CI_HALF_WIDTH_RATIO_MAX else "evaluation_noise_dominates"
    summary = {
        "experiment": "W3 诊断 ③：评测噪声",
        "config": {
            **BASE_CONFIG,
            "episodes": args.episodes,
            "eval_episodes": args.eval_episodes,
            "eval_noise_seeds": args.eval_noise_seeds,
            "device": str(device),
        },
        "seed": args.seed,
        "eval_noise": [float(value) for value in values],
        "mean": float(values.mean()),
        "median": median,
        "min": float(values.min()),
        "max": float(values.max()),
        "ci95_low": low,
        "ci95_high": high,
        "ci_half_width": half,
        "ci_half_width_over_median": ratio,
        "threshold": CI_HALF_WIDTH_RATIO_MAX,
        "verdict": verdict,
        "repro": collect(
            "W3 诊断 ③：评测噪声",
            seeds=str(args.seed),
            elapsed_s=elapsed,
            notes=[
                "诊断，不是验收；阈值见 research/rstdp_spiking/diagnosis-plan.md。",
                "同一份最终权重、20 个评测种子各跑一次，bootstrap 1000 次。",
            ],
            device=device,
        ).render_block(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"中位数 {median:.2f}，范围 {values.min():.1f}–{values.max():.1f}，"
        f"95% CI [{low:.2f}, {high:.2f}]，半宽/中位数 {ratio:.1%} → {verdict}"
    )
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""W3 脉冲化的判别实验：**诊断，不是验收**。

预注册阈值见 [`diagnosis-plan.md`](diagnosis-plan.md)；③ 评测噪声先做，因为它在三条里
最便宜、也最先该排除——在噪声里比较算法等于浪费 GPU。``actor`` 是 A1（信用分配的
符号检验，纯 CPU），预注册见 ``sweep_results/rstdp_spiking_actor_diagnosis_prereg.md``。

跑法::

    uv run python -m research.rstdp_spiking.diagnose --experiment noise --device cuda
    uv run python -m research.rstdp_spiking.diagnose --experiment noise --smoke
    uv run python -m research.rstdp_spiking.diagnose --experiment actor

产物：``sweep_results/rstdp_spiking_diagnosis/*/summary.json``（以及 actor 的
``sweep_results/rstdp_spiking_actor_diagnosis/summary.json``）+ 自证块。
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
from research.rstdp_spiking.actor import SpikingActor
from research.rstdp_spiking.cartpole import train
from research.rstdp_spiking.critic import LinearCritic
from research.rstdp_spiking.plasticity import pair_based_stdp
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
    parser.add_argument(
        "--experiment",
        choices=("noise", "entropy", "critic", "actor", "actor-symbol"),
        default="noise",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-noise-seeds", type=int, default=20)
    parser.add_argument("--critic", choices=("td", "pop"), default="td")
    parser.add_argument("--actor-windows", type=int, default=20)
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
        if args.experiment == "actor":
            args.out = OUT.parent / "rstdp_spiking_actor_diagnosis"
        elif args.experiment == "actor-symbol":
            args.out = OUT.parent.parent / "rstdp_spiking_actor_fix" / "a1_symbol_consistency"
        else:
            name = args.experiment if args.experiment != "critic" else f"critic_{args.critic}"
            args.out = OUT.parent / name

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()

    if args.experiment == "actor":
        windows = 3 if args.smoke else args.actor_windows
        rows: list[dict[str, Any]] = []
        for index in range(windows):
            generator = torch.Generator(device="cpu").manual_seed(index)
            actor = SpikingActor(
                BASE_CONFIG["n_features"],
                2,
                steps=BASE_CONFIG["steps"],
                learning_rate=1e-4,
                generator=generator,
            )
            features = torch.rand(
                BASE_CONFIG["n_features"],
                generator=torch.Generator(device="cpu").manual_seed(10_000 + index),
            )
            pre, post = actor.window_spikes(features, generator=generator)
            counts = post.sum(dim=0).squeeze(0)
            chosen = int(counts.argmax())
            other = 1 - chosen
            columns = pair_based_stdp(pre, post, actor.window).sum(dim=0)
            tie = float(counts[chosen]) == float(counts[other])
            rows.append(
                {
                    "window": index,
                    "counts": [float(value) for value in counts],
                    "chosen": chosen,
                    "columns": [float(value) for value in columns],
                    "tie": tie,
                    "favor": bool(columns[chosen] > columns[other]),
                }
            )
        passed = sum(1 for row in rows if row["favor"] and not row["tie"])
        verdict = "credit_assignment_ok" if passed == len(rows) else "credit_assignment_inverted"
        summary = {
            "experiment": "W3 诊断 A1：Actor 信用分配的符号检验",
            "config": {
                "features": BASE_CONFIG["n_features"],
                "steps": BASE_CONFIG["steps"],
                "windows": len(rows),
                "device": "cpu",
            },
            "rows": rows,
            "passed": passed,
            "verdict": verdict,
            "repro": collect(
                "W3 诊断 A1：Actor 信用分配的符号检验",
                seeds=str(args.seed),
                elapsed_s=time.perf_counter() - started,
                notes=[
                    "诊断，不是验收；预注册见 sweep_results/rstdp_spiking_actor_diagnosis_prereg.md。",
                    "正信号下'被选中动作'的资格列和必须更大；20/20 才算通过。",
                ],
            ).render_block(),
        }
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"{passed}/{len(rows)} 个窗口的资格列偏向被选中动作 → {verdict}")
        print(f"written to {args.out}")
        return 0

    if args.experiment == "actor-symbol":
        # A1′：符号一致的更新方向。用两条独立初始化的 actor（同种子 → 同权重）分别
        # 施加 +1 / −1，量**实际写进权重**的增量，而不是只看痕迹的值。
        windows = 3 if args.smoke else args.actor_windows
        rows: list[dict[str, Any]] = []
        for index in range(windows):
            features = torch.rand(
                BASE_CONFIG["n_features"],
                generator=torch.Generator(device="cpu").manual_seed(10_000 + index),
            )
            actors = []
            for signal in (1.0, -1.0):
                actor = SpikingActor(
                    BASE_CONFIG["n_features"],
                    2,
                    steps=BASE_CONFIG["steps"],
                    learning_rate=1.0,
                    trace_rule="sign-consistent",
                    normalize=False,
                    generator=torch.Generator(device="cpu").manual_seed(index),
                )
                actors.append((signal, actor))
            pre, post = actors[0][1].window_spikes(
                features, generator=torch.Generator(device="cpu").manual_seed(index)
            )
            counts = post.sum(dim=0).squeeze(0)
            chosen = int(counts.argmax())
            other = 1 - chosen
            tie = float(counts[chosen]) == float(counts[other])

            deltas = {}
            for signal, actor in actors:
                actor.trace.accumulate_window(pre, post, actor.window, chosen)
                before = actor.layer.weight.clone()
                actor.apply_success_signal(signal)
                deltas[signal] = actor.layer.weight - before

            positive_delta = deltas[1.0]
            negative_delta = deltas[-1.0]
            rows.append(
                {
                    "window": index,
                    "counts": [float(value) for value in counts],
                    "chosen": chosen,
                    "tie": tie,
                    "positive_selected_sum": float(positive_delta[:, chosen].sum()),
                    "positive_other_abs": float(positive_delta[:, other].abs().sum()),
                    "positive_all_nonneg": bool(torch.all(positive_delta >= 0).item()),
                    "negative_selected_sum": float(negative_delta[:, chosen].sum()),
                    "negative_other_abs": float(negative_delta[:, other].abs().sum()),
                    "negative_all_nonpos": bool(torch.all(negative_delta <= 0).item()),
                    "positive_ok": bool(
                        float(positive_delta[:, chosen].sum()) > 0
                        and float(positive_delta[:, other].abs().sum()) == 0.0
                    ),
                    "negative_ok": bool(
                        float(negative_delta[:, chosen].sum()) <= 0
                        and float(negative_delta[:, other].abs().sum()) == 0.0
                    ),
                }
            )

        positive_passed = sum(1 for row in rows if row["positive_ok"])
        negative_passed = sum(1 for row in rows if row["negative_ok"])
        negative_strict = sum(1 for row in rows if row["negative_selected_sum"] < 0)
        gating_passed = sum(
            1
            for row in rows
            if row["positive_other_abs"] == 0.0 and row["negative_other_abs"] == 0.0
        )
        ok = (
            positive_passed == len(rows)
            and negative_passed == len(rows)
            and gating_passed == len(rows)
        )
        verdict = "symbol_consistent_ok" if ok else "symbol_consistent_failed"
        summary = {
            "experiment": "W3 诊断 A1′：Actor 符号一致化的更新方向",
            "config": {
                "features": BASE_CONFIG["n_features"],
                "steps": BASE_CONFIG["steps"],
                "windows": len(rows),
                "device": "cpu",
                "normalize": False,
                "learning_rate": 1.0,
            },
            "rows": rows,
            "positive_passed": positive_passed,
            "negative_passed": negative_passed,
            "negative_strict": negative_strict,
            "gating_passed": gating_passed,
            "verdict": verdict,
            "repro": collect(
                "W3 诊断 A1′：Actor 符号一致化的更新方向",
                seeds=str(args.seed),
                elapsed_s=time.perf_counter() - started,
                notes=[
                    "诊断，不是验收；预注册见 sweep_results/rstdp_spiking_actor_fix_prereg.md。",
                    "正信号 20/20、负信号 20/20、未选列逐位为 0，才算通过。",
                ],
            ).render_block(),
        }
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(
            f"正 {positive_passed}/{len(rows)}、负 {negative_passed}/{len(rows)}"
            f"（严格负 {negative_strict}）、门控 {gating_passed}/{len(rows)} → {verdict}"
        )
        print(f"written to {args.out}")
        return 0

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

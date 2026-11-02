"""在 CartPole 上跑储层演员（v3 的 H1/H2），不改动已验收的 train()。

    uv run python research/rstdp_spiking/run_reservoir.py --seeds 0,1 --episodes 300

H1：每回合结束后用**回合回报**作为全局奖励，对沿途每个 (计数, 动作) 做两因子更新。
H2：先收集 N 个回合的 (计数, 动作)，岭回归拟合读出，再评估（读出冻结）。
"""

from __future__ import annotations

import argparse
import statistics

import torch

from .cartpole import CARTPOLE_HALF_WIDTHS, _encode, population_centers
from .reservoir_actor import ReservoirActor

STEPS = 25
N_FEATURES = 64
SIGMA = 0.2


def run_episode(env, actor: ReservoirActor, centers, *, generator, learn: bool) -> float:
    raw, _ = env.reset(seed=int(torch.randint(0, 2**31 - 1, (1,), generator=generator)))
    features = _encode(raw, centers, SIGMA, None)
    trace: list[tuple[torch.Tensor, int]] = []
    total = 0.0
    for _ in range(500):
        counts = actor.counts(actor.hidden_spikes(features, generator=generator))
        logits = actor.scores(counts)
        action = int(logits.argmax()) if not learn else actor.step(features, generator=generator)
        if learn:
            trace.append((counts, action))
        raw, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        features = _encode(raw, centers, SIGMA, None)
        if terminated or truncated:
            break
    if learn:
        for counts, action in trace:
            actor.reinforce(counts, action, reward=total / 500.0)
    return total


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--h2", action="store_true", help="岭回归读出（越红线，仅作上界参照）")
    parser.add_argument("--collect-episodes", type=int, default=60)
    args = parser.parse_args(argv)

    import gymnasium

    results = {}
    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        centers = population_centers(
            N_FEATURES, half_widths=CARTPOLE_HALF_WIDTHS, generator=generator
        )
        env = gymnasium.make("CartPole-v1")
        actor = ReservoirActor(N_FEATURES, args.hidden, 2, steps=STEPS, seed=seed)

        if args.h2:
            counts_rows, actions_rows = [], []
            for _ in range(args.collect_episodes):
                raw, _ = env.reset(seed=int(torch.randint(0, 2**31 - 1, (1,), generator=generator)))
                feats = _encode(raw, centers, SIGMA, None)
                for _ in range(500):
                    c = actor.counts(actor.hidden_spikes(feats, generator=generator))
                    a = int(torch.randint(0, 2, (1,), generator=generator))
                    counts_rows.append(c)
                    actions_rows.append(a)
                    raw, _, term, trunc, _ = env.step(a)
                    feats = _encode(raw, centers, SIGMA, None)
                    if term or trunc:
                        break
            actor.fit_decoder(torch.stack(counts_rows), torch.tensor(actions_rows), alpha=1e-2)
            print(f"  seed {seed}: H2 拟合了 {len(actions_rows)} 个样本", flush=True)

        for episode in range(args.episodes):
            run_episode(env, actor, centers, generator=generator, learn=True)
            if (episode + 1) % 100 == 0:
                scores = [
                    run_episode(env, actor, centers, generator=generator, learn=False)
                    for _ in range(args.eval_episodes)
                ]
                print(
                    f"  seed {seed} ep {episode + 1}: 评测中位 {statistics.median(scores):.1f}",
                    flush=True,
                )

        final = [
            run_episode(env, actor, centers, generator=generator, learn=False)
            for _ in range(args.eval_episodes)
        ]
        results[seed] = statistics.median(final)
        print(f"seed {seed}: 最终评测中位 {results[seed]:.1f}", flush=True)
        env.close()

    print("\nH1（局部读出）各 seed：", results)
    if results:
        print(f"中位数：{statistics.median(results.values()):.1f}（判据 ≥200）")
        print(
            f"过 200 的 seed：{sum(1 for v in results.values() if v >= 200)}/{len(results)}（判据 ≥8/10）"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))

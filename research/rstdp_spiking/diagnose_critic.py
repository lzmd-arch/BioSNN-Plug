"""诊断：Critic 到底有没有学会预测回报。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

## 为什么需要它

"CartPole 分数低"不能区分两件事：**Critic 不预测**，还是**Actor 用不好这个信号**。
这个脚本直接量前者：训练之后，在冻结策略下比较每个状态的 ``V(s_t)`` 与**蒙特卡洛回报**
``G_t = r_t + γ r_{t+1} + …``，报相关系数与两者的取值范围。

* 相关高（≈1）、``V`` 的动态范围与回报相当 → Critic 学到了，问题在别处；
* 相关≈0 或为负、``V`` 几乎是常数 → Critic 没学到，Actor 收到的是噪声。

## 实测（2026-10-17，800 回合，种子 0）

| Critic | corr(V, G) | V 的范围 | G 的范围 |
| :--- | ---: | :--- | :--- |
| 线性 TD(0) | **+0.965** | 0.8 – 17.2 | 1.0 – 10.5 |
| 群体（gradient 后因子，行归一化） | **−0.496** | 18.3 – 19.9 | 1.0 – 10.5 |
| 群体（同上，**关掉**行归一化） | +0.217 | 12.0 – 18.2 | 1.0 – 27.5 |

⚠️ 群体 Critic 的 ``V`` 动态范围只有 1.6（上限 200），**几乎是常数**——这正是
"它收不到有用信号"的直接证据。
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from research.components.cartpole_encoding import (
    CARTPOLE_HALF_WIDTHS,
    encode_state,
    population_centers,
)
from research.rstdp_spiking.actor import SpikingActor
from research.rstdp_spiking.critic import LinearCritic
from research.rstdp_spiking.pop_critic import PopulationCritic

__all__ = ["diagnose", "train_and_correlate"]


def _build_critic(
    kind: str,
    n_features: int,
    *,
    critic_rate: float,
    normalize: bool,
    units: int = 64,
    bias: float = 0.4,
    gain: float = 8.0,
    generator=None,
):
    if kind == "td":
        return LinearCritic(n_features, gamma=0.99, learning_rate=critic_rate)
    return PopulationCritic(
        n_features,
        n_units=units,
        gain=gain,
        bias=bias,
        gamma=0.99,
        learning_rate=critic_rate,
        trace_post_factor="gradient",
        normalize=normalize,
        generator=generator,
    )


def train_and_correlate(
    kind: str,
    *,
    seed: int = 0,
    episodes: int = 800,
    critic_rate: float = 0.1,
    normalize: bool = True,
    eval_episodes: int = 20,
    features: int = 64,
    pop_units: int = 64,
    pop_bias: float = 0.4,
    pop_gain: float = 8.0,
) -> dict:
    """训练后报 ``corr(V, G)`` 与两者的取值范围。"""
    import gymnasium

    generator = torch.Generator().manual_seed(seed)
    env = gymnasium.make("CartPole-v1")
    centers = population_centers(features, half_widths=CARTPOLE_HALF_WIDTHS, generator=generator)
    actor = SpikingActor(features, 2, steps=25, learning_rate=1e-3, generator=generator)
    critic = _build_critic(
        kind,
        features,
        critic_rate=critic_rate,
        normalize=normalize,
        units=pop_units,
        bias=pop_bias,
        gain=pop_gain,
        generator=generator,
    )

    for episode in range(episodes):
        raw, _ = env.reset(seed=episode)
        f = encode_state(torch.tensor(raw), centers, 1.0)
        for _ in range(500):
            action = actor.step(f, generator=generator, learn_trace=True)
            raw2, reward, terminated, truncated, _ = env.step(action)
            f2 = encode_state(torch.tensor(raw2), centers, 1.0)
            delta = critic.update(f, float(reward), f2, terminated=terminated)
            actor.apply_success_signal(delta)
            f = f2
            if terminated or truncated:
                break
        actor.end_episode()
        if hasattr(critic, "reset_trace"):
            critic.reset_trace()

    values: list[float] = []
    returns: list[float] = []
    for episode in range(1000, 1000 + eval_episodes):
        raw, _ = env.reset(seed=episode)
        f = encode_state(torch.tensor(raw), centers, 1.0)
        trajectory = []
        for _ in range(500):
            action = actor.step(f, generator=generator, greedy=True, learn_trace=False)
            raw2, reward, terminated, truncated, _ = env.step(action)
            f2 = encode_state(torch.tensor(raw2), centers, 1.0)
            trajectory.append((critic.value(f), float(reward)))
            f = f2
            if terminated or truncated:
                break
        running = 0.0
        for value, reward in reversed(trajectory):
            running = reward + 0.99 * running
            values.append(value)
            returns.append(running)

    v = np.asarray(values)
    g = np.asarray(returns)
    corr = float(np.corrcoef(v, g)[0, 1]) if v.std() > 0 and g.std() > 0 else float("nan")
    env.close()
    return {
        "critic": kind,
        "normalize": normalize,
        "units": pop_units,
        "bias": pop_bias,
        "gain": pop_gain,
        "corr": corr,
        "v_min": float(v.min()),
        "v_max": float(v.max()),
        "g_min": float(g.min()),
        "g_max": float(g.max()),
        "n": int(v.size),
    }


def diagnose(**kwargs) -> list[dict]:
    """跑三种 Critic，返回三行结果。"""
    return [
        train_and_correlate("td", **kwargs),
        train_and_correlate("pop", normalize=True, **kwargs),
        train_and_correlate("pop", normalize=False, **kwargs),
    ]


def operating_point(**kwargs) -> dict:
    """只跑一个群体 Critic 配置，供筛选/确认使用。"""
    return train_and_correlate("pop", normalize=False, **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Critic 是否学会预测回报")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--critic-rate", type=float, default=0.1)
    parser.add_argument("--pop-units", type=int, default=64)
    parser.add_argument("--pop-bias", type=float, default=0.4)
    parser.add_argument("--pop-gain", type=float, default=8.0)
    parser.add_argument("--single", action="store_true", help="只跑一个群体配置（当前工作点）")
    args = parser.parse_args(argv)

    rows = (
        [
            operating_point(
                seed=args.seed,
                episodes=args.episodes,
                critic_rate=args.critic_rate,
                pop_units=args.pop_units,
                pop_bias=args.pop_bias,
                pop_gain=args.pop_gain,
            )
        ]
        if args.single
        else diagnose(seed=args.seed, episodes=args.episodes, critic_rate=args.critic_rate)
    )
    for row in rows:
        print(
            f"{row['critic']:3s} units={row['units']:3d} bias={row['bias']:+.1f} gain={row['gain']:4.1f} "
            f"normalize={row['normalize']!s:5s} "
            f"corr(V,G)={row['corr']:+.3f}  "
            f"V∈[{row['v_min']:.1f}, {row['v_max']:.1f}]  "
            f"G∈[{row['g_min']:.1f}, {row['g_max']:.1f}]"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

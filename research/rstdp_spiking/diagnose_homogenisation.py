"""诊断：群体 Critic 的单元是否在训练中同质化。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

预注册：[`sweep_results/rstdp_spiking_homogenisation_prereg.md`](../../sweep_results/rstdp_spiking_homogenisation_prereg.md)。

## 它量什么

假设：`gradient` 后因子下痕迹恒非负、读出一律为正 → 每个单元的权重都被推向同一批方向，
彼此趋同，于是 `V` 失去区分状态的能力（实测训练后 `V` 只在一个 3.2 宽的带里）。

三个量，训练前后各算一次：

| 量 | 随机基线 | 同质化的方向 |
| :--- | :--- | :--- |
| 单元间余弦的均值 | ≈ 0，标准差 ≈ 1/√d | 均值绝对值升高 |
| 有效秩 ``exp(H)/n`` | 接近 1 | 下降 |
| 单元活动的两两相关 | ≈ 0 | 升高 |

⚠️ **有效秩用归一化特征值的熵**，不是矩阵的秩——同质化时矩阵的秩仍然是 64。

⚠️ **没有线性 TD 作对照**：它只有一条权重向量，没有"单元之间"可言，同质化对它无定义。
强行比较会给出一个没有意义的数（全部相同的行 ⇒ 余弦恒为 1）。
"""

from __future__ import annotations

import numpy as np
import torch

from research.rstdp_spiking.actor import SpikingActor
from research.rstdp_spiking.encoding import encode_state, population_centers
from research.rstdp_spiking.pop_critic import PopulationCritic

__all__ = ["effective_rank", "mean_pairwise_cosine", "measure", "train_and_measure"]


def mean_pairwise_cosine(weights: torch.Tensor) -> float:
    """``cos(w_i, w_j)``（i<j）的均值。随机单位向量应 ≈ 0。"""
    w = weights.detach().double()
    unit = w / (w.norm(dim=1, keepdim=True) + 1e-12)
    gram = unit @ unit.T
    n = unit.shape[0]
    off_diagonal = gram[~torch.eye(n, dtype=torch.bool, device=unit.device)]
    return float(off_diagonal.mean())


def effective_rank(weights: torch.Tensor) -> float:
    """归一化特征值的熵取指数、再除以单元数。1 = 完全正交，越小越同质。"""
    w = weights.detach().double()
    gram = w @ w.T
    eigenvalues = torch.linalg.eigvalsh(gram).clamp(min=0.0)
    total = eigenvalues.sum()
    if float(total) <= 0:
        return float("nan")
    p = eigenvalues / total
    p = p[p > 0]
    entropy = float(-(p * p.log()).sum())
    return float(np.exp(entropy) / w.shape[0])


def _activity_correlation(matrix: torch.Tensor) -> float:
    """各状态上单元活动的两两相关均值（去均值、按单元标准化）。"""
    a = matrix.detach().double()
    a = a - a.mean(dim=0, keepdim=True)
    a = a / (a.std(dim=0, keepdim=True) + 1e-12)
    corr = (a.T @ a) / a.shape[0]
    n = corr.shape[0]
    off = corr[~torch.eye(n, dtype=torch.bool, device=corr.device)]
    return float(off.mean())


def train_and_measure(
    *,
    seed: int = 0,
    episodes: int = 800,
    units: int = 64,
    bias: float = 0.0,
    gain: float = 16.0,
    critic_rate: float = 0.1,
    features: int = 64,
) -> dict:
    """训练一个群体 Critic，返回训练前后的同质化度量。"""
    import gymnasium

    generator = torch.Generator().manual_seed(seed)
    env = gymnasium.make("CartPole-v1")
    centers = population_centers(features, generator=generator)
    actor = SpikingActor(features, 2, steps=25, learning_rate=1e-3, generator=generator)
    critic = PopulationCritic(
        features,
        n_units=units,
        gain=gain,
        bias=bias,
        gamma=0.99,
        learning_rate=critic_rate,
        trace_post_factor="gradient",
        normalize=False,
        generator=generator,
    )

    before = {
        "cosine": mean_pairwise_cosine(critic.weights),
        "rank": effective_rank(critic.weights),
    }

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
        critic.reset_trace()

    after = {"cosine": mean_pairwise_cosine(critic.weights), "rank": effective_rank(critic.weights)}

    activities = []
    values = []
    for episode in range(1000, 1020):
        raw, _ = env.reset(seed=episode)
        f = encode_state(torch.tensor(raw), centers, 1.0)
        for _ in range(500):
            activities.append(critic.rates(f).detach().clone())
            values.append(critic.value(f))
            action = actor.step(f, generator=generator, greedy=True, learn_trace=False)
            raw2, _reward, terminated, truncated, _ = env.step(action)
            f = encode_state(torch.tensor(raw2), centers, 1.0)
            if terminated or truncated:
                break
    env.close()

    return {
        "before": before,
        "after": after,
        "activity_correlation": _activity_correlation(torch.stack(activities)),
        "value_range": (float(min(values)), float(max(values))),
        "mean_activity": float(torch.stack(activities).mean()),
    }


def measure(**kwargs) -> dict:
    return train_and_measure(**kwargs)


def main() -> int:
    row = train_and_measure()
    print(
        f"单元间余弦 {row['before']['cosine']:+.4f} → {row['after']['cosine']:+.4f}   "
        f"有效秩 {row['before']['rank']:.3f} → {row['after']['rank']:.3f}   "
        f"活动相关 {row['activity_correlation']:+.3f}   "
        f"平均活动 {row['mean_activity']:.3f}   "
        f"V∈[{row['value_range'][0]:.1f}, {row['value_range'][1]:.1f}]"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

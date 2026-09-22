"""容量探针：这个编码上**存在**能解 CartPole 的线性策略吗？

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

## 为什么需要它

W3 的验收没通过（见 [`README.md`](README.md)）。一个待查的假设是：**瓶颈在设计容量，
不在学习规则**——Actor 是一个线性策略，作用在固定的随机群体编码上。R-STDP 解 CartPole
的常见做法是用大得多的特征集，或让 STDP 参与塑造特征本身。

这个假设可以直接判定，不必靠调参猜：

* **若在这个编码上不存在能解 CartPole 的线性策略**，那么无论学习规则多好都不可能达标，
  容量就是瓶颈，该改的是编码/容量，不是规则；
* **若存在**，容量不是瓶颈，问题在信用分配或学习信号那一侧。

## 怎么判定

用 CartPole 的经典启发式控制器当**靶子**：`θ + k·θ̇ > 0` 时向右推，否则向左推。
先确认它本身能跑满分，再用它产生的 (状态, 动作) 训练一个**线性分类器**——分类器的
形式与 :class:`~research.rstdp.rstdp.RSTDPActor` 的动作选择完全一致（`argmax(x·W)`）。
然后在环境里评测这个线性策略能跑多少步。

**这个线性策略能跑多远，就是该编码的线性容量上限。** 它与 R-STDP 学到多远之间的差距，
就是"学习规则没做到"的部分；上限本身低于 200 步的话，那就是"任何规则都做不到"的部分。

## 这不是验收

本模块只做诊断，不产生验收数字。它的输出用来判断"该往哪个方向改"。
"""

from __future__ import annotations

import argparse

import torch

from research.rstdp.cartpole import EPISODE_LIMIT, encode_state

__all__ = ["fit_linear_policy", "heuristic_action", "run_policy"]


def heuristic_action(state: torch.Tensor, velocity_weight: float = 0.5) -> int:
    """经典 CartPole 启发式：``θ + k·θ̇ > 0`` 时向右推（动作 1）。

    CartPole 的状态是 ``[车位置, 车速度, 杆角度, 杆角速度]``。标准做法是朝杆倾斜的方向推，
    并加一点角速度阻尼来抑制振荡。
    """
    angle, angular_velocity = float(state[2]), float(state[3])
    return 1 if angle + velocity_weight * angular_velocity > 0 else 0


def run_policy(env, policy, centers: torch.Tensor, sigma: float, *, episodes: int = 20) -> float:
    """评测一个 ``state -> action`` 的策略，返回平均存活步数。"""
    lengths = []
    for _ in range(episodes):
        raw, _ = env.reset()
        state = torch.tensor(raw, dtype=torch.float32)
        length = 0
        for _ in range(EPISODE_LIMIT):
            action = policy(state, centers, sigma)
            raw, _, terminated, truncated, _ = env.step(action)
            state = torch.tensor(raw, dtype=torch.float32)
            length += 1
            if terminated or truncated:
                break
        lengths.append(length)
    return sum(lengths) / len(lengths)


def collect_expert_data(
    env, centers: torch.Tensor, sigma: float, *, episodes: int, noise: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """跑启发式控制器，收集 ``(编码后的状态, 专家动作)``。

    ``noise`` 给动作加一点随机翻转，用来覆盖状态空间——纯专家轨迹只会经过它自己能走到的
    状态，训练出的分类器在分布外的表现无从判断。
    """
    generator = torch.Generator().manual_seed(seed)
    features, labels = [], []
    for _ in range(episodes):
        raw, _ = env.reset()
        state = torch.tensor(raw, dtype=torch.float32)
        for _ in range(EPISODE_LIMIT):
            action = heuristic_action(state)
            features.append(encode_state(state, centers, sigma))
            labels.append(action)
            if noise > 0 and float(torch.rand(1, generator=generator).item()) < noise:
                action = 1 - action
            raw, _, terminated, truncated, _ = env.step(action)
            state = torch.tensor(raw, dtype=torch.float32)
            if terminated or truncated:
                break
    return torch.stack(features), torch.tensor(labels, dtype=torch.long)


def fit_linear_policy(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    steps: int = 800,
    learning_rate: float = 0.05,
) -> torch.Tensor:
    """逻辑回归拟合 ``(n_features, 2)`` 的线性策略，形式与 Actor 一致（``argmax(x·W)``）。

    第 0 列固定为第 1 列的相反数——两动作的线性打分在 CartPole 里天然是反对称的
    （推左/推右），这样权重就只有 ``n_features`` 个自由度，与 Actor 的同构。
    """
    n_features = features.shape[-1]
    direction = torch.zeros(n_features, requires_grad=True)
    optimizer = torch.optim.Adam([direction], lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad()
        logits = torch.stack([-(features @ direction), features @ direction], dim=-1)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        weights = torch.stack([-direction.detach(), direction.detach()], dim=-1)
    return weights


def linear_policy_from_weights(weights: torch.Tensor):
    """把 ``(n_features, 2)`` 权重包成 ``(state, centers, sigma) -> action`` 的策略。"""

    def policy(state: torch.Tensor, centers: torch.Tensor, sigma: float) -> int:
        features = encode_state(state, centers, sigma)
        return int((features @ weights).argmax().item())

    return policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--feature-counts", type=int, nargs="+", default=[16, 64, 256, 1024])
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument("--expert-episodes", type=int, default=40)
    parser.add_argument("--noise", type=float, default=0.15)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    import gymnasium

    args = build_parser().parse_args(argv)
    env = gymnasium.make("CartPole-v1")
    env.reset(seed=args.seed)

    print("专家启发式本身的表现：")
    expert_mean = run_policy(
        env, lambda s, c, sg: heuristic_action(s), None, 0.0, episodes=args.eval_episodes
    )
    print(f"  启发式控制器：平均 {expert_mean:.1f} 步（上限 {EPISODE_LIMIT}）\n")

    print("该编码上**线性**策略的容量上限：")
    print(f"{'特征数 N':>8} {'σ':>6} {'拟合精度':>10} {'线性策略平均步数':>18}")
    for n_features in args.feature_counts:
        for sigma in args.sigmas:
            generator = torch.Generator().manual_seed(args.seed)
            centers = (torch.rand(n_features, 4, generator=generator) - 0.5) * 3.0
            features, labels = collect_expert_data(
                env,
                centers,
                sigma,
                episodes=args.expert_episodes,
                noise=args.noise,
                seed=args.seed,
            )
            weights = fit_linear_policy(features, labels)
            with torch.no_grad():
                accuracy = float(
                    ((features @ weights).argmax(dim=-1) == labels).float().mean().item()
                )
            mean_steps = run_policy(
                env,
                linear_policy_from_weights(weights),
                centers,
                sigma,
                episodes=args.eval_episodes,
            )
            print(f"{n_features:>8} {sigma:>6} {accuracy:>10.4f} {mean_steps:>18.1f}")

    env.close()
    print(
        "\n读法：线性策略那一列若普遍远低于 200，说明瓶颈在**容量**（该编码上不存在好的线性\n"
        "策略），改学习规则无用；若接近或超过 200，则容量不是瓶颈，问题在信用分配一侧。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

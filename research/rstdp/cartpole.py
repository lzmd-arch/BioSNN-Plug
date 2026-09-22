"""W3 验收运行脚本：R-STDP + TD-LTP Critic 在 CartPole 上。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

用法::

    uv run python -m research.rstdp.cartpole                  # 验收运行（本机 GPU/CPU）
    uv run python -m research.rstdp.cartpole --smoke          # CI 冒烟：几十秒内跑完

验收标准（计划书 §七 第一阶段）：**CartPole 200 步以上，成功偏移 < 10%σR**。

## 为什么 Critic 不是可选项

计划书 §3.2 引 Frémaux et al. (2010)：R-STDP 的成功偏移达 σR 的约 25% 就无法学习，
偏移 < −0.4σR 时性能甚至低于学习前。该敏感性是通用特性，调 STDP 窗口或换权重依赖模型
都解决不了；唯一的结构性解法是**刺激特异性奖励预测**。所以 Actor 的成功信号取
``S = R − V(s)``，``V`` 由 TD-LTP Critic 给出。

## 状态编码是本项目自己的选择

CartPole 的 4 维连续状态用**高斯群体编码**：``N`` 个编码神经元各有各的偏好中心，
活动 ``x_i = exp(−‖s − c_i‖² / 2σ²)``。换成别的编码结果会变，所以种子固定并写进复现记录。

用速率型单元（而不是脉冲型）是一个**化简**：速率型下没有"脉冲先后"，所以 ``STDP(Δt)``
的窗口形状退化成同时刻的乘积，TD-LTP 与 TD-STDP 那处"有无 post-before-pre 分量"的差别
在本实现里无从体现。这一点写进结论的边界。
"""

from __future__ import annotations

import argparse
import time

import torch

from research.common.device import describe_device, measure_peak_memory, select_device
from research.common.provenance import DegradationLog, collect
from research.common.seeding import SeedBook
from research.rstdp.measure_bias import BiasTracker
from research.rstdp.rstdp import RSTDPActor
from research.rstdp.td_ltp import TDLCritic, td_error

__all__ = ["CartPoleAgent", "encode_state", "run_episode"]

#: 计划书 §七 第一阶段的两项判据。
ACCEPTANCE_STEPS = 200
EPISODE_LIMIT = 500


def encode_state(state: torch.Tensor, centers: torch.Tensor, sigma: float) -> torch.Tensor:
    """高斯群体编码：``(4,)`` 或 ``(batch, 4)`` → 活动 ``(N,)`` 或 ``(batch, N)``。

    ``x_i = exp(−‖s − c_i‖² / (2σ²))``。**不做归一化**——归一化会抹掉"离所有中心都远"
    这一信息，而那正是 CartPole 状态空间边缘的情形。
    """
    squared = ((state.unsqueeze(-2) - centers) ** 2).sum(dim=-1)
    return torch.exp(-squared / (2.0 * sigma**2))


class CartPoleAgent:
    """R-STDP Actor + TD-LTP Critic。

    Attributes:
        actor: :class:`~research.rstdp.rstdp.RSTDPActor`。
        critic: :class:`~research.rstdp.td_ltp.TDLCritic`。
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int = 2,
        *,
        actor_learning_rate: float = 1e-2,
        critic_learning_rate: float = 5e-4,
        critic_value_scale: float = 60.0,
        trace_decay: float = 0.9,
        critic_trace_decay: float = 0.9,
        discount: float = 0.99,
        success_signal: str = "td_error",
        normalize_weights: bool = True,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        if success_signal not in ("reward_minus_value", "td_error"):
            raise ValueError(
                f"success_signal 只能是 'reward_minus_value' 或 'td_error'，"
                f"收到 {success_signal!r}。"
            )
        device = device or torch.device("cpu")
        self.actor = RSTDPActor(
            n_features,
            n_actions,
            learning_rate=actor_learning_rate,
            trace_decay=trace_decay,
            normalize=normalize_weights,
            device=device,
            generator=generator,
        )
        self.critic = TDLCritic(
            n_features,
            learning_rate=critic_learning_rate,
            value_scale=critic_value_scale,
            trace_decay=critic_trace_decay,
            device=device,
            # 必须传一个**已播种**的 generator：Critic 是随机初始化（零初始化会零点锁死，
            # 见 td_ltp.py），不播种的话它的初值就不在复现记录覆盖的种子里。
            generator=generator,
        )
        self.discount = float(discount)
        self.success_signal = success_signal

    @torch.no_grad()
    def behave(
        self, features: torch.Tensor, *, generator: torch.Generator, exploration: float
    ) -> tuple[int, bool]:
        """选动作（含探索）。

        Returns:
            ``(action, explored)``。**是否探索必须由这里报出来**，不能让调用方自己重算——
            重算需要复现同一个随机数，而随机数已经被消耗掉了，重算必然错位。上一版就是
            这么写的，结果探索步的判定恒为假，等于把探索动作也当成 Actor 的选择去强化。
        """
        return self.actor.select_action(
            features, generator=generator, exploration=exploration, return_explored=True
        )

    @torch.no_grad()
    def learn_step(
        self,
        features: torch.Tensor,
        action: int,
        reward: float,
        next_features: torch.Tensor,
        *,
        terminated: bool,
        exploring: bool = False,
    ) -> tuple[float, float]:
        """一步学习：先算 δ 与 S，再更新 Critic 与 Actor。

        Returns:
            ``(delta, success_signal)``。

        **探索步不更新 Actor。** 那一步的动作不是 Actor 选的，用它去强化等于在训练噪声；
        这一点不写清楚很容易被当成"探索有助于探索"而留下。
        """
        value = self.critic.value(features)
        if terminated:
            next_value = torch.zeros_like(value)
            bootstrap_reward = reward
        else:
            next_value = self.critic.value(next_features)
            bootstrap_reward = reward

        delta = td_error(bootstrap_reward, value, next_value, discount=self.discount)
        signal = delta if self.success_signal == "td_error" else float(reward - value)

        self.critic.update(features, value, delta)
        if not exploring:
            self.actor.update(features, action, signal)
        return delta, signal

    def end_episode(self) -> None:
        """回合结束：清空两条资格痕迹，不让它跨回合累积。"""
        self.actor.reset_trace()
        self.critic.reset_trace()


def run_episode(
    env,
    agent: CartPoleAgent,
    centers: torch.Tensor,
    sigma: float,
    *,
    generator: torch.Generator,
    exploration: float,
    learn: bool,
    tracker: BiasTracker | None = None,
) -> int:
    """跑一个回合，返回存活步数。"""
    raw_state, _ = env.reset()
    state = torch.tensor(raw_state, dtype=torch.float32)
    features = encode_state(state, centers, sigma)

    steps = 0
    for _ in range(EPISODE_LIMIT):
        action, explored = agent.behave(features, generator=generator, exploration=exploration)
        raw_next, reward, terminated, truncated, _ = env.step(action)
        next_state = torch.tensor(raw_next, dtype=torch.float32)
        next_features = encode_state(next_state, centers, sigma)

        if learn:
            delta, signal = agent.learn_step(
                features,
                action,
                float(reward),
                next_features,
                terminated=terminated or truncated,
                exploring=explored,
            )
            if tracker is not None:
                tracker.record(signal)
            _ = delta

        features = next_features
        steps += 1
        if terminated or truncated:
            break

    if learn:
        agent.end_episode()
    return steps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--n-features", type=int, default=64, help="状态编码神经元数 N")
    parser.add_argument("--encoding-sigma", type=float, default=0.5)
    parser.add_argument("--actor-learning-rate", type=float, default=3e-3)
    parser.add_argument(
        "--critic-learning-rate",
        type=float,
        default=5e-4,
        help="Critic 学习率。**这一项极敏感**：1e-3 与 5e-4 在本任务上差别巨大，见 README",
    )
    parser.add_argument(
        "--critic-value-scale",
        type=float,
        default=60.0,
        help="Critic 输出的放大倍数。权重被归一到 L2=1，所以值域由它决定；"
        "CartPole 的回报可达数百，需要几十倍放大",
    )
    parser.add_argument("--trace-decay", type=float, default=0.9)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument(
        "--success-signal",
        default="td_error",
        choices=["reward_minus_value", "td_error"],
        help="Actor 的成功信号。默认用 TD 误差：CartPole 每步都给 +1 的**密集恒定奖励**，"
        "此时 S = R − ⟨R⟩ 恒为 0 附近、毫无对比度——计划书 §3.2 那个式子是给"
        "稀疏终末奖励的设定写的（Frémaux 2010 的原始场景）",
    )
    parser.add_argument("--exploration-start", type=float, default=0.3)
    parser.add_argument("--exploration-end", type=float, default=0.02)
    parser.add_argument("--evaluation-interval", type=int, default=20)
    parser.add_argument("--evaluation-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument("--smoke", action="store_true", help="CI 冒烟：缩到几十秒")
    return parser


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    args.episodes = 6
    args.n_features = 16
    args.evaluation_interval = 3
    args.evaluation_episodes = 2
    args.device = "cpu"


def main(argv: list[str] | None = None) -> int:
    import gymnasium

    args = build_parser().parse_args(argv)
    if args.smoke:
        apply_smoke_overrides(args)

    book = SeedBook(base=args.seed)
    device = select_device(args.device)

    generator = torch.Generator().manual_seed(book.derive("群体编码中心"))
    # CartPole 的 4 维状态大致落在 [-3, 3]（位置/角度/速度各不同量纲），
    # 这里用 [-1.5, 1.5] 的均匀中心覆盖，编码 σ 由参数控制。
    centers = (torch.rand(args.n_features, 4, generator=generator) - 0.5) * 3.0
    centers = centers.to(device)

    agent = CartPoleAgent(
        args.n_features,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        critic_value_scale=args.critic_value_scale,
        trace_decay=args.trace_decay,
        discount=args.discount,
        success_signal=args.success_signal,
        device=device,
        generator=torch.Generator().manual_seed(book.derive("Actor 与 Critic 初始权重")),
    )

    env = gymnasium.make("CartPole-v1")
    explore_rng = torch.Generator().manual_seed(book.derive("动作探索"))
    tracker = BiasTracker()

    started = time.perf_counter()
    history: list[int] = []
    with measure_peak_memory(device) as memory:
        for episode in range(1, args.episodes + 1):
            progress = episode / args.episodes
            exploration = args.exploration_start + progress * (
                args.exploration_end - args.exploration_start
            )
            steps = run_episode(
                env,
                agent,
                centers,
                args.encoding_sigma,
                generator=explore_rng,
                exploration=exploration,
                learn=True,
                tracker=tracker,
            )
            history.append(steps)

            if episode % args.evaluation_interval == 0:
                mean_recent = sum(history[-args.evaluation_interval :]) / len(
                    history[-args.evaluation_interval :]
                )
                print(
                    f"episode {episode:4d}  近 {args.evaluation_interval} 回合平均步数 "
                    f"{mean_recent:7.1f}  探索率 {exploration:.3f}  "
                    f"痕迹 rms {agent.actor.trace_rms():.4f}",
                    flush=True,
                )

    # 验收：贪心策略下的平均存活步数（不再探索）
    evaluation = [
        run_episode(
            env,
            agent,
            centers,
            args.encoding_sigma,
            generator=explore_rng,
            exploration=0.0,
            learn=False,
        )
        for _ in range(args.evaluation_episodes)
    ]
    env.close()
    elapsed = time.perf_counter() - started
    mean_steps = sum(evaluation) / len(evaluation)

    print(f"\n验收评测（贪心，{args.evaluation_episodes} 个回合）：平均 {mean_steps:.1f} 步")
    print(
        f"训练末期平均（最后 {args.evaluation_interval} 个回合）："
        f"{sum(history[-args.evaluation_interval :]) / args.evaluation_interval:.1f} 步"
    )
    print(f"\n成功偏移：\n{tracker.render()}")

    degradation = DegradationLog()
    if args.smoke:
        degradation.scale_reduction = True
        degradation.notes.append(f"冒烟模式：只用 {args.episodes} 个回合")

    record = collect(
        "rstdp/cartpole",
        seeds=book.render(),
        elapsed_s=elapsed,
        gpu=describe_device(device).render(),
        peak_mb=memory["peak_mb"],
        degradation=degradation,
        notes=[
            f"N={args.n_features}, sigma={args.encoding_sigma}, value_scale={args.critic_value_scale}, "
            f"eta_actor={args.actor_learning_rate}, eta_critic={args.critic_learning_rate}, "
            f"trace_decay={args.trace_decay}, gamma={args.discount}, "
            f"success_signal={args.success_signal}",
            "状态编码：4 维连续状态的高斯群体编码（本项目自己的选择）",
            "速率型单元：STDP 窗口形状与 TD-LTP/TD-STDP 的差别在此化简下无从体现",
        ],
    )
    print("\n" + record.render_block())

    if args.smoke:
        return 0

    passed_steps = mean_steps >= ACCEPTANCE_STEPS
    passed_bias = tracker.tail.passed
    print(
        f"\n验收（计划书 §七）：平均步数 {mean_steps:.1f} >= {ACCEPTANCE_STEPS} → "
        f"{'达到' if passed_steps else '未达到'}；"
        f"成功偏移 |偏移|/σR {tracker.tail.ratio:.4f} < 0.10 → "
        f"{'达到' if passed_bias else '未达到'}。"
    )
    return 0 if (passed_steps and passed_bias) else 1


if __name__ == "__main__":
    raise SystemExit(main())

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
import inspect
import time
from dataclasses import dataclass

import torch

from research.common.device import describe_device, measure_peak_memory, select_device
from research.common.provenance import DegradationLog, collect
from research.common.seeding import SeedBook
from research.rstdp.measure_bias import OFFSET_THRESHOLD, BiasTracker
from research.rstdp.rstdp import RSTDPActor
from research.rstdp.td_ltp import PopulationCritic, TDLCritic, td_error

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
        n_features: int = 64,
        n_actions: int = 2,
        *,
        actor_learning_rate: float = 3e-3,
        critic_learning_rate: float = 5e-4,
        critic_value_scale: float = 200.0,
        trace_decay: float = 0.9,
        critic_trace_decay: float = 0.9,
        discount: float = 0.99,
        success_signal: str = "td_error",
        normalize_weights: bool = True,
        critic_kind: str = "population",
        critic_units: int = 64,
        critic_init_directions: torch.Tensor | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        if critic_kind not in ("population", "single"):
            raise ValueError(f"critic_kind 只能是 'population' 或 'single'，收到 {critic_kind!r}。")
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
        if critic_kind == "population":
            self.critic = PopulationCritic(
                n_features,
                n_units=critic_units,
                learning_rate=critic_learning_rate,
                trace_decay=critic_trace_decay,
                value_scale=critic_value_scale,
                init_directions=critic_init_directions,
                device=device,
                generator=generator,
            )
        else:
            self.critic = TDLCritic(
                n_features,
                learning_rate=critic_learning_rate,
                value_scale=critic_value_scale,
                trace_decay=critic_trace_decay,
                device=device,
                generator=generator,
            )
        self.critic_kind = critic_kind
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
        truncated: bool = False,
        exploring: bool = False,
    ) -> tuple[float, float]:
        """一步学习：先算 δ 与 S，再更新 Critic 与 Actor。

        Returns:
            ``(delta, success_signal)``。

        **探索步不更新 Actor。** 那一步的动作不是 Actor 选的，用它去强化等于在训练噪声；
        这一点不写清楚很容易被当成"探索有助于探索"而留下。

        **终止与截断必须分开**（``terminated`` / ``truncated``）。只有**终止**才把 ``V(s')``
        置零；**截断**是「时间上限到了，回合没结束」，要照常自举。两者混为一谈时，一个
        跑满 ``EPISODE_LIMIT`` 步的回合会在最后一步拿到

            δ = r + γ·0 − V(s') ≈ 1 − 99.3 = **−98**

        （``V≈(1−0.99⁵⁰⁰)/0.01``），而 Actor 的更新量 ``η·δ·e`` 在这一步约为 56 倍典型权重
        ——**策略越好，这一下打得越狠**。实测到的「峰值 500 步 → 崩到 14 步」正是这个形状：
        一旦够到上限，惩罚开始点火。所以这里不合并这两个标志。
        """
        value = self.critic.value(features)
        # 终止（真正结束）才置零；截断（撞上时间上限）照常自举。
        next_value = torch.zeros_like(value) if terminated else self.critic.value(next_features)

        delta = td_error(reward, value, next_value, discount=self.discount)
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
) -> tuple[int, float]:
    """跑一个回合。

    Returns:
        ``(存活步数, 回合内 Actor 资格痕迹的最大幅值)``。

        痕迹必须在**回合进行中**取最大值：``end_episode()`` 会清空它，回合一结束再读就
        恒为 0——第一版就是这么打印的，日志里那一列永远是 0.0000，看起来像"痕迹没在动"，
        实际是测量时机错了。
    """
    raw_state, _ = env.reset()
    # **状态张量必须与 centers 同设备。** centers 会被搬到 agent 所在的设备上，而
    # gymnasium 返回的是 CPU 上的 numpy 数组——不搬就会在 encode_state 里炸
    # "Expected all tensors to be on the same device"。这个 bug 潜伏了很久没被发现，
    # 因为 W3 的验收一直是显式带 --device cpu 跑的，而**默认**设备是 CUDA。
    state = torch.tensor(raw_state, dtype=torch.float32, device=centers.device)
    features = encode_state(state, centers, sigma)

    steps = 0
    max_trace = 0.0
    for _ in range(EPISODE_LIMIT):
        action, explored = agent.behave(features, generator=generator, exploration=exploration)
        raw_next, reward, terminated, truncated, _ = env.step(action)
        next_state = torch.tensor(raw_next, dtype=torch.float32, device=centers.device)
        next_features = encode_state(next_state, centers, sigma)

        if learn:
            delta, signal = agent.learn_step(
                features,
                action,
                float(reward),
                next_features,
                terminated=terminated,
                truncated=truncated,
                exploring=explored,
            )
            if tracker is not None:
                tracker.record(signal)
            max_trace = max(max_trace, agent.actor.trace.abs().max().item())
            _ = delta

        features = next_features
        steps += 1
        if terminated or truncated:
            break

    if learn:
        agent.end_episode()
    return steps, max_trace


#: CLI 的 ``dest`` → :class:`CartPoleAgent` 的构造参数名（两者不一定同名，
#: 例如 ``--actor-normalize`` 的 dest 是 ``actor_normalize``）。
#:
#: **CLI 的默认值一律从类的签名派生**，不在这里另写一遍字面量。
#: 起因是一个真实踩到的坑：\`CartPoleAgent.__init__\` 的 \`actor_learning_rate\` 默认是
#: 1e-2、\`critic_value_scale\` 默认是 60.0，而 argparse 里写的是 3e-3 与 200.0——直接
#: 构造 agent 的人（消融脚本、将来的使用者）会**静默**拿到与 CLI 不同的行为。两个独立的
#: 消融臂第一版脚本都因此对不上基线，各浪费了一轮。
#:
#: 有两份默认值就一定会漂移，所以这里只留一份：类的签名。
CLI_TO_AGENT_PARAM = {
    "n_features": "n_features",
    "actor_learning_rate": "actor_learning_rate",
    "critic_learning_rate": "critic_learning_rate",
    "critic_value_scale": "critic_value_scale",
    "trace_decay": "trace_decay",
    "discount": "discount",
    "success_signal": "success_signal",
    "actor_normalize": "normalize_weights",
    "critic_kind": "critic_kind",
    "critic_units": "critic_units",
}


def agent_default(parameter: str):
    """取 :class:`CartPoleAgent` 某个构造参数的默认值。

    Raises:
        KeyError: 没有这个构造参数，**或者该参数没有默认值**。后一种必须显式报错：
            ``inspect.Parameter.empty`` 是个哨兵对象，拿它当 argparse 的 ``default``
            不会报错，只会让 ``args.<name>`` 变成那个哨兵——实测就是这么炸的
            （``n_features`` 当时是唯一没有默认值的构造参数）。
    """
    signature = inspect.signature(CartPoleAgent.__init__)
    if parameter not in signature.parameters:
        raise KeyError(f"CartPoleAgent 没有构造参数 {parameter!r}。")
    default = signature.parameters[parameter].default
    if default is inspect.Parameter.empty:
        raise KeyError(
            f"CartPoleAgent 的构造参数 {parameter!r} 没有默认值，无法从签名派生 CLI 默认值。"
            f"要么给它一个默认值，要么在 build_parser 里写独立字面量并说明原因。"
        )
    return default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # —— 只有这几项不属于 CartPoleAgent，才写独立字面量 ——
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--encoding-sigma", type=float, default=0.5)
    # —— 以下默认值全部来自类签名，见 CLI_TO_AGENT_PARAM 的说明 ——
    parser.add_argument(
        "--n-features",
        type=int,
        default=agent_default("n_features"),
        help="状态编码神经元数 N",
    )
    parser.add_argument(
        "--actor-learning-rate", type=float, default=agent_default("actor_learning_rate")
    )
    parser.add_argument(
        "--actor-normalize",
        action=argparse.BooleanOptionalAction,
        default=agent_default("normalize_weights"),
        help="Actor 的逐动作列 L1 权重归一化（计划书 §3.2「权重总和恒定」）。"
        "用 --no-actor-normalize 关掉——注意关掉会同时解除容量约束**与**尺度锁定，"
        "权重尺度从 L1=1 变成约 51，所以等价的相对步长要大 lr 约 51 倍",
    )
    parser.add_argument(
        "--critic-learning-rate",
        type=float,
        default=agent_default("critic_learning_rate"),
        help="Critic 学习率。**这一项极敏感**：1e-3 与 5e-4 在本任务上差别巨大，见 README",
    )
    parser.add_argument(
        "--critic-kind",
        default=agent_default("critic_kind"),
        choices=["population", "single"],
        help="Critic 结构：'population' 是论文那样的群体 + 固定读出；'single' 是单单元版"
        "（保留用于对照——两者的差别见 README）",
    )
    parser.add_argument(
        "--critic-units",
        type=int,
        default=agent_default("critic_units"),
        help="群体 Critic 的单元数",
    )
    parser.add_argument(
        "--critic-value-scale",
        type=float,
        default=agent_default("critic_value_scale"),
        help="Critic 输出的量程。群体版里它经**固定读出**换算成 V 的值域上限；"
        "CartPole 在 γ=0.99 下满分策略的值约 100，所以取 200 留余量",
    )
    parser.add_argument("--trace-decay", type=float, default=agent_default("trace_decay"))
    parser.add_argument("--discount", type=float, default=agent_default("discount"))
    parser.add_argument(
        "--success-signal",
        default=agent_default("success_signal"),
        choices=["reward_minus_value", "td_error"],
        help="Actor 的成功信号。默认用 TD 误差：CartPole 每步都给 +1 的**密集恒定奖励**，"
        "此时 S = R − ⟨R⟩ 恒为 0 附近、毫无对比度——计划书 §3.2 那个式子是给"
        "稀疏终末奖励的设定写的（Frémaux 2010 的原始场景）",
    )
    parser.add_argument("--exploration-start", type=float, default=0.3)
    parser.add_argument("--exploration-end", type=float, default=0.02)
    parser.add_argument(
        "--actor-lr-final-fraction",
        type=float,
        default=1.0,
        help="Actor 学习率在训练末降到初始值的这个比例（1.0 = 不衰减）。"
        "实测贪心策略在恒定学习率下剧烈震荡，衰减是冲着这个去的",
    )
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


@dataclass
class TrialResult:
    """一次完整训练 + 贪心评测的结果。

    字段刻意同时包含**逐种子可比的量**（``mean_steps``）与**分布信息**（``min/max``）：
    本问题的种子方差极大（同一配置内实测能差 3.5 倍），只看均值会被离群点带走。
    """

    seed: int
    mean_steps: float
    min_steps: int
    max_steps: int
    offset_ratio: float
    offset: float
    sigma: float
    elapsed_s: float
    peak_mb: float
    seeds_used: str


def build_agent_inputs(
    book: SeedBook,
    *,
    n_features: int,
    encoding_sigma: float,
    critic_units: int,
    device: torch.device,
):
    """采样群体编码中心与 Critic 单元的偏好方向（感受野）。

    抽成独立函数是为了让 :mod:`research.rstdp.sweep` 与 CLI 走**同一条**构造路径——
    两份实现必然会漂移，而种子派生路径一漂移，数字就不可比且不会报错。
    """
    import gymnasium

    generator = torch.Generator().manual_seed(book.derive("群体编码中心"))
    # CartPole 的 4 维状态大致落在 [-3, 3]（位置/角度/速度各不同量纲），
    # 这里用 [-1.5, 1.5] 的均匀中心覆盖，编码 σ 由参数控制。
    centers = ((torch.rand(n_features, 4, generator=generator) - 0.5) * 3.0).to(device)

    direction_rng = torch.Generator().manual_seed(book.derive("Critic 感受野采样"))
    sample_env = gymnasium.make("CartPole-v1")
    sample_env.reset(seed=book.derive("感受野采样环境"))
    sampled = []
    for _ in range(40):
        raw, _ = sample_env.reset()
        for _ in range(EPISODE_LIMIT):
            sampled.append(
                encode_state(
                    torch.tensor(raw, dtype=torch.float32, device=centers.device),
                    centers,
                    encoding_sigma,
                )
            )
            raw, _, term, trunc, _ = sample_env.step(
                int(torch.randint(2, (1,), generator=direction_rng).item())
            )
            if term or trunc:
                break
    sample_env.close()
    sampled = torch.stack(sampled)
    choice = torch.randperm(len(sampled), generator=direction_rng)[:critic_units]
    return centers, sampled[choice].to(device)


def run_trial(
    seed: int,
    *,
    episodes: int = 800,
    n_features: int = 64,
    encoding_sigma: float = 0.5,
    actor_learning_rate: float | None = None,
    actor_normalize: bool | None = None,
    critic_learning_rate: float | None = None,
    critic_kind: str | None = None,
    critic_units: int | None = None,
    critic_value_scale: float | None = None,
    trace_decay: float | None = None,
    discount: float | None = None,
    success_signal: str | None = None,
    exploration_start: float = 0.3,
    exploration_end: float = 0.02,
    actor_lr_final_fraction: float = 1.0,
    evaluation_episodes: int = 10,
    evaluation_interval: int = 20,
    device: torch.device | None = None,
    verbose: bool = False,
    return_agent: bool = False,
):
    """训练一个种子并做贪心评测。

    ``None`` 表示"用 :class:`CartPoleAgent` 的构造默认值"——**不在这里重复写默认值**，
    理由见 :data:`CLI_TO_AGENT_PARAM` 的说明：两份默认值一定会漂移。

    Returns:
        :class:`TrialResult`；``return_agent=True`` 时返回 ``(result, agent)``。
    """
    import gymnasium

    book = SeedBook(base=seed)
    device = device if device is not None else select_device(None)
    defaults = {name: agent_default(name) for name in CLI_TO_AGENT_PARAM.values()}

    if not 0.0 < actor_lr_final_fraction <= 1.0:
        raise ValueError(
            f"actor_lr_final_fraction 应在 (0, 1] 内，收到 {actor_lr_final_fraction}。"
        )

    resolved = {
        "actor_learning_rate": actor_learning_rate,
        "normalize_weights": actor_normalize,
        "critic_learning_rate": critic_learning_rate,
        "critic_kind": critic_kind,
        "critic_units": critic_units,
        "critic_value_scale": critic_value_scale,
        "trace_decay": trace_decay,
        "discount": discount,
        "success_signal": success_signal,
    }
    for key, value in list(resolved.items()):
        if value is None:
            resolved[key] = defaults[key]

    centers, init_directions = build_agent_inputs(
        book,
        n_features=n_features,
        encoding_sigma=encoding_sigma,
        critic_units=resolved["critic_units"],
        device=device,
    )

    agent = CartPoleAgent(
        n_features,
        actor_learning_rate=resolved["actor_learning_rate"],
        critic_learning_rate=resolved["critic_learning_rate"],
        critic_value_scale=resolved["critic_value_scale"],
        trace_decay=resolved["trace_decay"],
        discount=resolved["discount"],
        success_signal=resolved["success_signal"],
        normalize_weights=resolved["normalize_weights"],
        critic_kind=resolved["critic_kind"],
        critic_units=resolved["critic_units"],
        critic_init_directions=init_directions,
        device=device,
        generator=torch.Generator().manual_seed(book.derive("Actor 与 Critic 初始权重")),
    )

    env = gymnasium.make("CartPole-v1")
    # **必须给环境播种。** gymnasium 的 CartPole 有自己的 RNG，不播种的话每次运行的
    # 初始状态与状态转移都不同——上面那一串种子就覆盖不到实验的一半。实测发现过：
    # 同一个 seed 两次跑出 20.4 步与 9.3 步。
    env.reset(seed=book.derive("环境随机种子"))
    env.action_space.seed(book.derive("环境动作空间种子"))
    explore_rng = torch.Generator().manual_seed(book.derive("动作探索"))
    tracker = BiasTracker()

    started = time.perf_counter()
    history: list[int] = []
    with measure_peak_memory(device) as memory:
        actor_lr_base = agent.actor.learning_rate
        for episode in range(1, episodes + 1):
            progress = episode / episodes
            exploration = exploration_start + progress * (exploration_end - exploration_start)
            # **Actor 学习率线性衰减。** 此前它全程恒定，而实测贪心策略在训练中剧烈震荡
            # （一条种子 150 → 73 → 16 → 13 → 500 → 14），说明过程根本不收敛：恒定步长
            # 下它反复找到好解又丢掉，终值只取决于震荡停在哪一相。衰减是直接针对这个的
            # 标准做法——让后期的更新幅度小到能把好的解留住。
            agent.actor.learning_rate = actor_lr_base * (
                1.0 - progress * (1.0 - actor_lr_final_fraction)
            )
            steps, max_trace = run_episode(
                env,
                agent,
                centers,
                encoding_sigma,
                generator=explore_rng,
                exploration=exploration,
                learn=True,
                tracker=tracker,
            )
            history.append(steps)

            if verbose and episode % evaluation_interval == 0:
                recent = history[-evaluation_interval:]
                print(
                    f"episode {episode:4d}  近 {evaluation_interval} 回合平均步数 "
                    f"{sum(recent) / len(recent):7.1f}  探索率 {exploration:.3f}  "
                    f"回合内痕迹峰值 {max_trace:.4f}",
                    flush=True,
                )

    evaluation = [
        run_episode(
            env,
            agent,
            centers,
            encoding_sigma,
            generator=explore_rng,
            exploration=0.0,
            learn=False,
        )[0]
        for _ in range(evaluation_episodes)
    ]
    env.close()
    elapsed = time.perf_counter() - started

    result = TrialResult(
        seed=seed,
        mean_steps=sum(evaluation) / len(evaluation),
        min_steps=min(evaluation),
        max_steps=max(evaluation),
        offset_ratio=tracker.tail.ratio,
        offset=tracker.tail.offset,
        sigma=tracker.tail.sigma,
        elapsed_s=elapsed,
        peak_mb=memory["peak_mb"],
        seeds_used=book.render(),
    )
    if return_agent:
        return result, agent
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        apply_smoke_overrides(args)

    device = select_device(args.device)
    result = run_trial(
        args.seed,
        episodes=args.episodes,
        n_features=args.n_features,
        encoding_sigma=args.encoding_sigma,
        actor_learning_rate=args.actor_learning_rate,
        actor_normalize=args.actor_normalize,
        critic_learning_rate=args.critic_learning_rate,
        critic_kind=args.critic_kind,
        critic_units=args.critic_units,
        critic_value_scale=args.critic_value_scale,
        trace_decay=args.trace_decay,
        discount=args.discount,
        success_signal=args.success_signal,
        exploration_start=args.exploration_start,
        exploration_end=args.exploration_end,
        actor_lr_final_fraction=args.actor_lr_final_fraction,
        evaluation_episodes=args.evaluation_episodes,
        evaluation_interval=args.evaluation_interval,
        device=device,
        verbose=True,
    )

    print(
        f"\n验收评测（贪心，{args.evaluation_episodes} 个回合）："
        f"平均 {result.mean_steps:.1f} 步（最小 {result.min_steps} / 最大 {result.max_steps}）"
    )
    print(
        f"成功偏移：偏移 {result.offset:+.5f}  σR {result.sigma:.5f}  "
        f"|偏移|/σR {result.offset_ratio:.4f}（阈值 < {OFFSET_THRESHOLD}）"
    )

    degradation = DegradationLog()
    if args.smoke:
        degradation.scale_reduction = True
        degradation.notes.append(f"冒烟模式：只用 {args.episodes} 个回合")

    record = collect(
        "rstdp/cartpole",
        seeds=result.seeds_used,
        elapsed_s=result.elapsed_s,
        gpu=describe_device(device).render(),
        peak_mb=result.peak_mb,
        degradation=degradation,
        notes=[
            f"N={args.n_features}, sigma={args.encoding_sigma}, "
            f"value_scale={args.critic_value_scale}, "
            f"eta_actor={args.actor_learning_rate}, eta_critic={args.critic_learning_rate}, "
            f"actor_normalize={args.actor_normalize}, "
            f"trace_decay={args.trace_decay}, gamma={args.discount}, "
            f"success_signal={args.success_signal}",
            "状态编码：4 维连续状态的高斯群体编码（本项目自己的选择）",
            "速率型单元：STDP 窗口形状与 TD-LTP/TD-STDP 的差别在此化简下无从体现",
        ],
    )
    print("\n" + record.render_block())

    if args.smoke:
        return 0

    passed_steps = result.mean_steps >= ACCEPTANCE_STEPS
    passed_bias = result.offset_ratio < OFFSET_THRESHOLD
    print(
        f"\n验收（计划书 §七）：平均步数 {result.mean_steps:.1f} >= {ACCEPTANCE_STEPS} → "
        f"{'达到' if passed_steps else '未达到'}；"
        f"成功偏移 |偏移|/σR {result.offset_ratio:.4f} < {OFFSET_THRESHOLD} → "
        f"{'达到' if passed_bias else '未达到'}。"
    )
    return 0 if (passed_steps and passed_bias) else 1


if __name__ == "__main__":
    raise SystemExit(main())

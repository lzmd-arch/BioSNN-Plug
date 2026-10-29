"""脉冲执行层的 CartPole 训练与评测（计划书 §七 第三阶段「执行层脉冲化」）。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

用法::

    uv run python -m research.rstdp_spiking.cartpole --smoke     # 冒烟
    uv run python -m research.rstdp_spiking.cartpole             # 完整运行

## 评测协议：冻结随机流（计划书修订 ⑪）

脉冲网络的输出本身是随机的（输入是伯努利脉冲编码），**同一份权重跑两次会给出不同动作**。
所以贪心评测必须**冻结随机流**：每个评测回合用固定种子重采同一串输入脉冲、并把环境也按
固定种子重置，使同一份权重给出同一个动作序列。不这么做，量到的是一个随机策略——
速率版在 `boltzmann` 分支上已经踩过一次同类坑。

## 成功信号：先用手写基线，不是 TD-LTP Critic

计划书 §3.2 引 Frémaux et al. (2010)：成功偏移达 σR 的约 25% 就无法学习，结构性解法是
**刺激特异性奖励预测**。真正的 TD-LTP 群体 Critic 是第三阶段的另一个任务，**本文件没有**
——这里用**回报的指数滑动平均**当基线，``S = G − baseline``。

⚠️ 这条边界要写进结论：本文件的数字说明的是「脉冲化之后的 R-STDP 能不能学 CartPole」，
**不能**用来论证 TD-LTP 的必要性——那需要把 Critic 接上再跑一次。

## 与速率版的关系

并列，不是替换。`research/rstdp/` 的速率版保留为对照基线（验收 237.8 步）。
两个主题**互不 import**（仓库约定：只共享 `research/common/`）。
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from research.common.provenance import collect
from research.components.cartpole_encoding import (
    CARTPOLE_HALF_WIDTHS,
    encode_state,
    population_centers,
)
from research.rstdp_spiking.actor import SpikingActor
from research.rstdp_spiking.critic import LinearCritic
from research.rstdp_spiking.linear_policy import LinearRatePolicy
from research.rstdp_spiking.pop_critic import PopulationCritic

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "rstdp_spiking"

#: 计划书 §七 第一阶段的判据，速率版与脉冲版共用。
ACCEPTANCE_STEPS = 200
EPISODE_LIMIT = 500

__all__ = ["evaluate", "run_episode", "train"]


@dataclass(frozen=True)
class TrainResult:
    seed: int
    returns: list[float]
    eval_scores: list[dict]
    best_steps: float
    selected_episode: int
    eval_noise: tuple[float, ...] = ()
    """同一份最终权重、20 个评测种子各跑一次的平均步数（诊断 ③ 用）。空 = 没跑。"""
    entropies: tuple[float, ...] = ()
    """每回合 Boltzmann 动作分布的平均熵（诊断 ② 用）。空 = 没记。"""
    probes: tuple[dict, ...] = ()
    """每 ``probe_every`` 回合在固定极角 ±0.2 上的窗口探针（D5 诊断用）。空 = 没记。"""
    actor: Any = field(default=None, repr=False, compare=False)
    """``keep_actor=True`` 时的最终 actor（已载入最好权重）。诊断 ① 用。"""
    centers: Any = field(default=None, repr=False, compare=False)
    """与 actor 配套的群体编码中心。诊断 ① 必须用同一组中心。"""


def _state_tensor(raw, device: torch.device) -> torch.Tensor:
    return torch.tensor(raw, dtype=torch.float32, device=device)


def run_episode(
    env,
    actor: SpikingActor,
    centers: torch.Tensor,
    sigma: float,
    *,
    generator: torch.Generator,
    learn: bool,
    greedy: bool = False,
    env_seed: int | None = None,
    critic: LinearCritic | PopulationCritic | None = None,
) -> float:
    """跑一个回合，返回存活步数。

    ``env_seed`` 非 ``None`` 时按该种子重置环境——评测路径必须传它，与固定的 ``generator``
    一起构成**冻结随机流**。
    """
    raw, _ = env.reset(seed=env_seed)
    features = encode_state(_state_tensor(raw, centers.device), centers, sigma)

    steps = 0
    for _ in range(EPISODE_LIMIT):
        action = actor.step(features, generator=generator, greedy=greedy, learn_trace=learn)
        raw_next, reward, terminated, truncated, _ = env.step(action)
        next_features = encode_state(_state_tensor(raw_next, centers.device), centers, sigma)
        if learn and critic is not None:
            # 逐步的演员-评论家：δ 就是成功信号，每一步都更新
            delta = critic.update(features, float(reward), next_features, terminated=terminated)
            actor.apply_success_signal(delta)
        features = next_features
        steps += 1
        if terminated or truncated:
            break
    return float(steps)


def evaluate(
    actor: SpikingActor,
    centers: torch.Tensor,
    sigma: float,
    *,
    episodes: int,
    seed: int,
) -> float:
    """冻结随机流下的贪心评测，返回平均存活步数。

    每个评测回合：环境 ``reset(seed=seed + k)``、脉冲发生器 ``manual_seed(seed * 1000 + k)``。
    同一份权重、同一个 ``seed`` → 同一串动作。
    """
    import gymnasium

    env = gymnasium.make("CartPole-v1")
    scores: list[float] = []
    for k in range(episodes):
        generator = torch.Generator(device="cpu").manual_seed(seed * 1000 + k)
        scores.append(
            run_episode(
                env,
                actor,
                centers,
                sigma,
                generator=generator,
                learn=False,
                greedy=True,
                env_seed=seed + k,
            )
        )
    env.close()
    return float(sum(scores) / len(scores))


def train(
    *,
    seed: int,
    episodes: int,
    eval_every: int,
    eval_episodes: int,
    n_features: int,
    sigma: float,
    steps: int,
    learning_rate: float,
    baseline_decay: float,
    critic_kind: str,
    critic_rate: float,
    gamma: float,
    pop_post: str = "rate",
    pop_readout: str = "uniform",
    pop_normalize: bool = True,
    pop_units: int = 64,
    pop_bias: float = 0.4,
    pop_gain: float = 8.0,
    pop_homeostasis: bool = False,
    pop_homeostatic_rate: float = 0.01,
    pop_activity_target: float = 0.1,
    pop_trace_demean: bool = False,
    pop_trace_demean_rate: float = 0.01,
    pop_mask_density: float | None = None,
    warmup: int,
    device: torch.device,
    eval_noise_seeds: int = 0,
    entropy_log: bool = False,
    probe_every: int = 0,
    keep_actor: bool = False,
    actor_after: str = "best",
    trace_rule: str = "pair-stdp",
    policy: str = "spiking",
    readout: str = "count",
    logit_scale_start: float | None = 1.0,
    logit_scale_end: float | None = 40.0,
) -> TrainResult:
    """训练一个种子。返回逐回合回报与评测曲线。"""
    import gymnasium

    generator = torch.Generator(device="cpu").manual_seed(seed)
    env = gymnasium.make("CartPole-v1")

    centers = population_centers(
        n_features, half_widths=CARTPOLE_HALF_WIDTHS, generator=generator, device=device
    )
    if policy == "spiking":
        actor = SpikingActor(
            n_features,
            n_actions=2,
            steps=steps,
            learning_rate=learning_rate,
            trace_rule=trace_rule,
            readout=readout,
            generator=generator,
            device=device,
        )
    elif policy == "linear":
        actor = LinearRatePolicy(
            n_features,
            2,
            learning_rate=learning_rate,
            logit_scale=1.0,
            generator=generator,
            device=device,
        )
    else:
        raise ValueError(f"未知的 policy：{policy!r}；可选 'spiking' 或 'linear'。")

    if critic_kind == "td":
        critic = LinearCritic(n_features, gamma=gamma, learning_rate=critic_rate)
    elif critic_kind == "pop":
        critic = PopulationCritic(
            n_features,
            n_units=pop_units,
            gain=pop_gain,
            bias=pop_bias,
            gamma=gamma,
            learning_rate=critic_rate,
            trace_post_factor=pop_post,
            readout=pop_readout,
            normalize=pop_normalize,
            homeostasis=pop_homeostasis,
            homeostatic_rate=pop_homeostatic_rate,
            activity_target=pop_activity_target,
            trace_demean=pop_trace_demean,
            trace_demean_rate=pop_trace_demean_rate,
            mask_density=pop_mask_density,
            generator=generator,
            device=device,
        )
    else:
        critic = None

    returns: list[float] = []
    entropies: list[float] = []
    probes: list[dict] = []
    eval_scores: list[dict] = []
    baseline = 0.0
    best_steps = -1.0
    selected_episode = -1
    # 选中最好一步时**快照权重**。只记 best_steps 而不存权重，会让诊断 ③ 去评测
    # "训练结束时已经游走的权重"，而不是报告里那个 156.9 所对应的策略。
    best_state: dict[str, torch.Tensor] | None = None

    for episode in range(episodes):
        if policy == "linear" and logit_scale_start is not None and logit_scale_end is not None:
            progress = episode / episodes
            actor.logit_scale = math.exp(
                math.log(logit_scale_start)
                + progress * (math.log(logit_scale_end) - math.log(logit_scale_start))
            )
        steps_alive = run_episode(
            env,
            actor,
            centers,
            sigma,
            generator=generator,
            learn=True,
            greedy=False,
            critic=critic,
            # ⚠️ 训练回合也必须给环境种子。不给的话 env.reset(seed=None) 会用 gymnasium
            # 在 make() 时由系统熵初始化的内部 RNG，**同一种子两次运行轨迹完全不同**
            # （实测：两次数值全不一样）。这曾让本模块所有数字都不是逐位可复现的。
            env_seed=seed * 100_000 + episode,
        )
        returns.append(steps_alive)
        if entropy_log:
            entropies.append(actor.episode_entropy)

        if critic is None:
            # 无 Critic 的对照：回报的指数滑动平均当基线
            baseline = (1 - baseline_decay) * baseline + baseline_decay * steps_alive
            actor.apply_success_signal(steps_alive - baseline)
        actor.end_episode()
        if hasattr(critic, "reset_trace"):
            critic.reset_trace()

        if probe_every and (episode + 1) % probe_every == 0:
            probes.append(_probe_actor(actor, centers, sigma, seed=seed, episode=episode + 1))

        if warmup <= episode and (episode + 1) % eval_every == 0:
            score = evaluate(actor, centers, sigma, episodes=eval_episodes, seed=10_000 + seed)
            eval_scores.append({"episode": episode + 1, "mean_steps": score})
            if score > best_steps:
                best_steps = score
                selected_episode = episode + 1
                best_state = {
                    key: value.detach().clone() for key, value in actor.state_dict().items()
                }

    env.close()

    # 诊断 ③：同一份权重、不同评测种子各跑一次，量"评测口径本身有多吵"。
    noise: list[float] = []
    if eval_noise_seeds > 0:
        if best_state is not None:
            actor.load_state_dict(best_state)
        for index in range(eval_noise_seeds):
            noise.append(
                evaluate(
                    actor,
                    centers,
                    sigma,
                    episodes=eval_episodes,
                    seed=20_000 + index,
                )
            )

    # 诊断（如 D4 状态依赖）应看**最终**权重：best_state 是按评测最大值选的，
    # 会被噪声抬高，用它做诊断等于挑了一个幸运快照。
    if keep_actor and best_state is not None and actor_after == "best":
        actor.load_state_dict(best_state)

    return TrainResult(
        seed=seed,
        returns=returns,
        eval_scores=eval_scores,
        best_steps=best_steps,
        selected_episode=selected_episode,
        eval_noise=tuple(noise),
        entropies=tuple(entropies),
        probes=tuple(probes),
        actor=actor if keep_actor else None,
        centers=centers if keep_actor else None,
    )


@torch.no_grad()
def _probe_actor(
    actor: SpikingActor,
    centers: torch.Tensor,
    sigma: float,
    *,
    seed: int,
    episode: int,
    repeats: int = 5,
) -> dict:
    """在固定极角 ±0.2 上采几个冻结窗口：后突触脉冲数与贪心动作（D5 诊断用）。"""
    record: dict = {"episode": episode}
    for label, angle in (("pos", 0.2), ("neg", -0.2)):
        state = torch.tensor([0.0, 0.0, angle, 0.0])
        features = encode_state(state, centers, sigma)
        generator = torch.Generator(device="cpu").manual_seed(seed * 10_000 + episode + (angle > 0))
        left = 0
        spikes = 0.0
        for _ in range(repeats):
            _, post = actor.window_spikes(features, generator=generator)
            counts = post.sum(dim=0).squeeze(0)
            spikes += float(counts.sum())
            left += int(counts.argmax().item() == 0)
        record[f"p_left_{label}"] = left / repeats
        record[f"spikes_{label}"] = spikes / repeats
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="脉冲执行层 CartPole")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=25, help="每个决策的脉冲窗口长度")
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument(
        "--policy",
        choices=("spiking", "linear"),
        default="spiking",
        help="动作策略：spiking=脉冲计数（默认）；linear=D2 的线性 Boltzmann 对照",
    )
    parser.add_argument(
        "--readout",
        choices=("count", "rate"),
        default="count",
        help="策略读出：count=窗口内硬计数（默认）；rate=窗口末低通发放率（D3 实验）",
    )
    parser.add_argument("--logit-scale-start", type=float, default=1.0)
    parser.add_argument("--logit-scale-end", type=float, default=40.0)
    parser.add_argument(
        "--trace-rule",
        choices=("pair-stdp", "sign-consistent", "rate-features", "count-aligned"),
        default="pair-stdp",
        help="Actor 痕迹规则：pair-stdp=历史的有符号规则（默认）；"
        "sign-consistent=按成功信号符号选 LTP/LTD 两相；"
        "rate-features=速率版形状的 x·a（诊断对照）；count-aligned=计数同向 x⊗[z⊙(1[a]−π)]",
    )
    parser.add_argument("--baseline-decay", type=float, default=0.05)
    parser.add_argument(
        "--critic",
        choices=("td", "pop", "ema"),
        default="td",
        help="td=线性 TD；pop=TD-LTP 群体 Critic；ema=只用回报基线（对照）",
    )
    parser.add_argument("--critic-rate", type=float, default=1e-2, help="Critic 的步长 β")
    parser.add_argument(
        "--pop-post",
        choices=("rate", "gradient"),
        default="rate",
        help="群体 Critic 痕迹的第二因子",
    )
    parser.add_argument(
        "--pop-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="群体 Critic 权重是否行归一化",
    )
    parser.add_argument("--pop-units", type=int, default=64, help="群体 Critic 的单元数")
    parser.add_argument("--pop-bias", type=float, default=0.4, help="群体 Critic 的单元阈值 b")
    parser.add_argument("--pop-gain", type=float, default=8.0, help="群体 Critic 的增益 g")
    parser.add_argument(
        "--pop-homeostasis",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="单元级活动归一化（稳态可塑性）",
    )
    parser.add_argument("--pop-homeostatic-rate", type=float, default=0.01, help="稳态步长 β")
    parser.add_argument("--pop-activity-target", type=float, default=0.1, help="目标活动率 y*")
    parser.add_argument(
        "--pop-trace-demean",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="每单元痕迹去直流（完全局部）",
    )
    parser.add_argument("--pop-trace-demean-rate", type=float, default=0.01)
    parser.add_argument(
        "--pop-mask-density", type=float, default=None, help="固定输入子空间密度（None=不限制）"
    )
    parser.add_argument(
        "--pop-readout",
        choices=("uniform", "random"),
        default="uniform",
        help="群体 Critic 的读出（random=有符号，须配 gradient）",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true", help="小规模冒烟")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    if args.smoke:
        args.seeds = "0"
        args.episodes = 6
        args.eval_every = 3
        args.eval_episodes = 2
        args.warmup = 1
        # ⚠️ 冒烟**不许**落进默认结果目录。曾经发生过：一次 `--smoke --critic pop` 把
        # 6 回合的冒烟结果写进了 `sweep_results/rstdp_spiking/`，覆盖了 5 种子 × 1200 回合的
        # 正式记录，而那份残留还被当成结果提交了。默认目录只收正式运行。
        if args.out == OUT:
            args.out = Path(tempfile.mkdtemp(prefix="rstdp_spiking_smoke_"))
            print(f"冒烟结果写到临时目录：{args.out}（不进仓库）")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    device = torch.device(args.device)
    started = time.perf_counter()
    results = []
    for seed in seeds:
        outcome = train(
            seed=seed,
            episodes=args.episodes,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            n_features=args.features,
            sigma=args.sigma,
            steps=args.steps,
            learning_rate=args.learning_rate,
            baseline_decay=args.baseline_decay,
            warmup=args.warmup,
            critic_kind=args.critic,
            critic_rate=args.critic_rate,
            gamma=args.gamma,
            pop_post=args.pop_post,
            pop_readout=args.pop_readout,
            pop_normalize=args.pop_normalize,
            pop_units=args.pop_units,
            pop_bias=args.pop_bias,
            pop_gain=args.pop_gain,
            pop_homeostasis=args.pop_homeostasis,
            pop_homeostatic_rate=args.pop_homeostatic_rate,
            pop_activity_target=args.pop_activity_target,
            pop_trace_demean=args.pop_trace_demean,
            pop_trace_demean_rate=args.pop_trace_demean_rate,
            pop_mask_density=args.pop_mask_density,
            device=device,
            trace_rule=args.trace_rule,
            policy=args.policy,
            readout=args.readout,
            logit_scale_start=args.logit_scale_start,
            logit_scale_end=args.logit_scale_end,
        )
        results.append(outcome)
        print(
            f"seed {seed}: best eval {outcome.best_steps:.1f} steps"
            f" @ episode {outcome.selected_episode}"
        )

    summary = {
        "experiment": "spiking R-STDP on CartPole (episodic, EMA success signal)",
        "acceptance_steps": ACCEPTANCE_STEPS,
        "config": {
            "episodes": args.episodes,
            "eval_every": args.eval_every,
            "eval_episodes": args.eval_episodes,
            "features": args.features,
            "sigma": args.sigma,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "baseline_decay": args.baseline_decay,
            "critic": args.critic,
            "critic_rate": args.critic_rate,
            "pop_post": args.pop_post,
            "pop_readout": args.pop_readout,
            "pop_normalize": args.pop_normalize,
            "pop_units": args.pop_units,
            "pop_bias": args.pop_bias,
            "pop_gain": args.pop_gain,
            "pop_homeostasis": args.pop_homeostasis,
            "pop_homeostatic_rate": args.pop_homeostatic_rate,
            "pop_activity_target": args.pop_activity_target,
            "pop_trace_demean": args.pop_trace_demean,
            "pop_trace_demean_rate": args.pop_trace_demean_rate,
            "pop_mask_density": args.pop_mask_density,
            "gamma": args.gamma,
            "warmup": args.warmup,
            "trace_rule": "rate-centered" if args.policy == "linear" else args.trace_rule,
            "policy": args.policy,
            "readout": args.readout,
            "logit_scale_start": args.logit_scale_start,
            "logit_scale_end": args.logit_scale_end,
        },
        "per_seed": [
            {
                "seed": r.seed,
                "best_steps": r.best_steps,
                # best_steps 是 32 次评测取最大，**会被评测噪声抬高**；末次与末三段均值
                # 才是「有没有持续学会」。两者同报，别再只拿 best 下结论。
                "last_steps": r.eval_scores[-1]["mean_steps"] if r.eval_scores else None,
                "final3_mean_steps": (
                    sum(point["mean_steps"] for point in r.eval_scores[-3:])
                    / min(3, len(r.eval_scores))
                    if r.eval_scores
                    else None
                ),
                "selected_episode": r.selected_episode,
                "probes": list(r.probes),
                "eval_scores": r.eval_scores,
                "returns": r.returns,
            }
            for r in results
        ],
        "mean_best_steps": sum(r.best_steps for r in results) / len(results),
        "mean_last_steps": (
            sum(r.eval_scores[-1]["mean_steps"] for r in results if r.eval_scores)
            / max(1, sum(1 for r in results if r.eval_scores))
        ),
        "mean_final3_steps": (
            sum(
                sum(point["mean_steps"] for point in r.eval_scores[-3:])
                / min(3, len(r.eval_scores))
                for r in results
                if r.eval_scores
            )
            / max(1, sum(1 for r in results if r.eval_scores))
        ),
        "passed": sum(r.best_steps >= ACCEPTANCE_STEPS for r in results),
        "repro": collect(
            "W3 脉冲化：CartPole（第三阶段）",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "episodic R-STDP，成功信号 = 回报 − EMA 基线（**TD-LTP Critic 尚未接入**）",
                "评测冻结随机流（环境种子 + 脉冲发生器种子）",
                "Actor 痕迹规则："
                + ("rate-centered（线性策略自带）" if args.policy == "linear" else args.trace_rule),
                f"动作策略：{args.policy}",
            ],
            device=device,
        ).render_block(),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    for r in results:
        (args.out / f"seed{r.seed}.json").write_text(
            json.dumps(
                {
                    "seed": r.seed,
                    "best_steps": r.best_steps,
                    "selected_episode": r.selected_episode,
                    "probes": list(r.probes),
                    "eval_scores": r.eval_scores,
                    "returns": r.returns,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        f"\nmean best = {summary['mean_best_steps']:.1f} steps"
        f"  ({summary['passed']}/{len(results)} ≥ {ACCEPTANCE_STEPS})"
    )
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

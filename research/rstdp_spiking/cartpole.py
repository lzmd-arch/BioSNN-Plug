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
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from research.common.provenance import collect
from research.rstdp_spiking.actor import SpikingActor
from research.rstdp_spiking.critic import LinearCritic
from research.rstdp_spiking.encoding import encode_state, population_centers
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
    warmup: int,
    device: torch.device,
) -> TrainResult:
    """训练一个种子。返回逐回合回报与评测曲线。"""
    import gymnasium

    generator = torch.Generator(device="cpu").manual_seed(seed)
    env = gymnasium.make("CartPole-v1")

    centers = population_centers(n_features, generator=generator, device=device)
    actor = SpikingActor(
        n_features,
        n_actions=2,
        steps=steps,
        learning_rate=learning_rate,
        generator=generator,
        device=device,
    )

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
            generator=generator,
            device=device,
        )
    else:
        critic = None

    returns: list[float] = []
    eval_scores: list[dict] = []
    baseline = 0.0
    best_steps = -1.0
    selected_episode = -1

    for episode in range(episodes):
        steps_alive = run_episode(
            env,
            actor,
            centers,
            sigma,
            generator=generator,
            learn=True,
            greedy=False,
            critic=critic,
        )
        returns.append(steps_alive)

        if critic is None:
            # 无 Critic 的对照：回报的指数滑动平均当基线
            baseline = (1 - baseline_decay) * baseline + baseline_decay * steps_alive
            actor.apply_success_signal(steps_alive - baseline)
        actor.end_episode()
        if hasattr(critic, "reset_trace"):
            critic.reset_trace()

        if warmup <= episode and (episode + 1) % eval_every == 0:
            score = evaluate(actor, centers, sigma, episodes=eval_episodes, seed=10_000 + seed)
            eval_scores.append({"episode": episode + 1, "mean_steps": score})
            if score > best_steps:
                best_steps = score
                selected_episode = episode + 1

    env.close()
    return TrainResult(
        seed=seed,
        returns=returns,
        eval_scores=eval_scores,
        best_steps=best_steps,
        selected_episode=selected_episode,
    )


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
            device=device,
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
            "gamma": args.gamma,
            "warmup": args.warmup,
        },
        "per_seed": [
            {
                "seed": r.seed,
                "best_steps": r.best_steps,
                "selected_episode": r.selected_episode,
                "eval_scores": r.eval_scores,
                "returns": r.returns,
            }
            for r in results
        ],
        "mean_best_steps": sum(r.best_steps for r in results) / len(results),
        "passed": sum(r.best_steps >= ACCEPTANCE_STEPS for r in results),
        "repro": collect(
            "W3 脉冲化：CartPole（第三阶段）",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "episodic R-STDP，成功信号 = 回报 − EMA 基线（**TD-LTP Critic 尚未接入**）",
                "评测冻结随机流（环境种子 + 脉冲发生器种子）",
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

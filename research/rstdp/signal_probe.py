"""成功信号的质量探针：Critic 准不准、δ 是不是优势估计、Actor 的方向对不对。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

与 :mod:`research.rstdp.capacity_probe` 并列：那个问「这个编码上**存在**好策略吗」，这个问
「**学习信号**够不够好」。只诊断，不产生验收数字。

## 为什么这里能拿到**精确**的真值

CartPole-v1 的动力学是**确定性的**——只有初始状态是随机的。所以给定状态与策略，
``V^π(s)``、``Q^π(s,a)``、优势 ``A = Q − V^π`` 都能用**一次 rollout** 精确算出来
（把 ``env.unwrapped.state`` 摆到目标状态再 ``step``）。**这不是代理梯度**，是环境采样，
不碰「无全局反向传播」那条底线。

这把好几个「太贵没法量」的问题变成几秒钟的事——尤其是**逐状态**的偏置：计划书 §七 的
偏移指标只约束 δ 的**全局均值**，而终止状态只占约 1% 的步数，把它挪动不到 0.07。
全局达标完全盖不住逐状态的偏置。

## 要量的四件事

| 量 | 回答 |
| :--- | :--- |
| 解释方差 `EV(V, V*)` | Critic 到底准不准 |
| 逐状态偏置 `abs(mean(V−V*)) / std(V−V*)` | Actor 那个「必须为零才无偏」的量，在**每个状态**上是不是零 |
| `terminal_delta` | 终止步的 δ。`V` 的下界是 ``value_scale·σ(−gain·bias)``，旋转学不动它，所以真值约 1 时 Critic 只能给到地板 ⇒ 每回合最大的一次更新是常量惩罚 |
| `cos(U, T)` | 规则自己的平均更新方向，与**精确优势加权**的方向差多少 |

前三个是「Critic 坏没坏」，第四个是「Actor 的规则方向对不对」。两者要分开，因为修法
不相交。
"""

from __future__ import annotations

import argparse
import weakref
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from research.rstdp.cartpole import EPISODE_LIMIT, encode_state

__all__ = [
    "ProbeReport",
    "action_value",
    "compare_rules",
    "explained_variance",
    "greedy_policy",
    "probe_states",
    "project_to_l1_tangent",
    "state_value",
    "terminal_delta",
]

#: 一个 ``state -> action`` 的策略，接受 ``(4,)`` 的张量。
Policy = Callable[[torch.Tensor], int]

#: 危险区：真值低于这个数的状态。「动作对不对」在这里才真正决定成败。
DANGER_VALUE = 20.0


#: 已经 ``reset()`` 过的环境。``OrderEnforcing`` 包装器要求先 ``reset()`` 才能 ``step()``，
#: 而探针是**直接摆状态**、不用 ``reset()`` 的返回值的，所以得替它补一次。
_RESET_ENVS: weakref.WeakSet = weakref.WeakSet()


def _ensure_reset(env) -> None:
    """保证环境 ``reset()`` 过一次——**只在第一次**，否则每次 rollout 都会推进环境的 RNG。

    CartPole 的 ``step`` 没有随机性，所以这一次 ``reset()`` 不会影响任何被报告的数字；
    它只是为了让 ``OrderEnforcing`` 放行。
    """
    if env in _RESET_ENVS:
        return
    env.reset()
    _RESET_ENVS.add(env)


def _place(env, state) -> torch.Tensor:
    """把环境摆到指定状态，返回它的张量形式。

    ``_elapsed_steps`` 一并归零，好让 ``EPISODE_LIMIT`` 从探针状态**重新**计时——
    否则从回合中途取的状态会因为剩余预算不足而拿到偏小的值，跨检查点不可比。
    """
    # 先验类型再碰环境：类型不对时不该有任何副作用（连 ``reset()`` 都不该发生）。
    unwrapped = getattr(env, "unwrapped", env)
    if not hasattr(unwrapped, "state"):
        raise TypeError("探针只支持经典控制的 CartPole（需要 env.unwrapped.state）。")
    _ensure_reset(env)
    unwrapped.state = np.asarray(state, dtype=np.float64)
    unwrapped.steps_beyond_terminated = None
    if hasattr(unwrapped, "_elapsed_steps"):
        unwrapped._elapsed_steps = 0
    return torch.as_tensor(np.asarray(state, dtype=np.float32))


def rollout_return(
    env, state, policy: Policy, *, gamma: float = 0.99, max_steps: int = EPISODE_LIMIT
) -> float:
    """从 ``state`` 出发按 ``policy`` 滚一个回合，返回**折扣**回报。

    折扣与 Critic 的预测目标同口径（γ=0.99），否则 `V` 与 `V*` 不可比。
    """
    current = _place(env, state)
    total, discount = 0.0, 1.0
    for _ in range(max_steps):
        raw, reward, terminated, truncated, _ = env.step(policy(current))
        total += discount * float(reward)
        discount *= gamma
        current = torch.as_tensor(np.asarray(raw, dtype=np.float32))
        if terminated or truncated:
            break
    return total


def state_value(env, state, policy: Policy, *, gamma: float = 0.99) -> float:
    """``V^π(s)``——精确值，一次 rollout。"""
    return rollout_return(env, state, policy, gamma=gamma)


def action_value(env, state, action: int, policy: Policy, *, gamma: float = 0.99) -> float:
    """``Q^π(s,a)``——先强制走 ``a``，再按 ``policy`` 走。"""
    _place(env, state)  # 副作用才是要的：把环境摆到 state
    raw, reward, terminated, truncated, _ = env.step(int(action))
    if terminated or truncated:
        return float(reward)
    return float(reward) + gamma * state_value(env, raw, policy, gamma=gamma)


def advantage(env, state, action: int, policy: Policy, *, gamma: float = 0.99) -> float:
    """``A^π(s,a) = Q^π(s,a) − V^π(s)``。"""
    return action_value(env, state, action, policy, gamma=gamma) - state_value(
        env, state, policy, gamma=gamma
    )


def greedy_policy(actor, centers: torch.Tensor, sigma: float) -> Policy:
    """把 Actor 包成 ``state -> action`` 的确定性策略（评测用的就是它）。"""

    def policy(state: torch.Tensor) -> int:
        return int(actor.scores(encode_state(state, centers, sigma)).argmax().item())

    return policy


def explained_variance(predicted, true) -> float:
    """``1 − Var(真值 − 预测)/Var(真值)``。真值无方差时无定义，返回 ``nan``。

    返回 ``nan`` 而不是 0：那会让「无定义」被当成「很差」，两种情况要分辨得开。
    """
    predicted = np.asarray(predicted, dtype=float)
    true = np.asarray(true, dtype=float)
    variance = float(np.var(true))
    if variance <= 0:
        return float("nan")
    return float(1.0 - np.var(true - predicted) / variance)


def project_to_l1_tangent(direction: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """把 ``direction`` 投影到 ``|w|₁ = const`` 在 ``w`` 处的**切空间**。

    切条件是 ``Σ_i sign(w_i)·v_i = 0``，投影为 ``v − mean(sign(w)·v)·sign(w)``。

    **不投影这个比较就没有意义**：L1 归一化之后只有切向分量能留下来，径向分量会被下一步
    的归一化直接抹掉，而它的模长往往比切向分量大得多——余弦会被它主导。
    """
    direction = np.asarray(direction, dtype=float)
    sign = np.sign(np.asarray(weights, dtype=float))
    sign[sign == 0.0] = 1.0
    return direction - float(np.mean(sign * direction)) * sign


def _subsample(states: np.ndarray, limit: int) -> np.ndarray:
    """等距抽样到 ``limit`` 个。

    **抽样就要说出来**：报告的 ``n_states`` 是实际用的个数，所以「跑完了」与「抽样过了」
    在输出里分得开。等距而非随机，是为了让 ``(seed, 配置)`` 固定时结果可复现。
    """
    if len(states) <= limit:
        return states
    return states[np.linspace(0, len(states) - 1, limit).astype(int)]


def _linear_fit(x, y) -> tuple[float, float]:
    """最小二乘斜率与相关系数。样本不足或 ``x`` 无方差时返回 ``nan``。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or float(np.std(x)) == 0.0:
        return float("nan"), float("nan")
    slope = float(np.polyfit(x, y, 1)[0])
    return slope, float(np.corrcoef(x, y)[0, 1])


@dataclass
class ProbeReport:
    """一次探针的结论。字段名即报告里的列名。"""

    n_states: int
    explained_variance: float
    bias_ratio: float
    value_mae: float
    floor: float
    floor_hit_fraction: float
    negative_weight_fraction: float
    predicted_min: float
    true_value_min: float
    true_value_max: float
    true_value_std: float
    predicted_std: float
    value_min_gap: float
    #: 只在 ``V* >= DANGER_VALUE`` 的那些状态上算的解释方差。
    #:
    #: **为什么要分开算**：探针集被危险区状态主导，如果 ``Var(V*)`` 本来就很小时，EV 会被
    #: 残差方差主导，测出来的「Critic 很差」可能只是「这批状态上真值没什么变化」。

    explained_variance_high: float
    n_high: int
    terminal_delta: float
    interior_delta: float
    delta_advantage_slope: float
    delta_advantage_r: float
    danger_agreement: float
    n_danger: int

    def render(self) -> str:
        return "\n".join(
            [
                f"探针状态 {self.n_states} 个（危险区 {self.n_danger} 个）",
                f"  解释方差 EV(V, V*)        {self.explained_variance:+.4f}",
                f"  逐状态偏置 |mean|/std     {self.bias_ratio:.4f}",
                f"  V 的平均绝对误差           {self.value_mae:.4f}",
                f"  非负权重下的下界           {self.floor:.4f}"
                f"    贴着它的状态占比 {self.floor_hit_fraction:.4f}",
                f"  Critic 权重为负的占比      {self.negative_weight_fraction:.4f}"
                "    ← 不为 0 才说明 V 能低于那个下界",
                f"  V 的取值区间               [{self.predicted_min:.2f}, …]"
                f"    真值下界 {self.true_value_min:.2f}"
                f"    够不着的差距 {self.value_min_gap:+.2f}",
                f"  真值 V* 的区间 / 标准差     [{self.true_value_min:.2f}, "
                f"{self.true_value_max:.2f}] / {self.true_value_std:.2f}",
                f"  预测 V 的标准差            {self.predicted_std:.2f}",
                f"  高价值子集上的 EV          {self.explained_variance_high:+.4f}"
                f"（{self.n_high} 个状态，V* >= {DANGER_VALUE:.0f}）",
                f"  终止步 δ 均值              {self.terminal_delta:+.4f}",
                f"  内部步 δ 均值              {self.interior_delta:+.4f}",
                f"  δ 对**另一动作**优势的斜率 / r  {self.delta_advantage_slope:+.4f} / "
                f"{self.delta_advantage_r:+.4f}",
                "    （策略是确定性的，所以 A(s, 贪心动作) ≡ 0，不能用它做回归）",
                f"  危险区与启发式一致率       {self.danger_agreement:.4f}",
            ]
        )

    def verdict(self) -> str:
        """按预登记阈值给一句话结论（阈值见计划书，先于测量写死）。"""
        if not np.isfinite(self.explained_variance):
            return "解释方差无定义——真值在该探针集上没有方差，换一批状态再测。"
        if self.explained_variance < 0.5:
            return "EV < 0.5 ⇒ **Critic 是瓶颈**（分支 A）。"
        return "EV ≥ 0.5 ⇒ Critic 够准，**瓶颈在 Actor 的更新规则或步长**（分支 B）。"


def probe_states(
    env,
    policy: Policy,
    *,
    episodes: int = 20,
    seed: int = 0,
    danger_window: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """跑 ``episodes`` 个回合，返回 ``(P1 占用, P2 危险区)`` 两组探针状态。

    P1 是 Actor 实际会更新到的状态分布；P2 是**每次终止前 ``danger_window`` 步**，也就是
    「动作对不对」真正决定成败的那一段。
    """
    generator = np.random.default_rng(seed)
    occupied: list[np.ndarray] = []
    danger: list[np.ndarray] = []
    for _ in range(episodes):
        raw, _ = env.reset(seed=int(generator.integers(0, 2**31 - 1)))
        current = torch.as_tensor(np.asarray(raw, dtype=np.float32))
        trajectory: list[np.ndarray] = []
        for _ in range(EPISODE_LIMIT):
            trajectory.append(np.asarray(raw, dtype=float))
            raw, _, terminated, truncated, _ = env.step(policy(current))
            current = torch.as_tensor(np.asarray(raw, dtype=np.float32))
            if terminated or truncated:
                break
        occupied.extend(trajectory)
        danger.extend(trajectory[-danger_window:])
    return np.asarray(occupied), np.asarray(danger)


def expert_states(env, *, episodes: int = 10, noise: float = 0.15, seed: int = 0) -> np.ndarray:
    """P3：经典启发式控制器走过的状态——「好策略必须掌握」的区域。

    这里没有复用 :func:`research.rstdp.capacity_probe.collect_expert_data`：那个返回的是
    **编码后**的特征，而探针需要**原始状态**才能把环境摆回去。启发式规则本身是复用的。
    """
    from research.rstdp.capacity_probe import heuristic_action

    generator = np.random.default_rng(seed)
    states: list[np.ndarray] = []
    for _ in range(episodes):
        raw, _ = env.reset(seed=int(generator.integers(0, 2**31 - 1)))
        current = torch.as_tensor(np.asarray(raw, dtype=np.float32))
        for _ in range(EPISODE_LIMIT):
            states.append(np.asarray(raw, dtype=float))
            action = heuristic_action(current)
            if noise > 0 and float(generator.random()) < noise:
                action = 1 - action
            raw, _, terminated, truncated, _ = env.step(action)
            current = torch.as_tensor(np.asarray(raw, dtype=np.float32))
            if terminated or truncated:
                break
    return np.asarray(states)


def terminal_delta(
    env,
    policy: Policy,
    critic,
    centers: torch.Tensor,
    sigma: float,
    *,
    episodes: int = 20,
    seed: int = 0,
) -> tuple[float, float]:
    """返回 ``(终止步 δ 的均值, 内部步 δ 的均值)``。

    终止步的 δ 是 ``r + γ·0 − V(s_T)``——``V(s')`` 被置零是**终止**的正当处理，问题在于
    ``V`` 本身有个学不动的下界，所以这个数会稳定在一个负值上，且**与动作无关**。
    """
    generator = np.random.default_rng(seed)
    terminal: list[float] = []
    interior: list[float] = []
    for _ in range(episodes):
        raw, _ = env.reset(seed=int(generator.integers(0, 2**31 - 1)))
        current = torch.as_tensor(np.asarray(raw, dtype=np.float32))
        for _ in range(EPISODE_LIMIT):
            action = policy(current)
            features = encode_state(current, centers, sigma)
            value = float(critic.value(features))
            raw, reward, terminated, truncated, _ = env.step(action)
            current = torch.as_tensor(np.asarray(raw, dtype=np.float32))
            if terminated:
                terminal.append(float(reward) - value)
                break
            if truncated:
                interior.append(
                    float(reward)
                    + 0.99 * float(critic.value(encode_state(current, centers, sigma)))
                    - value
                )
                break
            interior.append(
                float(reward)
                + 0.99 * float(critic.value(encode_state(current, centers, sigma)))
                - value
            )
    mean = float(np.mean(terminal)) if terminal else float("nan")
    return mean, float(np.mean(interior)) if interior else float("nan")


def probe(
    env,
    agent,
    centers: torch.Tensor,
    sigma: float,
    *,
    seed: int = 0,
    episodes: int = 20,
    danger_value: float = DANGER_VALUE,
    max_states: int = 300,
) -> ProbeReport:
    """跑一次完整探针。``env`` 必须是 CartPole 实例，调用方负责关闭它。

    ``max_states`` 给探针状态总数封顶：每个状态要一次完整 rollout 才算得出真值，不封顶的
    话一次探针会滚到几十万步。**这是抽样，就要说出来**——报告的 ``n_states`` 是实际用的
    个数，采样是确定性的（按种子等距取），所以同一个 ``(seed, 配置)`` 结果可复现。
    """
    policy = greedy_policy(agent.actor, centers, sigma)
    occupied, danger = probe_states(env, policy, episodes=episodes, seed=seed)
    expert = expert_states(env, episodes=max(2, episodes // 2), seed=seed + 1)

    # **按组分别封顶，不整体封顶**：整体等距抽样会让状态最多的那一组（专家轨迹有上千个
    # 状态）挤掉别的两组，而三组问的正是不同的问题。
    per_group = max(1, max_states // 3)
    states = np.concatenate(
        [
            _subsample(occupied, per_group),
            _subsample(danger, per_group),
            _subsample(expert, per_group),
        ],
        axis=0,
    )
    true_values = np.array([state_value(env, s, policy) for s in states])
    predicted = np.array(
        [
            float(
                agent.critic.value(
                    encode_state(torch.as_tensor(s, dtype=torch.float32), centers, sigma)
                )
            )
            for s in states
        ]
    )
    residual = predicted - true_values
    std = float(np.std(residual))
    bias_ratio = abs(float(np.mean(residual))) / std if std > 0 else float("inf")

    floor = (
        agent.critic.value_floor()
        if hasattr(agent.critic, "value_floor")
        else float("-inf")  # 单单元版没有这个地板
    )
    floor_hits = float(np.mean(predicted < floor + 0.5)) if np.isfinite(floor) else 0.0

    # **下界够不够得着是要测的**：``value_floor()`` 只在权重非负时成立，而 ``w += η·δ·e``
    # 会把权重推成负数。所以这里直接量「权重还非负吗」与「V 的最低值比真值的最低值高多少」。
    weights = agent.critic.weights.detach()
    negative_weight_fraction = float((weights < 0).float().mean().item())
    true_value_min = float(np.min(true_values))
    predicted_min = float(np.min(predicted))
    value_min_gap = true_value_min - predicted_min
    high_mask = true_values >= danger_value

    # δ 对精确优势。**注意这里只能取「非贪心动作」的优势**：策略是确定性的，所以
    # ``A(s, π(s)) ≡ 0`` 恒成立（强制走策略自己会走的动作、再按策略走，那就是 ``V^π`` 的定义），
    # 拿它去回归会得到一个全零的自变量——第一版就是这么写的，斜率与 r 都是 NaN。
    # 有意义的对照是**另一个动作**的优势：它问的是「δ 有没有区分开两个动作」。
    deltas, advantages_ = [], []
    for s in states:
        action = policy(torch.as_tensor(s, dtype=torch.float32))
        features = encode_state(torch.as_tensor(s, dtype=torch.float32), centers, sigma)
        value = float(agent.critic.value(features))
        next_state = _step_once(env, s, action)
        if next_state is None:
            delta = 1.0 - value
        else:
            delta = (
                1.0
                + 0.99
                * float(
                    agent.critic.value(
                        encode_state(
                            torch.as_tensor(next_state, dtype=torch.float32), centers, sigma
                        )
                    )
                )
                - value
            )
        other = 1 - int(action)
        deltas.append(delta)
        advantages_.append(action_value(env, s, other, policy) - state_value(env, s, policy))
    slope, correlation = _linear_fit(advantages_, deltas)

    # 危险区按**真值**判定，而不是「终止前 N 步」这个代理——代理里混着一些其实还安全的
    # 状态，而探针要问的是「动作对不对真正决定成败的那一段」。
    from research.rstdp.capacity_probe import heuristic_action

    danger_mask = true_values < danger_value
    danger_states = states[danger_mask]
    agreement = (
        float(
            np.mean(
                [
                    policy(torch.as_tensor(s, dtype=torch.float32))
                    == heuristic_action(torch.as_tensor(s, dtype=torch.float32))
                    for s in danger_states
                ]
            )
        )
        if len(danger_states)
        else float("nan")
    )

    term, interior = terminal_delta(
        env, policy, agent.critic, centers, sigma, episodes=episodes, seed=seed
    )
    return ProbeReport(
        n_states=len(states),
        explained_variance=explained_variance(predicted, true_values),
        bias_ratio=bias_ratio,
        value_mae=float(np.mean(np.abs(residual))),
        floor=floor,
        floor_hit_fraction=floor_hits,
        negative_weight_fraction=negative_weight_fraction,
        predicted_min=predicted_min,
        true_value_min=true_value_min,
        true_value_max=float(np.max(true_values)),
        true_value_std=float(np.std(true_values)),
        predicted_std=float(np.std(predicted)),
        value_min_gap=value_min_gap,
        explained_variance_high=explained_variance(predicted[high_mask], true_values[high_mask]),
        n_high=int(high_mask.sum()),
        terminal_delta=term,
        interior_delta=interior,
        delta_advantage_slope=slope,
        delta_advantage_r=correlation,
        danger_agreement=agreement,
        n_danger=int(danger_mask.sum()),
    )


def _step_once(env, state, action: int):
    """走一步；若这一步就结束则返回 ``None``（调用方按终止处理）。"""
    _place(env, state)  # 副作用才是要的：把环境摆到 state
    raw, _, terminated, truncated, _ = env.step(int(action))
    if terminated or truncated:
        return None
    return raw


def compare_rules(
    env,
    agent,
    centers: torch.Tensor,
    sigma: float,
    *,
    seed: int = 0,
    episodes: int = 20,
    max_states: int = 200,
) -> dict[str, float]:
    """规则自己的平均更新方向 ``U`` 与精确优势加权方向 ``T`` 的余弦。

        T_j = Σ_s w(s)·(Q(s,j) − V^π(s))·x(s)
        U_j = Σ_s w(s)·δ(s, a(s))·1{a(s)=j}·x(s)

    两者都先投影到该列 L1 球面的切空间再比。``cos ≥ 0.5`` ⇒ 方向没问题，去怪信号或步长；
    ``≤ 0.2`` ⇒ 方向不对，去怪 Actor 的规则。**十秒钟就能替掉一轮 10 种子扫描。**
    """
    policy = greedy_policy(agent.actor, centers, sigma)
    occupied, danger = probe_states(env, policy, episodes=episodes, seed=seed)
    per_group = max(1, max_states // 2)
    states = np.concatenate(
        [_subsample(occupied, per_group), _subsample(danger, per_group)], axis=0
    )

    n_features = agent.actor.n_features
    n_actions = agent.actor.n_actions
    target = np.zeros((n_features, n_actions))
    rule = np.zeros((n_features, n_actions))
    for s in states:
        state_tensor = torch.as_tensor(s, dtype=torch.float32)
        features = encode_state(state_tensor, centers, sigma).numpy()
        value_true = state_value(env, s, policy)
        value_critic = float(agent.critic.value(encode_state(state_tensor, centers, sigma)))
        action = policy(state_tensor)
        next_state = _step_once(env, s, action)
        delta = (
            1.0 - value_critic
            if next_state is None
            else 1.0
            + 0.99
            * float(
                agent.critic.value(
                    encode_state(torch.as_tensor(next_state, dtype=torch.float32), centers, sigma)
                )
            )
            - value_critic
        )
        for j in range(n_actions):
            target[:, j] += (action_value(env, s, j, policy) - value_true) * features
            if j == action:
                rule[:, j] += delta * features

    weights = agent.actor.weights.numpy()
    out: dict[str, float] = {}
    for j in range(n_actions):
        u = project_to_l1_tangent(rule[:, j], weights[:, j])
        t = project_to_l1_tangent(target[:, j], weights[:, j])
        denominator = float(np.linalg.norm(u) * np.linalg.norm(t))
        out[f"cos_col{j}"] = float(u @ t / denominator) if denominator > 0 else float("nan")
    cosines = [v for k, v in out.items() if k.startswith("cos_col")]
    out["cos_mean"] = float(np.nanmean(cosines))
    out["rule_norm_over_target_norm"] = (
        float(np.linalg.norm(rule) / np.linalg.norm(target))
        if np.linalg.norm(target) > 0
        else float("nan")
    )
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", default="0-4", help="种子，形如 0-4 或 0,1,2")
    parser.add_argument("--episodes", type=int, default=800, help="训练回合数")
    parser.add_argument("--probe-episodes", type=int, default=20, help="探针 rollout 的回合数")
    parser.add_argument("--max-states", type=int, default=300, help="探针状态总数上限")
    parser.add_argument(
        "--compare-rules", action="store_true", help="额外跑 U/T 余弦面板（逐状态两个优势，较慢）"
    )
    parser.add_argument("--actor-lr-final-fraction", type=float, default=0.1)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return parser


def main(argv: list[str] | None = None) -> int:
    import gymnasium

    from research.common.device import select_device
    from research.rstdp.cartpole import run_trial
    from research.rstdp.sweep import parse_seeds

    args = build_parser().parse_args(argv)
    device = select_device(args.device)
    env = gymnasium.make("CartPole-v1")

    rows: list[tuple[int, float, ProbeReport]] = []
    for seed in parse_seeds(args.seeds):
        # **走 run_trial 而不是自己写训练循环**：诊断必须与训练共用同一条构造路径，
        # 否则「探针量的」可能不是「训练用的」，而那种错误不会报错。
        result, context = run_trial(
            seed,
            episodes=args.episodes,
            actor_lr_final_fraction=args.actor_lr_final_fraction,
            device=device,
            return_context=True,
        )
        report = probe(
            env,
            context.agent,
            context.centers,
            context.encoding_sigma,
            seed=seed,
            episodes=args.probe_episodes,
            max_states=args.max_states,
        )
        print()
        print(f"=== seed {seed}（贪心评测 {result.mean_steps:.1f} 步）===")
        print(report.render())
        if args.compare_rules:
            cosines = compare_rules(
                env,
                context.agent,
                context.centers,
                context.encoding_sigma,
                seed=seed,
                episodes=args.probe_episodes,
                max_states=args.max_states,
            )
            print(
                f"  规则方向 vs 优势方向 cos = {cosines['cos_mean']:+.4f}"
                f"（|U|/|T| = {cosines['rule_norm_over_target_norm']:.4f}）"
            )
        rows.append((seed, result.mean_steps, report))

    env.close()
    _print_summary(rows)
    return 0


def _median(values) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.median(finite)) if finite else float("nan")


def _print_summary(rows: list) -> None:
    """跨种子汇总。判据用**中位数**——本问题的种子分布明显右偏。"""
    if not rows:
        return
    print()
    print("=" * 72)
    print(f"种子 {len(rows)} 个    步数中位数 {_median([s for _, s, _ in rows]):.1f}")
    print(f"  解释方差 EV 中位数        {_median([r.explained_variance for _, _, r in rows]):+.4f}")
    print(f"  逐状态偏置中位数           {_median([r.bias_ratio for _, _, r in rows]):.4f}")
    print(f"  终止步 δ 中位数            {_median([r.terminal_delta for _, _, r in rows]):+.4f}")
    print(
        f"  负数权重占比中位数         "
        f"{_median([r.negative_weight_fraction for _, _, r in rows]):.4f}"
    )
    print(f"  危险区一致率中位数         {_median([r.danger_agreement for _, _, r in rows]):.4f}")
    ev = _median([r.explained_variance for _, _, r in rows])
    verdict = (
        "EV < 0.5 ⇒ Critic 是瓶颈（分支 A）"
        if np.isfinite(ev) and ev < 0.5
        else "EV ≥ 0.5 ⇒ 瓶颈在 Actor 的更新规则或步长（分支 B）"
    )
    print(f"  分叉：{verdict}")
    print("=" * 72)


if __name__ == "__main__":
    raise SystemExit(main())

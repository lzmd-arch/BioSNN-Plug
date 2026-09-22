"""R-STDP：Actor 的学习规则与权重归一化（计划书 §3.2）。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

    Δw_ij = STDP(Δt_ij) · S         （计划书 §3.2 的写法）

其中 ``S`` 是**无偏的成功信号**。Frémaux et al. (2013) 对这个结构的描述：

> In R-STDP, the effects of classic STDP are stored into an exponentially decaying, medium
> term, synapse-specific memory, called an eligibility trace. This trace is only imprinted
> into the actual synaptic weights when a global, neuromodulatory success signal is sent to
> the synapses. In R-STDP, the neuromodulatory signal is the reward minus a baseline,
> i.e., [R − b]. It was shown that for R-STDP to maximize reward, **the baseline must
> precisely match the mean (or expected) reward**.

最后那句就是"Critic 不是可选项"的来源：计划书 §3.2 引 Frémaux et al. (2010) 指出，成功
偏移达 σR 的约 25% 就无法学习，偏移 < −0.4σR 时性能甚至低于学习前。这个敏感性是通用
特性，调 STDP 窗口或换权重依赖模型都解决不了——**唯一的结构性解法是刺激特异性奖励预测**。

## 成功信号取哪一个

两种都实现，由调用方选：

* ``S = R − V(s)``（奖励预测误差）——计划书 §3.2 的读法（"S = R − ⟨R⟩"，Critic 给出 ⟨R⟩）；
* ``S = δ = R + γV(s') − V(s)``（TD 误差）——Frémaux 2013 指出的自然延伸：既然 δ 也是
  奖励预测误差信号，用它替代 S 就把 R-STDP 变成 TD 调制的规则。

两者都无偏（只要 ``V`` 收敛到期望奖励）。差别在视野：前者只看当下，后者往前看一步。

## 速率型下的落地

离散动作的 Actor：每个动作一个单元，后突触活动取**被选中动作的指示量**
``a_j ∈ {0, 1}``。资格痕迹按

    e_ij ← λ·e_ij + x_i · a_j

累积，权重按 ``w_ij ← w_ij + η·S·e_ij`` 更新。这与"经典 STDP 的效果先存进痕迹、只在成功
信号到达时写入权重"是同构的。

**这同样是化简**：速率型下没有"脉冲先后"可言，所以 `STDP(Δt)` 的具体窗口形状在这里退化
成"同时刻的乘积"。窗口参数（τ_+、τ_−）因此不是本实现的自由参数——这是速率型的代价。
"""

from __future__ import annotations

import torch

from research.common.seeding import device_generator

__all__ = ["RSTDPActor"]


class RSTDPActor:
    """离散动作的 R-STDP Actor，带逐神经元权重归一化。

    Attributes:
        weights: ``(n_features, n_actions)``。
        trace: 同形状的资格痕迹。
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        *,
        learning_rate: float = 1e-2,
        trace_decay: float = 0.9,
        weight_norm: float = 1.0,
        normalize: bool = True,
        action_sampling: str = "epsilon_greedy",
        logit_scale: float = 1.0,
        trace_center: str = "none",
        polyak_tau: float | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        if not 0.0 <= trace_decay < 1.0:
            raise ValueError(f"trace_decay 应在 [0, 1) 内，收到 {trace_decay}。")
        if weight_norm <= 0:
            raise ValueError(f"weight_norm 必须为正，收到 {weight_norm}。")
        if polyak_tau is not None and not 0.0 <= polyak_tau < 1.0:
            raise ValueError(f"polyak_tau 应在 [0, 1) 内或为 None，收到 {polyak_tau}。")
        if action_sampling not in ("epsilon_greedy", "boltzmann"):
            raise ValueError(
                f"action_sampling 只能是 'epsilon_greedy' 或 'boltzmann'，收到 {action_sampling!r}。"
            )
        if trace_center not in ("none", "sampling"):
            raise ValueError(f"trace_center 只能是 'none' 或 'sampling'，收到 {trace_center!r}。")
        if logit_scale <= 0:
            raise ValueError(f"logit_scale 必须为正，收到 {logit_scale}。")
        if trace_center == "sampling" and action_sampling != "boltzmann":
            # **这条不是洁癖，是算术。** ε-贪心下真正更新 Actor 的那些步的动作是确定性的
            # ``argmax``，于是 ``a_j ≡ π_j``、``a_j − π_j ≡ 0``——痕迹恒为零，Actor 一步都
            # 不更新。让它静默停机比报错危险得多。
            raise ValueError(
                "trace_center='sampling' 只在 action_sampling='boltzmann' 下有意义："
                "ε-贪心时更新步的动作是确定性的 argmax，a_j − π_j ≡ 0，痕迹会恒为零、"
                "Actor 完全停摆。"
            )

        device = device or torch.device("cpu")
        self.n_features = int(n_features)
        self.n_actions = int(n_actions)
        self.learning_rate = float(learning_rate)
        self.trace_decay = float(trace_decay)
        self.weight_norm = float(weight_norm)
        self.normalize = bool(normalize)
        self.action_sampling = action_sampling
        self.logit_scale = float(logit_scale)
        self.trace_center = trace_center

        generator = device_generator(generator, device)
        self.weights = torch.randn(n_features, n_actions, device=device, generator=generator)
        self.trace = torch.zeros(n_features, n_actions, device=device)
        if self.normalize:
            self._normalize_like_weights(self.weights)

        # **部署权重**（决定动作的权重）。默认时它与 ``self.weights`` 是**同一个张量对象**，
        # 不是副本——所以默认路径逐位不变、也不多占内存。给了 ``polyak_tau`` 才另开一份，
        # 按 ``w_deployed ← τ·w_deployed + (1−τ)·w`` 追踪在线权重（见 :meth:`_normalize`）。
        #
        # 为什么需要它：实测这条线的失败模式是**游走**而不是学不动——同一条轨迹里峰值中位数
        # 218.8 步而终值中位数只有 78.2。也就是「找得到好策略，留不住」。参数平均是这种情形
        # 最便宜的修法，而且它只改**部署**用的那一份：行为策略与学习仍走在线权重，所以不会把
        # 训练本身变成离策略的。
        self.polyak_tau = None if polyak_tau is None else float(polyak_tau)
        self.weights_deployed = self.weights if self.polyak_tau is None else self.weights.clone()

    @torch.no_grad()
    def _normalize_like_weights(self, weights: torch.Tensor) -> None:
        """把每个动作单元（每一列）的入权重 **L1 范数**归一到 ``weight_norm``。

        计划书 §3.2：「每个神经元层面保持权重总和恒定，防止突触动态失控」。取 L1 而不是
        L2：L1 约束的是"权重总和"，与原文措辞一致；L2 约束的是能量，在稀疏编码下会让
        个别权重吃掉全部预算。
        """
        column_sums = weights.abs().sum(dim=0, keepdim=True).clamp_min(1e-12)
        weights.mul_(self.weight_norm / column_sums)

    @torch.no_grad()
    def _normalize(self) -> None:
        """归一化在线权重，并把部署权重同步到最新的平均上。"""
        self._normalize_like_weights(self.weights)
        if self.polyak_tau is not None:
            self.weights_deployed.lerp_(self.weights, 1.0 - self.polyak_tau)
            # 平均之后再归一化一次：两列各自的 L1 范数在平均后不再是常数，而部署时
            # ``argmax`` 比较的是两列的原始得分，尺度不一致就会偏向范数大的那一列。
            self._normalize_like_weights(self.weights_deployed)

    @torch.no_grad()
    def scores(self, features: torch.Tensor) -> torch.Tensor:
        """各动作的得分，``(n_actions,)``。**用的是部署权重**，不是在线权重。"""
        return features @ self.weights_deployed

    @torch.no_grad()
    def action_probabilities(self, features: torch.Tensor) -> torch.Tensor:
        """**采样**动作的分布 ``π``，``(n_actions,)``。

        * ``boltzmann``：``softmax(logit_scale · scores)``。需要 ``logit_scale`` 是因为
          L1 归一化之后两列的得分差很小（量级 0.05–0.5），不放大的话 softmax 几乎是均匀的，
          等于没有策略。
        * ``epsilon_greedy``：``onehot(argmax)``——**注意这是「真正更新 Actor 的那一步」上
          的分布**，与带 ε 的行为分布不同（探索步压根不更新 Actor）。
        """
        scores = self.scores(features)
        if self.action_sampling == "boltzmann":
            return torch.softmax(self.logit_scale * scores, dim=-1)
        indicator = torch.zeros_like(scores)
        indicator[int(scores.argmax().item())] = 1.0
        return indicator

    @torch.no_grad()
    def select_action(
        self,
        features: torch.Tensor,
        *,
        generator: torch.Generator,
        exploration: float = 0.0,
        greedy: bool = False,
        return_explored: bool = False,
    ):
        """按得分选动作，``exploration`` 概率下随机探索。

        ``greedy=True`` 时**无视采样方案**直接返回 ``argmax``——**评测必须走这条路**。

        起因是一个真实踩到的坑：``boltzmann`` 分支根本不看 ``exploration``，所以评测代码
        传 ``exploration=0.0`` 时它**照样采样**。于是 boltzmann 各档的「贪心评测」量的是
        一个**随机策略**的存活步数，而那个尺度的天花板被压得很低——一个满分策略（启发式，
        500 步）在动作准确率 70% 时只值 97.4 步、60% 时只值 36.3 步。那一行结论因此是被
        评测路径本身污染的。

        ``return_explored=True`` 时返回 ``(action, explored)``——调用方需要知道这一步
        是不是探索步（探索步不该拿去强化 Actor），而**只有这里知道**：随机数在这里被消耗，
        外面无法复现。
        """
        if greedy:
            action = int(self.scores(features).argmax().item())
            return (action, False) if return_explored else action

        explored = False
        if self.action_sampling == "boltzmann":
            # **没有「探索步」这回事了**：被采样的动作就是策略自己的选择，所以每一步都该
            # 拿去更新 Actor（``explored`` 恒为 False）。``exploration`` 参数在这个方案下
            # 不使用——探索由 ``logit_scale``（温度的倒数）决定。
            probabilities = self.action_probabilities(features)
            action = int(torch.multinomial(probabilities, 1, generator=generator).item())
        elif exploration > 0 and float(torch.rand(1, generator=generator).item()) < exploration:
            action = int(torch.randint(self.n_actions, (1,), generator=generator).item())
            explored = True
        else:
            action = int(self.scores(features).argmax().item())
        return (action, explored) if return_explored else action

    @torch.no_grad()
    def update(self, features: torch.Tensor, action: int, success_signal: float) -> None:
        """累积资格痕迹，并用成功信号调制后写入权重。

        Args:
            features: ``(n_features,)`` 当前状态的活动。
            action: 被选中的动作下标。
            success_signal: 全局标量 ``S``（见模块 docstring 对两个取法的说明）。
        """
        indicator = torch.zeros(self.n_actions, device=self.weights.device)
        indicator[action] = 1.0
        second_factor = indicator
        if self.trace_center == "sampling":
            # ``a_j − π_j``：这就是策略梯度的得分函数项。它的期望是 0，因此
            # ``E[Δw] = η·E[x·(a−π)·S]`` 里那个「需要 Critic 逐状态无偏」的偏置项
            # ``E[S|s]·x·π`` 被精确消掉；而且未选中的那一列会被**削弱**，这正是当前规则
            # 结构上缺失的「动作比较」。
            second_factor = indicator - self.action_probabilities(features)
        self.trace.mul_(self.trace_decay).add_(torch.outer(features, second_factor))
        self.weights.add_(self.learning_rate * success_signal * self.trace)
        if self.normalize:
            self._normalize()

    @torch.no_grad()
    def reset_trace(self) -> None:
        """回合结束时清空痕迹。"""
        self.trace.zero_()

    @torch.no_grad()
    def trace_rms(self) -> float:
        """痕迹的均方根，用于观察它有没有失控（计划书 §3.2 的"突触动态失控"）。"""
        return float(self.trace.pow(2).mean().sqrt().item())

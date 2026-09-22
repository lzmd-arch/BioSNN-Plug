"""TD-LTP：Critic 的学习规则（Frémaux et al. 2013, Eq. 17）。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

出处由 **ADR-0007** 结项（计划书 §十一 的 P0 待办）。原文逐字：

> Because it has, roughly, the form of "TD error signal × Hebbian LTP", we call this
> learning rule **TD-LTP**.

规则本体（三因子）：

    Δw_i ∝ δ(t) · κ ∗ [ pre_i(t) · post(t) ]      ← 符合窗**仅计 pre-before-post**

## 三因子各是什么

| 因子 | 内容 | 局部性 |
| :--- | :--- | :--- |
| 1 | 前突触脉冲串 | 突触本地 |
| 2 | 后突触脉冲串 | 突触本地 |
| 3 | **δ(t)，全局标量** TD 误差 | 经多巴胺式广播到达每个突触 |

原文对第三因子的说明（这句是"纯局部"主张的依据）：

> The first one is the TD error term, which is the same for all synapses, and can thus be
> considered as a global factor, possibly transmitted by one or more neuromodulators.

**非局部量只有这一个标量**，不含误差向量的反向传播。

## 与 TD-STDP 的实质差别

原文 Figure 2A 的图注：

> Bottom: TD-STDP is a TD-modulated variant of R-STDP. The main difference with TD-LTP is
> the presence of a post-before-pre component in the coincidence window.

即：**TD-LTP 的符合窗只计 pre-before-post，TD-STDP 还含 post-before-pre 分量。**
ADR-0007 的备选 C 就是退回 TD-STDP（改动仅此一处），若本实现在 CartPole 上受阻则启用，
并另开 ADR。

## 速率型实现下的形式

本项目用速率型单元（CartPole 的状态是连续量，群体编码后做速率读出），所以"pre 脉冲串与
post 脉冲串的符合"落地为两者的乘积，再经指数核 κ 滤波成资格痕迹：

    e_i(t) = λ·e_i(t−1) + x_i(t)·y(t)          λ = exp(−dt/τ_κ)
    Δw_i   = η · δ(t) · e_i(t)

**这是一个化简**：脉冲串之间的"符合窗"在速率型下退化成同时刻的乘积，"仅计
pre-before-post"这一条因此不再是可区分的结构（速率型下没有"先后"）。所以本模块
**无法**体现 TD-LTP 与 TD-STDP 的那处差别——如果将来要退回 TD-STDP，那处差别需要
在脉冲型的实现里才谈得上。这一点写进结论的边界，不含糊。
"""

from __future__ import annotations

import math

import torch

from research.common.seeding import device_generator

__all__ = ["PopulationCritic", "TDLCritic", "td_error"]


class TDLCritic:
    """线性 Critic，用 TD-LTP 训练。

    ``V(s) = Σ_i v_i · x_i``（``x`` 是状态编码的活动），按

        e_i ← λ·e_i + x_i·V
        v_i ← v_i + η·δ·e_i

    更新。``δ`` 由外部给出（见 :func:`td_error`），本类只管"用 δ 调制资格痕迹"这件事。

    Attributes:
        weights: ``(n_features,)`` 的 ``v``。
        trace: ``(n_features,)`` 的资格痕迹 ``e``。
    """

    def __init__(
        self,
        n_features: int,
        *,
        learning_rate: float = 1e-3,
        trace_decay: float = 0.9,
        weight_decay: float = 0.0,
        init_scale: float = 0.01,
        weight_norm: float | None = 1.0,
        value_scale: float = 60.0,
        output_bias: float = 0.0,
        bias_learning_rate: float | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        if not 0.0 <= trace_decay < 1.0:
            raise ValueError(f"trace_decay 应在 [0, 1) 内，收到 {trace_decay}。")
        if init_scale <= 0:
            raise ValueError(f"init_scale 必须为正，收到 {init_scale}。")
        if weight_norm is not None and weight_norm <= 0:
            raise ValueError(f"weight_norm 必须为正或 None，收到 {weight_norm}。")
        if value_scale <= 0:
            raise ValueError(f"value_scale 必须为正，收到 {value_scale}。")

        self.n_features = int(n_features)
        self.learning_rate = float(learning_rate)
        self.trace_decay = float(trace_decay)
        self.weight_decay = float(weight_decay)
        self.weight_norm = None if weight_norm is None else float(weight_norm)
        self.value_scale = float(value_scale)
        self.bias_learning_rate = bias_learning_rate
        device = device or torch.device("cpu")
        # 加性偏置 ``V = … + out_bias``。零初始化，且 ``x + 0.0`` 是精确的，所以**默认路径
        # 逐位不变**。它是「纯局部」的：一个标量，与 δ 同样广播，不含非局部向量。
        self.out_bias = torch.tensor(float(output_bias), device=device)

        # **不能用零初始化。** 资格痕迹是 ``e ← λ·e + x·V``，即被 Critic 自己的输出
        # ``V`` 门控；若 ``V ≡ 0``（零初始化且不更新），痕迹恒为 0，权重更新量
        # ``η·δ·e`` 也就恒为 0——Critic 永远学不动。这是三因子结构里"后突触活动作为
        # 第二因子"带来的**零点锁死**，不是调参能绕过的。所以这里用小幅随机初始化，
        # 与论文的随机初始化一致。
        #
        # **还要做权重归一化**（计划书 §3.2 那一句「每个神经元层面保持权重总和恒定，
        # 防止突触动态失控」）。不做的话上面那个正值反馈会失控：V 变大 → e = x·V 变大
        # → Δw 变大 → V 更大。实测就是权重发散成 NaN。把 ||w||₂ 钉住，V 就被
        # ``value_scale · ‖x‖`` 界住，反馈回路断开。
        #
        # ``value_scale`` 用来恢复被归一化拿掉的那个自由度（Critic 的整体增益）：
        # CartPole 的回报可达数百，而 ‖x‖ ≈ √N，所以需要几十倍的放大。
        generator = device_generator(generator, device)
        self.weights = torch.randn(n_features, device=device, generator=generator) * init_scale
        self.trace = torch.zeros(n_features, device=device)
        self._normalize()

    @torch.no_grad()
    def _normalize(self) -> None:
        """把权重向量归一到 L2 范数 ``weight_norm``（计划书 §3.2 的权重归一化）。"""
        if self.weight_norm is None:
            return
        norm = self.weights.norm().clamp_min(1e-12)
        self.weights.mul_(self.weight_norm / norm)

    @torch.no_grad()
    def value(self, features: torch.Tensor) -> torch.Tensor:
        """``V(s) = value_scale · (w·x)``。

        ``features`` 是 ``(n_features,)`` 或 ``(batch, n_features)``。
        """
        return self.value_scale * (features @ self.weights) + self.out_bias

    @torch.no_grad()
    def update(self, features: torch.Tensor, value: torch.Tensor, td_error: float) -> None:
        """累积资格痕迹并用 ``δ`` 调制，就地更新权重。

        Args:
            features: ``(n_features,)`` 当前状态的活动。
            value: 标量 ``V(s)``（本步的预测）。
            td_error: 全局标量 ``δ``。
        """
        # 第二因子用**缩放前**的活动 ``w·x``，不是 ``V`` 本身：否则 trace 会随
        # ``value_scale`` 一起放大，把增益重复计一次。
        activity = value / self.value_scale
        self.trace.mul_(self.trace_decay).add_(features * activity)
        self.weights.add_(self.learning_rate * td_error * self.trace)
        if self.weight_decay:
            self.weights.mul_(1.0 - self.weight_decay)
        self._normalize()
        self._learn_bias(td_error)

    @torch.no_grad()
    def _learn_bias(self, td_error: float) -> None:
        """``out_bias += η_θ·δ``——线性值函数偏置项的教科书半梯度更新。

        ``∂V/∂θ = 1``，半梯度只会经 ``V`` 回传（不含自举项 ``γV(s')``），所以更新就是
        ``+η_θ·δ`` 这一行——与权重更新同号，也和「δ>0 说明 V 偏低、该把 V 抬上去」一致。

        **要注意 η_θ 的量级**：内部步里 δ 只通过 ``(1−γ)`` 依赖 θ，所以那里偏置几乎不动；
        真正把它标定住的是**终止步**（那里 ``V(s')`` 被强制为 0，``∂δ/∂θ = 1``）。而终止步
        每回合只有一次，所以 η_θ 需要比 η_w 大得多才来得及。实测的比例在 10² 量级。
        """
        if self.bias_learning_rate is not None:
            self.out_bias += self.bias_learning_rate * td_error

    @torch.no_grad()
    def reset_trace(self) -> None:
        """回合结束时清空痕迹——资格痕迹不该跨回合累积。"""
        self.trace.zero_()


def td_error(
    reward: float,
    value: torch.Tensor,
    next_value: torch.Tensor,
    *,
    discount: float = 1.0,
) -> float:
    """一步 TD 误差 ``δ = r + γ·V(s') − V(s)``。

    ``value`` 与 ``next_value`` 是标量张量。回合终止时 ``V(s')`` 应传 0。
    """
    return float(reward + discount * float(next_value) - float(value))


class PopulationCritic:
    """**群体 + 固定读出**的 Critic——照论文的结构（Frémaux 2013）。

    论文里的 Critic 是一**群**脉冲神经元，值由它们的活动经**固定**权重读出
    （"Its activity, together with actual rewards, conditions the delivery of a
    neuromodulatory TD signal to itself and to the actor"）。本类是它的速率型对应：

        y_j      = σ( g · (w_j · x − b) )        第 j 个单元的发放率 ∈ (0, 1)，b 固定
        V(s)     = Σ_j u_j · y_j                 读出 u 固定
        e_ji     ← λ·e_ji + x_i · y_j            第二因子是**单元自己的发放率**
        w_ji     ← w_ji + η·δ·e_ji               TD-LTP

    ## 为什么这个结构能修掉单单元版的两个病

    :class:`TDLCritic` 把"单元输出"与"值"当成了同一个量，于是：

    1. **正值反馈**：`e = x·V` 且 `V = w·x` → `V` 变大、痕迹变大、更新变大、`V` 更大。
       实测权重发散成 NaN。
    2. **尺度自由度被归一化抹掉**：为了堵住上一条我加了 `‖w‖₂ = 1`，结果 Critic 只能
       旋转、不能设定自己的输出尺度，于是 `V` 永远顶到上界 `value_scale·‖x‖`，
       `δ` 退化成常数（实测：scale=60 时 `V` 到 ~50，scale=500 时到 ~500）。

    群体结构两个都治：第二因子是 `y_j ∈ (0,1)`——**有界且非负**，正值反馈断掉；
    而值的尺度由固定的 `u_j` 承担，`w_j` 只管形状，所以归一化 `‖w_j‖₂ = 1` 不再
    夺走任何它需要的自由度。

    ## 读数不是调参旋钮

    ``gain`` 与 ``value_scale`` 一起决定 `V` 的值域：`‖x‖ ≤ 1.11`、`‖w_j‖ = 1` 时，
    `g·(w_j·x) ∈ [−4.4, 4.4]`（取 g=4），所以 `y_j ∈ (0.012, 0.988)`，
    `V ∈ (0.012·value_scale, 0.988·value_scale)`。取 `value_scale = 200` 覆盖 CartPole
    的值域（γ=0.99 下满分约 100）。**这两者设一次就固定，不靠调。**

    Attributes:
        weights: ``(n_units, n_features)``，每行是一个单元的 ``w_j``。
        trace: 同形状的资格痕迹 ``e_ji``。
        readout: ``(n_units,)`` 固定读出 ``u``，构造后不再改变。
    """

    def __init__(
        self,
        n_features: int,
        n_units: int = 64,
        *,
        learning_rate: float = 5e-3,
        trace_decay: float = 0.9,
        gain: float = 8.0,
        bias: float = 0.4,
        value_scale: float = 200.0,
        output_bias: float = 0.0,
        bias_learning_rate: float | None = None,
        trace_post_factor: str = "rate",
        readout: str = "uniform",
        init_directions: torch.Tensor | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        if not 0.0 <= trace_decay < 1.0:
            raise ValueError(f"trace_decay 应在 [0, 1) 内，收到 {trace_decay}。")
        if n_units < 1:
            raise ValueError(f"n_units 必须 >= 1，收到 {n_units}。")
        if gain <= 0 or value_scale <= 0:
            raise ValueError(f"gain 与 value_scale 必须为正，收到 {gain}、{value_scale}。")
        if trace_post_factor not in ("rate", "gradient"):
            raise ValueError(
                f"trace_post_factor 只能是 'rate' 或 'gradient'，收到 {trace_post_factor!r}。"
            )
        if readout not in ("uniform", "random"):
            raise ValueError(f"readout 只能是 'uniform' 或 'random'，收到 {readout!r}。")
        if readout == "random" and trace_post_factor != "gradient":
            # 见 readout 的说明：全正那条符号论证只在 rate 后因子下成立。有符号读出配 rate
            # 后因子会让一半单元的更新方向反过来——**这不是保守，是避免一个静默的错**。
            raise ValueError(
                "readout='random'（有符号）必须配 trace_post_factor='gradient'："
                "rate 后因子下 TD 梯度与更新方向同号要求 u_j > 0。"
            )

        self.n_features = int(n_features)
        self.n_units = int(n_units)
        self.learning_rate = float(learning_rate)
        self.trace_decay = float(trace_decay)
        self.gain = float(gain)
        self.bias = float(bias)
        self.value_scale = float(value_scale)
        self.bias_learning_rate = bias_learning_rate
        self.trace_post_factor = trace_post_factor
        self.readout_kind = readout
        device = device or torch.device("cpu")
        # 加性偏置 ``V = Σ u_j y_j + out_bias``。零初始化，``x + 0.0`` 精确，所以**默认路径
        # 逐位不变**。它与 ``bias`` 不是一回事：``bias`` 是每个单元的**输入阈值**，影响的是
        # 发放率；``out_bias`` 是**输出侧的一个平移**，影响的是值本身。
        #
        # 为什么需要它：真值在临死那一步约等于 1，而群体读出到不了那么低（实测 V 的最低值
        # 13.4、真值 1.0）。够不着的后果是**每回合的终止步都稳定地拿到一个大负 δ**——实测
        # −11.3，而内部步只有 +0.07，差 160 倍。那是整个回合里最大的一次 Actor 更新，打在
        # 痕迹视野内「死前那几步」上，且与动作无关。
        self.out_bias = torch.tensor(float(output_bias), device=device)

        # 单元的"偏好方向"（感受野）最好**铺开**，而不是取随机方向。调用方可以传入代表性
        # 状态的编码（``init_directions``），这是 RBF 中心的标准做法，也是论文里那群体靠
        # 学习形成的调谐的初始化对应物。
        #
        # **但这条的收益取决于特征分布，据实记录**：在**稠密均匀**特征下随机方向几乎无
        # 分辨力（``V`` 的跨度只有 10，而感受野约 30）；换成**稀疏局域凸起**特征后差别就
        # 基本消失（28.2 vs 29.9）。所以 ``init_directions`` 是建议而非必需——别把
        # "随机方向一定不行"当成普遍结论，它只在我最初那个特征分布下成立。
        #
        # 确有一条稳定成立：感受野让**同一输入**下的群体响应铺得更开（实测组内发放率范围
        # 0.039–0.712 vs 0.075–0.348），也就是每个输入有更独特的群体编码。
        generator = device_generator(generator, device)
        if init_directions is not None:
            if init_directions.shape != (n_units, n_features):
                raise ValueError(
                    f"init_directions 的形状应为 ({n_units}, {n_features})，"
                    f"收到 {tuple(init_directions.shape)}。"
                )
            self.weights = init_directions.detach().clone().to(device)
        else:
            self.weights = torch.rand(n_units, n_features, device=device, generator=generator)
        self.trace = torch.zeros(n_units, n_features, device=device)
        self._normalize()
        # 固定读出。``uniform`` 是 ``value_scale/n_units`` 的**全正**均匀读出；``random``
        # 是固定但**有符号**的随机读出（``Σ|u_j| = value_scale``，值域与前者可比）。
        #
        # 为什么要有符号那一档：把读出换成自由最小二乘解，**同一批单元活动**的解释方差
        # 从 −0.9 变成 **+0.6**（实测，见 signal_probe 的「自由线性读出的 EV」一列）。
        # 特征里有信息，是「全正且均匀」这个读出把它浪费掉了。论文要求读出**固定**，
        # 没有要求全正；原先取全正的理由是 rate 后因子下的符号论证（``u_j > 0`` 才保证
        # ``δ·y_j·x`` 与 TD 梯度同号），那条论证在 gradient 后因子下不再成立，所以
        # 有符号读出必须配 gradient（上面有断言挡着）。
        generator = device_generator(generator, device)
        if readout == "random":
            raw = torch.rand(n_units, device=device, generator=generator) * 2.0 - 1.0
            self.readout = value_scale * raw / raw.abs().sum().clamp_min(1e-12)
        else:
            self.readout = torch.full((n_units,), value_scale / n_units, device=device)

    @torch.no_grad()
    def _normalize(self) -> None:
        """把每个单元的权重行归一到单位 L2 范数。

        有了它，``g·(w_j·x)`` 就落在有界区间里，sigmoid 不会饱和、也不会退化成常数。
        """
        norms = self.weights.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.weights.mul_(1.0 / norms)

    def trace_second_factor(self, rates: torch.Tensor) -> torch.Tensor:
        """资格痕迹的第二因子：``"rate"`` 用 ``y_j``，``"gradient"`` 用 ``y_j(1−y_j)``。

        半梯度是 ``δ · u_j·g·y_j(1−y_j) · x_i``，而实现用的是 ``δ · y_j · x_i``。常数
        ``u_j``（固定读出）与 ``g`` 的差别只是每个单元**同一个**倍率，会被学习率吸收；
        真正的**逐单元**差别只有 ``(1−y_j)``：

        * ``y_j`` 已饱和（≈1）的单元几乎改不动 ``V``（导数为 0），却按 ``y_j≈1`` 拿到大更新；
        * ``y_j`` 在中段的单元最能改 ``V``（导数最大），却只按 ``y_j≈0.5`` 拿到一半的更新。

        也就是这一版把**最没用的单元**加权得最重。``"gradient"`` 把它换回真实导数的形状。

        **量级要注意**：``y(1−y) ≤ 0.25`` 而 ``y`` 典型约 0.5，所以切过去等于把 Critic 的
        有效学习率大约减半——比较时必须把这一条一起看，不能只看中位数。
        """
        if self.trace_post_factor == "rate":
            return rates
        # gradient：``u_j·g·y_j(1−y_j)`` 的形状。``u_j`` 必须**留在里面**——有符号读出的
        # 那一档正是靠它把每个单元的符号带进更新方向。
        return self.readout * self.gain * rates * (1.0 - rates)

    def value_floor(self) -> float:
        """``V`` 在**权重非负**这一前提下的下界：``value_scale · σ(−gain · bias)``。

        权重与编码活动都非负时 ``cos ∈ [0, 1]``，于是每个单元至少贡献 ``u_j·σ(−g·b)``，
        **旋转 ``w_j`` 学不动这个下界**。默认参数下是 ``200·σ(−3.2) = 7.83``。

        **它不是绝对下界。** ``update()`` 做的是 ``w += η·δ·e``，δ 为负时权重会被推成
        负数；此后 ``cos`` 可以一直到 −1，``V`` 也就能一路降到约 0（``200·σ(−11.2) ≈ 0.003``）。
        所以「Critic 到底够不够得着真值的低段」是个**要测的问题**，不是能推的问题——
        :mod:`research.rstdp.signal_probe` 报的负数权重占比与 ``V`` 的实际取值区间就是冲
        这个来的。

        真要紧的是它可能造成什么：一个还能活 ``T`` 步的策略，真值是 ``(1−γ^T)/(1−γ)``，
        临死那一步约为 1。若 ``V`` 因为权重仍非负而卡在 7.83 之上，那么每回合的终止步都会
        稳定地拿到 ``δ = 1 − V ≈ −6.8``，且**与动作无关**——那是整个回合里最大的一次 Actor
        更新，打在痕迹视野内「死前那几步」上，恰恰是最需要知道「哪个动作能救回来」的地方。
        """
        # ``Σ_j u_j·σ(−g·b)``——均匀全正读出下就是 ``value_scale·σ(−g·b)``；有符号读出下
        # 它不再是「可达下界」（正负项会互相抵消），所以那个档位下这个数只作参照。
        return float(self.readout.sum()) / (1.0 + math.exp(self.gain * self.bias))

    @torch.no_grad()
    def rates(self, features: torch.Tensor) -> torch.Tensor:
        """各单元的发放率 ``y_j``，``(n_units,)``。

        ``y_j = σ( g · (cos(w_j, x) − b) )``。**对特征做归一化**（用余弦而不是内积）是
        必需的：输入活动的模长随状态起伏（实测 ``‖x‖ ∈ [0, 0.65, 1.11]``），而 ``b`` 是一个
        固定常数——不做归一化的话同一个 ``b`` 在不同状态下含义不同，单元会整体饱和。
        实测过：不归一化时 ``V`` 恒等于其上限。归一化之后 ``cos ∈ [0, 1]``（权重与活动都
        非负），``b = 0.5`` 是真正居中的操作点。
        """
        normalized = features / features.norm().clamp_min(1e-12)
        return torch.sigmoid(self.gain * ((self.weights @ normalized) - self.bias))

    @torch.no_grad()
    def value(self, features: torch.Tensor) -> torch.Tensor:
        """``V(s) = Σ_j u_j·y_j + out_bias``，0 维张量。"""
        return self.readout @ self.rates(features) + self.out_bias

    @torch.no_grad()
    def update(self, features: torch.Tensor, value: torch.Tensor, td_error: float) -> None:
        """三因子更新：``e_ji ← λe_ji + x_i·y_j``，``w_ji ← w_ji + η·δ·e_ji``。

        ``value`` 参数保留是为了与 :class:`TDLCritic` 的签名一致；本类不用它——
        第二因子取的是单元自己的发放率，不是值。这正是两个类关键的结构差别。
        """
        rates = self.rates(features)
        self.trace.mul_(self.trace_decay).add_(
            torch.outer(self.trace_second_factor(rates), features)
        )
        self.weights.add_(self.learning_rate * td_error * self.trace)
        self._normalize()
        self._learn_bias(td_error)

    @torch.no_grad()
    def _learn_bias(self, td_error: float) -> None:
        """``out_bias += η_θ·δ``——线性值函数偏置项的教科书半梯度更新。

        ``∂V/∂θ = 1``，半梯度只经 ``V`` 回传（不含自举项 ``γV(s')``），所以更新就是
        ``+η_θ·δ`` 这一行——与权重更新同号，也和「δ>0 说明 V 偏低、该把 V 抬上去」一致。

        **η_θ 的量级要注意**：内部步里 δ 只通过 ``(1−γ)`` 依赖 θ，偏置在那里几乎不动；
        真正把它标定住的是**终止步**（那里 ``V(s')`` 被强制为 0，``∂δ/∂θ = 1``）。而终止步
        每回合只有一次，所以 η_θ 需要比 η_w 大得多才来得及，实测比例在 10² 量级。
        """
        if self.bias_learning_rate is not None:
            self.out_bias += self.bias_learning_rate * td_error

    @torch.no_grad()
    def reset_trace(self) -> None:
        """回合结束时清空痕迹。"""
        self.trace.zero_()

    @torch.no_grad()
    def active_unit_fraction(self, features: torch.Tensor, *, threshold: float = 0.5) -> float:
        """发放率超过阈值的单元占比，用于观察有没有单元死掉（计划书 §3.1 的关切）。"""
        return float((self.rates(features) > threshold).float().mean().item())

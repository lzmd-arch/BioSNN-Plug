"""TD-LTP 群体 Critic（Frémaux et al. 2013 的结构，脉冲目录内的实现）。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

## 结构

论文里的 Critic 是一**群**神经元，值由它们的活动经**固定**权重读出。本实现：

```text
y_j   = σ( g·(w_j · x − b) )        第 j 个单元的活动 ∈ (0, 1)，b 固定
V(s)  = Σ_j u_j·y_j + out_bias      u 固定（均匀）
e_ji ← λ·e_ji + x_i·y_j             第二因子是**单元自己的活动**，有界且非负
w_j  ← w_j + η·δ·e_j，再归一化 ‖w_j‖₂ = 1
```

## 为什么不是单权重线性 Critic

`LinearCritic` 只有一条权重向量，等于把"单元输出"和"值"当成同一个量：

1. **正值反馈**：`e = x·V` 且 `V = w·x` → 更新变大、`V` 更大、痕迹再变大；
2. **尺度自由度被归一化抹掉**：只能旋转、不能设定输出尺度，`δ` 容易退化成常数。

群体结构两条都治：第二因子有界非负，正值反馈断掉；值的尺度由固定的 `u` 承担，
`w_j` 只管形状，行归一化不再夺走自由度。

## 与速率版的关系

`research/rstdp/td_ltp.py` 里有一个功能更全的 `PopulationCritic`（含 bias 学习、有符号读出、
gradient 后因子等）。**本目录不 import 它**（仓库约定：各主题只共享 `research/common/`），
这里是一个**精简移植**：只保留 CartPole 这条线用得到的部分，参数默认值按本线的量纲取。

⚠️ **这不是等价实现**：速率版里针对"终止步稳定拿到大负 δ"的 `output_bias` 学习、以及
有符号读出的限制，这里都**没有**——诊断那两个病是速率版的历史，脉冲线还没走到那一步。
"""

from __future__ import annotations

import torch

__all__ = ["PopulationCritic", "td_error"]


def td_error(
    reward: float, value: float, next_value: float, *, terminated: bool, gamma: float
) -> float:
    """``δ = r + γ·V(s′) − V(s)``；终止状态不加 ``V(s′)``。"""
    target = float(reward) + (0.0 if terminated else gamma * next_value)
    return target - value


class PopulationCritic:
    """群体 + 固定读出的 Critic。

    ⚠️ **默认工作点不是最好的那个。** 实测（见 README「工作点」一节）：`bias=0.4, gain=8`
    会让 76% 的单元活动 < 0.05——偏置比前激活尺度（0.16）大一个数量级，单元贴在饱和尾巴上。
    `bias=0, gain=16` 把 `corr(V,G)` 从 −0.50 提到 +0.70。默认值保留原样是为了不悄悄改变
    任何既有结果；要用好的那组请显式传参。

    ## 稳态（homeostasis，可选）

    ``b_j ← b_j + β(y_j − y*)``：**只用该单元自己的活动与一个常数目标**，不读别单元、
    不经过 ``δ``。它是与 TD-LTP **并行的另一条局部规则**，不是 TD 的一部分。

    实测（800 回合）：把平均活动从 0.003 拉回 0.089、最差种子从 33.2 抬到 52.4，
    **但没有打破权重同质化**（单元间余弦 +0.82 → +0.78，有效秩 0.025 → 0.024）。
    即「单元静默」不是同质化的原因；共模推力来自痕迹的符号结构。
    默认 **关闭**——它是实验性规则，不改任何既有结果。

    Attributes:
        weights: ``(n_units, n_features)``，每行是一个单元的 ``w_j``。
        trace: 同形状的资格痕迹。
        readout: ``(n_units,)``，固定均匀读出。
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
        gamma: float = 0.99,
        bias_learning_rate: float = 1e-2,
        trace_post_factor: str = "rate",
        readout: str = "uniform",
        normalize: bool = True,
        homeostasis: bool = False,
        homeostatic_rate: float = 0.01,
        activity_target: float = 0.1,
        trace_demean: bool = False,
        trace_demean_rate: float = 0.01,
        mask_density: float | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        if n_features <= 0 or n_units < 1:
            raise ValueError(f"n_features 与 n_units 必须为正，收到 {n_features}、{n_units}。")
        if not 0.0 <= trace_decay < 1.0:
            raise ValueError(f"trace_decay 应在 [0, 1) 内，收到 {trace_decay}。")
        if gain <= 0 or value_scale <= 0:
            raise ValueError(f"gain 与 value_scale 必须为正，收到 {gain}、{value_scale}。")
        if trace_post_factor not in ("rate", "gradient"):
            raise ValueError(
                f"trace_post_factor 只能是 'rate' 或 'gradient'，收到 {trace_post_factor!r}。"
            )
        if readout not in ("uniform", "random"):
            raise ValueError(f"readout 只能是 'uniform' 或 'random'，收到 {readout!r}。")
        if readout == "random" and trace_post_factor != "gradient":
            # 与速率版同一条约束：全正读出的符号论证只在 rate 后因子下成立。
            # 有符号读出配 rate 后因子会让一半单元的更新方向反过来——不是保守，是避免静默的错。
            raise ValueError("readout='random'（有符号）必须配 trace_post_factor='gradient'。")

        device = device or torch.device("cpu")
        self.n_features = int(n_features)
        self.n_units = int(n_units)
        self.learning_rate = float(learning_rate)
        self.trace_decay = float(trace_decay)
        self.gain = float(gain)
        self.bias = float(bias)
        self.value_scale = float(value_scale)
        self.gamma = float(gamma)
        self.bias_learning_rate = float(bias_learning_rate)
        self.trace_post_factor = trace_post_factor
        self.readout_kind = readout
        self.normalize = bool(normalize)
        self.homeostasis = bool(homeostasis)
        self.homeostatic_rate = float(homeostatic_rate)
        self.activity_target = float(activity_target)
        if homeostasis and not 0.0 < activity_target < 1.0:
            raise ValueError(f"activity_target 应在 (0, 1) 内，收到 {activity_target}。")
        # 稳态需要**每个单元自己的**阈值，所以 bias 扩成向量（只在开启时用）
        self.bias_per_unit = torch.full((n_units,), float(bias), device=device)
        self.trace_demean = bool(trace_demean)
        self.trace_demean_rate = float(trace_demean_rate)
        if trace_demean and not 0.0 < trace_demean_rate <= 1.0:
            raise ValueError(f"trace_demean_rate 应在 (0, 1] 内，收到 {trace_demean_rate}。")
        if mask_density is not None and not 0.0 < mask_density <= 1.0:
            raise ValueError(f"mask_density 应在 (0, 1] 内，收到 {mask_density}。")

        gen = generator or torch.Generator(device=device)
        weights = torch.randn(n_units, n_features, generator=gen, device=device)
        self.weights = weights / (weights.norm(dim=1, keepdim=True) + 1e-12)

        # 每个单元一个固定的输入子空间（结构约束，不训练）。像固定连接概率：
        # 它不涉及任何学习信号，因此不影响局部性主张（与 ADR-0012 的"固定随机投影"同一取舍）。
        self.mask: torch.Tensor | None = None
        if mask_density is not None:
            draws = torch.rand(n_units, n_features, generator=gen, device=device)
            self.mask = (draws < float(mask_density)).to(self.weights.dtype)
            # 保证每行至少一个连接，否则该单元永远是死的
            empty = self.mask.sum(dim=1) == 0
            if bool(empty.any()):
                self.mask[empty, 0] = 1.0
            self.weights = self.weights * self.mask
        # 每单元自己的痕迹历史均值（局部量：只用该单元自己的痕迹）
        self.trace_mean = torch.zeros_like(self.weights)
        self.trace = torch.zeros_like(self.weights)
        if readout == "uniform":
            self.readout = torch.full((n_units,), value_scale / n_units, device=device)
        else:
            signs = torch.randint(0, 2, (n_units,), generator=gen, device=device) * 2 - 1
            # dtype 必须与 weights 一致：torch.dot 不接受 float64 读出去点 float32 活动
            self.readout = signs.to(self.weights.dtype) * (value_scale / n_units)
        # 输出侧可学偏置。**不是**单元阈值：它平移的是值本身。
        # 没有它，群体读出到不了"临死那步真值≈0"，终止步会稳定拿到一个大负 δ，
        # 而那一次更新落在痕迹视野内、与动作无关（速率版记载为 −11.3 vs 内部 +0.07）。
        self.out_bias = torch.zeros((), device=device)

    # ------------------------------------------------------------------ 前向
    def rates(self, features: torch.Tensor) -> torch.Tensor:
        """``(n_units,)`` 的单元活动。"""
        x = features.detach().flatten()
        thresholds = self.bias_per_unit if self.homeostasis else self.bias
        return torch.sigmoid(self.gain * (self.weights @ x - thresholds))

    def _post_factor(self, y: torch.Tensor) -> torch.Tensor:
        """痕迹的第二因子：``rate`` 用发放率，``gradient`` 用 σ 对输入的导数。"""
        if self.trace_post_factor == "rate":
            return y
        return y * (1.0 - y)

    def value(self, features: torch.Tensor) -> float:
        return float(torch.dot(self.readout, self.rates(features)) + self.out_bias)

    # ------------------------------------------------------------------ 学习
    def update(
        self,
        features: torch.Tensor,
        reward: float,
        next_features: torch.Tensor,
        *,
        terminated: bool,
    ) -> float:
        """TD-LTP 一步：累积痕迹 → 用 δ 调制权重。返回 δ（同时是 Actor 的成功信号）。"""
        x = features.detach().flatten()
        y = self.rates(features)
        delta = td_error(
            reward,
            self.value(features),
            self.value(next_features),
            terminated=terminated,
            gamma=self.gamma,
        )
        self.trace.mul_(self.trace_decay).add_(torch.outer(self._post_factor(y), x))
        if self.trace_demean:
            # 每单元自己的直流分量：μ_j ← (1−κ)μ_j + κ·e_j，更新用 e_j − μ_j。
            # **完全局部**——只读该单元自己的痕迹历史，不涉及任何群体统计量。
            self.trace_mean.mul_(1.0 - self.trace_demean_rate).add_(
                self.trace_demean_rate * self.trace
            )
            effective_trace = self.trace - self.trace_mean
        else:
            effective_trace = self.trace
        self.weights.add_(self.learning_rate * delta * effective_trace)
        if self.mask is not None:
            self.weights.mul_(self.mask)
        if self.normalize:
            norms = self.weights.norm(dim=1, keepdim=True) + 1e-12
            self.weights.div_(norms)
        self.out_bias.add_(self.bias_learning_rate * delta)
        if self.homeostasis:
            # 稳态：只用**该单元自己的**活动与一个常数目标，不读别单元、不经过 δ。
            # y_j > y* ⇒ 抬高阈值（降活动）；反之降低。
            self.bias_per_unit.add_(self.homeostatic_rate * (y - self.activity_target))
        return delta

    def reset_trace(self) -> None:
        """清空痕迹（回合边界用；**权重保留**）。"""
        self.trace.zero_()

"""脉冲版 W3 的可塑性：**带真实 Δt 窗口的 STDP** + 资格痕迹。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

## 这个模块存在的理由

计划书 §3.2 写的更新规则是

    Δw_ij = STDP(Δt_ij) · (R − V(s))

**带 Δt**。而 `research/rstdp/` 的速率型实现里，Δt 是不存在的——它的 README 逐字写着
「速率型下没有『脉冲先后』，所以 `STDP(Δt)` 的窗口形状退化成同时刻的乘积，**TD-LTP 与
TD-STDP 那处『有无 post-before-pre 分量』的差别在本实现里无从体现**」。

本模块把那处差别**做出来**：pre 与 post 的配对按**实际脉冲时刻之差**加权，窗口由
``τ_+`` 与 ``τ_−`` 两个时间常数决定——这两个参数在速率型下根本不是自由参数。

## 实现：用迹（trace）把成对 STDP 写成一遍扫描

成对 STDP 的定义是「对每一对 (pre, post) 脉冲，按 Δt 加权求和」——朴素实现是
``O(n_pre · n_post · n_spike²)``。标准做法是把它改写成两条迹的递推，本模块采用：

    LTP_ij ∝ A_+ · Σ_t  z_j(t) · x̂_i(t⁻)
    LTD_ij ∝ A_− · Σ_t  x_i(t) · ẑ_j(t⁻)

其中 ``x̂``/``ẑ`` 是各自**指数衰减的迹**——每步**先衰减、再用于配对、最后加上本步脉冲**，
所以 ``x̂(t)`` 只含 ``t`` 之前的 pre 脉冲，只有**严格先后**的配对才计分：

* 某个 pre 脉冲比 post 早 Δt 步 → 对 LTP 贡献 ``exp(−Δt/τ_+)``；
* 某个 post 脉冲比 pre 早 Δt 步 → 对 LTD 贡献 ``exp(−Δt/τ_−)``。

这个「恰好是 exp(−Δt/τ)」不是凑出来的，是上面那个更新顺序换来的；测试直接对着指数断言。
"""

from __future__ import annotations

import math

import torch

__all__ = ["EligibilityTrace", "STDPWindow", "pair_based_stdp"]


class STDPWindow:
    """STDP 窗口的两个时间常数与两个幅度。

    Args:
        tau_plus: pre-before-post（LTP 侧）的时间常数，毫秒。
        tau_minus: post-before-pre（LTD 侧）的时间常数，毫秒。
        a_plus: LTP 幅度。
        a_minus: LTD 幅度。**必须是正数**——符号由 :func:`pair_based_stdp` 里的减号承担，
            这样调用方不会因为少写一个负号而静默地把 LTD 变成第二个 LTP（那正是本仓库
            在别处踩过的「静默失效」那类坑）。
        dt: 时间步长。
    """

    def __init__(
        self,
        *,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        a_plus: float = 1.0,
        a_minus: float = 1.0,
        dt: float = 1.0,
    ) -> None:
        if tau_plus <= 0 or tau_minus <= 0:
            raise ValueError(f"时间常数必须为正，收到 τ+={tau_plus}、τ−={tau_minus}。")
        if a_plus < 0 or a_minus < 0:
            raise ValueError(f"幅度必须非负（符号由实现承担），收到 A+={a_plus}、A−={a_minus}。")
        self.tau_plus = float(tau_plus)
        self.tau_minus = float(tau_minus)
        self.a_plus = float(a_plus)
        self.a_minus = float(a_minus)
        self.decay_plus = math.exp(-dt / tau_plus)
        self.decay_minus = math.exp(-dt / tau_minus)


@torch.no_grad()
def pair_based_stdp(
    pre_spikes: torch.Tensor,
    post_spikes: torch.Tensor,
    window: STDPWindow,
) -> torch.Tensor:
    """一遍扫描算出成对 STDP 的权重变化，``(n_pre, n_post)``。

    Args:
        pre_spikes: ``(T, batch, n_pre)`` 的 0/1 张量。
        post_spikes: ``(T, batch, n_post)`` 的 0/1 张量。
        window: 窗口参数。

    Returns:
        ``Δw``，形状 ``(n_pre, n_post)``，**已经对时间与 batch 求和**。

    Note:
        迹在**同一时刻**的贡献被排除（用更新前的迹），所以同时发放的一对 pre/post
        不计入任何一侧——这正是「先后」两个字的字面意思。
    """
    if pre_spikes.shape[0] != post_spikes.shape[0]:
        raise ValueError(
            f"pre 与 post 的时间步数不一致：{pre_spikes.shape[0]} 对 {post_spikes.shape[0]}。"
        )
    if pre_spikes.shape[1] != post_spikes.shape[1]:
        raise ValueError(
            f"pre 与 post 的 batch 不一致：{pre_spikes.shape[1]} 对 {post_spikes.shape[1]}。"
        )

    steps = pre_spikes.shape[0]
    n_pre, n_post = pre_spikes.shape[2], post_spikes.shape[2]
    ltp = torch.zeros(n_pre, n_post, device=pre_spikes.device, dtype=pre_spikes.dtype)
    ltd = torch.zeros_like(ltp)

    pre_trace = torch.zeros_like(pre_spikes[0])
    post_trace = torch.zeros_like(post_spikes[0])
    for t in range(steps):
        # 先衰减再使用，最后加上本步的脉冲。这个顺序让「相差 Δt 步的一对」的贡献**恰好**
        # 是 exp(−Δt/τ)——不必在文档里写「差一步」，测试也能直接对着指数断言。
        pre_trace = pre_trace * window.decay_plus
        post_trace = post_trace * window.decay_minus
        ltp += pre_trace.transpose(0, 1) @ post_spikes[t]
        ltd += pre_spikes[t].transpose(0, 1) @ post_trace
        pre_trace = pre_trace + pre_spikes[t]
        post_trace = post_trace + post_spikes[t]

    return window.a_plus * ltp - window.a_minus * ltd


class EligibilityTrace:
    """跨决策累积的资格痕迹。

    与速率版 ``research/rstdp/`` 的同名概念同构：``e ← λ·e + Δw``，权重只在成功信号
    到达时按 ``η·S·e`` 更新——这对应论文里「STDP 的效果先存进痕迹、只在成功信号到达
    时写入权重」的那一步。
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        *,
        decay: float = 0.9,
        device: torch.device | None = None,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay 应在 [0, 1) 内，收到 {decay}。")
        self.decay = float(decay)
        self.value = torch.zeros(n_pre, n_post, device=device)

    def accumulate(self, delta: torch.Tensor) -> None:
        """``e ← λ·e + Δw``。"""
        if delta.shape != self.value.shape:
            raise ValueError(
                f"形状不符：痕迹 {tuple(self.value.shape)}，增量 {tuple(delta.shape)}。"
            )
        self.value.mul_(self.decay).add_(delta)

    def rms(self) -> float:
        """痕迹的均方根，用于观察它有没有失控（计划书 §3.2 的「突触动态失控」）。"""
        return float(self.value.pow(2).mean().sqrt().item())

    def reset(self) -> None:
        self.value.zero_()

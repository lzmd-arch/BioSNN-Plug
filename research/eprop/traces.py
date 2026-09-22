"""资格痕迹（Bellec et al. 2020 的 Eq. 25）。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

    ψ_j^t      = (dampening_factor / thr) · max(0, 1 − |v_scaled_j^t|)      不应期时为 0
    ε_v,ij^t   = α · ε_v,ij^{t−1} + z_i^t                                   α = exp(−dt/τ)
    ε_a,ij^t   = (ρ − β·ψ_j^t) · ε_a,ij^{t−1} + ψ_j^t · ε_v,ij^t            ρ = exp(−dt/τ_a)
    e_ij^t     = ψ_j^t · ( ε_v,ij^t − β · ε_a,ij^t )

    dE/dW_ij   = Σ_t L_j^t · e_ij^t          （L_j^t = dE/dz_j^t，见 eprop.py）

其中 ``i`` 是前突触、``j`` 是后突触。``β = 0`` 时 ``ε_a`` 项消失，退化为 LIF 的
经典 e-prop 痕迹。

## 关于时间对齐：不猜，用 BPTT 裁决

官方实现用 ``tf.scan`` 写这两条递推，而 ``tf.scan`` 与 ``psi[:-1]`` 切片组合下的索引对齐
（``ε_a`` 究竟与 ``ε_v`` 同步还是滞后一步）从代码上读不可靠——读错不会报错，只会让痕迹
系统性偏一点。

所以本模块按**论文的时间对齐形式**实现，并用一条数值验证来裁决：把 ``dE/dz`` 取成
autograd 给出的真实学习信号，检查 ``Σ_t L_j^t e_ij^t`` 是否等于 BPTT 的梯度。作者自己的
脚本（``numerical_verification_eprop_factorization_vs_BPTT.py``）做的就是这件事，预期相对
误差在 1e-14 量级。**若对齐错了，这条验证会当场失败**——这就是它存在的意义。

测试见 ``tests/test_traces.py::TestFactorizationMatchesBPTT``。
"""

from __future__ import annotations

import torch

__all__ = ["eligibility_traces", "eprop_gradient", "exp_convolve", "refractory_mask"]


def refractory_mask(z_post: torch.Tensor, n_refractory: int) -> torch.Tensor:
    """哪些时刻哪些神经元处在不应期，``(T, batch, n_post)`` 的布尔张量。

    **判据是"上一步之后的计数器"，不是"本步之后的"**——这一点错了不会报错，但会让
    ψ 在错误的时刻被置零，痕迹系统性偏掉（实测相对差 ~20%，见
    ``tests/test_traces.py::TestFactorizationMatchesBPTT``）。

    动力学里第 t 步的不应期判据是 ``r_{t-1} > 0``（状态携带的是上一步结束时的计数器），
    而 ``r_t`` 已经**包含**了本步的脉冲。所以这里必须先把当前计数器当作判据、再更新它：

        mask[t] = counter_{t-1} > 0
        counter_t = clip(counter_{t-1} + n_ref·z_t − 1, 0, n_ref)

    官方实现用 ``tf.scan`` 扫 ``z_post[:-1]`` 再在头部补一个零，得到正是 ``counter_{t-1}``；
    本函数用同样的语义写成显式循环。
    """
    if n_refractory < 0:
        raise ValueError(f"n_refractory 必须 >= 0，收到 {n_refractory}。")

    steps, batch, n_post = z_post.shape
    if n_refractory == 0:
        return torch.zeros_like(z_post, dtype=torch.bool)

    counter = torch.zeros(batch, n_post, device=z_post.device, dtype=z_post.dtype)
    mask = torch.empty(steps, batch, n_post, device=z_post.device, dtype=torch.bool)
    for t in range(steps):
        mask[t] = counter > 0  # 先判据（这是 r_{t-1}），再让本步脉冲更新计数器
        counter = torch.clamp(
            counter + n_refractory * z_post[t] - 1.0, min=0.0, max=float(n_refractory)
        )
    return mask


def eligibility_traces(
    v_scaled: torch.Tensor,
    z_pre: torch.Tensor,
    z_post: torch.Tensor,
    *,
    alpha: float,
    rho: float,
    beta: float,
    threshold: float,
    dampening_factor: float,
    n_refractory: int = 0,
    is_recurrent: bool = False,
) -> torch.Tensor:
    """在线计算资格痕迹，``(T, batch, n_pre, n_post)``。

    Args:
        v_scaled: ``(T, batch, n_post)``，相对阈值且已按适应后阈值缩放的膜电位。
        z_pre: ``(T, batch, n_pre)`` 前突触脉冲。
        z_post: ``(T, batch, n_post)`` 后突触脉冲（用于不应期判定）。
        alpha: ``exp(−dt/τ)``。
        rho: ``exp(−dt/τ_a)``。
        beta: 适应强度；0 即 LIF。
        threshold: 基阈值，用于把伪导数缩放到与 ``z`` 同量纲。
        dampening_factor: 阻尼因子。
        n_refractory: 不应期步数。
        is_recurrent: 是否循环连接。为真时把对角（自连接）置零——官方实现如此，
            因为 ``w_rec`` 的对角在动力学里已被断开，痕迹也必须一致。

    Returns:
        ``e_ij^t``，形状 ``(T, batch, n_pre, n_post)``。
    """
    if v_scaled.dim() != 3 or z_pre.dim() != 3 or z_post.dim() != 3:
        raise ValueError(
            f"三个输入都必须是 (T, batch, n)，收到 {tuple(v_scaled.shape)}、"
            f"{tuple(z_pre.shape)}、{tuple(z_post.shape)}。"
        )
    steps, batch, n_post = v_scaled.shape
    if z_post.shape != v_scaled.shape:
        raise ValueError(
            f"z_post 与 v_scaled 的形状必须一致，收到 {tuple(z_post.shape)} 与 "
            f"{tuple(v_scaled.shape)}。"
        )
    n_pre = z_pre.shape[-1]

    psi_full = (dampening_factor / threshold) * torch.clamp(1.0 - v_scaled.abs(), min=0.0)
    psi = torch.where(refractory_mask(z_post, n_refractory), torch.zeros_like(psi_full), psi_full)

    # ε_v,ij^t = α·ε_v,ij^{t−1} + z_i^t，对所有后突触神经元 j 相同（递推里不含 j）。
    epsilon_v = torch.empty(
        steps, batch, n_pre, n_post, device=v_scaled.device, dtype=v_scaled.dtype
    )
    running = z_pre[0][:, :, None].expand(batch, n_pre, n_post).clone()
    epsilon_v[0] = running
    for t in range(1, steps):
        running = alpha * running + z_pre[t][:, :, None]
        epsilon_v[t] = running

    # ε_a,ij^t = (ρ − β·ψ_j^t)·ε_a,ij^{t−1} + ψ_j^t·ε_v,ij^t，ε_a^0 = 0。
    epsilon_a = torch.zeros_like(epsilon_v)
    running_a = torch.zeros(batch, n_pre, n_post, device=v_scaled.device, dtype=v_scaled.dtype)
    for t in range(steps):
        psi_t = psi[t][:, None, :]  # (batch, 1, n_post)，对前突触维广播
        running_a = (rho - beta * psi_t) * running_a + psi_t * epsilon_v[t]
        epsilon_a[t] = running_a

    # 全程时间主序：psi 是 (T, batch, n_post)，插一个前突触维后广播到 (T, batch, n_pre, n_post)。
    e_trace = psi[:, :, None, :] * (epsilon_v - beta * epsilon_a)

    if is_recurrent:
        identity = torch.eye(n_pre, device=e_trace.device, dtype=e_trace.dtype)
        e_trace = e_trace * (1.0 - identity)
    return e_trace


def exp_convolve(tensor: torch.Tensor, decay: float) -> torch.Tensor:
    """沿时间维做指数滤波：``y^t = decay·y^{t−1} + (1 − decay)·x^t``，``y^0`` 取零。

    官方实现用它把资格痕迹平滑后再与学习信号做外积（``compute_loss_gradient`` 的
    ``decay_out``）。Bellec et al. 的 Figure 1 说明这一步对应突触后电位的时间常数，
    是可选的。
    """
    steps = tensor.shape[0]
    out = torch.empty_like(tensor)
    running = torch.zeros_like(tensor[0])
    for t in range(steps):
        running = decay * running + (1.0 - decay) * tensor[t]
        out[t] = running
    return out


def eprop_gradient(
    v_scaled: torch.Tensor,
    z_pre: torch.Tensor,
    z_post: torch.Tensor,
    learning_signal: torch.Tensor,
    *,
    alpha: float,
    rho: float,
    beta: float,
    threshold: float,
    dampening_factor: float,
    n_refractory: int = 0,
    is_recurrent: bool = False,
) -> torch.Tensor:
    """**在线**累积权重梯度 ``Σ_t L_j^t·e_ij^t``，``(n_pre, n_post)``。

    与 :func:`eligibility_traces` 数学上等价（有测试对照），但**不保存时间维**：

        eligibility_traces 的内存  O(T · batch · n_pre · n_post)
        eprop_gradient 的内存      O(batch · n_pre · n_post)

    在 sMNIST 的验收配置（T=28、batch=64、n_rec=256）下，前者是约 470 MB **每个**中间
    张量、且每次更新要建好几个；后者是约 33 MB。这不是微优化——按前者算，30 个 epoch
    要跑一个多小时，按后者几分钟。

    ## 与计划书 §3.1 的 Trace Propagation 是什么关系

    这是**在线累积**，把时间维省掉了，但每个突触仍要存一份资格痕迹，所以内存仍是
    **O(N²)** 量级（随连接数）。§3.1「关键修正 1」要求的 Trace Propagation 更进一步：
    结合逐层对比损失、无需辅助逐层矩阵，把存储降到 **O(N)**。

    **所以这一条还没有实现。** 本函数解决的是"能不能在 8GB 上跑完"（§6.2 的硬约束），
    没有解决"痕迹存储随突触数增长"（§3.1 的要求）。两者的区别要写进结论的边界。
    """
    steps, batch, n_post = v_scaled.shape
    if z_post.shape != v_scaled.shape:
        raise ValueError(
            f"z_post 与 v_scaled 的形状必须一致，收到 {tuple(z_post.shape)} 与 "
            f"{tuple(v_scaled.shape)}。"
        )
    if learning_signal.shape[:2] != (steps, batch):
        raise ValueError(
            f"learning_signal 的前两维应是 (T, batch)=({steps}, {batch})，"
            f"收到 {tuple(learning_signal.shape)}。"
        )
    n_pre = z_pre.shape[-1]

    hard_refractory = refractory_mask(z_post, n_refractory)
    psi_full = (dampening_factor / threshold) * torch.clamp(1.0 - v_scaled.abs(), min=0.0)
    psi = torch.where(hard_refractory, torch.zeros_like(psi_full), psi_full)

    gradients = torch.zeros(n_pre, n_post, device=v_scaled.device, dtype=v_scaled.dtype)
    # **``epsilon_v`` 只需 ``(batch, n_pre, 1)``**：它的递推 ``eps_v ← α·eps_v + z_pre[t]``
    # **不含后突触下标 j**，所以同一时刻对所有 j 是同一个值，广播即可。存成
    # ``(batch, n_pre, n_post)`` 是纯粹的浪费——实测**逐位等价**（``torch.equal`` 为真、
    # ``max|diff| = 0``），验收形状（T=28, batch=64, n_rec=256）下墙钟 **−18%**。
    #
    # **``epsilon_a`` 则真的需要 ``(batch, n_pre, n_post)``**：它的系数 ``(ρ − β·ψ_j)`` 含 j，
    # 不可因式化。所以**这一段的内存大头由 ALIF 的适应项决定**；LIF（β=0）下两者都退化成
    # 外积，整个迹的状态可降到 O(batch·N)。
    epsilon_v = z_pre[0][:, :, None].clone()  # (batch, n_pre, 1)
    epsilon_a = torch.zeros(batch, n_pre, n_post, device=v_scaled.device, dtype=v_scaled.dtype)

    for t in range(steps):
        if t > 0:
            epsilon_v = alpha * epsilon_v + z_pre[t][:, :, None]
        psi_t = psi[t][:, None, :]  # (batch, 1, n_post)
        epsilon_a = (rho - beta * psi_t) * epsilon_a + psi_t * epsilon_v
        e_trace = psi_t * (epsilon_v - beta * epsilon_a)  # (batch, n_pre, n_post)
        gradients += torch.einsum("bj,bij->ij", learning_signal[t], e_trace)

    if is_recurrent:
        identity = torch.eye(n_pre, device=gradients.device, dtype=gradients.dtype)
        gradients = gradients * (1.0 - identity)
    return gradients

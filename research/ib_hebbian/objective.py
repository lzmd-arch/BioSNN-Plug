"""核化信息瓶颈的局部目标（pHSIC）。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

对应 Pogodin & Latham (2020) 的 Eq. (9)/(15)/(26)/(34)/(35)。每层最小化

    pHSIC(Z_k, Z_k) − γ · pHSIC(Y, Z_k),   k = 1, ..., L          (9)

``pHSIC`` 是论文提出的 HSIC 近似（"plausible HSIC"），其经验估计为

    p̂HSIC(A,B) = (1/m²) Σ_ij k(a_i,a_j)k(b_i,b_j)
                − (1/m²) Σ_ij k(a_i,a_j) · (1/m²) Σ_ql k(b_q,b_l)  (34)

即**减去两个核矩阵均值的乘积**。它与真 HSIC 的差别在于：真 HSIC 要把两个核矩阵都
中心化，而 pHSIC 只中心化一个（等价写法：``mean((A − mean(A)) · B)``）。这样神经元
不必记住多个数据点上的活动——这就是"plausible"的来源。代价是 pHSIC 是 HSIC 的
上界（论文 Eq. 32），当输出核已中心化时两者相等。

## 教学信号是二值的（Eq. 35）

输出核 k(y_i, y_j) 取**居中标签的余弦相似度**。类别数 n、类别均衡时，

    k(y_i, y_j) = 1            若 y_i = y_j
    k(y_i, y_j) = −1/(n−1)     否则                              (35)

论文原话：

> We will use exactly this signal in our experiments, as the datasets we used are balanced.

所以本模块**直接实现这个二值信号**，不绕道 one-hot 再算余弦——省一次矩阵运算，
也让"教学信号只有一个标量"这件事在代码里一眼可见（它是三因子规则的第三因子中
唯一来自标签的部分）。

## 高斯核的实现

Eq. (26) 是 ``k(a_i,a_j) = exp(−‖a_i−a_j‖²/(2σ²))``。本模块用展开式
``‖a−b‖² = ‖a‖² + ‖b‖² − 2aᵀb`` 算平方距离，而不是官方的 ``torch.pdist``：
``pdist`` 在 ``i = j`` 处梯度未定义（0/0），而展开式在整个批上都有干净的梯度。
两者数学上等价，有测试对照朴素直算验证这一点。
"""

from __future__ import annotations

import torch

__all__ = [
    "biased_hsic",
    "centered_label_kernel",
    "cosine_kernel",
    "gaussian_kernel",
    "kernelized_bottleneck_objective",
    "phsic",
    "squared_distances",
]

#: 论文 D.5 用的高斯核带宽。
DEFAULT_SIGMA = 5.0
#: 论文 D.5 用的瓶颈平衡参数（§5.2 说别的取值更差）。
DEFAULT_GAMMA = 2.0


def squared_distances(x: torch.Tensor) -> torch.Tensor:
    """批内两两平方欧氏距离，``(m, d)`` → ``(m, m)``。

    用展开式而非 ``torch.pdist``：后者在 ``i = j`` 处不可导（两个相同向量的距离对
    输入的导数是 0/0），而本模块需要对该矩阵求梯度。``clamp_min(0)`` 兜住浮点误差
    导致的极小负值。
    """
    if x.dim() != 2:
        raise ValueError(f"需要 (m, d) 的二维输入，收到形状 {tuple(x.shape)}。")
    sq_norm = x.pow(2).sum(dim=-1, keepdim=True)
    # 展开式在数值上会有极小负值（尤其两点很接近时），夹到 0 以免 sqrt 出 NaN
    return (sq_norm + sq_norm.transpose(0, 1) - 2.0 * (x @ x.transpose(0, 1))).clamp_min(0.0)


def gaussian_kernel(x: torch.Tensor, sigma: float = DEFAULT_SIGMA) -> torch.Tensor:
    """高斯核 ``k(a_i,a_j) = exp(−‖a_i−a_j‖²/(2σ²))``（Eq. 15 / Eq. 26）。"""
    if sigma <= 0:
        raise ValueError(f"sigma 必须为正，收到 {sigma}。")
    return torch.exp(-squared_distances(x) / (2.0 * sigma**2))


def cosine_kernel(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """余弦核 ``k(x_i, x_j) = <x_i, x_j> / (‖x_i‖·‖x_j‖)``——论文的 "cossim"（Eq. 25）。

    与高斯核一样给出一个**正半定**的格拉姆矩阵，所以 pHSIC 的推导一字不改。

    ⚠️ **论文的默认核是余弦，不是高斯**：官方仓库 ``--hsic-kernel-z`` / ``--hsic-kernel-y``
    的默认值都是 ``cossim``；论文 Table 1/表 3 把 ``pHSIC: cossim`` 与 ``pHSIC: Gaussian``
    两组列并列报告。本仓库的默认取的是**高斯**，因为验收数字（`train_mnist.py` 的 97.79%）
    对应的正是论文 Table 3 里 MNIST 的 **Gaussian + grp + div** 那一列——余弦核在这里作为
    **对照臂**接进来。这一默认差异写在 README 的「已知边界」里。

    ``eps`` 只在范数为 0 时起作用（全零行），照官方 ``CosineSimilarityKernel`` 的写法。
    """
    norm = x.norm(dim=1, keepdim=True)
    safe = torch.where(norm > 0, norm, torch.ones_like(norm))
    normalized = x / safe
    return normalized @ normalized.t()


def centered_label_kernel(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    """教学信号 ``k(y_i,y_j)``（Eq. 35）：同类 1，异类 ``−1/(n−1)``。

    论文用"居中标签的余弦相似度"定义它，并指出类别均衡时化简为这个二值信号；论文
    实验里用的就是它。``n_classes = 1`` 时异类项无定义，此时恒为 1——调用方应避免
    这种退化情形。

    Args:
        labels: ``(m,)`` 的整数类别标签。
        n_classes: 类别数 n。

    Returns:
        ``(m, m)`` 的对称矩阵。
    """
    if n_classes < 2:
        raise ValueError(f"n_classes 必须 >= 2，收到 {n_classes}（异类信号 −1/(n−1) 需要 n > 1）。")
    labels = labels.reshape(-1)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    off_diagonal_value = -1.0 / (n_classes - 1)
    return torch.where(
        same,
        torch.ones_like(same, dtype=torch.float64),
        torch.full_like(same, off_diagonal_value, dtype=torch.float64),
    ).to(torch.get_default_dtype())


def phsic(a_kernel: torch.Tensor, b_kernel: torch.Tensor) -> torch.Tensor:
    """pHSIC 的经验估计（Eq. 34）。

    ``mean(A·B) − mean(A)·mean(B)``，与论文的
    ``(1/m²)Σ_ij k_ij^a k_ij^b − (1/m²)Σ k_ij^a · (1/m²)Σ k_ij^b`` 是同一件事。
    """
    if a_kernel.shape != b_kernel.shape:
        raise ValueError(
            f"两个核矩阵形状必须一致，收到 {tuple(a_kernel.shape)} 与 {tuple(b_kernel.shape)}。"
        )
    return (a_kernel * b_kernel).mean() - a_kernel.mean() * b_kernel.mean()


def biased_hsic(a_kernel: torch.Tensor, b_kernel: torch.Tensor) -> torch.Tensor:
    """带偏 HSIC 估计。

    论文对比了 HSIC 与 pHSIC，结论是**优化 HSIC 并不改善性能**，而原始 HSIC 瓶颈
    表述（Eq. 2）差得多。保留这个函数是为了让那个对比在代码里可复现，不是主路径。
    """
    if a_kernel.shape != b_kernel.shape:
        raise ValueError(
            f"两个核矩阵形状必须一致，收到 {tuple(a_kernel.shape)} 与 {tuple(b_kernel.shape)}。"
        )
    a_mean = a_kernel.mean(dim=0)
    b_mean = b_kernel.mean(dim=0)
    return (
        (a_kernel * b_kernel).mean() - 2 * (a_mean * b_mean).mean() + a_mean.mean() * b_mean.mean()
    )


def kernelized_bottleneck_objective(
    z: torch.Tensor,
    label_kernel: torch.Tensor,
    *,
    sigma: float = DEFAULT_SIGMA,
    gamma: float = DEFAULT_GAMMA,
    mode: str = "plausible",
    kernel: str = "gaussian",
) -> torch.Tensor:
    """单层的局部目标 ``pHSIC(Z,Z) − γ·pHSIC(Y,Z)``（Eq. 9）。

    这就是每层各自最小化的东西。它的**梯度**具有三因子 Hebbian 形式（论文
    Eq. 10→12→18 的推导），但梯度里第三因子的具体形状由本函数的结构决定——
    见 :mod:`research.ib_hebbian.layers` 里对"局部性到底指什么"的说明。

    Args:
        z: ``(m, d)``，该层的活动（尚未做除法归一化）。
        label_kernel: ``(m, m)``，由 :func:`centered_label_kernel` 得到。
        sigma: 高斯核带宽 σ。
        gamma: 瓶颈平衡参数 γ。
        mode: ``"plausible"``（pHSIC，实验用的）或 ``"biased"``（HSIC，对照用）。
            命名与取值照抄官方仓库的 ``--hsic-estimate-mode``（默认 ``plausible``）。
            注意**论文全文没有 "biased" 这个词**——它是官方代码的用语，论文只写 HSIC 与 pHSIC。
        kernel: ``"gaussian"`` 或 ``"cossim"``。**官方默认是 cossim**，本仓库默认高斯
            （与验收数字对应的是论文 Table 3 里 MNIST 的 Gaussian 列）——见 :func:`cosine_kernel`。

    Returns:
        标量损失。
    """
    if mode not in ("plausible", "biased"):
        raise ValueError(f"mode 只能是 'plausible' 或 'biased'，收到 {mode!r}。")
    if kernel not in ("gaussian", "cossim"):
        raise ValueError(f"kernel 只能是 'gaussian' 或 'cossim'，收到 {kernel!r}。")
    estimator = phsic if mode == "plausible" else biased_hsic

    z_kernel = gaussian_kernel(z, sigma=sigma) if kernel == "gaussian" else cosine_kernel(z)
    return estimator(z_kernel, z_kernel) - gamma * estimator(label_kernel, z_kernel)

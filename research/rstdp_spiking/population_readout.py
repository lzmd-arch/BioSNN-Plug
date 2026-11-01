"""群体线性解码读出：整段试次的发放计数 → 岭回归 → 动作分数。

这是预注册 ``prereg-readout-v2.md`` 里"换读出结构"的那一半。与旧读出的区别：
旧的是**单点速率 → 阈值**（逐单元、非线性、参数靠奖励塑形去逼）；这个是
**整段计数 → 线性解码**（多→多、参数由最小二乘一次解出、评估时冻结）。

为什么可能有用：D15 的标度（K 窗平均提升 ∝ K^0.6）说明瓶颈是单点的信噪比；
群体解码把 K 个单元一次用完，而不是先压成一个数。

这里的 :class:`RidgeReadout` 只做"拟合"与"打分"，不碰训练循环——
把它接进 actor 是下一步的事，先让这块本身可测。
"""

from __future__ import annotations

import torch

__all__ = ["RidgeReadout"]


class RidgeReadout:
    """岭回归线性解码器：``scores = counts @ W + b``。

    Args:
        alpha: 岭系数。**必须在训练集内部选**（留一或按块留出），不能用评估集。
    """

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha < 0:
            raise ValueError(f"alpha 必须 ≥ 0，收到 {alpha!r}。")
        self.alpha = float(alpha)
        self.weight: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None

    def fit(self, counts: torch.Tensor, targets: torch.Tensor) -> RidgeReadout:
        """最小二乘一次解出。

        Args:
            counts: ``(n, neurons)`` 的整段发放计数。
            targets: ``(n, actions)`` 的目标分数（可用 one-hot 或回报）。
        """
        if counts.ndim != 2 or targets.ndim != 2:
            raise ValueError(
                f"counts 与 targets 都必须是 2 维，收到 {counts.shape} 与 {targets.shape}。"
            )
        if counts.shape[0] != targets.shape[0]:
            raise ValueError("counts 与 targets 的样本数必须一致。")
        n = counts.shape[0]
        x = torch.cat([counts, torch.ones(n, 1)], dim=1).double()
        y = targets.double()
        eye = torch.eye(x.shape[1], dtype=x.dtype)
        eye[-1, -1] = 0.0  # 偏置不参与正则
        gram = x.T @ x + self.alpha * eye
        solution = torch.linalg.solve(gram, x.T @ y)
        self.weight = solution[:-1].to(torch.float32)
        self.bias = solution[-1].to(torch.float32)
        return self

    def scores(self, counts: torch.Tensor) -> torch.Tensor:
        if self.weight is None or self.bias is None:
            raise RuntimeError("解码器还没 fit。")
        return counts @ self.weight + self.bias

    def choose(self, counts: torch.Tensor) -> int:
        return int(torch.argmax(self.scores(counts)))

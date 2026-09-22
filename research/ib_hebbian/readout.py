"""读出层：岭回归闭式解（累积二阶统计量 + 一次求解）。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

## 它替代的是什么

原来这里的读出是一个 ``nn.Linear``，用 SGD + 交叉熵训练（``.backward()``、学习率、动量、
100 个 epoch）。现在改为**闭式解**：

    W = (XᵀX/n + λI)⁻¹ XᵀY          X 是末层特征（增广一列常数），Y 是 one-hot 标签

没有 ``.backward()``、没有学习率、没有 epoch、没有迭代。

## 边界：**它不是「局部学习的读出」，不要那样写**

这一点必须写死在这里，因为很容易被说成比实际更强的东西：

* **它确实不是逐层反传**，也不迭代——累积完解一次方程就完了；
* **两个累积量都是逐样本可加的 Hebbian 式相关**：``h hᵀ`` 是「前后突触活动之积」，``h yᵀ``
  是「前突触活动 × 教学信号」，与设计里的三因子形式同族；
* **但它仍然是监督的**（用了标签），而且 ``XᵀX`` 是 ``(H+1)²`` 的矩阵、需要**全体训练样本**
  才能累积出来——那是一个**全局量**，不是突触局部量。最后那步求逆也是全局的。

**所以正确的说法是**：「**无反向传播、非迭代的闭式读出**」（a closed-form, non-iterative
readout with no backpropagation）。**不应**说「局部学习的读出」或「纯局部分类」。
它相对于 SGD 读出的改进是**去掉了反向传播与迭代**，不是去掉了全局性。

对照之下，隐藏层那条路径的局部性主张不受影响：各层用**自己的**目标函数就地更新、
输入被切断梯度——这一点由 ``layers.py`` 的断言与测试保证（读出反传后各隐藏层权重
``.grad`` 必须是 ``None``；换成闭式解之后连这一步反传都不存在了）。
"""

from __future__ import annotations

import torch

__all__ = ["RidgeReadout"]


class RidgeReadout:
    """累积 ``(XᵀX, XᵀY)`` 并一次解出岭回归权重。

    增广一列常数以容纳偏置（论文 D.1 的读出有偏置），并**不对偏置项做正则化**——
    这是岭回归的标准做法：正则化要收缩的是权重，不是截距。

    Attributes:
        xtx: ``(width+1, width+1)``，``Σ_m [h;1][h;1]ᵀ``。
        xty: ``(width+1, n_classes)``，``Σ_m [h;1]·onehot(y)ᵀ``。
        count: 已累积的样本数。
    """

    def __init__(
        self,
        width: int,
        n_classes: int,
        *,
        ridge: float = 1e-2,
        device: torch.device | None = None,
    ) -> None:
        # None 是调用方的错，但报出来要能看懂：这个 ridge 只在 `solve()` 不带参数时当兜底，
        # 调用方该传一个具体数。给 None 会让下面那句变成一句与参数名无关的 TypeError。
        if ridge is None:
            raise ValueError("ridge 不能是 None；λ 在 solve() 时显式给出，或在这里给一个具体值。")
        if ridge < 0:
            raise ValueError(f"ridge 必须非负，收到 {ridge}。")
        self.width = int(width)
        self.n_classes = int(n_classes)
        self.ridge = float(ridge)
        self.device = device or torch.device("cpu")
        # 累积量用 **float64**：XᵀX 要跨 5.4 万样本求和，float32 的累加误差会进到解里。
        # 这一步只做一次，多花的内存（1025² × 8 B ≈ 8.4 MB）换来的是解的可信度。
        self.xtx = torch.zeros(
            self.width + 1, self.width + 1, dtype=torch.float64, device=self.device
        )
        self.xty = torch.zeros(
            self.width + 1, self.n_classes, dtype=torch.float64, device=self.device
        )
        self.count = 0

    @torch.no_grad()
    def accumulate(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """累加一个批次的贡献。**逐样本可加**，是这里唯一"像 Hebbian"的一步。

        Args:
            features: ``(m, width)`` 末层特征（已 detach）。
            labels: ``(m,)`` 类别下标。
        """
        augmented = torch.cat(
            [
                features.to(torch.float64),
                torch.ones(features.shape[0], 1, dtype=torch.float64, device=features.device),
            ],
            dim=1,
        )
        onehot = torch.nn.functional.one_hot(labels, self.n_classes).to(torch.float64)
        self.xtx += augmented.T @ augmented
        self.xty += augmented.T @ onehot
        self.count += int(features.shape[0])

    @torch.no_grad()
    def solve(self, ridge: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """解出 ``(weight, bias)``，形状 ``(n_classes, width)`` 与 ``(n_classes,)``。

        Args:
            ridge: 覆盖构造时的 λ。用来在**验证集**上比较不同的 λ——Gram 矩阵只累积一次，
                多解几个几乎不花时间（一次求逆 1025³ ≈ 10⁹ flops，GPU 上毫秒级）。

        Raises:
            ValueError: 一个样本都没累积，或 λ 为负。
        """
        if self.count == 0:
            raise ValueError("还没累积任何样本，无法求解。")
        ridge = self.ridge if ridge is None else float(ridge)
        if ridge < 0:
            raise ValueError(f"ridge 必须非负，收到 {ridge}。")

        gram = self.xtx / self.count
        target = self.xty / self.count
        regulariser = ridge * torch.eye(self.width + 1, dtype=torch.float64, device=self.device)
        # **偏置不正则化**：要收缩的是权重，不是截距。正则化它会把读出整体往零拉。
        regulariser[-1, -1] = 0.0

        solution = torch.linalg.solve(gram + regulariser, target)  # (width+1, n_classes)
        weight = solution[:-1].T.contiguous()  # (n_classes, width)
        bias = solution[-1]  # (n_classes,)
        return weight.float(), bias.float()

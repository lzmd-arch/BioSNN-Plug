"""多层堆叠 + 线性读出（论文 §5.2 的小规模全连接网络）。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

结构照论文 D.1 与 §5.2：

* 隐藏层：3 层、每层 1024 个神经元，`LocalObjectiveLayer`（无偏置）；
* 读出层：一个**有偏置**的线性分类器，用**交叉熵**训练——论文 D.1 原话：

  > None of the hidden layers had the bias term, but the output layer did. The last
  > layer (or the whole network for backprop) was trained with the cross-entropy loss.

所以读出的训练方式与隐藏层**不同**：隐藏层用各自的局部 pHSIC 目标，读出用监督交叉熵。
这不是取巧——论文表 1 里 pHSIC 那几列就是这个配置（隐藏层局部规则 + 读出交叉熵），
也是"感知层用局部规则提取特征，读出来做分类"这一评估方式的由来。

读出的输入是隐藏层**已归一化**的活动 r，且已被切断梯度，所以交叉熵的梯度只落在
读出权重上，不会回传进隐藏层——与隐藏层之间的局部性一致。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from research.ib_hebbian.layers import LocalObjectiveLayer
from research.ib_hebbian.objective import DEFAULT_GAMMA, DEFAULT_SIGMA, centered_label_kernel

__all__ = ["IBHebbianPerceptron", "preprocess_mnist"]

#: 论文 D.4：MNIST 图像按 0.5 居中、按 0.5 归一化。
MNIST_SHIFT = 0.5
MNIST_SCALE = 0.5


def preprocess_mnist(images: torch.Tensor) -> torch.Tensor:
    """论文 D.4 的 MNIST 预处理：``(x/255 − 0.5) / 0.5``，并展平成 ``(m, 784)``。

    Args:
        images: ``(m, 28, 28)`` 或 ``(m, 784)``，uint8 或浮点。

    Returns:
        ``(m, 784)`` 的浮点张量，取值在 ``[−1, 1]``。
    """
    x = images.reshape(images.shape[0], -1).float() / 255.0
    return (x - MNIST_SHIFT) / MNIST_SCALE


class IBHebbianPerceptron(nn.Module):
    """隐藏层用核化 IB 局部规则、读出层用交叉熵的多层感知机。

    Attributes:
        layers: 隐藏层（每层自带优化器）。
        readout: 线性读出（有偏置）。它的参数**不在**任何隐藏层的优化器里。
        readout_optimizer: 读出的专属优化器。
    """

    def __init__(
        self,
        n_in: int = 28 * 28,
        *,
        width: int = 1024,
        n_layers: int = 3,
        n_classes: int = 10,
        n_groups: int = 16,
        sigma: float = DEFAULT_SIGMA,
        gamma: float = DEFAULT_GAMMA,
        divnorm_power: float = 0.2,
        smoothing_delta: float = 1.0,
        hidden_learning_rate: float = 0.4,
        hidden_momentum: float = 0.95,
        readout_learning_rate: float = 5e-3,
        readout_momentum: float = 0.95,
        negative_slope: float = 0.01,
        dropout_p: float = 0.0,
        center_activity: bool = True,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers 必须 >= 1，收到 {n_layers}。")

        self.n_in = int(n_in)
        self.width = int(width)
        self.n_layers = int(n_layers)
        self.n_classes = int(n_classes)
        self.n_groups = int(n_groups)

        sizes = [n_in, *([width] * n_layers)]
        self.layers = nn.ModuleList(
            LocalObjectiveLayer(
                sizes[i],
                sizes[i + 1],
                n_groups=n_groups,
                sigma=sigma,
                gamma=gamma,
                divnorm_power=divnorm_power,
                smoothing_delta=smoothing_delta,
                learning_rate=hidden_learning_rate,
                momentum=hidden_momentum,
                negative_slope=negative_slope,
                dropout_p=dropout_p,
                center_activity=center_activity,
            )
            for i in range(n_layers)
        )

        # 读出层有偏置（论文 D.1），且用交叉熵训练。
        self.readout = nn.Linear(width, n_classes, bias=True)
        self.readout_optimizer = torch.optim.SGD(
            self.readout.parameters(), lr=readout_learning_rate, momentum=readout_momentum
        )

    def label_kernel(self, labels: torch.Tensor) -> torch.Tensor:
        """当前批的二值教学信号（Eq. 35）。"""
        return centered_label_kernel(labels, self.n_classes)

    def hidden_activations(self, x: torch.Tensor, labels: torch.Tensor | None, *, update: bool):
        """逐层前向。``update=True`` 时每层在自己的前向内更新一次。"""
        kernel = self.label_kernel(labels) if labels is not None else None
        h = x
        for layer in self.layers:
            h = layer(h, kernel, update=update)
        return h

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        update_hidden: bool = False,
        update_readout: bool = False,
    ) -> torch.Tensor:
        """前向，可选地在过程中更新隐藏层与读出层。

        Args:
            x: ``(m, n_in)``，**不携带梯度**。
            labels: ``(m,)`` 的类别下标。更新隐藏层或读出层时必需。
            update_hidden: 是否让每个隐藏层就地更新一次。
            update_readout: 是否更新读出层的交叉熵。

        Returns:
            ``(m, n_classes)`` 的 logits。``update_readout=True`` 时返回的是已 detach
            的版本（本步的损失另有返回值可取——见 :meth:`step`）。
        """
        if (update_hidden or update_readout) and labels is None:
            raise ValueError("更新需要 labels。")

        h = self.hidden_activations(x, labels, update=update_hidden)
        logits = self.readout(h)

        if update_readout:
            self.readout_optimizer.zero_grad()
            functional.cross_entropy(logits, labels).backward()
            self.readout_optimizer.step()
            logits = logits.detach()

        return logits

    def step(self, x: torch.Tensor, labels: torch.Tensor, *, update_readout: bool = True) -> float:
        """一个训练步：更新全部隐藏层（以及可选地更新读出层）。

        Args:
            update_readout: 为假时**只更新隐藏层**。闭式解读出（``readout.RidgeReadout``）
                模式下用它——读出不是逐 epoch 学的，而是训练完之后一次解出来的，
                所以训练期间它不该被动。

        Returns:
            更新**之前**的读出交叉熵。``update_readout=False`` 时读出的权重还没解出来，
            这个数**没有意义**（调用方应忽略它）。
        """
        with torch.no_grad():
            cross_entropy = float(functional.cross_entropy(self.forward(x, labels), labels).item())
        self.forward(x, labels, update_hidden=True, update_readout=update_readout)
        return cross_entropy

    @torch.no_grad()
    def hidden_objective(self, x: torch.Tensor, labels: torch.Tensor) -> float:
        """各隐藏层**局部目标之和**。

        闭式解读出模式下**这才是在被优化的量**：读出不是逐 epoch 学的，所以在它被解出来
        之前，任何用读出算的交叉熵/准确率都没有意义（会随隐藏层特征的漂移而乱走）。
        进度日志要打印的是这个，不是那个。
        """
        if labels is None:
            raise ValueError("局部目标需要 labels。")
        kernel = self.label_kernel(labels)
        h = x
        total = 0.0
        for layer in self.layers:
            total += layer.measure_local_objective(h, kernel)
            h = layer(h, kernel, update=False)
        return float(total)

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """末层特征 ``h``（不更新任何权重）。闭式解读出累积 ``XᵀX / XᵀY`` 时用它。"""
        return self.hidden_activations(x, None, update=False)

    @torch.no_grad()
    def set_readout(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        """把闭式解出来的读入写进读出层。

        Raises:
            ValueError: 形状对不上——那是"解出来的东西"与"要被写进去的层"不是一回事，
                必须当场暴露。
        """
        expected = (self.n_classes, self.width)
        if tuple(weight.shape) != expected or tuple(bias.shape) != (self.n_classes,):
            raise ValueError(
                f"读出权重/偏置的形状应为 {expected} / ({self.n_classes},)，"
                f"收到 {tuple(weight.shape)} / {tuple(bias.shape)}。"
            )
        self.readout.weight.copy_(weight.to(self.readout.weight.dtype))
        self.readout.bias.copy_(bias.to(self.readout.bias.dtype))

    @torch.no_grad()
    def evaluate(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
        """评测：不更新任何权重，返回 ``(交叉熵, 准确率)``。"""
        logits = self.forward(x, labels, update_hidden=False, update_readout=False)
        loss = float(functional.cross_entropy(logits, labels).item())
        accuracy = float((logits.argmax(dim=-1) == labels).float().mean().item())
        return loss, accuracy

    @torch.no_grad()
    def layer_objectives(self, x: torch.Tensor, labels: torch.Tensor) -> list[float]:
        """各隐藏层在**自己那份实际输入**上的局部目标值。

        逐层前向、逐层测量：第 k 层拿到的是第 k−1 层归一化后的输出（即它真实的输入），
        而不是原始输入。不更新任何权重——测量不该改变被测量的东西。
        """
        kernel = self.label_kernel(labels)
        h = x
        values: list[float] = []
        for layer in self.layers:
            values.append(layer.measure_local_objective(h, kernel))
            h = layer(h, kernel, update=False)
        return values

    @torch.no_grad()
    def pre_normalization_activity(self, x: torch.Tensor) -> torch.Tensor:
        """最后一层隐藏层**除法归一化之前**的活动 z，``(m, width)``。

        计划书 §9 的「活跃神经元比例」是 SNN 指标；本层是速率型（LReLU）不是脉冲型，
        所以取它的类比量，而且必须量在**归一化之前**：归一化按组居中了活动，之后
        恒有一半左右的单元为负——在那个量上算"活跃比例"只是在数分组，跟网络学没学到
        东西无关。归一化前的 LReLU 输出为正，表示该单元处在非负（线性）区，
        这才对应"这个单元还参与编码"。
        """
        h = x
        for index, layer in enumerate(self.layers):
            z = layer.nonlinearity(layer.linear(h))
            if index == len(self.layers) - 1:
                return z
            h = layer.divisive_norm(z)
        raise AssertionError("n_layers >= 1 已在构造时校验，这里不该走到。")

"""融合模型骨架：**把 SNN 的池状态接进小语言模型的隐藏层**（FiLM 式状态调制）。

## 为什么是 FiLM 而不是加 logits 偏置

我们已经有两条实测：加性偏置在真实骨架上两次为零（S0）；乘性/标定那一族一次拿到
+0.26 nats（S0b）。所以状态该乘进表示，而不是加到输出上。

## "拆不开"的定义

`p = 0`（或把 `state_dim` 全零送进去）时，每层的调制量恰好为 0 ⇒ 模型退化成**原模型**。
反过来，只要 `p ≠ 0`，**每一层的前向都被它改过**——没有"关掉 SNN 跑一遍"这种运行模式。

## 可训性

只有 `StateModulator`（几万参数）与（可选）`lm_head` 可训；主干默认冻结。
这是 8 GB 上唯一能全参走完前向+反向的形状。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StateModulator(nn.Module):
    """`p → 每层的 (scale, shift)`，**低秩 + 无偏置**。

    两条设计约束，都是为了"拆不开"这件事在**训练之后**仍然成立：

    1. **无偏置**：`p = 0 ⇒ z = 0 ⇒ 调制恰好为 0`。留了偏置的话，训完之后
       `p=0` 也会给出非零调制，"没有状态时就是原模型"这句话就失效了；
    2. **低秩**（`rank` 维）：直接 `16 → 2·L·H` 是 154M 参数，在 8 GB 上没法训。
       低秩把可训参数压到两万量级，仍然表达"每层不同的缩放/平移"。

    `w1` 零初始化 ⇒ 训练开始时调制为 0（与"建了融合头但没给状态"同一条纪律）。
    """

    def __init__(
        self, state_dim: int, n_layers: int, hidden: int, alpha: float = 0.1, rank: int = 8
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.n_layers = int(n_layers)
        self.hidden = int(hidden)
        self.rank = int(rank)
        self.w1 = nn.Parameter(torch.zeros(state_dim, n_layers * 2 * rank))
        self.u = nn.Parameter(torch.randn(2, rank, hidden) / hidden**0.5)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        z = (p @ self.w1).view(p.shape[0], self.n_layers, 2, self.rank)
        mod = torch.einsum("blcr,crh->blch", z, self.u)
        return self.alpha * torch.tanh(mod)  # (B, L, 2, H)，有界


class FusionLM(nn.Module):
    """小 LM + 状态调制。`forward(input_ids, p)`；`p=None` 时等价于原模型。"""

    def __init__(
        self, lm, state_dim: int, alpha: float = 0.1, train_head: bool = False, rank: int = 8
    ):
        super().__init__()
        self.lm = lm
        for prm in self.lm.parameters():
            prm.requires_grad = False
        cfg = lm.config
        layers = lm.model.layers
        self.modulator = StateModulator(state_dim, len(layers), cfg.hidden_size, alpha, rank)
        self._mod = None
        self._hooks = [
            layer.register_forward_hook(self._make_hook(i)) for i, layer in enumerate(layers)
        ]
        self.head_trainable = bool(train_head)
        if train_head:
            lm.lm_head.weight.requires_grad = True
            if lm.lm_head.bias is not None:
                lm.lm_head.bias.requires_grad = True

    def _make_hook(self, idx: int):
        def hook(_module, _inputs, output):
            if self._mod is None:
                return output
            h = output[0] if isinstance(output, tuple) else output
            # **必须跟着隐藏层的 dtype 走**：调制器是 fp32，主干是 bf16，
            # 直接乘会把 hidden 提升成 fp32，下一个 Linear 立刻报 dtype 不匹配。
            dt = h.dtype
            scale = self._mod[:, idx, 0].to(dt).unsqueeze(1)
            shift = self._mod[:, idx, 1].to(dt).unsqueeze(1)
            h = h * (1.0 + scale) + shift
            return (h, *output[1:]) if isinstance(output, tuple) else h

        return hook

    def set_state(self, p: torch.Tensor | None) -> None:
        self._mod = None if p is None else self.modulator(p.to(self.modulator.w1.dtype))

    def forward(self, input_ids: torch.Tensor, p: torch.Tensor | None = None, **kw):
        self.set_state(p)
        try:
            return self.lm(input_ids=input_ids, **kw)
        finally:
            self._mod = None

    def trainable_parameters(self):
        yield from self.modulator.parameters()
        if self.head_trainable:
            yield from (prm for prm in self.lm.lm_head.parameters() if prm.requires_grad)


def load_fusion(
    model_dir: str,
    state_dim: int,
    *,
    alpha: float = 0.1,
    train_head: bool = False,
    rank: int = 8,
    dtype=torch.bfloat16,
    device: str = "cuda",
):
    from transformers import AutoModelForCausalLM

    lm = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype)
    lm.to(device).eval()
    return FusionLM(lm, state_dim, alpha=alpha, train_head=train_head, rank=rank).to(device)

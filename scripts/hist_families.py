"""自造历史族：族 k 的符号 k 占 `conc`，其余均分。

绕开 `seeded_dirichlet` 的 `PREF_SEED=0` 混淆（同 α 的历史共享同一个 `pref`），
并且让"历史的内容"变成一个**可命名的标签**，供解码与盆地指派使用。
"""

from __future__ import annotations

# ruff: noqa: N803
import torch


def make_family_hist(
    dominant: int,
    seed: int,
    T: int,
    B: int,
    dev: str,
    n_sym: int = 8,
    conc: float = 0.7,
) -> torch.Tensor:
    """返回 `(B, T)` 的 token 流。同一族共享概率向量，不同族的主导符号不同。"""
    g = torch.Generator().manual_seed(seed)
    p = torch.full((n_sym,), (1.0 - conc) / max(n_sym - 1, 1))
    p[dominant] = conc
    idx = torch.multinomial(p, B * T, replacement=True, generator=g).reshape(B, T)
    return idx.to(dev)


def spatial_bins(r: torch.Tensor, side: int = 32, blocks: int = 8) -> torch.Tensor:
    """`(B, N)` 的发放 → `(B, blocks²)` 的空间桶均值。"""
    b = r.shape[0]
    step = side // blocks
    x = r.view(b, side, side).view(b, blocks, step, blocks, step).mean(dim=(2, 4))
    return x.reshape(b, blocks * blocks)

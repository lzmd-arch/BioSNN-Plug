"""树读出的融合核：把「gather + 点积」和「gather + 信号累加」各压成**一次** DRAM 读。

为什么需要它：PyTorch 路径下 `weight.index_select(0, flat)` 会把 ``(B·depth, M)`` 物化出来
（一次写、一次读），紧接着的 `einsum` 再读一遍——**同一批字节走三次 DRAM**。
实测（`bench_hierarchical.py`）：只 gather 是 936 GB/s，gather+点积掉到 72 GB/s。

两个核的分工（合起来把一步的 6 趟压到 4 趟）：

- :func:`gather_dot`：每个 ``(b, j)`` 一个 program，沿 M 累加，**行读进来就直接乘掉**，
  不落临时张量；
- :func:`gather_signal`：每个 ``(b, m 分块)`` 一个 program，沿 depth 累加，
  行读进来就直接加权累加。

更新那一步仍然是 `index_add_`（读-改-写 2 趟）。要再省就得按节点分组（一个节点的行装进
shared memory，读一趟写一趟），那是下一步。

没有 CUDA 或没有 Triton 时**自动退回 PyTorch 路径**，所以 CPU 测试照跑。
"""

from __future__ import annotations

import torch

try:  # pragma: no cover - 取决于运行环境
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False

__all__ = ["HAS_TRITON", "available", "gather_dot", "gather_signal"]


def available(device: torch.device) -> bool:
    return HAS_TRITON and torch.device(device).type == "cuda"


if HAS_TRITON:

    @triton.jit
    def _gather_dot_kernel(
        f_ptr,
        w_ptr,
        idx_ptr,
        out_ptr,
        width,
        depth,
        stride_bm,
        stride_wm,
        block_m: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_j = tl.program_id(1)
        v = tl.load(idx_ptr + pid_b * depth + pid_j)
        acc = tl.zeros([block_m], dtype=tl.float32)
        for m0 in range(0, width, block_m):
            offs = m0 + tl.arange(0, block_m)
            mask = offs < width
            f = tl.load(f_ptr + pid_b * stride_bm + offs, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + v * stride_wm + offs, mask=mask, other=0.0).to(tl.float32)
            acc += f * w
        tl.store(out_ptr + pid_b * depth + pid_j, tl.sum(acc))

    @triton.jit
    def _gather_signal_kernel(
        d_ptr,
        w_ptr,
        idx_ptr,
        out_ptr,
        width,
        depth,
        stride_bm,
        stride_wm,
        block_m: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs = pid_m * block_m + tl.arange(0, block_m)
        mask = offs < width
        acc = tl.zeros([block_m], dtype=tl.float32)
        for j in range(0, depth):
            v = tl.load(idx_ptr + pid_b * depth + j)
            dlt = tl.load(d_ptr + pid_b * depth + j)
            w = tl.load(w_ptr + v * stride_wm + offs, mask=mask, other=0.0).to(tl.float32)
            acc += dlt * w
        tl.store(out_ptr + pid_b * stride_bm + offs, acc, mask=mask)


def _pick_block(m: int) -> int:
    for block in (2048, 1024, 512, 256, 128, 64, 32):
        if block <= max(m, 32):
            return block
    return 32


def gather_dot(features: torch.Tensor, weight: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """``(B, D)`` 的 logit；``ids`` 是 ``(B, D)`` 的节点下标（``-1`` 补位请先夹到 0）。"""
    batch, depth = ids.shape
    weight = weight.contiguous()
    out = torch.empty(batch, depth, dtype=torch.float32, device=features.device)
    block = _pick_block(int(weight.shape[1]))
    _gather_dot_kernel[(batch, depth)](
        features,
        weight,
        ids,
        out,
        int(weight.shape[1]),
        int(depth),
        int(features.stride(0)),
        int(weight.stride(0)),
        block_m=block,
        num_warps=8,
    )
    return out


def gather_signal(delta: torch.Tensor, weight: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """``(B, M)`` 的信号 ``Σ_j delta[b, j] · W[ids[b, j], :]``，**不物化 gather 结果**。"""
    batch, depth = ids.shape
    width = int(weight.shape[1])
    weight = weight.contiguous()
    out = torch.zeros(batch, width, dtype=torch.float32, device=delta.device)
    block = _pick_block(width)
    _gather_signal_kernel[(batch, triton.cdiv(width, block))](
        delta,
        weight,
        ids,
        out,
        width,
        int(depth),
        int(out.stride(0)),
        int(weight.stride(0)),
        block_m=block,
        num_warps=8,
    )
    return out

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

__all__ = [
    "HAS_TRITON",
    "available",
    "gather_dot",
    "gather_signal",
    "pack_paths",
    "packed_dot",
    "packed_signal",
    "scatter_update",
]


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
        if v < 0:  # 补位：**早退**，一行都不读。实测这一步就是 2.4x
            return
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
            live = mask & (v >= 0)
            v = tl.where(v >= 0, v, 0)  # 夹到 0 只是为了地址合法，live=False 时不会真的读
            w = tl.load(w_ptr + v * stride_wm + offs, mask=live, other=0.0).to(tl.float32)
            acc += dlt * w
        tl.store(out_ptr + pid_b * stride_bm + offs, acc, mask=mask)


def _pick_block(m: int) -> int:
    """实测（V=32,000、M=31,250、B=128、depth 18）的 dot 耗时：

    512 → 78 ms，1024 → 70 ms，2048 → 40 ms，**4096 → 22.3 ms**，8192 → 22.9 ms。

    ⇒ 大宽度上 4096 是拐点；再大不涨。配套 `num_warps=4`（8 会掉到 49 ms）。
    """
    for block in (4096, 2048, 1024, 512, 256, 128, 64, 32):
        if block <= max(m, 32):
            return block
    return 32


def gather_dot(
    features: torch.Tensor,
    weight: torch.Tensor,
    ids: torch.Tensor,
    *,
    block: int = 0,
    warps: int = 8,
) -> torch.Tensor:
    """``(B, D)`` 的 logit；``ids`` 是 ``(B, D)`` 的节点下标（``-1`` 补位请先夹到 0）。"""
    batch, depth = ids.shape
    weight = weight.contiguous()
    out = torch.empty(batch, depth, dtype=torch.float32, device=features.device)
    block = block or _pick_block(int(weight.shape[1]))
    warps = warps if warps != 8 else (4 if block >= 4096 else 8)
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
        num_warps=warps,
    )
    return out


def gather_signal(
    delta: torch.Tensor,
    weight: torch.Tensor,
    ids: torch.Tensor,
    *,
    block: int = 0,
    warps: int = 8,
) -> torch.Tensor:
    """``(B, M)`` 的信号 ``Σ_j delta[b, j] · W[ids[b, j], :]``，**不物化 gather 结果**。"""
    batch, depth = ids.shape
    width = int(weight.shape[1])
    weight = weight.contiguous()
    out = torch.zeros(batch, width, dtype=torch.float32, device=delta.device)
    block = block or _pick_block(width)
    warps = warps if warps != 8 else (4 if block >= 4096 else 8)
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


# -- ragged（只处理真实存在的 (b, j) 对） ----------------------------------------
#
# 树的矩形布局要按 `depth_max` 补到最深的那个 token。实测 SkyPile BPE（V=32,000）：
# **加权深度 12.24，最深 50** —— 矩形实现有 **4.1x** 的算力花在补位上。
# 下面这一组只走真实存在的配对：`Np = Σ_b length[b] ≈ B·12.24`。


def pack_paths(paths, targets: torch.Tensor):
    """把 ``(B, D_max)`` 的路径压成 ``Np`` 个真实节点。

    返回 ``(flat, rows, bits, offs)``：``flat[n]`` 是节点下标，``rows[n]`` 是它属于哪个样本，
    ``bits[n]`` 是走向，``offs`` 是 ``(B+1,)`` 的分段边界（**同一样本的点是连续的**）。
    """
    ids = paths.nodes.index_select(0, targets)
    valid = ids >= 0
    flat = ids[valid]
    bits = paths.bits.index_select(0, targets)[valid]
    rows = torch.arange(ids.shape[0], device=ids.device).unsqueeze(1).expand_as(ids)[valid]
    counts = valid.sum(dim=1)
    offs = torch.zeros(ids.shape[0] + 1, dtype=torch.int32, device=ids.device)
    offs[1:] = counts.cumsum(0)
    return flat, rows, bits, offs


if HAS_TRITON:

    @triton.jit
    def _packed_dot_kernel(
        f_ptr,
        w_ptr,
        flat_ptr,
        rows_ptr,
        out_ptr,
        width,
        stride_bm,
        stride_wm,
        block_m: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = tl.load(rows_ptr + pid)
        v = tl.load(flat_ptr + pid)
        acc = tl.zeros([block_m], dtype=tl.float32)
        for m0 in range(0, width, block_m):
            offs = m0 + tl.arange(0, block_m)
            mask = offs < width
            f = tl.load(f_ptr + b * stride_bm + offs, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + v * stride_wm + offs, mask=mask, other=0.0).to(tl.float32)
            acc += f * w
        tl.store(out_ptr + pid, tl.sum(acc))

    @triton.jit
    def _packed_signal_kernel(
        d_ptr,
        w_ptr,
        flat_ptr,
        off_ptr,
        out_ptr,
        width,
        stride_bm,
        stride_wm,
        block_m: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs = pid_m * block_m + tl.arange(0, block_m)
        mask = offs < width
        start = tl.load(off_ptr + pid_b)
        end = tl.load(off_ptr + pid_b + 1)
        acc = tl.zeros([block_m], dtype=tl.float32)
        for n in range(start, end):
            v = tl.load(flat_ptr + n)
            dlt = tl.load(d_ptr + n)
            w = tl.load(w_ptr + v * stride_wm + offs, mask=mask, other=0.0).to(tl.float32)
            acc += dlt * w
        tl.store(out_ptr + pid_b * stride_bm + offs, acc, mask=mask)


def packed_dot(
    features: torch.Tensor, weight: torch.Tensor, flat: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    """``(Np,)`` 的 logit，每个 program 处理一个真实配对。"""
    weight = weight.contiguous()
    out = torch.empty(flat.numel(), dtype=torch.float32, device=features.device)
    block = _pick_block(int(weight.shape[1]))
    _packed_dot_kernel[(flat.numel(),)](
        features,
        weight,
        flat,
        rows,
        out,
        int(weight.shape[1]),
        int(features.stride(0)),
        int(weight.stride(0)),
        block_m=block,
        num_warps=8,
    )
    return out


def packed_signal(
    delta: torch.Tensor,
    weight: torch.Tensor,
    flat: torch.Tensor,
    offs: torch.Tensor,
    batch: int,
) -> torch.Tensor:
    """``(B, M)`` 的信号，按 ``offs`` 分段累加，**不物化 gather 结果**。"""
    weight = weight.contiguous()
    width = int(weight.shape[1])
    block = _pick_block(width)
    out = torch.zeros(batch, width, dtype=torch.float32, device=delta.device)
    _packed_signal_kernel[(batch, triton.cdiv(width, block))](
        delta,
        weight,
        flat,
        offs,
        out,
        width,
        int(out.stride(0)),
        int(weight.stride(0)),
        block_m=block,
        num_warps=8,
    )
    return out


if HAS_TRITON:

    @triton.jit
    def _scatter_update_kernel(
        w_ptr,
        f_ptr,
        flat_ptr,
        rows_ptr,
        scale_ptr,
        lr,
        width,
        stride_wm,
        stride_bm,
        block_m: tl.constexpr,
    ):
        """``W[flat[n]] -= lr · scale[n] · f[rows[n]]``，按元素原子加。

        为什么不用 `index_add_`：profiler 实测它在 1B 形状上要 **43 ms/步**（占整步一半），
        1,024 行 × 125 KB 的散排行只跑到 ~12 GB/s。这里是逐元素 atomic_add，
        同一节点被多个样本命中时结果与 `index_add_` 相同（累加）。
        """
        pid = tl.program_id(0)
        pid_m = tl.program_id(1)
        v = tl.load(flat_ptr + pid)
        b = tl.load(rows_ptr + pid)
        s = tl.load(scale_ptr + pid)
        offs = pid_m * block_m + tl.arange(0, block_m)
        mask = offs < width
        f = tl.load(f_ptr + b * stride_bm + offs, mask=mask, other=0.0).to(tl.float32)
        tl.atomic_add(w_ptr + v * stride_wm + offs, -lr * s * f, mask=mask)


def scatter_update(
    weight: torch.Tensor,
    features: torch.Tensor,
    flat: torch.Tensor,
    rows: torch.Tensor,
    scale: torch.Tensor,
    lr: float,
    *,
    block: int = 0,
) -> None:
    """原地更新权重行；``flat`` / ``rows`` / ``scale`` 都是 ``(Np,)``。"""
    width = int(weight.shape[1])
    block = block or _pick_block(width)
    warps = 4 if block >= 4096 else 8
    _scatter_update_kernel[(flat.numel(), triton.cdiv(width, block))](
        weight,
        features,
        flat,
        rows,
        scale,
        float(lr),
        width,
        int(weight.stride(0)),
        int(features.stride(0)),
        block_m=block,
        num_warps=warps,
    )

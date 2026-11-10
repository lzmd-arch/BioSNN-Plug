"""K1：把 LIF 步进的**逐元素链**融合成一个 Triton 内核（实验性，默认不接入）。

为什么只融这一段：`TokenEProp.step` 里每步有 **~15 个 (B,N) 逐元素内核**
（膜电位、阈值、不应期、EMA、痕迹衰减），而 matmul 走 cuBLAS 已经很高效。
2026-11-10 的教训是"跳零"（稀疏）在 PyTorch 里必输；**能赢的只有融合**。

守卫：本模块的 ``fused_lif`` 与 ``TokenEProp`` 里同名的逐元素运算**逐位一致**（同 dtype、同顺序）。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["fused_lif"]


@triton.jit
def _lif_kernel(
    v_ptr,
    z_prev_ptr,
    r_ptr,
    readout_ptr,
    current_ptr,
    v_out_ptr,
    z_out_ptr,
    r_out_ptr,
    readout_out_ptr,
    alpha,
    theta,
    decay_read,
    n_refr,
    n_cols,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * block_size + tl.arange(0, block_size)
    mask = cols < n_cols
    base = row * n_cols
    offset = base + cols

    v = tl.load(v_ptr + offset, mask=mask, other=0.0)
    z_prev = tl.load(z_prev_ptr + offset, mask=mask, other=0.0)
    r = tl.load(r_ptr + offset, mask=mask, other=0.0)
    readout = tl.load(readout_ptr + offset, mask=mask, other=0.0)
    current = tl.load(current_ptr + offset, mask=mask, other=0.0)

    new_v = alpha * v + current - z_prev * theta
    scaled = (new_v - theta) / theta
    fresh = tl.where(scaled > 0.0, 1.0, 0.0)
    new_z = tl.where(r > 0.0, 0.0, fresh)
    new_r = r + n_refr * new_z - 1.0
    new_r = tl.minimum(tl.maximum(new_r, 0.0), n_refr)
    new_readout = decay_read * readout + new_z

    tl.store(v_out_ptr + offset, new_v, mask=mask)
    tl.store(z_out_ptr + offset, new_z, mask=mask)
    tl.store(r_out_ptr + offset, new_r, mask=mask)
    tl.store(readout_out_ptr + offset, new_readout, mask=mask)


def fused_lif(
    v: torch.Tensor,
    z_prev: torch.Tensor,
    refractory: torch.Tensor,
    readout: torch.Tensor,
    current: torch.Tensor,
    *,
    alpha: float,
    theta: float,
    decay_read: float,
    n_refractory: float,
    block: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 ``(v, z, refractory, readout)`` 的**新张量**（不原地改）。"""
    rows, cols = v.shape
    grid = (rows, triton.cdiv(cols, block))
    v_out = torch.empty_like(v)
    z_out = torch.empty_like(z_prev)
    r_out = torch.empty_like(refractory)
    readout_out = torch.empty_like(readout)
    _lif_kernel[grid](
        v,
        z_prev,
        refractory,
        readout,
        current,
        v_out,
        z_out,
        r_out,
        readout_out,
        alpha,
        theta,
        decay_read,
        n_refractory,
        cols,
        block_size=block,
    )
    return v_out, z_out, r_out, readout_out


def reference_lif(
    v: torch.Tensor,
    z_prev: torch.Tensor,
    refractory: torch.Tensor,
    readout: torch.Tensor,
    current: torch.Tensor,
    *,
    alpha: float,
    theta: float,
    decay_read: float,
    n_refractory: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """与 ``TokenEProp.step`` 同一顺序的参考实现（guard 用）。"""
    new_v = alpha * v + current - z_prev * theta
    scaled = (new_v - theta) / theta
    fresh = (scaled > 0).to(v.dtype)
    new_z = torch.where(refractory > 0, torch.zeros_like(fresh), fresh)
    new_r = torch.clamp(refractory + n_refractory * new_z - 1.0, min=0.0, max=n_refractory)
    new_readout = decay_read * readout + new_z
    return new_v, new_z, new_r, new_readout

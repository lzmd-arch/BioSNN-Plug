"""H-0/H-2 的实测：稠密读出 vs 树读出，**同一形状、同一设备、逐个矩阵乘**。

H-0 只在 V=32,000 上测过（7.1x）。这个脚本把同一个对照搬到任意 ``(V, M, B)``，
用来回答两个问题：

1. 训练形状（V=3,001）上树到底快多少 —— 预注册 amend2/amend3 用"触了多少行"估过，
   但那条式子只在**稠密路径已经访存受限**时成立；我们这个形状上稠密是**算力受限**的，
   所以估算与实测会差开。**以本脚本的实测为准。**
2. 1B 形状（V=32,000、M=16,384、B=256）上 H-2 的 3x 能不能过。

用法::

    uv run python -m research.sel.lm.bench_hierarchical --vocab 3001 --width 2048 --batch 256
"""

from __future__ import annotations

import argparse
import time

import torch

from research.sel.lm.hierarchical import (
    TreeReadout,
    build_huffman_paths,
    path_deltas,
    path_logits,
    path_nll,
)


def zipf_counts(vocab: int, exponent: float = 1.0) -> torch.Tensor:
    """按 Zipf 造频次 —— 均匀频次会把树压成满二叉树，深度失真。"""
    rank = torch.arange(1, vocab + 1, dtype=torch.float32)
    return 1.0 / rank.pow(exponent)


def timeit(fn, *, warmup: int = 3, repeat: int = 10) -> float:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(repeat):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / repeat  # 毫秒
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - start) * 1000.0 / repeat


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="稠密读出 vs 树读出（前向 / 信号 / 更新）")
    ap.add_argument("--vocab", type=int, default=3001)
    ap.add_argument("--width", type=int, default=2048, help="M：读出特征维度（τ 数 × N）")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--dtype", default="fp32", choices=("fp32", "bf16"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    gen = torch.Generator().manual_seed(args.seed)
    v, m, b = args.vocab, args.width, args.batch

    counts = zipf_counts(v)
    paths = build_huffman_paths(counts).to(device)
    share = counts / counts.sum()
    depth = float((share * paths.length.float().cpu()).sum())

    features = torch.randn(b, m, generator=gen).to(device).to(dtype)
    # 与模型里的 `w_out` 同形：**稠密侧是 (M, V)**，树侧是 (n_nodes, M)——行主序那条结论
    dense_w = (torch.randn(m, v, generator=gen) / m**0.5).to(device).to(dtype)
    readout = TreeReadout(paths, m, generator=gen, device=device, dtype=dtype)
    targets = torch.multinomial(share.repeat(b, 1), 1, generator=gen).squeeze(1).to(device)

    # 稠密侧：梯度矩阵用 (B, V) 的稠密"概率 - onehot"模拟，与模型里的形状一致
    grad = torch.randn(b, v, generator=gen).to(device).to(dtype) / b

    def dense_forward():
        return features @ dense_w

    def dense_signal():
        return grad @ dense_w.T

    def dense_update():
        return torch.addmm(dense_w, features.T, grad, alpha=-0.5, beta=1.0, out=dense_w)

    def dense_step():
        out = features @ dense_w
        sig = grad @ dense_w.T
        torch.addmm(dense_w, features.T, grad, alpha=-0.5, beta=1.0, out=dense_w)
        return out, sig

    def tree_step():
        return readout.step(features, targets, update=True)

    def tree_forward():
        return path_logits(features, readout.weight, paths, targets, readout.bias)

    # 树侧的三段分开测，与稠密的三段一一对应（不把整步都塞进"更新"里）
    _, prob0, bits0, valid0, safe0 = path_nll(
        features, readout.weight, paths, targets, readout.bias
    )
    delta = path_deltas(prob0, bits0, valid0)
    step_denom = features.float().pow(2).sum(dim=1).mean().clamp(min=1e-3).to(dtype)

    def tree_signal():
        return readout.step(features, targets, update=False)

    def tree_update():
        return readout.step(features, targets, update=True)

    # ---- 扁平实现：把 (B·depth) 个节点摊成一次 gather / 一次 index_add ----
    flat = safe0.reshape(-1)
    rows_index = torch.arange(b, device=device).repeat_interleave(safe0.shape[1])
    delta_flat = delta.reshape(-1)

    def flat_forward():
        w = readout.weight.index_select(0, flat).view(b, safe0.shape[1], m)
        return torch.einsum("bm,bkm->bk", features, w)

    def flat_signal():
        w = readout.weight.index_select(0, flat)
        out = torch.zeros(b, m, device=device, dtype=dtype)
        out.index_add_(0, rows_index, w * delta_flat.unsqueeze(1))
        return out

    def flat_update():
        src = features.index_select(0, rows_index) * (delta_flat / (step_denom * b)).unsqueeze(1)
        readout.weight.index_add_(0, flat, src, alpha=-readout.lr_out)

    rows = []
    for name, dense_fn, tree_fn, tree_flat in (
        ("前向", dense_forward, tree_forward, flat_forward),
        ("信号", dense_signal, tree_signal, flat_signal),
        ("更新", dense_update, tree_update, flat_update),
        ("**整步**", dense_step, tree_step, tree_step),
    ):
        d_ms = timeit(dense_fn)
        t_ms = timeit(tree_fn)
        f_ms = timeit(tree_flat)
        rows.append(
            {
                "step": name,
                "dense_ms": round(d_ms, 3),
                "tree_ms": round(t_ms, 3),
                "flat_ms": round(f_ms, 3),
                "speedup": round(d_ms / max(f_ms, 1e-9), 2),
                "slot_speedup": round(d_ms / max(t_ms, 1e-9), 2),
                "dense_us_per_token": round(d_ms * 1000 / b, 1),
                "tree_us_per_token": round(f_ms * 1000 / b, 1),
            }
        )

    total_dense = rows[-1]["dense_ms"]
    total_tree = rows[-1]["flat_ms"]
    print(
        f"V={v:,} M={m:,} B={b} {args.dtype} on {device}；树：节点 {paths.n_nodes:,}、"
        f"加权深度 {depth:.2f}、最深 {paths.depth}",
        flush=True,
    )
    for r in rows:
        print(
            f"  {r['step']}: 稠密 {r['dense_ms']:8.3f} ms ({r['dense_us_per_token']:6.1f} µs/token)"
            f"  树·逐层 {r['tree_ms']:8.3f} ms ({r['slot_speedup']:5.2f}x)"
            f"  树·扁平 {r['flat_ms']:8.3f} ms ({r['tree_us_per_token']:6.1f} µs/token)"
            f"  ⇒ {r['speedup']:5.2f}x",
            flush=True,
        )
    print(
        f"  整步：稠密 {total_dense:.3f} ms / 树 {total_tree:.3f} ms"
        f" ⇒ **{total_dense / max(total_tree, 1e-9):.2f}x**（H-2 的判据看这一行）",
        flush=True,
    )
    print(
        f"  （`V/(B·depth)` = {v / (b * depth):.2f}x 只是**访存受限时的下界**，"
        f"实测常高于它，见 amend4）",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

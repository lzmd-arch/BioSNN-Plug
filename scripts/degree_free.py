"""真正的无标度：**度分布**是幂律，不是权重幂律。

预注册：`research/sel/prereg-deg-free-v1.md`

上一轮（`scale_free_wiring.py`）测错了东西：所有神经元入度都固定为 8，
只是把**权重**改成幂律。Albert–Jeong–Barabási 那条结果是关于**度分布**的。

本轮：`P(k) ∝ k^(−γ)`，k ∈ [2, kmax]。少数枢纽接很多，多数只接几个。
用稠密权重矩阵 `W`（N=1024，每步一次 matmul，便宜）。
"""

from __future__ import annotations

# ruff: noqa: N806, N803
import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.bio_all import make_task
from scripts.instrument import FIRING_WINDOW, scale_grid

OUT = _pathlib.Path("sweep_results/degree_free")
L, R, V = 32, 8, 64
N = L * L


@torch.no_grad()
def build_deg_free(gamma: float, lam: float, kmin: int, kmax: int, seed: int, dev: str):
    """`P(k) ∝ k^(−γ)`；每个神经元按 `exp(−d/λ)` 从候选窗里选 `k_i` 个伙伴。"""
    gen = torch.Generator().manual_seed(seed)
    idx = torch.arange(N)
    x = (idx % L).float().to(dev)
    y = (idx // L).float().to(dev)
    ax = torch.arange(-R, R + 1).float()
    ox, oy = torch.meshgrid(ax, ax, indexing="ij")
    ox, oy = ox.reshape(-1).to(dev), oy.reshape(-1).to(dev)
    C = ox.numel()
    cx = (x.unsqueeze(1) + ox.unsqueeze(0)) % L
    cy = (y.unsqueeze(1) + oy.unsqueeze(0)) % L
    d = torch.sqrt((ox.unsqueeze(0) ** 2 + oy.unsqueeze(0) ** 2).expand(N, C))
    w = torch.exp(-d / lam)
    w[:, (ox == 0) & (oy == 0)] = 0.0
    lin = (cy * L + cx).long()

    # Pareto 抽样：P(k) ∝ k^(−γ)
    u = torch.rand(N, generator=gen).clamp(min=1e-6)
    k = (kmin * u ** (-1.0 / (gamma - 1.0))).ceil().clamp(max=kmax).long().to(dev)

    W = torch.zeros(N, N, device=dev)
    p = w.cpu()
    for i in range(N):
        ki = int(k[i])
        sel = torch.multinomial(p[i], ki, replacement=False)
        W[i, lin[i, sel.to(dev)]] = 1.0 / ki
    # 归一化到总驱动 1.2
    W = W * (1.2 * (1.0 / W.sum(1, keepdim=True).clamp(min=1e-9)))
    w_in = (torch.randn(V, N, generator=gen) / N**0.5).to(dev)
    return w_in, W, k


@torch.no_grad()
def run_w(w_in, W, toks, dev, alive=None):
    b, T = toks.shape
    v = torch.zeros((b, N), device=dev)
    z = torch.zeros((b, N), device=dev)
    ref = torch.zeros((b, N), device=dev)
    rate = 0.0
    for t in range(T):
        rec = z @ W.T
        nv = 0.95 * v + w_in[toks[:, t]] + rec - z * 0.5
        fresh = ((nv - 0.5) / 0.5 > 0).float()
        nz = torch.where(ref > 0, torch.zeros_like(fresh), fresh)
        if alive is not None:
            nz = nz * alive
        ref = torch.clamp(ref + 2.0 * nz - 1.0, 0.0, 2.0)
        v, z = nv, nz
        rate += float(z.sum() / (alive.sum() * b)) if alive is not None else float(z.mean())
    return rate / T


def n_viable(w_in, W, toks, dev, alive, scales):
    lo, hi = FIRING_WINDOW
    return sum(1 for s in scales if lo <= run_w(w_in * s, W, toks, dev, alive) <= hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=48)
    ap.add_argument("--n-seq", type=int, default=32)
    ap.add_argument("--gammas", default="2,2.5,3")
    ap.add_argument("--kmax", type=int, default=64)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(x) for x in args.seeds.split(",")]
    scales = scale_grid(0.4, 12.0, 1.15)
    res: dict = {"cells": {}}

    for gamma in [float(x) for x in args.gammas.split(",")]:
        vi, vl, degs = [], [], []
        for sd in seeds:
            toks, _ = make_task(args.n_seq, args.seq, 8, 10 + sd, dev)
            w_in, W, kdeg = build_deg_free(gamma, 4.0, 2, args.kmax, sd, dev)
            degs.append((float(kdeg.float().mean()), int(kdeg.max())))
            gen = torch.Generator().manual_seed(7000 + sd)
            alive = (torch.rand(N, generator=gen) >= 0.5).float().to(dev)
            vi.append(n_viable(w_in, W, toks, dev, None, scales))
            vl.append(n_viable(w_in, W, toks, dev, alive, scales))
            del w_in, W, toks
            torch.cuda.empty_cache()
        tag = f"gamma={gamma:g}"
        res["cells"][tag] = {
            "intact": vi,
            "lesion": vl,
            "mi": sum(vi) / len(vi),
            "ml": sum(vl) / len(vl),
            "deg_mean_max": degs,
        }
        print(
            f"  {tag:10s} 度 均值/最大 {[f'{a:.0f}/{b}' for a, b in degs]}  "
            f"完整 {vi}  损伤50% {vl}",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
    )

    print("")
    print("判（对照：固定度 8 的 exp 接线 = 完整 2.3 / 损伤 7.3）：")
    cc = res["cells"]
    d0 = all(v["mi"] > 0 for v in cc.values())
    print(f"  D-0 所有 γ 在完整网络都能工作 => {'过' if d0 else '不过 => 作废'}")
    if not d0:
        return 0
    best = max(cc, key=lambda k: cc[k]["ml"])
    d1 = cc[best]["ml"] > 7.3
    print(
        f"  D-1 度异质更抗损伤：最好 {best} 损伤 {cc[best]['ml']:.1f} vs 固定度 7.3 "
        f"=> {'过' if d1 else '不过'}"
    )
    seq = [cc[k]["ml"] for k in sorted(cc, key=lambda k: float(k.split("=")[1]))]
    print(f"  D-2 剂量–反应：损伤 n_viable 随 γ = {[round(x, 1) for x in seq]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

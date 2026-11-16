"""幂律（无标度）接线：分形那条路的第一步。

预注册：`research/sel/prereg-scale-free-wiring-v1.md`

现在 `w ∝ exp(−d/λ)` **有特征尺度 ⇒ 不是分形**。
无标度是 `w ∝ d^(−α)`。网络科学里有解析结果：
**幂律度分布对随机节点失效鲁棒** —— 而我们的 50% 损伤正是随机失效。

测：`n_viable`（落在工作区的尺度数），完整 vs 50% 损伤。
"""

from __future__ import annotations

# ruff: noqa: N806
import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.bio_all import make_task
from scripts.global_field import run_gf
from scripts.instrument import FIRING_WINDOW, scale_grid

OUT = _pathlib.Path("sweep_results/scale_free_wiring")
L = 32
N = L * L
R = 8
V, DEG = 64, 8


def build_power(alpha: float, seed: int, dev: str):
    """`w ∝ d^(−α)`（α=0 时退化为均匀 = 指数接线的极限对照）。"""
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
    w = torch.where(d > 0, d.clamp(min=1.0).pow(-alpha), torch.zeros_like(d))
    score = w * (1.0 + 0.5 * torch.rand(N, C, generator=gen).to(dev))
    idx_top = score.topk(DEG, dim=1).indices
    lin = (cy * L + cx).long()
    pre = lin.gather(1, idx_top)
    wrec = torch.rand(N, DEG, generator=gen).to(dev)
    wrec = wrec * (1.2 / wrec.sum(1, keepdim=True))
    w_in = (torch.randn(V, N, generator=gen) / N**0.5).to(dev)
    return w_in, pre, wrec


def build_exp(lam: float, seed: int, dev: str):
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
    score = w * (1.0 + 0.5 * torch.rand(N, C, generator=gen).to(dev))
    idx_top = score.topk(DEG, dim=1).indices
    lin = (cy * L + cx).long()
    pre = lin.gather(1, idx_top)
    wrec = torch.rand(N, DEG, generator=gen).to(dev)
    wrec = wrec * (1.2 / wrec.sum(1, keepdim=True))
    w_in = (torch.randn(V, N, generator=gen) / N**0.5).to(dev)
    return w_in, pre, wrec


def n_viable(w_in, pre, wrec, toks, dev, alive, scales):
    lo, hi = FIRING_WINDOW
    cnt = 0
    for s in scales:
        fr = run_gf(w_in * s, pre, wrec, toks, dev, 0.0, alive)
        if lo <= fr <= hi:
            cnt += 1
    return cnt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=48)
    ap.add_argument("--n-seq", type=int, default=32)
    ap.add_argument("--alphas", default="0.5,1,2,3")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(x) for x in args.seeds.split(",")]
    scales = scale_grid(0.4, 12.0, 1.15)
    res: dict = {"cells": {}}

    def run_arm(tag, builder):
        vi, vl = [], []
        for sd in seeds:
            toks, _ = make_task(args.n_seq, args.seq, 8, 10 + sd, dev)
            w_in, pre, wrec = builder(sd, dev)
            gen = torch.Generator().manual_seed(7000 + sd)
            alive = (torch.rand(N, generator=gen) >= 0.5).float().to(dev)
            vi.append(n_viable(w_in, pre, wrec, toks, dev, None, scales))
            vl.append(n_viable(w_in, pre, wrec, toks, dev, alive, scales))
            del w_in, pre, wrec, toks
            torch.cuda.empty_cache()
        res["cells"][tag] = {
            "intact": vi,
            "lesion": vl,
            "mi": sum(vi) / len(vi),
            "ml": sum(vl) / len(vl),
        }
        print(f"  {tag:12s} 完整 n_viable {vi}  损伤50% {vl}", flush=True)

    run_arm("exp(lam=4)", lambda sd, d: build_exp(4.0, sd, d))
    for a in [float(x) for x in args.alphas.split(",")]:
        run_arm(f"power(a={a:g})", lambda sd, d, _a=a: build_power(_a, sd, d))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
    )

    print("")
    print("判：")
    cc = res["cells"]
    base = cc["exp(lam=4)"]
    s0 = base["mi"] > 0 and all(v["mi"] > 0 for v in cc.values())
    print(f"  S-0 两族在完整网络都能工作 => {'过' if s0 else '不过 => 作废'}")
    if not s0:
        return 0
    pw = {k: v for k, v in cc.items() if k.startswith("power")}
    best = max(pw, key=lambda k: pw[k]["ml"])
    s1 = pw[best]["ml"] > base["ml"]
    print(
        f"  S-1 无标度更抗损伤：最好 {best} 损伤 {pw[best]['ml']:.1f} vs exp {base['ml']:.1f} "
        f"=> {'过' if s1 else '不过'}"
    )
    seq = [pw[k]["ml"] for k in sorted(pw, key=lambda k: float(k.split("=")[1].rstrip(")")))]
    print(f"  S-2 剂量–反应：power 的损伤 n_viable 随 α = {[round(x, 1) for x in seq]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

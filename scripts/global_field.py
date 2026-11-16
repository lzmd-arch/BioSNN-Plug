"""全局场 g_field：测「刀锋」本身 —— 有多少个输入尺度能让网络落在工作区。

预注册：`research/sel/prereg-global-field-v1.md`

```
Φ(t) = κ · mean_i z_i(t)          细胞外场的粗粒化
输入 += g_field · Φ(t)
```

`n_viable` = 尺度网格里落在 [0.03, 0.20] 的点数。**它直接量化"离临界点有多少余量"。**
"""

from __future__ import annotations

# ruff: noqa: N806
import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.bio_all import build, make_task
from scripts.cortical_fold import N
from scripts.instrument import FIRING_WINDOW, scale_grid

OUT = _pathlib.Path("sweep_results/global_field")


@torch.no_grad()
def run_gf(
    w_in, pre, wrec, toks, dev, g_field: float, alive=None, mode: str = "fixed", fstar: float = 0.08
):
    """带全局场的 LIF。返回每存活神经元的平均发放。

    `mode="fixed"`：`cur += g_field·Φ`（v14 原样）
    `mode="homeo"`：`cur += −g0·(f_run/f*)·Φ` —— 稳态反馈，活动高就压
    """
    b, T = toks.shape
    n = w_in.shape[1]
    v = torch.zeros((b, n), device=dev)
    z = torch.zeros((b, n), device=dev)
    ref = torch.zeros((b, n), device=dev)
    f_run = fstar if mode == "homeo" else 0.0
    rate = 0.0
    for t in range(T):
        rec = torch.zeros_like(z)
        for j in range(pre.shape[1]):
            rec = rec + z[:, pre[:, j]] * wrec[:, j].unsqueeze(0)
        cur = w_in[toks[:, t]]
        phi = float(z.mean())  # 场是全局标量（粗粒化 LFP）
        if mode == "homeo":
            g_eff = g_field * (f_run / max(fstar, 1e-9))
            cur = cur - g_eff * phi
            f_run = 0.9 * f_run + 0.1 * phi
        elif g_field != 0.0:  # **不能写 > 0**：负的（抑制性）场要真的施加
            cur = cur + g_field * phi
        nv = 0.95 * v + cur + rec - z * 0.5
        fresh = ((nv - 0.5) / 0.5 > 0).float()
        nz = torch.where(ref > 0, torch.zeros_like(fresh), fresh)
        if alive is not None:
            nz = nz * alive
        ref = torch.clamp(ref + 2.0 * nz - 1.0, 0.0, 2.0)
        v, z = nv, nz
        rate += float(z.sum() / (alive.sum() * b)) if alive is not None else float(z.mean())
    return rate / T


def n_viable(w_in, pre, wrec, toks, dev, g_field, alive, scales, mode="fixed"):
    lo, hi = FIRING_WINDOW
    cnt = 0
    for s in scales:
        fr = run_gf(w_in * s, pre, wrec, toks, dev, g_field, alive, mode=mode)
        if lo <= fr <= hi:
            cnt += 1
    return cnt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=48)
    ap.add_argument("--n-seq", type=int, default=32)
    ap.add_argument("--gfs", default="0,0.5,1,2,4")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--mode", default="fixed", choices=("fixed", "homeo"))
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(x) for x in args.seeds.split(",")]
    scales = scale_grid(0.4, 12.0, 1.15)
    res: dict = {"scales": len(scales), "cells": {}}

    for gf in [float(x) for x in args.gfs.split(",")]:
        vi_int, vi_les = [], []
        for sd in seeds:
            toks, _ = make_task(args.n_seq, args.seq, 8, 10 + sd, dev)
            w_in, pre, wrec, _a, _d, _p = build(N, set(), sd, dev, 8)
            gen = torch.Generator().manual_seed(7000 + sd)
            alive = (torch.rand(N, generator=gen) >= 0.5).float().to(dev)
            vi_int.append(n_viable(w_in, pre, wrec, toks, dev, gf, None, scales, args.mode))
            vi_les.append(n_viable(w_in, pre, wrec, toks, dev, gf, alive, scales, args.mode))
            del w_in, pre, wrec, toks
            torch.cuda.empty_cache()
        res["cells"][f"{gf:.1f}"] = {
            "viable_intact": vi_int,
            "viable_lesion": vi_les,
            "mean_intact": sum(vi_int) / len(vi_int),
            "mean_lesion": sum(vi_les) / len(vi_les),
        }
        print(
            f"  [{args.mode}] g={gf:5.1f}  完整 n_viable {vi_int}  损伤50% {vi_les}",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
    )

    print("")
    print(f"判（尺度网格 {len(scales)} 个点）：")
    cc = res["cells"]
    base = cc.get("0.0")
    if base is None:
        print("  ✗ 缺 g_field=0 对照")
        return 1
    f0 = base["mean_intact"] > 0 and base["mean_lesion"] < base["mean_intact"]
    print(
        f"  F-0 g=0：完整 {base['mean_intact']:.1f} vs 损伤 {base['mean_lesion']:.1f} "
        f"=> {'过（复现 v8）' if f0 else '不过 => 作废'}"
    )
    if not f0:
        return 0
    best = max(cc, key=lambda k: cc[k]["mean_lesion"])
    ratio = cc[best]["mean_lesion"] / max(base["mean_lesion"], 1e-9)
    f1 = ratio >= 2.0
    print(
        f"  F-1 场扩大余量：最好 g={best} 的损伤 n_viable {cc[best]['mean_lesion']:.1f} "
        f"= g=0 的 {ratio:.2f} 倍（要 ≥2）=> {'过' if f1 else '不过'}"
    )
    seq = [cc[k]["mean_lesion"] for k in sorted(cc, key=float)]
    print(f"  F-2 剂量–反应：损伤 n_viable 随 g = {[round(x, 1) for x in seq]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

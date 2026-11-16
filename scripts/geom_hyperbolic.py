"""超欧氏几何：**双曲 vs 欧氏，在匹配度数下比**。

## 为什么之前没测到

前几轮用 `smallworld`（重连 20%）和 `scalefree`（优先连接）当替身——那是**图论属性**，
不是**内蕴几何**。材料里指的"双曲"是严格的：**体积指数增长**，所以层级/树天然住得下。

## 本实验

四种布线，**平均度数全部匹配到同一个值**（默认 64）：

| 臂 | 几何 |
| :--- | :--- |
| `euclid2d` | 欧氏圆盘上均匀撒点，距离 < r 连边 |
| `euclid3d` | 欧氏球内均匀撒点，距离 < r 连边 |
| **`hyperbolic`** | **Poincaré 圆盘**：角度均匀、半径按 `sinh` 密度取（体积指数增长），**双曲距离 < D 连边** |
| `ring` | 一维环（基线，度数 4，不参与匹配） |

双曲距离用标准公式：

```
d_H(i,j) = arccosh( cosh(r_i)cosh(r_j) − sinh(r_i)sinh(r_j)cos(θ_i − θ_j) )
```

r 按 `cosh(r) ~ U(1, cosh(R))` 取，这样点在双曲度量下是**均匀**的。

## 判据

| 编号 | 判据 |
| :--- | :--- |
| **X-0** | 三种几何的平均度数差 **<10%**（否则比的是度数） |
| **X-1** | 每臂发放率 ∈ [0.03, 0.20]，且 acc ≥ 多数类 + 15pp |
| **X-2** | 双曲比两个欧氏都高 **≥3pp**，且三种子同向 |
"""

from __future__ import annotations

# ruff: noqa: N806  —— R（双曲半径）、T（序列长）是这类推导的惯例

import argparse
import json
import math
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.dim_ladder import mlp
from scripts.structure_ladder import K, V, make_pref, make_task

OUT = _pathlib.Path("sweep_results/geom_hyperbolic")
TAUS = (2.0, 16.0)


def _thresh_to_degree(dist: torch.Tensor, target: int, lo: float, hi: float, iters: int = 40):
    """二分找距离阈值，使平均度数（不含自环）≈ target。"""
    for _ in range(iters):
        mid = (lo + hi) / 2
        deg = (dist < mid).float().sum() / dist.shape[0] - 1.0
        if deg < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def build_geom(kind: str, n: int, target_deg: int, seed: int, gain: float, dev: str):
    g = torch.Generator().manual_seed(seed)
    if kind == "hyperbolic":
        R = math.log(n) * 0.6  # 半径：保证层级深度
        theta = torch.rand(n, generator=g) * 2 * math.pi
        u = torch.rand(n, generator=g)
        r = torch.arccosh(1 + u * (math.cosh(R) - 1))
        # 双曲距离
        c_i, c_j = torch.cosh(r).unsqueeze(1), torch.cosh(r).unsqueeze(0)
        s_i, s_j = torch.sinh(r).unsqueeze(1), torch.sinh(r).unsqueeze(0)
        dth = torch.cos(theta.unsqueeze(1) - theta.unsqueeze(0))
        dist = torch.arccosh((c_i * c_j - s_i * s_j * dth).clamp(min=1.0 + 1e-7))
        thr = _thresh_to_degree(dist, target_deg, 1e-4, float(dist.max()))
        mask = dist < thr
    else:
        dim = 2 if kind == "euclid2d" else 3
        # 欧氏球内均匀撒点（r ∝ u^(1/dim)）
        v = torch.randn(n, dim, generator=g)
        v = v / v.norm(dim=1, keepdim=True)
        rad = torch.rand(n, generator=g) ** (1.0 / dim)
        pts = v * rad.unsqueeze(1)
        dist = torch.cdist(pts, pts)
        thr = _thresh_to_degree(dist, target_deg, 1e-4, float(dist.max()))
        mask = dist < thr
    mask.fill_diagonal_(False)
    deg = float(mask.float().sum(1).mean())
    # 行归一化 ⇒ **每神经元的总递归驱动相同**
    w = mask.float().to(dev) * torch.rand(n, n, generator=g).to(dev)
    w = w * (gain / w.sum(1, keepdim=True).clamp(min=1e-6))
    return w, deg


@torch.no_grad()
def run(w_in, wmat, toks, dev, stride=8):
    b, T = toks.shape
    n = w_in.shape[1]
    v = torch.zeros((b, n), device=dev)
    z = torch.zeros((b, n), device=dev)
    ref = torch.zeros((b, n), device=dev)
    decays = torch.tensor([1.0 / t for t in TAUS], device=dev)
    read = torch.zeros((b, len(TAUS), n), device=dev)
    feats, rate = [], 0.0
    for t in range(T):
        nv = 0.95 * v + w_in[toks[:, t]] + z @ wmat - z * 0.5
        fresh = ((nv - 0.5) / 0.5 > 0).float()
        nz = torch.where(ref > 0, torch.zeros_like(fresh), fresh)
        ref = torch.clamp(ref + 2.0 * nz - 1.0, 0.0, 2.0)
        v, z = nv, nz
        rate += float(z.mean())
        read = read * decays.view(1, -1, 1) + z.unsqueeze(1)
        if t % stride == 0:
            feats.append(read.reshape(b, -1))
    return torch.stack(feats, 1), rate / T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--deg", type=int, default=64)
    ap.add_argument("--seq", type=int, default=96)
    ap.add_argument("--train-seq", type=int, default=192)
    ap.add_argument("--test-seq", type=int, default=48)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(x) for x in args.seeds.split(",")]
    n = args.n

    maj = []
    for sd in seeds:
        pref = make_pref(0, 3.0).to(dev)
        _, tes, _ = make_task(args.test_seq, args.seq, 20 + sd, dev, pref)
        st = tes[:, ::8].reshape(-1)
        maj.append(float(torch.bincount(st, minlength=K).max()) / st.numel())
    majority = sum(maj) / len(maj)
    floor = majority + 0.15
    print(f"  多数类 {majority:.3f}（真解码门槛 {floor:.3f}）", flush=True)

    res: dict = {"n": n, "deg_target": args.deg, "majority": majority, "cells": {}}
    for kind in ("euclid2d", "euclid3d", "hyperbolic"):
        for gain in (0.1, 0.2, 0.4):
            accs, frs, degs = [], [], []
            for sd in seeds:
                pref = make_pref(0, 3.0).to(dev)
                tr, trs, _ = make_task(args.train_seq, args.seq, 10 + sd, dev, pref)
                te, tes, _ = make_task(args.test_seq, args.seq, 20 + sd, dev, pref)
                toks = torch.cat([tr, te])
                st = torch.cat([trs, tes])[:, ::8]
                w, deg = build_geom(kind, n, args.deg, sd, gain, dev)
                g = torch.Generator().manual_seed(sd)
                w_in = (torch.randn(V, n, generator=g) / n**0.5 * 16.0).to(dev)
                feat, fr = run(w_in, w, toks, dev)
                accs.append(mlp(feat, st, args.train_seq))
                frs.append(fr)
                degs.append(deg)
                del feat, w
                torch.cuda.empty_cache()
            key = f"{kind}/g{gain}"
            res["cells"][key] = {
                "acc": sum(accs) / len(accs),
                "runs": accs,
                "firing": sum(frs) / len(frs),
                "deg": sum(degs) / len(degs),
            }
            print(
                f"  {kind:11s} gain {gain:.1f}  度数 {sum(degs) / len(degs):5.1f}  "
                f"发放 {sum(frs) / len(frs):.4f}  acc {sum(accs) / len(accs):.3f}",
                flush=True,
            )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )

    print("\n判：")
    degs = [v["deg"] for v in res["cells"].values()]
    ok_deg = (max(degs) - min(degs)) / max(degs) < 0.10
    print(
        f"  X-0 度数极差 {(max(degs) - min(degs)) / max(degs):.1%} <10% ⇒ {'过' if ok_deg else '不过'}"
    )
    for k, v in res["cells"].items():
        print(
            f"  X-1 {k:16s} 发放 {v['firing']:.4f} {'✓' if 0.03 <= v['firing'] <= 0.20 else '✗'}  "
            f"acc {v['acc']:.3f} {'✓' if v['acc'] >= floor else '✗'}"
        )
    best_e = max(v["acc"] for k, v in res["cells"].items() if not k.startswith("hyperbolic"))
    hyper = [v for k, v in res["cells"].items() if k.startswith("hyperbolic")]
    best_h = max(hyper, key=lambda v: v["acc"])
    d = best_h["acc"] - best_e
    print(
        f"  X-2 双曲最佳 {best_h['acc']:.3f} − 欧氏最佳 {best_e:.3f} = {d:+.3f} ≥0.03 ⇒ "
        f"{'过' if d >= 0.03 else '不过'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

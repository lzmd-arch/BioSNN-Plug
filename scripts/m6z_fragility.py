"""M6z：**局部场能不能把"脆性"带回来** —— 这可能是"自维持系统"的关键一跳。

## 动机

今晚已经量到三件事，全都指向同一个方向：

| 测量 | 结果 |
| :--- | :--- |
| M5b 切掉 50% 神经元 | 解码**变好**（0.34 → 0.69–0.91） |
| M8 给子代加噪声 | 解码**变好**（三个种子全部高于亲代） |
| M6v 状态流形 | **PR 2.49，低维** |

**一个自维持的系统应该是脆的**：破坏要有系统性代价。这里没有。

而 M6w 已经证明：把全局场换成局部场，PR 从 2.52 抬到 **61.32**。
**那么脆性会不会一起回来？**

## 判据

- **Z-0** 两档发放都在窗内；
- **Z-1 局部场下有脆性** ★：`radius=2` 时随机切 50% 后解码 **低于** 基线（≥2/3 种子）；
- **Z-2 全局场下没有** ：`radius=32` 时切后 **不低于** 基线（对照，≥2/3 种子）。
"""

from __future__ import annotations

# ruff: noqa: N806
import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.hist_families import make_family_hist, spatial_bins
from scripts.instrument import majority_baseline, probe
from scripts.m6_pools import N, build_pools, run_spont

OUT = _pathlib.Path("sweep_results/m6z_fragility")
K = 4
T_HIST, T_SPONT = 1024, 2048
FAMS = list(range(8))


@torch.no_grad()
def decode(net, alive, seed: int, radius: int, per_fam: int, dev: str):
    feats, labels = [], []
    for fam in FAMS:
        toks = make_family_hist(fam, seed * 41 + fam * 11 + 3, T_HIST, per_fam, dev)
        r = run_spont(
            seed,
            0.5,
            0,
            0.7,
            0.02,
            dev,
            K,
            0.0,
            0.0,
            -1.0,
            hist_in=toks,
            t_spont=T_SPONT,
            net=net,
            alive=alive,
            field_radius=radius,
            **field_args(radius),
        )[0]
        feats.append(spatial_bins(r).cpu())
        labels.append(torch.full((r.shape[0],), fam))
    X, Y = torch.cat(feats), torch.cat(labels)
    perm = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(5 + seed))
    X, Y = X[perm], Y[perm]
    cut = int(X.shape[0] * 0.75)
    acc, _ = probe(X, Y, cut, len(FAMS))
    return acc, majority_baseline(Y[:cut], Y[cut:], len(FAMS))


def field_args(radius: int) -> dict:
    """`field_radius` 与 `g_field/g_local` 的对应关系。

    **2026-11-16 踩过**：加了 `g_local` 之后全局项永远施加，只传 `field_radius`
    不会打开局部场 —— M6x/M6z 第一版因此全部跑成了全局场（PR 2.34 与 radius=32 一致）。
    """
    if radius >= 32:
        return {"g_field": 1.0, "g_local": 0.0}
    return {"g_field": 0.0, "g_local": 1.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--radii", default="2,32")
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--per-fam", type=int, default=16)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": {}, "frac": args.frac}
    for rad in [int(x) for x in args.radii.split(",")]:
        rows = []
        for sd in range(args.seeds):
            net = build_pools(sd, dev, K, 0.0)
            ones = torch.ones(N, device=dev)
            base, floor = decode(net, ones, sd, rad, args.per_fam, dev)
            g = torch.Generator().manual_seed(sd + 77)
            mask = (torch.randperm(N, generator=g).to(dev) >= int(args.frac * N)).float()
            post, _ = decode(net, mask, sd, rad, args.per_fam, dev)
            rows.append({"seed": sd, "base": base, "post": post, "floor": floor})
            print(
                f"  radius={rad:2d} seed {sd}: 基线 {base:.3f} → 切后 {post:.3f} "
                f"（{post - base:+.3f}）",
                flush=True,
            )
        key = f"r={rad}"
        res["cells"][key] = {
            "radius": rad,
            "base": sum(r["base"] for r in rows) / len(rows),
            "post": sum(r["post"] for r in rows) / len(rows),
            "hurt": sum(1 for r in rows if r["post"] < r["base"]),
            "n": args.seeds,
            "rows": rows,
        }
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )

    print("")
    print("判：")
    cc = res["cells"]
    if "r=2" in cc:
        v = cc["r=2"]
        print(
            f"  Z-1 局部场下有脆性（切后 < 基线，≥2/3）：{v['hurt']}/{v['n']} "
            f"（{v['base']:.3f} → {v['post']:.3f}）=> {'过' if v['hurt'] >= 2 else '不过'}"
        )
    if "r=32" in cc:
        v = cc["r=32"]
        print(
            f"  Z-2 全局场下没有（切后 ≥ 基线，≥2/3）：{v['n'] - v['hurt']}/{v['n']} "
            f"（{v['base']:.3f} → {v['post']:.3f}）=> {'过' if v['n'] - v['hurt'] >= 2 else '不过'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

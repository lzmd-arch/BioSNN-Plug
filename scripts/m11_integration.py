"""M11（Round H）：**整合–分离平衡**。

判据在 `research/sel/night-2026-11-16.md`（含修订 6 —— 口径与切断对照都定死了）。

- 整合 = 两半球逐时刻平均发放的时间相关（绝对值）；
- 分离 = 半球内部的空间变异系数；
- **切断对照**：把跨半球的突触目标换成同半球内的随机目标（真切断，不改行和）；
- 场强阶梯 `g_field ∈ {1.0, 0.25}`（`run_spont` 的参数）。
"""

from __future__ import annotations

# ruff: noqa: N806
import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.hist_families import make_family_hist
from scripts.m6_pools import N, build_pools, run_spont

OUT = _pathlib.Path("sweep_results/m11_integration")
K, SIDE = 4, 32
T_HIST, T_SPONT, REC = 1024, 1024, 256


def halves(dev: str) -> torch.Tensor:
    return ((torch.arange(N) % SIDE) >= 16).long().to(dev)


def cut_cross(pre: torch.Tensor, dev: str, seed: int) -> torch.Tensor:
    """跨半球的目标换成同半球内的随机目标。"""
    g = torch.Generator().manual_seed(seed)
    h = halves(dev)
    idx = torch.arange(N, device=dev)
    new = pre.clone()
    for i in range(N):
        m = h[pre[i]] != h[i]
        if bool(m.any()):
            cand = idx[h == h[i]]
            pick = cand[torch.randint(0, cand.numel(), (int(m.sum()),), generator=g).to(dev)]
            new[i, m] = pick
    return new


@torch.no_grad()
def metrics(net, seed: int, dev: str, g_field: float):
    toks = make_family_hist(0, seed * 61 + 8, T_HIST, 32, dev)
    _, snaps, _ = run_spont(
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
        record_len=REC,
        net=net,
        g_field=g_field,
    )
    Z = snaps.mean(0)  # (N, REC)
    h = halves(dev).cpu()
    a = Z[h == 0].mean(0)
    b = Z[h == 1].mean(0)
    a = a - a.mean()
    b = b - b.mean()
    integ = float((a * b).sum() / (a.norm() * b.norm() + 1e-12))
    segs = []
    for k in (0, 1):
        H = Z[h == k]
        segs.append(float((H.std(0) / (H.mean(0) + 1e-9)).mean()))
    return abs(integ), sum(segs) / 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--gfields", default="1.0,0.25")
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": {}}
    for gf in [float(x) for x in args.gfields.split(",")]:
        rows = []
        for sd in range(args.seeds):
            w_in, pre, wrec, pid = build_pools(sd, dev, K, 0.0)
            i_full, s_full = metrics((w_in, pre, wrec, pid), sd, dev, gf)
            pre_cut = cut_cross(pre, dev, sd)
            i_cut, s_cut = metrics((w_in, pre_cut, wrec, pid), sd, dev, gf)
            rows.append(
                {
                    "seed": sd,
                    "integ_full": i_full,
                    "integ_cut": i_cut,
                    "seg_full": s_full,
                    "seg_cut": s_cut,
                }
            )
            print(
                f"  g_field={gf:g} seed {sd}: 整合 完整 {i_full:.3f} / 切断 {i_cut:.3f} | "
                f"分离 完整 {s_full:.3f} / 切断 {s_cut:.3f}",
                flush=True,
            )
        key = f"g_field={gf:g}"
        res["cells"][key] = {
            "rows": rows,
            "win_integ": sum(1 for r in rows if r["integ_full"] > r["integ_cut"]),
            "win_seg": sum(1 for r in rows if r["seg_full"] >= 0.8 * r["seg_cut"]),
            "n": args.seeds,
        }
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )

    print("")
    print("判：")
    cc = res["cells"]
    ok = []
    for k, v in cc.items():
        both = v["win_integ"] >= max(5, 0.83 * v["n"]) and v["win_seg"] >= max(5, 0.83 * v["n"])
        print(
            f"  {k}: 整合更高 {v['win_integ']}/{v['n']}、分离保住 {v['win_seg']}/{v['n']} "
            f"=> {'过' if both else '不过'}"
        )
        if both:
            ok.append(k)
    print(f"  M11 存在某档同时满足：{'过 ' + str(ok) if ok else '不过'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

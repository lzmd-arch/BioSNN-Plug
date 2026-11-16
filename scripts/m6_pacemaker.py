"""M6 内源驱动：起搏神经元 + 零输入下的自发行为。

预注册：`research/sel/prereg-m6-pacemaker-v1.md`

起搏神经元真实存在（丘脑 relay、多巴胺能、下橄榄核）—— 它们不需要突触输入就能放电。
给一部分神经元持续去极化电流，然后**把输入全关掉**，看系统还能不能产生活动，
以及那个活动是不是它自己的历史。
"""

from __future__ import annotations

# ruff: noqa: N803
import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.m7_history import (
    A_HIGH,
    A_LOW,
    T_HIST,
    N,
    build_net,
    run_seg,
)

OUT = _pathlib.Path("sweep_results/m6_pacemaker")
T_SPONT = 256


def make_pace(p: float, seed: int, dev: str):
    g = torch.Generator().manual_seed(seed + 31337)
    return (torch.rand(N, generator=g) < p).float().to(dev)


@torch.no_grad()
def spontaneous(
    seed: int,
    alpha_hist: float,
    hist_seed: int,
    scale: float,
    p: float,
    dev: str,
    gfield: float = 1.0,
):
    """跑历史 → **输入全零** → 记录自发模式（每神经元平均发放）。"""
    w_in, pre, wrec = build_net(seed, dev)
    w_in = w_in * scale
    alive = torch.ones(N, device=dev)
    gain = torch.ones(N, device=dev)
    pace = make_pace(p, seed, dev) * 0.5  # I_pace = 0.5

    st = (
        torch.zeros((32, N), device=dev),
        torch.zeros((32, N), device=dev),
        torch.zeros((32, N), device=dev),
    )
    hist, _ = make_task_hist(T_HIST, hist_seed, dev, alpha_hist)
    st, _, _ = run_seg(w_in, pre, wrec, hist, dev, st, alive, gain, T_HIST)
    # **零输入**：tokens 全为一个固定值，但 w_in 贡献被 pace 之外的通道抵消不了 ——
    # 所以这里直接把输入项关掉，只留起搏电流
    zeros = torch.zeros((32, T_SPONT), dtype=torch.long, device=dev)
    v, z, ref = st
    rates_early = torch.zeros(N, device=dev)
    rates_late = torch.zeros(N, device=dev)
    zmat = []
    for step in range(T_SPONT):
        rec = torch.zeros_like(z)
        for j in range(pre.shape[1]):
            rec = rec + z[:, pre[:, j]] * wrec[:, j].unsqueeze(0)
        phi = float(z.mean())
        cur = pace.unsqueeze(0) - gfield * phi  # **没有 w_in[tok]**
        nv = 0.95 * v + cur + rec - z * 0.5
        fresh = ((nv - 0.5) / 0.5 > 0).float()
        nz = torch.where(ref > 0, torch.zeros_like(fresh), fresh) * alive
        ref = torch.clamp(ref + 2.0 * nz - 1.0, 0.0, 2.0)
        v, z = nv, nz
        if step < 64:
            rates_early += z.mean(0)
        elif step >= T_SPONT - 64:
            rates_late += z.mean(0)
            zmat.append(z.mean(0))
    del w_in, pre, wrec, hist, zeros
    torch.cuda.empty_cache()
    return rates_early / 64, rates_late / 64, torch.stack(zmat, 1)


def make_task_hist(T: int, seed: int, dev: str, alpha: float):
    from scripts.bio_all import make_task

    return make_task(32, T, 8, seed, dev, alpha)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--ps", default="0,0.02")
    ap.add_argument("--gfs", default="0,0.25,0.5,1.0")
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda"
    res: dict = {"cells": {}}

    for gf in [float(x) for x in args.gfs.split(",")]:
        for p in [float(x) for x in args.ps.split(",")]:
            rec = {"early": ([], [], []), "late": ([], [], [])}
            zcs, zcb, zc2 = [], [], []
            frs, sds = [], []
            for sd in range(args.seeds):
                ea, la, za = spontaneous(sd, A_LOW, 100 + sd, 0.7, p, dev, gf)
                eb, lb, zb = spontaneous(sd, A_HIGH, 100 + sd, 0.7, p, dev, gf)
                ea2, la2, za2 = spontaneous(sd, A_LOW, 900 + sd, 0.7, p, dev, gf)
                for w, x, y, z2 in (("early", ea, eb, ea2), ("late", la, lb, la2)):
                    rec[w][0].append(x)
                    rec[w][1].append(y)
                    rec[w][2].append(z2)
                frs.append(float(la.mean()))
                sds.append(float(la.std()))
                # **减掉全局公共分量**后剩下的才是"图案"（不是一起涨落）
                zc = za - za.mean(0, keepdim=True)
                zcs.append(zc)
                zcb.append(zb - zb.mean(0, keepdim=True))
                zc2.append(za2 - za2.mean(0, keepdim=True))

            def _ratio(w, _rec=rec):
                xs, ys, zs = _rec[w]
                rh = [
                    float((xs[i] - ys[i]).norm() / (0.5 * (xs[i].norm() + ys[i].norm()) + 1e-12))
                    for i in range(len(xs))
                ]
                rn = [
                    float((xs[i] - zs[i]).norm() / (0.5 * (xs[i].norm() + zs[i].norm()) + 1e-12))
                    for i in range(len(xs))
                ]
                return (sum(rh) / len(rh)) / max(sum(rn) / len(rn), 1e-12), sum(
                    1 for a, b in zip(rh, rn, strict=False) if a > b
                )

            # 结构：去掉公共分量后的参与比（>1.5 说明不止一个模式）
            zall = torch.stack(zcs)  # (n_seed, N, T)
            zcall = zall.reshape(-1, zall.shape[-1])
            zcall = zcall - zcall.mean(0, keepdim=True)
            ev = (
                torch.linalg.eigvalsh(zcall @ zcall.T / zcall.shape[0]).clamp(min=0)
                if zcall.shape[0] < zcall.shape[1]
                else torch.linalg.eigvalsh(zcall.T @ zcall / zcall.shape[0]).clamp(min=0)
            )
            pr_res = float((ev.sum() ** 2) / max(float((ev**2).sum()), 1e-12))
            # 残余图案的历史/对照比
            rh2 = [
                float((zcs[i] - zcb[i]).norm() / (0.5 * (zcs[i].norm() + zcb[i].norm()) + 1e-12))
                for i in range(len(zcs))
            ]
            rn2 = [
                float((zcs[i] - zc2[i]).norm() / (0.5 * (zcs[i].norm() + zc2[i].norm()) + 1e-12))
                for i in range(len(zcs))
            ]
            r_pat = (sum(rh2) / len(rh2)) / max(sum(rn2) / len(rn2), 1e-12)
            w_pat = sum(1 for a, b in zip(rh2, rn2, strict=False) if a > b)

            r_e, w_e = _ratio("early")
            r_l, w_l = _ratio("late")
            mf = sum(frs) / len(frs)
            res["cells"][f"gf={gf:g}/p={p:g}"] = {
                "firing_late": mf,
                "spatial_sd": sum(sds) / len(sds),
                "ratio_early": r_e,
                "ratio_late": r_l,
                "wins_early": w_e,
                "wins_late": w_l,
                "n": args.seeds,
                "pr_residual": pr_res,
                "ratio_pattern": r_pat,
                "wins_pattern": w_pat,
            }
            print(
                f"  gf={gf:4.2f} p={p:4.2f}  后期发放 {mf:.4f}  空间SD {sum(sds) / len(sds):.4f}  "
                f"前64 {r_e:.2f}x({w_e})  后64 {r_l:.2f}x({w_l})  "
                f"残余PR {pr_res:.2f}  图案比 {r_pat:.2f}x({w_pat})",
                flush=True,
            )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
    )

    print("")
    print("判：")
    cc = res["cells"]
    ok1 = [k for k, v in cc.items() if 0.01 <= v["firing_late"] <= 0.20]
    print(f"  M6-1 后期自发活动在 [0.01,0.20]：{ok1 or '无'} => {'过' if ok1 else '不过 => 作废'}")
    if not ok1:
        return 0
    best = max(ok1, key=lambda k: cc[k]["ratio_late"])
    v = cc[best]
    m2 = v["spatial_sd"] > 0.5 * v["firing_late"]
    print(
        f"  M6-2 结构化（SD > 0.5×均值）：{v['spatial_sd']:.4f} vs {0.5 * v['firing_late']:.4f}"
        f" => {'过' if m2 else '不过'}"
    )
    m3 = v["ratio_late"] > 1.5 and v["wins_late"] == v["n"]
    print(
        f"  M6-3 带自己的历史（**后 64 步**）：{best} 比 {v['ratio_late']:.2f}x "
        f"{v['wins_late']}/{v['n']} => {'过' if m3 else '不过'}"
    )
    return 0
    best = max(ok1, key=lambda k: cc[k]["ratio"])
    v = cc[best]
    m2 = v["spatial_sd"] > 0.5 * v["firing"]
    print(
        f"  M6-2 结构化（SD > 0.5×均值）：{v['spatial_sd']:.4f} vs {0.5 * v['firing']:.4f} "
        f"=> {'过' if m2 else '不过'}"
    )
    m3 = v["ratio"] > 1.5 and v["wins"] == v["n"]
    print(
        f"  M6-3 带自己的历史：{best} 比 {v['ratio']:.2f}x {v['wins']}/{v['n']} => {'过' if m3 else '不过'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

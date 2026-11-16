"""M6w：**把全局抑制换成局部抑制** —— 直接去造高维状态。

## 诊断在前

M6v 量到状态流形的内在维度只有 **PR ≈ 2.45**（128 条历史、`g_inh=0`）。
瓶颈明确：抑制是**一个全局标量** `−G_FIELD·φ`，对每个神经元完全一样 ⇒
整个网络只能有一个活动水平 ⇒ **状态必然是低维的**。

## 干预

`field_radius < 32` 时，每个神经元被**自己邻域**的平均活动压住，而不是全网络的平均。
半径越小，空间上不同区域越能维持不同的活动水平 ⇒ 状态空间才有维度。

判据：
- **W-0** 发放在窗内；
- **W-1 高维** ★：`radius=2` 的 PR **> 4**；
- **W-2 剂量–反应** ★：PR 随半径单调下降（Spearman = −1）。
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
from scripts.m6_pools import build_pools, run_spont
from scripts.m6v_dimension import pr_of, two_nn

OUT = _pathlib.Path("sweep_results/m6w_local_field")
K = 4
T_HIST, T_SPONT = 1024, 2048


def spearman(xs, ys) -> float:
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for p, i in enumerate(o):
            r[i] = float(p)
        return r

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=False))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / max(vx * vy, 1e-12)


@torch.no_grad()
def cell(seed: int, radius: int, n_fam: int, per_fam: int, dev: str):
    net = build_pools(seed, dev, K, 0.0)
    states, frs = [], []
    for fam in range(n_fam):
        toks = make_family_hist(fam, seed * 307 + fam * 31 + 11, T_HIST, per_fam, dev)
        r, _, _ = run_spont(
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
            field_radius=radius,
        )
        states.append(r.cpu())
        frs.append(float(r.mean()))
    S = torch.cat(states)
    return pr_of(S), two_nn(S), sum(frs) / len(frs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--radii", default="2,4,8,16,32")
    ap.add_argument("--n-fam", type=int, default=8)
    ap.add_argument("--per-fam", type=int, default=16)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": {}}
    for rad in [int(x) for x in args.radii.split(",")]:
        prs, tns, frs = [], [], []
        for sd in range(args.seeds):
            pr, tn, fr = cell(sd, rad, args.n_fam, args.per_fam, dev)
            prs.append(pr)
            tns.append(tn)
            frs.append(fr)
            print(
                f"  radius={rad:2d} seed {sd}: PR={pr:6.2f}  two-NN={tn:6.2f}  发放 {fr:.4f}",
                flush=True,
            )
        m = lambda v: sum(v) / len(v)  # noqa: E731
        res["cells"][f"r={rad}"] = {
            "radius": rad,
            "pr": prs,
            "pr_mean": m(prs),
            "two_nn_mean": m(tns),
            "firing": m(frs),
            "seeds": args.seeds,
        }
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )

    print("")
    print("判：")
    cc = res["cells"]
    ks = sorted(cc, key=lambda k: cc[k]["radius"])
    print("  PR 随半径：" + "  ".join(f"{cc[k]['radius']}→{cc[k]['pr_mean']:.2f}" for k in ks))
    ok = [k for k in ks if 0.01 <= cc[k]["firing"] <= 0.20]
    print(f"  W-0 发放在窗内：{len(ok)}/{len(ks)} 档 => {'过' if ok else '不过 => 作废'}")
    if not ok:
        return 0
    k0 = ks[0]
    w1 = cc[k0]["pr_mean"] > 4.0
    print(
        f"  W-1 高维（radius={cc[k0]['radius']} 的 PR > 4）：{cc[k0]['pr_mean']:.2f} "
        f"=> {'过' if w1 else '不过'}"
    )
    rho = spearman([cc[k]["radius"] for k in ks], [cc[k]["pr_mean"] for k in ks])
    print(
        f"  W-2 剂量–反应（PR 随半径单调下降）：Spearman={rho:+.2f} => "
        f"{'过' if rho <= -0.9 else '不过'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

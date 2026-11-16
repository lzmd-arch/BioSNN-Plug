"""M6s：**数吸引子的形态** —— "个体坐进不同状态"在这个动力学族里到底存不存在。

预注册：`research/sel/prereg-m6s-attractors-v1.md`（含**开跑前**修订 1）

## 为什么不再扫参数

冻结的增益只能挪动唯一那个不动点（仿射映射 `(I−A)v = b`）。M6r 的 1.68× 到底是
"个体落进不同状态"还是"暂态被拉长"，`rel` 距离这个口径**分不出来**。

能分开的是直接数：同一网络、零输入，从大量**随机初值**出发跑到收敛，
然后看**终态两两距离的分布形态**（不是簇数 —— 连续统的簇数恒等于初值数）。

| 形态 | 判据 | 含义 |
| :--- | :--- | :--- |
| 单一吸引子 | `far < 0.1` | 殊途同归 —— 个体没有选择 |
| **连续吸引子** | `near < 0.1` 且 `far >= 0.5` | 终态随初值连续变化 —— 个体状态存在 |
| 离散多吸引子 | `near >= 0.1` 且 `far >= 0.1` | 双峰 —— 少数几个盆 |
"""

from __future__ import annotations

# ruff: noqa: N806
import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.m6_pools import N, run_spont

OUT = _pathlib.Path("sweep_results/m6s_attractors")
K = 4
T_END = 1024
T_PREV = 960  # 与 T_END 相邻的 64 步窗
T_LONG = 2048  # 持久性检验


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / (0.5 * (a.norm() + b.norm()) + 1e-12))


def classify(near: float, far: float) -> str:
    if far < 0.1:
        return "单一吸引子"
    if near < 0.1:
        return "连续吸引子"
    return "离散多吸引子"


@torch.no_grad()
def run_cfg(seed: int, g_inh: float, n_init: int, sigma: float, dev: str) -> dict:
    g = torch.Generator().manual_seed(seed + 7717)
    ends, prevs, longs = [], [], []
    done = 0
    while done < n_init:
        b = min(32, n_init - done)
        v0 = (torch.randn(b, N, generator=g) * sigma).to(dev)
        args = (seed, 0.5, 100 + seed, 0.7, 0.02, dev, K, 0.0, g_inh, -1.0)
        ends.append(run_spont(*args, t_hist=0, t_spont=T_END, v0=v0)[0])
        prevs.append(run_spont(*args, t_hist=0, t_spont=T_PREV, v0=v0)[0])
        longs.append(run_spont(*args, t_hist=0, t_spont=T_LONG, v0=v0)[0])
        done += b
    F = torch.cat(ends)
    P = torch.cat(prevs)
    L = torch.cat(longs)
    n = F.shape[0]
    conv = torch.tensor([rel(P[i], F[i]) for i in range(n)])
    pers = torch.tensor([rel(F[i], L[i]) for i in range(n)])
    keep = conv < 0.05
    idx = keep.nonzero().squeeze(1).tolist()
    ds = [rel(F[i], F[j]) for a, i in enumerate(idx) for j in idx[a + 1 :]]
    near = sum(1 for d in ds if d < 0.05) / max(len(ds), 1)
    far = sum(1 for d in ds if d > 0.5) / max(len(ds), 1)
    srt = sorted(ds)
    med = sorted(pers.tolist())[len(pers) // 2]
    return {
        "g_inh": g_inh,
        "sigma": sigma,
        "n_init": n,
        "firing": float(F.mean()),
        "converged": int(keep.sum()),
        "near_frac": near,
        "far_frac": far,
        "form": classify(near, far),
        "persist_median": med,
        "pair_min": srt[0] if srt else float("nan"),
        "pair_med": srt[len(srt) // 2] if srt else float("nan"),
        "pair_max": srt[-1] if srt else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--ginhs", default="0,1,2,4")
    ap.add_argument("--n-init", type=int, default=64)
    ap.add_argument("--sigmas", default="1.0,4.0")
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": {}, "n_init": args.n_init, "K": K}
    for sigma in [float(x) for x in args.sigmas.split(",")]:
        for g_inh in [float(x) for x in args.ginhs.split(",")]:
            rows = [run_cfg(sd, g_inh, args.n_init, sigma, dev) for sd in range(args.seeds)]
            key = f"g={g_inh:g}/sigma={sigma:g}"
            agg = {
                k: sum(r[k] for r in rows) / len(rows)
                for k in rows[0]
                if isinstance(rows[0][k], (int, float))
            }
            agg["sigma"] = sigma
            agg["rows"] = rows
            agg["form"] = rows[0]["form"] if len({r["form"] for r in rows}) == 1 else "混合"
            res["cells"][key] = agg
            print(
                f"  {key:16s} 发放 {agg['firing']:.4f} 收敛 {agg['converged']:.0f}/{agg['n_init']}"
                f"  近 {agg['near_frac']:.2f} 远 {agg['far_frac']:.2f} **{agg['form']}**"
                f"  持久 {agg['persist_median']:.4f}  配对距离 "
                f"{agg['pair_min']:.3f}/{agg['pair_med']:.3f}/{agg['pair_max']:.3f}",
                flush=True,
            )
            (args.out / "result.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
            )

    print("")
    print("判：")
    cc = res["cells"]
    ok = [k for k, v in cc.items() if 0.01 <= v["firing"] <= 0.20]
    print(f"  S-0 发放在窗内：{len(ok)}/{len(cc)} 格 => {'过' if ok else '不过 => 作废'}")
    if not ok:
        return 0
    conv = [k for k in ok if cc[k]["converged"] / cc[k]["n_init"] >= 0.5]
    print(f"  S-1 半数以上收敛：{len(conv)}/{len(ok)} 格 => {'过' if conv else '不过'}")
    if not conv:
        print("      => 不收敛就谈不上吸引子；这条本身是结论。")
        return 0
    pers = [k for k in conv if cc[k]["persist_median"] < 0.05]
    print(
        f"  P-1 持久（1024 vs 2048 中位 <0.05）：{len(pers)}/{len(conv)} 格 "
        f"=> {'过' if pers else '不过'}"
    )
    forms: dict = {}
    for k in conv:
        forms.setdefault(cc[k]["form"], []).append(k)
    for f, ks in forms.items():
        print(f"  S-2' {f}：{len(ks)} 格 —— {ks}")
    verdict = (
        "**存在'个体坐进不同状态'**"
        if any(f != "单一吸引子" for f in forms)
        else "**只有单一吸引子：个体选择在这个动力学族里数学上不存在**"
    )
    print(f"      => {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

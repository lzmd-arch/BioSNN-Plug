"""M6u（Round B）：**盆地指派** —— 历史把个体送进哪个盆。

判据在 `research/sel/night-2026-11-16.md`（跑之前）。

M6s 已证 `g_inh ≥ 1` 时终态是**离散多吸引子**（近配对 0.11–0.15、远 0.85–0.89，粗估 6–9 个盆）。
那么正确的问法就不是连续的 `rel` 距离，而是**离散的盆地下标**：
历史族能不能决定个体落进哪个盆？

这比 `rel` 干净 —— 它不问"状态差多少"，只问"**有没有被送到不同的地方**"。
"""

from __future__ import annotations

# ruff: noqa: N806
import argparse
import json
import math
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.hist_families import make_family_hist
from scripts.m6_pools import rel, run_spont

OUT = _pathlib.Path("sweep_results/m6u_basins")
K = 4
T_HIST = 1024
T_SPONT = 2048
CLUSTER_TOL = 0.05


def basin_ids(states: torch.Tensor) -> list[int]:
    """单链接聚类：与任一已开簇的代表距离 < `CLUSTER_TOL` 就并入。"""
    n = states.shape[0]
    reps: list[int] = []
    out = [-1] * n
    for i in range(n):
        hit = -1
        for c, r in enumerate(reps):
            if rel(states[i], states[r]) < CLUSTER_TOL:
                hit = c
                break
        if hit < 0:
            reps.append(i)
            hit = len(reps) - 1
        out[i] = hit
    return out


def mutual_info(fams: list[int], basins: list[int]) -> float:
    n = len(fams)
    pf: dict = {}
    pb: dict = {}
    pj: dict = {}
    for f, b in zip(fams, basins, strict=False):
        pf[f] = pf.get(f, 0) + 1
        pb[b] = pb.get(b, 0) + 1
        pj[(f, b)] = pj.get((f, b), 0) + 1
    mi = 0.0
    for (f, b), c in pj.items():
        p = c / n
        mi += p * math.log(p / ((pf[f] / n) * (pb[b] / n)))
    return mi


@torch.no_grad()
def run_cell(seed: int, g_inh: float, per_fam: int, n_fam: int, dev: str) -> dict:
    states, fams = [], []
    for fam in range(n_fam):
        toks = make_family_hist(fam, seed * 7001 + fam * 37, T_HIST, per_fam, dev)
        r, _, _ = run_spont(
            seed,
            0.5,
            0,
            0.7,
            0.02,
            dev,
            K,
            0.0,
            g_inh,
            -1.0,
            hist_in=toks,
            t_spont=T_SPONT,
        )
        states.append(r.cpu())
        fams.extend([fam] * r.shape[0])
    S = torch.cat(states)
    bids = basin_ids(S)
    mi = mutual_info(fams, bids)
    g = torch.Generator().manual_seed(seed + 313)
    null = []
    fl = list(fams)
    for _ in range(200):
        p = torch.randperm(len(fl), generator=g).tolist()
        null.append(mutual_info([fl[i] for i in p], bids))
    null.sort()
    return {
        "g_inh": g_inh,
        "n": len(fams),
        "n_basins": len(set(bids)),
        "mi": mi,
        "null_p95": null[int(0.95 * len(null))],
        "firing": float(S.mean()),
        "occupancy": bids,
        "fams": fams,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--ginhs", default="1,2")
    ap.add_argument("--n-fam", type=int, default=8)
    ap.add_argument("--per-fam", type=int, default=16)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": {}, "n_fam": args.n_fam, "per_fam": args.per_fam}
    for g_inh in [float(x) for x in args.ginhs.split(",")]:
        rows = [run_cell(sd, g_inh, args.per_fam, args.n_fam, dev) for sd in range(args.seeds)]
        key = f"g={g_inh:g}"
        wins = sum(1 for r in rows if r["mi"] > r["null_p95"])
        res["cells"][key] = {
            "g_inh": g_inh,
            "n_basins": sum(r["n_basins"] for r in rows) / len(rows),
            "mi": [r["mi"] for r in rows],
            "null_p95": [r["null_p95"] for r in rows],
            "wins": wins,
            "n": args.seeds,
            "firing": sum(r["firing"] for r in rows) / len(rows),
            "rows": rows,
        }
        v = res["cells"][key]
        print(
            f"  {key:6s} 发放 {v['firing']:.4f}  盆 {v['n_basins']:.1f}  "
            f"MI {[round(x, 3) for x in v['mi']]}  "
            f"零分布95% {[round(x, 3) for x in v['null_p95']]}  "
            f"=> 过线 {wins}/{args.seeds}",
            flush=True,
        )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )

    print("")
    print("判：")
    cc = res["cells"]
    ok = [k for k, v in cc.items() if 0.01 <= v["firing"] <= 0.20]
    print(f"  U-0 发放在窗内：{len(ok)}/{len(cc)} 档 => {'过' if ok else '不过 => 作废'}")
    if not ok:
        return 0
    best = max(ok, key=lambda k: cc[k]["wins"])
    u1 = cc[best]["wins"] >= 5
    print(
        f"  U-1 指派（MI > 置换 95% 分位，≥5/6）：{best} "
        f"{cc[best]['wins']}/{cc[best]['n']} => {'过' if u1 else '不过'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""M9（Round F）：**广播广度** —— 少数共享模式覆盖多少神经元。

判据在 `research/sel/night-2026-11-16.md`（含**修订 4a**）。

## 修订 4a（开跑前）

全局抑制场强制了一个共同模态，第一主成分本来就会载荷到几乎所有神经元 ⇒ 广度≈1，
测的是"场存在"而不是"广播"。**在扣掉群体均值之后的残余上做**，
并要求残余广度显著高于"逐神经元独立打乱时间"的零分布。
"""

from __future__ import annotations

# ruff: noqa: N803, N806
import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.hist_families import make_family_hist
from scripts.m6_pools import build_pools, run_spont

OUT = _pathlib.Path("sweep_results/m9_broadcast")
K = 4
T_HIST = 1024
T_SPONT = 1024
REC = 256
FAMS = list(range(8))


def breadth(Z: torch.Tensor, target: float = 0.8, load: float = 0.3) -> float:
    """`Z` 是 `(N, T)`；返回解释 `target` 方差的主成分覆盖的神经元比例。"""
    Zc = Z - Z.mean(1, keepdim=True)
    C = Zc @ Zc.T / max(Zc.shape[1], 1)
    ev, evec = torch.linalg.eigh(C.double())
    order = ev.argsort(descending=True)
    ev, evec = ev[order], evec[:, order]
    ev = ev.clamp(min=0)
    k = int((ev.cumsum(0) / max(float(ev.sum()), 1e-12) < target).sum().item()) + 1
    n = Z.shape[0]
    hit = torch.zeros(n)
    for j in range(k):
        v = evec[:, j]
        hit = torch.maximum(hit, (v.abs() > load).float())
    return float(hit.mean())


@torch.no_grad()
def run_one(net, seed: int, fam: int, dev: str) -> torch.Tensor:
    toks = make_family_hist(fam, seed * 71 + fam * 13 + 4, T_HIST, 32, dev)
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
    )
    return snaps.mean(0).cpu()  # (N, REC)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": []}
    for sd in range(args.seeds):
        net = build_pools(sd, dev, K, 0.0)
        raw, res_, nul = [], [], []
        for fam in FAMS:
            Z = run_one(net, sd, fam, dev)
            raw.append(breadth(Z))
            Zr = Z - Z.mean(0, keepdim=True)  # 扣掉每时刻的群体均值
            res_.append(breadth(Zr))
            g = torch.Generator().manual_seed(sd * 7 + fam)
            perm = torch.stack(
                [torch.randperm(Zr.shape[1], generator=g) for _ in range(Zr.shape[0])]
            )
            Zs = Zr.gather(1, perm)
            nul.append(breadth(Zs))
        m = lambda v: sum(v) / len(v)  # noqa: E731
        row = {
            "seed": sd,
            "breadth_raw": m(raw),
            "breadth_residual": m(res_),
            "breadth_null": m(nul),
        }
        res["cells"].append(row)
        print(
            f"  seed {sd}: 原始 {row['breadth_raw']:.3f}  残余 {row['breadth_residual']:.3f}  "
            f"打乱零分布 {row['breadth_null']:.3f}",
            flush=True,
        )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )

    print("")
    print("判：")
    rows = res["cells"]
    br = sum(r["breadth_residual"] for r in rows) / len(rows)
    bn = sum(r["breadth_null"] for r in rows) / len(rows)
    print(f"  M9-1 残余广播广度 > 0.5：{br:.3f} => {'过' if br > 0.5 else '不过'}")
    wins = sum(1 for r in rows if r["breadth_residual"] > r["breadth_null"] + 0.15)
    print(
        f"  M9-2 高于打乱零分布（+0.15，≥5/6）：{br - bn:+.3f} => {wins}/{len(rows)} "
        f"{'过' if wins >= max(5, 0.83 * len(rows)) else '不过'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

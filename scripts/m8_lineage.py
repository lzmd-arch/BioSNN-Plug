"""M8（Round E）：**谱系** —— 子代继承能力，但彼此可区分。

判据在 `research/sel/night-2026-11-16.md`（跑之前）+ 本文件顶部修订说明。

## 修订（**开跑前**）

预注册 M8-2 写的是"子代之间的终态差异 > 同期亲代自身的重复差异"。
**那条在确定性系统里恒真**：同一权重、同一历史跑两遍逐位相同，重复差异恒为 0。
改成：**子代之间的终态差异 > 亲代在"同族但不同随机流"两条历史之间的差异** ——
即"变异"要大于"历史本身的随机性"。
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
from scripts.m6_pools import build_pools, rel, run_spont

OUT = _pathlib.Path("sweep_results/m8_lineage")
K = 4
T_HIST = 1024
T_MEAS = 1024
FAMS = list(range(8))


@torch.no_grad()
def run_one(net, seed: int, fam: int, per_fam: int, salt: int, dev: str):
    toks = make_family_hist(fam, seed * 53 + fam * 19 + salt, T_HIST, per_fam, dev)
    r, _, _ = run_spont(
        seed, 0.5, 0, 0.7, 0.02, dev, K, 0.0, 0.0, -1.0, hist_in=toks, t_spont=T_MEAS, net=net
    )
    return r


@torch.no_grad()
def decode(net, seed: int, per_fam: int, salt: int, dev: str):
    feats, labels = [], []
    for fam in FAMS:
        r = run_one(net, seed, fam, per_fam, salt, dev)
        feats.append(spatial_bins(r).cpu())
        labels.append(torch.full((r.shape[0],), fam))
    X = torch.cat(feats)
    Y = torch.cat(labels)
    perm = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(3 + seed))
    X, Y = X[perm], Y[perm]
    cut = int(X.shape[0] * 0.75)
    acc, _ = probe(X, Y, cut, len(FAMS))
    return acc, majority_baseline(Y[:cut], Y[cut:], len(FAMS))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--children", type=int, default=6)
    ap.add_argument("--sigmas", default="0.05,0.2")
    ap.add_argument("--per-fam", type=int, default=8)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": []}
    for sd in range(args.seeds):
        w_in, pre, wrec, pid = build_pools(sd, dev, K, 0.0)
        parent = (w_in, pre, wrec, pid)
        pacc, floor = decode(parent, sd, args.per_fam, 0, dev)
        # 亲代自身的"历史随机性"基线：同族、两条不同随机流之间的终态距离
        a = run_one(parent, sd, 0, 16, 0, dev)
        b = run_one(parent, sd, 0, 16, 999, dev)
        d_parent = sum(rel(a[i], b[i]) for i in range(a.shape[0])) / a.shape[0]
        print(
            f"  seed {sd}: 亲代 {pacc:.3f}（底板 {floor:.3f}）历史随机性 {d_parent:.3f}", flush=True
        )
        for sigma in [float(x) for x in args.sigmas.split(",")]:
            accs = []
            g = torch.Generator().manual_seed(sd * 31 + int(sigma * 1000))
            states = []
            for _c in range(args.children):
                noise = torch.randn(wrec.shape, generator=g).to(dev) * sigma
                child = (w_in, pre, (wrec + noise).clamp(min=0.0), pid)
                acc, _ = decode(child, sd, args.per_fam, 0, dev)
                accs.append(acc)
                states.append(run_one(child, sd, 0, 16, 0, dev).cpu())
            d_child = sum(
                rel(states[i], states[j])
                for i in range(len(states))
                for j in range(i + 1, len(states))
            ) / max(len(states) * (len(states) - 1) / 2, 1)
            row = {
                "seed": sd,
                "sigma": sigma,
                "parent_acc": pacc,
                "child_acc_mean": sum(accs) / len(accs),
                "child_acc_min": min(accs),
                "d_parent": d_parent,
                "d_child": d_child,
            }
            res["cells"].append(row)
            print(
                f"    sigma={sigma:g}: 子代均值 {row['child_acc_mean']:.3f} "
                f"最低 {row['child_acc_min']:.3f} | 变异 {d_child:.3f} vs 历史随机性 {d_parent:.3f}",
                flush=True,
            )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )

    print("")
    print("判：")
    rows = res["cells"]
    q = [r["child_acc_mean"] / max(r["parent_acc"], 1e-9) for r in rows]
    m1 = all(x >= 0.80 for x in q)
    print(f"  M8-1 子代 ≥ 亲代 80%：{[round(x, 2) for x in q]} => {'过' if m1 else '不过'}")
    wins = sum(1 for r in rows if r["d_child"] > r["d_parent"])
    m2 = wins >= max(5, 0.83 * len(rows))
    print(
        f"  M8-2 变异 > 历史随机性：{[round(r['d_child'] / max(r['d_parent'], 1e-9), 2) for r in rows]} "
        f"=> {wins}/{len(rows)} {'过' if m2 else '不过'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

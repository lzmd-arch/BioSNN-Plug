"""M6ag：**用非线性/非参数读出重测高维状态** —— 线性探针是错的仪器。

## 用户的更正（2026-11-16）

> 高维隐状态本就不能读，更何况用线性，这是根本性错误。

正确。分布式高维编码的固有性质就是**线性不可读**；用 `n=128, d=1024` 的线性探针
去测它，失败是**仪器的必然**，不是"没有信息"。

## 三种读出，同一条数据

| 读出 | 说明 | 它对高维编码的要求 |
| :--- | :--- | :--- |
| **线性**（岭回归） | 上一轮用的 | 信息在一个**线性子空间**里 —— 高维编码通常不满足 |
| **1-NN**（非参数） | 测试态最近邻的训练态属于哪一族 | 只要**同族在状态空间里聚拢**，不需要任何参数形式 |
| **MLP**（非线性） | 隐层 256，早停 | 信息在**任意光滑**的映射里 |

**判据**：`radius=2` 下，**1-NN 或 MLP 的准确率 > 底板 + 0.15**
⇒ 信息在，只是线性读不出。

## 样本量

每条目 `n_fam × per_fam` 个状态。**只有当 `n` 与 `d` 同量级或更多时，
非线性读出才有意义。** 本轮 `per_fam=128`（n=1024），与 `d=1024` 相当；
`M6ab` 的 `n=128` 远远不够。
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
from scripts.instrument import majority_baseline, mlp_probe, probe
from scripts.m6_pools import build_pools, run_spont

OUT = _pathlib.Path("sweep_results/m6ag_nonlinear_readout")
K = 4
T_HIST, T_SPONT = 1024, 2048
FAMS = list(range(8))


def field_args(radius: int) -> dict:
    return {"g_field": 1.0, "g_local": 0.0} if radius >= 32 else {"g_field": 0.0, "g_local": 1.0}


@torch.no_grad()
def collect(seed: int, radius: int, per_fam: int, dev: str):
    net = build_pools(seed, dev, K, 0.0)
    X, Y = [], []
    done = 0
    while done < per_fam:
        b = min(32, per_fam - done)
        for fam in FAMS:
            toks = make_family_hist(fam, seed * 911 + fam * 37 + done, T_HIST, b, dev)
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
                field_radius=radius,
                **field_args(radius),
            )[0]
            X.append(r.cpu())
            Y.append(torch.full((r.shape[0],), fam))
        done += b
    return torch.cat(X), torch.cat(Y)


def knn_acc(Xtr, Ytr, Xte, Yte, k: int = 1) -> float:
    """1-NN：测试态最近的 k 个训练态投票。**非参数，对高维编码不该有先验偏见。**"""
    d = torch.cdist(Xte, Xtr)
    idx = d.topk(k, largest=False).indices
    votes = Ytr[idx]
    pred = votes.mode(dim=1).values if k > 1 else votes[:, 0]
    return float((pred == Yte).float().mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--radii", default="2,32")
    ap.add_argument("--per-fam", type=int, default=128)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": {}}
    for rad in [int(x) for x in args.radii.split(",")]:
        rows = []
        for sd in range(args.seeds):
            X, Y = collect(sd, rad, args.per_fam, dev)
            perm = torch.randperm(Y.shape[0], generator=torch.Generator().manual_seed(1 + sd))
            X, Y = X[perm], Y[perm]
            cut = int(Y.shape[0] * 0.75)
            lin, _ = probe(X, Y, cut, len(FAMS))
            nn = knn_acc(X[:cut], Y[:cut], X[cut:], Y[cut:])
            mlp = mlp_probe(X, Y, cut, len(FAMS))[0]
            floor = majority_baseline(Y[:cut], Y[cut:], len(FAMS))
            rows.append({"seed": sd, "linear": lin, "knn": nn, "mlp": mlp, "floor": floor})
            print(
                f"  radius={rad:2d} seed {sd}（n={X.shape[0]}, d={X.shape[1]}）: "
                f"线性 {lin:.3f}  1-NN {nn:.3f}  MLP {mlp:.3f}  底板 {floor:.3f}",
                flush=True,
            )
        m = lambda k, _r=rows: sum(r[k] for r in _r) / len(_r)  # noqa: E731
        res["cells"][f"r={rad}"] = {
            "radius": rad,
            "linear": m("linear"),
            "knn": m("knn"),
            "mlp": m("mlp"),
            "floor": m("floor"),
            "n": args.per_fam * len(FAMS),
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
        best = max(v["knn"], v["mlp"])
        ok = best > v["floor"] + 0.15
        print(
            f"  AG-1 局部场下非线性可读（1-NN {v['knn']:.3f} / MLP {v['mlp']:.3f}，"
            f"底板 {v['floor']:.3f}）：最好 {best:.3f} => "
            f"{'过（信息在，线性读不出）' if ok else '不过（换仪器也读不出）'}"
        )
        print(f"       线性对照 {v['linear']:.3f}")
    if "r=32" in cc:
        v = cc["r=32"]
        print(f"  全局场：线性 {v['linear']:.3f}、1-NN {v['knn']:.3f}、MLP {v['mlp']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

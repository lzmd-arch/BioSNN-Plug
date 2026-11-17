"""M6ba2：**残余里带不带历史 —— 用探针问，不用距离比问**。

## 第一版为什么无效（自我更正）

`m6ba` 用"距离比"：`rel(A,B) / rel(A,A')`。两处硬伤：

1. **三个条件各算各的 PCA 基** —— 它们不在同一个子空间里，距离不可比。
   这就是为什么"可读子空间"只有 1.13×（而它的解码是 0.604）。
2. **距离比这个口径本身没有分辨力** —— 它把一个二分类问题压成了一个标量。

## 正确做法

**用已经验证过的仪器：探针。** 同一个基底、同一个任务：

```
基底   = 训练集 PCA 前 m 维（**只用训练集算，一次算好，三者共用**）
可读端 = 投影到前 m 维        → 探针（应 ≈ 0.60，这是仪器自检）
残余端 = x − 投影             → 探针（> 底板+0.15 ⇒ 残余带史）
```

## 判据

- **R-0 仪器自检** ★：可读端的探针准确率 ≥ **0.45**（若不达，仪器或任务有问题，作废）；
- **R-1 残余带史** ★★：残余端 MLP > 底板 + **0.15**；
- **R-2 量化**：报残余端 / 可读端 的比值 —— 残余承载了多少比例的信息。
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

OUT = _pathlib.Path("sweep_results/m6ba2_residual_probe")
K = 4
T_HIST, T_SPONT = 1024, 2048
FAMS = list(range(8))
P_LONG = 0.5


@torch.no_grad()
def collect(seed: int, per_fam: int, dev: str):
    net = build_pools(seed, dev, K, 0.0, p_long=P_LONG)
    X, Y = [], []
    for fam in FAMS:
        toks = make_family_hist(fam, seed * 691 + fam * 67 + 41, T_HIST, per_fam, dev)
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
        )[0]
        X.append(r.cpu())
        Y.append(torch.full((r.shape[0],), fam))
    return torch.cat(X), torch.cat(Y)


def split_train_test_pca(Xtr: torch.Tensor, Xte: torch.Tensor, m: int):
    """**只用训练集**算基底，一次算好 —— 三者共用，这是第一版错的地方。"""
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True) + 1e-6
    A = (Xtr - mu) / sd
    B = (Xte - mu) / sd
    C = A.T @ A / max(A.shape[0], 1)
    ev, evec = torch.linalg.eigh(C.double())
    P = evec[:, ev.argsort(descending=True)[:m]].float()
    read_tr, read_te = A @ P, B @ P
    res_tr = A - read_tr @ P.T
    res_te = B - read_te @ P.T
    return read_tr, read_te, res_tr, res_te


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--per-fam", type=int, default=64)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": []}
    for sd in range(args.seeds):
        X, Y = collect(sd, args.per_fam, dev)
        perm = torch.randperm(Y.shape[0], generator=torch.Generator().manual_seed(71 + sd))
        X, Y = X[perm], Y[perm]
        cut = int(Y.shape[0] * 0.75)
        Rtr, Rte, Str, Ste = split_train_test_pca(X[:cut], X[cut:], args.m)
        read, _ = probe(Rtr, Y[:cut], int(Rtr.shape[0] * 0.8), len(FAMS))
        read_mlp = mlp_probe(torch.cat([Rtr, Rte]), Y, cut, len(FAMS))[0]
        res_mlp = mlp_probe(torch.cat([Str, Ste]), Y, cut, len(FAMS))[0]
        res_lin, _ = probe(Str, Y[:cut], int(Str.shape[0] * 0.8), len(FAMS))
        floor = majority_baseline(Y[:cut], Y[cut:], len(FAMS))
        res["cells"].append(
            {
                "seed": sd,
                "read_linear": read,
                "read_mlp": read_mlp,
                "res_linear": res_lin,
                "res_mlp": res_mlp,
                "floor": floor,
            }
        )
        print(
            f"  seed {sd}: 可读端 线性 {read:.3f} MLP {read_mlp:.3f} | "
            f"**残余端 线性 {res_lin:.3f} MLP {res_mlp:.3f}** | 底板 {floor:.3f}",
            flush=True,
        )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1, default=float),
            encoding="utf-8",
            newline=chr(10),
        )

    print("")
    print("判：")
    rows = res["cells"]
    m = lambda k, _r=rows: sum(r[k] for r in _r) / len(_r)  # noqa: E731
    r0 = m("read_mlp") >= 0.45
    print(
        f"  R-0 仪器自检（可读端 MLP ≥0.45）：{m('read_mlp'):.3f} => {'过' if r0 else '不过 => 作废'}"
    )
    if not r0:
        return 0
    gain = m("res_mlp") - m("floor")
    r1 = gain > 0.15
    print(
        f"  R-1 残余带史（MLP > 底板+0.15）：{m('res_mlp'):.3f} vs {m('floor'):.3f} "
        f"（{gain:+.3f}）=> {'过（残余带史）' if r1 else '不过（残余不带史）'}"
    )
    frac = (m("res_mlp") - m("floor")) / max(m("read_mlp") - m("floor"), 1e-9)
    print(f"  R-2 残余承载比例：{frac:.2f}（残余增益 / 可读端增益）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

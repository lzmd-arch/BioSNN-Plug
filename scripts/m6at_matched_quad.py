"""M6at：**维度匹配的二次读出** —— 用户 2026-11-17 指出 m6as 不匹配。

## 问题

`m6as` 在 64 维特征上做全二次展开（2080 维）。但：
- 状态的**内在维度不是 64**；若流形维度是 `m`，它的二次函数最多张成 `m(m+1)/2` 维；
- **2080 维里绝大部分方向是噪声** ⇒ `m6as` 测的是"在噪声里做二次"，不是"在流形上做二次"。

## 修法

**先投影到内在维度，再做二次**：

```
64 维 → PCA 前 m 个 → 二次展开 m(m+1)/2 维
m=4  → 10 维
m=8  → 36 维
m=16 → 136 维
m=32 → 528 维
```

`m` 扫一遍，看二次读出的增益在哪一档出现 —— **那一档就是状态的内在维度**。

## 判据

- **AT-0** 发放在窗内；
- **AT-1 存在匹配的 m 使二次 > 线性 + 0.15** ★★；
- **AT-2 增益随 m 先升后降**：升说明"信息需要非线性"，降说明"超过内在维度后是噪声"。
"""

from __future__ import annotations

# ruff: noqa: N803, N806
import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.hist_families import make_family_hist, spatial_bins
from scripts.instrument import majority_baseline, probe
from scripts.m6_pools import build_pools, run_spont

OUT = _pathlib.Path("sweep_results/m6at_matched_quad")
K = 4
T_HIST, T_SPONT = 1024, 2048
FAMS = list(range(8))
P_LONG = 0.5
ARMS = {
    "high_dim(gl=2)": {"field_radius": 2, "g_field": 0.0, "g_local": 2.0},
    "global": {"field_radius": 32, "g_field": 1.0, "g_local": 0.0},
}


def quad_m(Xtr: torch.Tensor, Xte: torch.Tensor, m: int):
    """在训练集上做 PCA 取 m 个主成分，再各自做二次展开（只用训练统计 ⇒ 无泄漏）。"""
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True) + 1e-6
    A = (Xtr - mu) / sd
    B = (Xte - mu) / sd
    # 训练集协方差的特征分解
    C = A.T @ A / max(A.shape[0], 1)
    ev, evec = torch.linalg.eigh(C.double())
    order = ev.argsort(descending=True)[:m]
    P = evec[:, order].float()
    Za, Zb = A @ P, B @ P
    iu = torch.triu_indices(m, m)
    return (Za[:, iu[0]] * Za[:, iu[1]]), (Zb[:, iu[0]] * Zb[:, iu[1]])


@torch.no_grad()
def collect(seed: int, arm: dict, per_fam: int, dev: str):
    w_in, pre, wrec, pid = build_pools(seed, dev, K, 0.0, p_long=P_LONG)
    net = (w_in, pre, wrec, pid)
    feats, labels, frs = [], [], []
    for fam in FAMS:
        toks = make_family_hist(fam, seed * 769 + fam * 47 + 23, T_HIST, per_fam, dev)
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
            **arm,
        )[0]
        feats.append(spatial_bins(r).cpu())
        labels.append(torch.full((r.shape[0],), fam))
        frs.append(float(r.mean()))
    return torch.cat(feats), torch.cat(labels), sum(frs) / len(frs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--per-fam", type=int, default=96)
    ap.add_argument("--ms", default="4,8,16,32")
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    ms = [int(x) for x in args.ms.split(",")]

    res: dict = {"cells": {}}
    for name, arm in ARMS.items():
        rows = []
        for sd in range(args.seeds):
            X, Y, fr = collect(sd, arm, args.per_fam, dev)
            perm = torch.randperm(Y.shape[0], generator=torch.Generator().manual_seed(29 + sd))
            X, Y = X[perm], Y[perm]
            cut = int(Y.shape[0] * 0.75)
            lin, _ = probe(X, Y, cut, len(FAMS))
            row = {
                "seed": sd,
                "linear": lin,
                "firing": fr,
                "floor": majority_baseline(Y[:cut], Y[cut:], len(FAMS)),
            }
            for m in ms:
                Qtr, Qte = quad_m(X[:cut], X[cut:], m)
                Xq = torch.cat([Qtr, Qte])
                row[f"quad_m{m}"] = probe(Xq, Y, cut, len(FAMS))[0]
            rows.append(row)
            print(
                f"  {name:16s} seed {sd}: 线性 {lin:.3f}  "
                + "  ".join(f"m{m} {row[f'quad_m{m}']:.3f}" for m in ms)
                + f"  底板 {row['floor']:.3f}",
                flush=True,
            )
        m = lambda k, _r=rows: sum(r[k] for r in _r) / len(_r)  # noqa: E731
        v = {"linear": m("linear"), "firing": m("firing"), "floor": m("floor"), "rows": rows}
        for mm in ms:
            v[f"quad_m{mm}"] = m(f"quad_m{mm}")
        res["cells"][name] = v
        print(
            f"  {name:16s} 均值：线性 {v['linear']:.3f}  "
            + "  ".join(f"m{mm} {v[f'quad_m{mm}']:.3f}" for mm in ms),
            flush=True,
        )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1, default=float),
            encoding="utf-8",
            newline=chr(10),
        )

    print("")
    print("判：")
    cc = res["cells"]
    h = cc["high_dim(gl=2)"]
    ok0 = 0.01 <= h["firing"] <= 0.20
    print(f"  AT-0 发放在窗内：{h['firing']:.4f} => {'过' if ok0 else '不过 => 作废'}")
    gains = {mm: h[f"quad_m{mm}"] - h["linear"] for mm in ms}
    best_m = max(gains, key=lambda k: gains[k])
    print(
        f"  AT-1 存在匹配的 m（二次 > 线性 +0.15）：最好 m={best_m} 增益 "
        f"{gains[best_m]:+.3f} => {'过' if gains[best_m] > 0.15 else '不过'}"
    )
    seq = [gains[mm] for mm in ms]
    peak = seq.index(max(seq))
    print(
        f"  AT-2 增益随 m：{[round(value, 3) for value in seq]} 峰值在 m={ms[peak]} "
        f"=> {'先升后降' if 0 < peak < len(seq) - 1 else '单调或端点'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

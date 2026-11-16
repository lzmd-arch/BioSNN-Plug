"""M6au：**高阶展开 + 随机涨落读出**（用户 2026-11-17 两条提案）。

## 提案

> 还有两个可以尝试：超欧展开的读出和量子随机涨落读出。

## 两条分别测什么

**① 超欧（高阶）展开**：把读出从二次推到**三次**。
`m6at` 已证低维状态是"~4 维 + 强非线性"（线性 0.104 → 4 维线性 0.547 → 4 维二次 **0.786**）。
若三次继续涨 ⇒ 非线性阶数还没到顶；若平/降 ⇒ 二次已经够。

**② 量子随机涨落读出**：用**真随机源**（`os.urandom`，见 `scripts/rng.py` 的三档）做
**随机投影集成**：R 次随机投影 → 每次一个弱分类器 → 投票。
测的是"涨落本身是不是资源"。这条**不依赖量子硬件**，只依赖不可预测的随机源。

## 判据

- **AU-0** 发放在窗内；
- **AU-1 高阶增益** ★：三次 > 二次 + 0.05（在某个 regime 上）；
- **AU-2 涨落读出** ★：随机投影集成 > 该维度上的确定性线性 + 0.10。
"""

from __future__ import annotations

# ruff: noqa: N803, N806
import argparse
import itertools
import json
import os
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.hist_families import make_family_hist, spatial_bins
from scripts.instrument import _ridge_fit, majority_baseline
from scripts.m6_pools import build_pools, run_spont

OUT = _pathlib.Path("sweep_results/m6au_expansion")
K = 4
T_HIST, T_SPONT = 1024, 2048
FAMS = list(range(8))
P_LONG = 0.5
ARMS = {
    "high_dim(gl=2)": {"field_radius": 2, "g_field": 0.0, "g_local": 2.0},
    "global": {"field_radius": 32, "g_field": 1.0, "g_local": 0.0},
}


def poly(Ztr, Zte, deg):
    """`deg` 阶多项式特征（所有 ≤deg 次的单项式，不含常数）。"""
    m = Ztr.shape[1]
    cols_tr, cols_te = [], []
    for d in range(1, deg + 1):
        for comb in itertools.combinations_with_replacement(range(m), d):
            v_tr = torch.ones(Ztr.shape[0])
            v_te = torch.ones(Zte.shape[0])
            for idx in comb:
                v_tr = v_tr * Ztr[:, idx]
                v_te = v_te * Zte[:, idx]
            cols_tr.append(v_tr)
            cols_te.append(v_te)
    return torch.stack(cols_tr, 1), torch.stack(cols_te, 1)


def pca_m(Xtr, Xte, m):
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    C = A.T @ A / max(A.shape[0], 1)
    ev, evec = torch.linalg.eigh(C.double())
    P = evec[:, ev.argsort(descending=True)[:m]].float()
    return A @ P, B @ P


def acc_ridge(Xtr, Ytr, Xte, Yte, n_class, lam=1.0):
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    W = _ridge_fit((Xtr - mu) / sd, Ytr, n_class, lam)
    pred = ((Xte - mu) / sd @ W).argmax(1)
    return float((pred == Yte).float().mean())


def stochastic_readout(Xtr, Ytr, Xte, Yte, m, n_class, reps=32):
    """**真随机源**驱动的随机投影集成：每次随机投影到 m 维，线性分类，多数投票。"""
    d = Xtr.shape[1]
    preds = torch.zeros(Yte.shape[0], n_class)
    for _ in range(reps):
        g = torch.Generator().manual_seed(int.from_bytes(os.urandom(8), "little") >> 1)
        P = torch.randn(d, m, generator=g) / d**0.5
        Ztr, Zte = Xtr @ P, Xte @ P
        mu, sd = Ztr.mean(0, keepdim=True), Ztr.std(0, keepdim=True) + 1e-6
        W = _ridge_fit((Ztr - mu) / sd, Ytr, n_class, 1.0)
        preds += torch.nn.functional.one_hot(((Zte - mu) / sd @ W).argmax(1), n_class).float()
    return float((preds.argmax(1) == Yte).float().mean())


@torch.no_grad()
def collect(seed: int, arm: dict, per_fam: int, dev: str):
    w_in, pre, wrec, pid = build_pools(seed, dev, K, 0.0, p_long=P_LONG)
    net = (w_in, pre, wrec, pid)
    feats, labels, frs = [], [], []
    for fam in FAMS:
        toks = make_family_hist(fam, seed * 733 + fam * 53 + 29, T_HIST, per_fam, dev)
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
    ap.add_argument("--ms", default="4,8")
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
            perm = torch.randperm(Y.shape[0], generator=torch.Generator().manual_seed(37 + sd))
            X, Y = X[perm], Y[perm]
            cut = int(Y.shape[0] * 0.75)
            row = {
                "seed": sd,
                "firing": fr,
                "floor": majority_baseline(Y[:cut], Y[cut:], len(FAMS)),
            }
            row["linear64"] = acc_ridge(X[:cut], Y[:cut], X[cut:], Y[cut:], len(FAMS))
            for m in ms:
                Ztr, Zte = pca_m(X[:cut], X[cut:], m)
                row[f"lin_m{m}"] = acc_ridge(Ztr, Y[:cut], Zte, Y[cut:], len(FAMS))
                for deg in (2, 3):
                    A, B = poly(Ztr, Zte, deg)
                    row[f"poly{deg}_m{m}"] = acc_ridge(A, Y[:cut], B, Y[cut:], len(FAMS))
                row[f"stoch_m{m}"] = stochastic_readout(
                    X[:cut], Y[:cut], X[cut:], Y[cut:], m, len(FAMS)
                )
            rows.append(row)
            keys = [k for k in row if k not in ("seed", "firing", "floor")]
            print(
                f"  {name:16s} seed {sd}: " + "  ".join(f"{k} {row[k]:.3f}" for k in keys),
                flush=True,
            )
        m_ = lambda k, _r=rows: sum(r[k] for r in _r) / len(_r)  # noqa: E731
        v = {"firing": m_("firing"), "floor": m_("floor"), "rows": rows}
        for k in rows[0]:
            if k in ("seed", "firing", "floor"):
                continue
            v[k] = m_(k)
        res["cells"][name] = v
        print(
            f"  {name:16s} 均值："
            + "  ".join(f"{k} {v[k]:.3f}" for k in v if k not in ("rows", "firing", "floor")),
            flush=True,
        )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1, default=float),
            encoding="utf-8",
            newline=chr(10),
        )

    print("")
    print("判：")
    for name, v in res["cells"].items():
        ok0 = 0.01 <= v["firing"] <= 0.20
        print(f"  {name}: 窗口 {'✅' if ok0 else '❌'}")
        for m in ms:
            g2 = v[f"poly2_m{m}"] - v[f"lin_m{m}"]
            g3 = v[f"poly3_m{m}"] - v[f"poly2_m{m}"]
            gs = v[f"stoch_m{m}"] - v[f"lin_m{m}"]
            print(f"    m={m}: 二次增益 {g2:+.3f}  三次增益 {g3:+.3f}  随机投影集成增益 {gs:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

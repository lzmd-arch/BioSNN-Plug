"""M6v：**状态流形的内在维度** —— "高维"从形容词变成数字。

## 为什么必须单独测

M6s/M6u/M12 都只数了"有多少个互不相同的持久态"（`g_inh=0` 时 128 条历史给 82–114 个）。
但**互不相同 ≠ 高维**：一条 1 维曲线上的 128 个不同位置也满足。

## 两个估计量 + 合成自检

- **参与比 PR**：状态协方差的 `(Σλ)²/Σλ²`；
- **two-NN**（Facco 2015）：`ID = n / Σ log(μ_i)`，`μ_i = d2/d1`。

**自检 D-0**：合成 d 维线性子空间上的点必须给出 `PR ≈ d`（d ∈ {1, 4, 16}）。
自检不过，后面的数全部作废。
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

OUT = _pathlib.Path("sweep_results/m6v_dimension")
K = 4
T_HIST, T_SPONT = 1024, 2048


def pr_of(X: torch.Tensor) -> float:
    """`X` 是 `(n, d)`；返回协方差的参与比（用 Gram 矩阵，n<d 时更稳）。"""
    Xc = X - X.mean(0, keepdim=True)
    n = Xc.shape[0]
    G = Xc @ Xc.T / n
    ev = torch.linalg.eigvalsh(G.double()).clamp(min=0)
    return float((ev.sum() ** 2) / max(float((ev**2).sum()), 1e-24))


def two_nn(X: torch.Tensor) -> float:
    """Facco 的 two-NN 内在维度估计。"""
    d = torch.cdist(X.double(), X.double())
    d.fill_diagonal_(float("inf"))
    two = d.topk(2, largest=False).values
    mu = two[:, 1] / two[:, 0].clamp(min=1e-12)
    return float(len(X) / mu.log().sum())


def self_check() -> bool:
    """**相对容差**：PR 在有限样本下系统性偏低（n/d 越小越偏），
    用绝对容差会把正常的有偏估计判成坏工具。"""
    ok = True
    for d in (1, 4, 16):
        g = torch.Generator().manual_seed(d)
        basis = torch.linalg.qr(torch.randn(512, d, generator=g).double())[0]
        X = (torch.randn(512, d, generator=g).double() @ basis.T).float()
        pr, tn = pr_of(X), two_nn(X)
        e1, e2 = abs(pr - d) / d, abs(tn - d) / d
        print(
            f"  合成 d={d:2d}: PR={pr:.2f}（相对误差 {e1:.2f}）  two-NN={tn:.2f}（{e2:.2f}）",
            flush=True,
        )
        ok = ok and e1 < 0.2 and e2 < 0.25
    print("  注：真实数据只有 128 个状态，PR 是**下界**，不是真值。", flush=True)
    return ok


@torch.no_grad()
def states_for(net, seed: int, g_inh: float, dev: str, n_fam: int, per_fam: int):
    out = []
    for fam in range(n_fam):
        toks = make_family_hist(fam, seed * 211 + fam * 29 + 7, T_HIST, per_fam, dev)
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
            net=net,
        )
        out.append(r.cpu())
    return torch.cat(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--ginhs", default="0,1")
    ap.add_argument("--n-fam", type=int, default=8)
    ap.add_argument("--per-fam", type=int, default=16)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    print("D-0 自检（已知维数的合成数据）：", flush=True)
    sc = self_check()
    print(f"  => {'过（工具可用）' if sc else '**不过 ⇒ 后面全部作废**'}", flush=True)
    res: dict = {"self_check": sc, "cells": {}}
    if not sc:
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )
        return 1

    for g_inh in [float(x) for x in args.ginhs.split(",")]:
        prs, tns, ns = [], [], []
        for sd in range(args.seeds):
            net = build_pools(sd, dev, K, 0.0)
            S = states_for(net, sd, g_inh, dev, args.n_fam, args.per_fam)
            prs.append(pr_of(S))
            tns.append(two_nn(S))
            ns.append(S.shape[0])
            print(
                f"  g_inh={g_inh:g} seed {sd}: n={S.shape[0]}  PR={prs[-1]:.2f}  "
                f"two-NN={tns[-1]:.2f}",
                flush=True,
            )
        m = lambda v: sum(v) / len(v)  # noqa: E731
        res["cells"][f"g={g_inh:g}"] = {
            "pr": prs,
            "two_nn": tns,
            "pr_mean": m(prs),
            "two_nn_mean": m(tns),
            "n_states": int(m(ns)),
            "seeds": args.seeds,
        }
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )

    print("")
    print("判：")
    cc = res["cells"]
    g0 = cc.get("g=0")
    if g0:
        d1 = g0["pr_mean"] > 4.0
        print(
            f"  D-1 高维（g_inh=0 的 PR > 4，{g0['n_states']} 个状态）："
            f"PR {g0['pr_mean']:.2f}、two-NN {g0['two_nn_mean']:.2f} => {'过' if d1 else '不过'}"
        )
    if "g=0" in cc and "g=1" in cc:
        d2 = cc["g=0"]["pr_mean"] > cc["g=1"]["pr_mean"]
        print(
            f"  D-2 交叉抑制压缩维度：g=0 {cc['g=0']['pr_mean']:.2f} vs "
            f"g=1 {cc['g=1']['pr_mean']:.2f} => {'过' if d2 else '不过'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

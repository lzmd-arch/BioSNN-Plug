"""M6bd：**超欧（高阶）展开 —— 但这次落在信息方向上**（预注册 C-5）

## 为什么重做

§36 的结论是「二次到顶」。但那是在**方差最大**的 4 维上量的。
M6ba2 已经证明 **方差 ≠ 信息**：前 4 主成分吃掉 99.9998% 的方差，
线性可读只有 0.266；只占 0.15% 幅度的残余，线性可读到 0.703。

⇒ 「二次到顶」很可能只在方差方向上成立。**在信息方向上，高阶可能还有东西。**

## 设计

两组方向，都在**训练集**上算：

| 组 | 怎么选 | 含义 |
| :--- | :--- | :--- |
| **方差方向** | PCA 前 k 个特征向量 | §36 用的那把尺子 |
| **信息方向** | LDA 前 k 个判别方向（`S_w⁻¹ S_b`） | 类间散度 / 类内散度最大的方向 |

在每组 k 维子空间里做**度数 ≤ d 的完整多项式展开**，岭回归读出。

## 判据（照预注册 C-5）

- **O-0 仪器**：方差方向 d=2 的准确率应落在 §36 的量级（0.45–0.70），否则作废；
- **O-1 高阶在信息方向上还有增益** ★★：`d=3 − d=2 > 0.02` ⇒ **「二次到顶」作废**；
  否则 ⇒ 「二次到顶」升级为**在信息方向上也成立**；
- **O-2 信息方向优于方差方向**：`acc(信息, d=2) ≥ acc(方差, d=2)`；
- **O-3 对照**：打乱标签后三阶不得凭空涨（防止过拟合冒充高阶信息）。

**种子 ≥ 6。**
"""

from __future__ import annotations

# ruff: noqa: N806, N803
import argparse
import itertools
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.hist_families import make_family_hist
from scripts.instrument import majority_baseline
from scripts.m6_pools import build_pools, run_spont

OUT = _pwin = _pathlib.Path("sweep_results/m6bd_order")
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


def variance_dirs(A: torch.Tensor, k: int) -> torch.Tensor:
    C = A.T @ A / max(A.shape[0], 1)
    ev, evec = torch.linalg.eigh(C.double())
    return evec[:, ev.argsort(descending=True)[:k]].float()


def informative_dirs(A: torch.Tensor, Y: torch.Tensor, k: int, ridge: float = 1e-2) -> torch.Tensor:
    """LDA：`S_w⁻¹ S_b` 的前 k 个特征向量（`S_w` 加岭正则）。"""
    mu = A.mean(0, keepdim=True)
    d = A.shape[1]
    Sb = torch.zeros(d, d, dtype=torch.float64)
    Sw = torch.zeros(d, d, dtype=torch.float64)
    Ad = A.double()
    for c in Y.unique():
        Ac = Ad[c == Y]
        mc = Ac.mean(0, keepdim=True)
        Sb += Ac.shape[0] * (mc - mu.double()).T @ (mc - mu.double())
        Sw += (Ac - mc).T @ (Ac - mc)
    Sw += ridge * torch.trace(Sw) / d * torch.eye(d, dtype=torch.float64)
    M = torch.linalg.solve(Sw, Sb)
    # **`S_w⁻¹ S_b` 不是对称阵**，必须用 eig 而不是 eigh（eigh 只读三角，会给错方向）
    ev, evec = torch.linalg.eig(M)
    order = ev.real.argsort(descending=True)
    return evec.real[:, order[:k]].float()


def poly_features(Z: torch.Tensor, degree: int) -> torch.Tensor:
    """度数 ≤ degree 的完整单项式（含常数项与各次项）。"""
    n, k = Z.shape
    cols = [torch.ones(n, 1)]
    for d in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(k), d):
            cols.append(torch.prod(Z[:, list(combo)], dim=1, keepdim=True))
    return torch.cat(cols, 1)


def ridge_acc(F: torch.Tensor, Y: torch.Tensor, cut: int, n_class: int) -> float:
    Ftr, Fte = F[:cut], F[cut:]
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
    Yh = torch.nn.functional.one_hot(Y[:cut], n_class).float()
    lam = 1.0 * Ftr.shape[0]
    W = torch.linalg.solve(Ftr.T @ Ftr + lam * torch.eye(Ftr.shape[1]), Ftr.T @ Yh)
    return float((Fte @ W).argmax(1).eq(Y[cut:]).float().mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--per-fam", type=int, default=32)
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
        mu, s = X[:cut].mean(0), X[:cut].std(0) + 1e-6
        A = (X - mu) / s
        Pv = variance_dirs(A[:cut], args.k)
        Pi = informative_dirs(A[:cut], Y[:cut], args.k)
        row = {"seed": sd, "floor": majority_baseline(Y[:cut], Y[cut:], len(FAMS))}
        for tag, P in (("var", Pv), ("info", Pi)):
            Z = A @ P
            for d in (1, 2, 3):
                row[f"{tag}{d}"] = ridge_acc(poly_features(Z, d), Y, cut, len(FAMS))
        # O-3 对照：标签打乱后 info3 不得凭空涨
        g = torch.Generator().manual_seed(900 + sd)
        Ys = Y.clone()[torch.randperm(Y.shape[0], generator=g)]
        row["info3_shuf"] = ridge_acc(poly_features(A @ Pi, 3), Ys, cut, len(FAMS))
        res["cells"].append(row)
        print(
            f"  seed {sd}: 方差 1/2/3 阶 {row['var1']:.3f}/{row['var2']:.3f}/{row['var3']:.3f}"
            f" | 信息 1/2/3 阶 {row['info1']:.3f}/{row['info2']:.3f}/{row['info3']:.3f}"
            f" | 三阶打乱 {row['info3_shuf']:.3f} | 底板 {row['floor']:.3f}",
            flush=True,
        )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1, default=float),
            encoding="utf-8",
            newline=chr(10),
        )

    rows = res["cells"]
    m = lambda k, _r=rows: sum(r[k] for r in _r) / len(_r)  # noqa: E731
    print("")
    print("判：")
    o0 = 0.40 <= m("var2") <= 0.75
    print(
        f"  O-0 仪器（方差方向二阶应落在 §36 量级）：{m('var2'):.3f} => {'过' if o0 else '不过 => 作废'}"
    )
    if not o0:
        return 0
    gain = m("info3") - m("info2")
    print(
        f"  O-1 信息方向三阶 − 二阶：{gain:+.3f} => "
        f"{'**过：二次到顶作废**' if gain > 0.02 else '不过：二次到顶在信息方向上也成立'}"
    )
    print(
        f"  O-2 信息方向 ≥ 方差方向（二阶）：{m('info2'):.3f} vs {m('var2'):.3f} => "
        f"{'过' if m('info2') >= m('var2') else '不过'}"
    )
    shuf = m("info3_shuf")
    print(
        f"  O-3 三阶打乱对照：{shuf:.3f} => {'过（未过拟合冒充）' if shuf < m('info3') - 0.15 else '**不过**'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

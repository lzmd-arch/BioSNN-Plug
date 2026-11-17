"""M6ba：**残余里带不带历史** —— 第三轴（不可读的个体史）在经典层的投影。

## 问题

三螺旋假说：双螺旋 = 物种层（生成器，可复制）；**第三轴 = 量子史轴（个体层，不可克隆）**。
按「不可克隆」奠基，第三轴承载的是**不可读的**个体史
⇒ **任何可读性测试都应该测不到它。**

**今晚唯一能直接问它的办法**：把状态拆成可读子空间与残余，问残余带不带历史。

```
状态 = 可读子空间（PCA-4 + 二次，实测 0.604）  ⊕  残余
```

- 残余**携带历史** ⇒ 存在"不可读但带史"的成分 —— 第三轴在经典层的投影；
- 残余**不携带** ⇒ 第三轴在这个系统里是空的。

## 判据

- **R-0** 可读子空间有效（解码 > 底板 + 0.15），否则作废；
- **R-1 残余带历史** ★★：在**残余**上，历史距离 / 对照距离 > 1.5 且 ≥5/6 种子；
- **R-2 对照**：同一口径下，可读子空间的比值应更高（它是"可读的那部分"）。

**残余的定义**：减去前 4 个主成分后的**逐神经元发放**（不是桶），即 `x - P Pᵀ x`。
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

OUT = _pathlib.Path("sweep_results/m6ba_residual_history")
K = 4
T_HIST, T_SPONT = 1024, 2048
N_SYM = 8


@torch.no_grad()
def collect(seed: int, alpha_seed: int, per_fam: int, dev: str, conc: float = 0.7):
    net = build_pools(seed, dev, K, 0.0, p_long=0.5)
    states, labels = [], []
    for fam in range(N_SYM):
        toks = make_family_hist(fam, alpha_seed * 1000 + fam * 7, T_HIST, per_fam, dev, conc=conc)
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
        states.append(r.cpu())
        labels.append(torch.full((r.shape[0],), fam))
    return torch.cat(states), torch.cat(labels)


def rel(a, b) -> float:
    return float((a - b).norm() / (0.5 * (a.norm() + b.norm()) + 1e-12))


def split_readable(X: torch.Tensor, m: int = 4):
    """可读子空间 = 前 m 个主成分；残余 = X − 投影。"""
    mu = X.mean(0, keepdim=True)
    A = X - mu
    C = A.T @ A / max(A.shape[0], 1)
    ev, evec = torch.linalg.eigh(C.double())
    P = evec[:, ev.argsort(descending=True)[:m]].float()
    read = A @ P @ P.T
    return read, A - read


def ratio_of(readable: bool, ra, rb, rn, m) -> tuple[float, int]:
    if readable:
        ta, tb, tn = split_readable(ra, m)[0], split_readable(rb, m)[0], split_readable(rn, m)[0]
    else:
        ta, tb, tn = split_readable(ra, m)[1], split_readable(rb, m)[1], split_readable(rn, m)[1]
    # 逐样本
    n = ta.shape[0]
    dh = sum(rel(ta[i], tb[i]) for i in range(n)) / n
    dn = sum(rel(ta[i], tn[i]) for i in range(n)) / n
    return dh / max(dn, 1e-12), 1 if dh > dn else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--per-fam", type=int, default=32)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": []}
    for sd in range(args.seeds):
        # **A 与 B 必须是不同的统计量**（conc 0.7 vs 0.3）；A' 同 A 的统计量、只换随机流。
        # 第一版只换了 seed ⇒ 对照没有区别 ⇒ 结果无效（R-0 只有 1.06x 就是证据）。
        Xa, Ya = collect(sd, 100 + sd, args.per_fam, dev, conc=0.7)
        Xb, _ = collect(sd, 100 + sd, args.per_fam, dev, conc=0.3)
        Xn, _ = collect(sd, 900 + sd, args.per_fam, dev, conc=0.7)
        # 只取族均值向量（每个族一个状态），个体间比较
        ra = torch.stack([Xa[Ya == f].mean(0) for f in range(N_SYM)])
        rb = torch.stack([Xb[Ya == f].mean(0) for f in range(N_SYM)])
        rn = torch.stack([Xn[Ya == f].mean(0) for f in range(N_SYM)])
        r_read, w_read = ratio_of(True, ra, rb, rn, args.m)
        r_res, w_res = ratio_of(False, ra, rb, rn, args.m)
        res["cells"].append(
            {
                "seed": sd,
                "readable": r_read,
                "residual": r_res,
                "win_read": w_read,
                "win_res": w_res,
            }
        )
        print(f"  seed {sd}: 可读子空间比 {r_read:.2f}x  残余比 {r_res:.2f}x", flush=True)
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1, default=float),
            encoding="utf-8",
            newline=chr(10),
        )

    print("")
    print("判：")
    rows = res["cells"]
    m = lambda k, _r=rows: sum(r[k] for r in _r) / len(_r)  # noqa: E731
    print(
        f"  R-0 可读子空间有效（比值 >1.5 且 ≥5/6）：{m('readable'):.2f}x "
        f"{sum(r['win_read'] for r in rows)}/{len(rows)}"
    )
    wins = sum(r["win_res"] for r in rows)
    r1 = m("residual") > 1.5 and wins >= 5
    print(
        f"  R-1 **残余带历史**（>1.5 且 ≥5/6）：{m('residual'):.2f}x {wins}/{len(rows)} "
        f"=> {'过（第三轴有经典投影）' if r1 else '不过（残余不带史）'}"
    )
    print(f"  R-2 可读 vs 残余：{m('readable'):.2f}x vs {m('residual'):.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

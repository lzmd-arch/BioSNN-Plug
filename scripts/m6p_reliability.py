"""M6p 的分批信度：批次均值那个"状态"是不是一个可估计的量。

**这不是预注册判据，是对已收数据的仪器检验**（写于 2026-11-16 深夜，结果出来之后）。

## 为什么必须做

M6p 的比值 `d_hist/d_null` 全部是被 `d_null` 撑起来的：`d_hist` 在所有臂上 ≈1.0，
而 `d_null` 在"好"臂里掉到 0.32。这只说明"同一统计下两次运行更像"，
**不说明那个"像"是一个稳定的状态**——如果 32 个样本各自乱跑，
它们的均值本身就是一个估计量，`d_null` 小可能只是估计噪声被平均掉了。

**分批信度**把这两件事分开：把 32 个样本劈成两半，各自求池向量，
量两半之间的距离。它与 `d_null` 同量级 ⇒ 那个"状态"就是估计噪声，
`d_hist` 的差异不可解释；它远小于 `d_null` ⇒ 状态是网络的真实属性。
"""

from __future__ import annotations

# ruff: noqa: N803, N806
import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.m6_pools import pool_id, pool_rates, rel, run_spont
from scripts.m7_history import A_HIGH, A_LOW

OUT = _pathlib.Path("sweep_results/m6_pools/reliability.json")
ARMS = [
    ("K=4/g=2/eta=-1", 4, 2.0, -1.0),
    ("K=4/g=2/eta=2", 4, 2.0, 2.0),
    ("K=8/g=0/eta=-1", 8, 0.0, -1.0),
    ("K=4/g=0/eta=-1", 4, 0.0, -1.0),
    ("K=8/g=1/eta=-1", 8, 1.0, -1.0),
]


def resid(R):
    return R - R.mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res: dict = {}

    for key, K, g, e in ARMS:
        pid = pool_id(dev, K)
        half, dh, dn, agrees, dh_ind, dn_ind, spread = [], [], [], [], [], [], []
        for sd in range(args.seeds):
            outs = []
            for alpha, hs in ((A_LOW, 100 + sd), (A_HIGH, 100 + sd), (A_LOW, 900 + sd)):
                r, _ = run_spont(sd, alpha, hs, 0.7, 0.02, dev, K, 0.0, g, e)
                outs.append(pool_rates(r, pid, K))  # (B, K) 逐样本
            Ra, Rb, Rn = outs
            ra, rb, rn = resid(Ra.mean(0)), resid(Rb.mean(0)), resid(Rn.mean(0))
            dh.append(rel(ra, rb))
            dn.append(rel(ra, rn))
            # 分批：偶数样本 vs 奇数样本
            A, Bb = resid(Ra[0::2].mean(0)), resid(Ra[1::2].mean(0))
            half.append(rel(A, Bb))
            agrees.append(1 if rel(A, Bb) < rel(ra, rn) else 0)
            # **逐个体**：同一下标、不同运行。历史若真的写进"个体"，这个比也该 >1。
            n_i = Ra.shape[0]
            dh_i = sum(rel(resid(Ra[i]), resid(Rb[i])) for i in range(n_i)) / n_i
            dn_i = sum(rel(resid(Ra[i]), resid(Rn[i])) for i in range(n_i)) / n_i
            dh_ind.append(dh_i)
            dn_ind.append(dn_i)
            # 单个个体离总体均值有多远（相对它的模）
            spread.append(
                float((Ra - Ra.mean(0, keepdim=True)).norm(dim=1).mean() / Ra.mean(0).norm())
            )
        m = lambda x: sum(x) / len(x)  # noqa: E731
        res[key] = {
            "K": K,
            "g_inh": g,
            "eta_gain": e,
            "d_hist": m(dh),
            "d_null": m(dn),
            "d_half": m(half),
            "ratio": m(dh) / max(m(dn), 1e-12),
            "half_over_null": m(half) / max(m(dn), 1e-12),
            "half_wins": sum(agrees),
            "d_hist_indiv": m(dh_ind),
            "d_null_indiv": m(dn_ind),
            "ratio_indiv": m(dh_ind) / max(m(dn_ind), 1e-12),
            "spread_indiv": m(spread),
            "n": args.seeds,
        }
        v = res[key]
        print(
            f"  {key:18s} d_hist {v['d_hist']:.3f}  d_null {v['d_null']:.3f}  "
            f"**d_half {v['d_half']:.3f}**  比 {v['ratio']:.2f}x  "
            f"| 逐个体 {v['d_hist_indiv']:.3f}/{v['d_null_indiv']:.3f} = "
            f"**{v['ratio_indiv']:.2f}x**  个体离总体 {v['spread_indiv']:.2f}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
    )
    print("")
    print("判读（**事后，非预注册**）：")
    for k, v in res.items():
        print(
            f"  {k:18s} 分批/对照 {v['half_over_null']:.2f}x；"
            f"总体均值比 {v['ratio']:.2f}x，**逐个体比 {v['ratio_indiv']:.2f}x**"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""M6av：**纠缠熵就是信息** —— 用互信息重测"整合"，而不是线性相关（用户 2026-11-17）。

## 论点

> 纠缠熵本就是信息。

我今晚量两个轴用的都是**方差与线性相关**：
- **维度**：PR（协方差的参与比）—— 方差（二阶统计量）；
- **整合**：两半球平均发放的 Pearson 相关 —— 线性相关。

**两者都是信息的下界，不是信息本身。** 而互信息 `I(A;B)` 数的是**存在于关联里的比特**。

**这与今晚已踩到两次的坑同型**：线性读出把低维状态的信息低估了 5 倍（0.101→0.571）。
**"低整合"和"低维"可能同样是度量太粗。**

## 测什么

1. **两半球互信息** `I(A;B)`（分位数分箱），与**同 r 的高斯变量隐含值** `−½ln(1−r²)` 比；
   实测 > 高斯值 ⇒ **存在线性相关看不见的关联**；
2. **随机神经元对的平均互信息**，同样与高斯值比；
3. 两个 regime 各测一次。

## 判据

- **AV-0** 发放在窗内；
- **AV-1 互信息超过高斯界** ★★：至少一个 regime 的实测 MI > 高斯隐含值 ×1.5；
- **AV-2 排序可能反转**：若高维 regime 的 MI 高于全局 regime，则"高维不整合"是度量假象。
"""

from __future__ import annotations

# ruff: noqa: N803, N806
import argparse
import json
import math
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.hist_families import make_family_hist
from scripts.m6_pools import N, build_pools, run_spont

OUT = _pathlib.Path("sweep_results/m6av_mutual_info")
K, SIDE = 4, 32
T_HIST, T_SPONT = 1024, 2048
FAMS = list(range(8))
P_LONG = 0.5
ARMS = {
    "high_dim(gl=2)": {"field_radius": 2, "g_field": 0.0, "g_local": 2.0},
    "global": {"field_radius": 32, "g_field": 1.0, "g_local": 0.0},
}


def mi_binned(a: torch.Tensor, b: torch.Tensor, nb: int = 8) -> float:
    """两个一维样本的互信息（分位数分箱）。"""
    qa = torch.quantile(a.float(), torch.linspace(0, 1, nb + 1)[1:-1])
    qb = torch.quantile(b.float(), torch.linspace(0, 1, nb + 1)[1:-1])
    ia = torch.bucketize(a.float(), qa)
    ib = torch.bucketize(b.float(), qb)
    n = a.shape[0]
    joint = torch.zeros(nb, nb)
    for x, y in zip(ia.tolist(), ib.tolist(), strict=False):
        joint[x, y] += 1
    joint = joint / n
    pa = joint.sum(1, keepdim=True)
    pb = joint.sum(0, keepdim=True)
    outer = pa @ pb
    mask = joint > 0
    return float((joint[mask] * (joint[mask] / outer[mask]).log()).sum())


def gauss_mi(r: float) -> float:
    return -0.5 * math.log(max(1.0 - min(abs(r), 0.999999) ** 2, 1e-12))


@torch.no_grad()
def collect(seed: int, arm: dict, per_fam: int, dev: str):
    w_in, pre, wrec, pid = build_pools(seed, dev, K, 0.0, p_long=P_LONG)
    net = (w_in, pre, wrec, pid)
    states, frs = [], []
    for fam in FAMS:
        toks = make_family_hist(fam, seed * 701 + fam * 59 + 31, T_HIST, per_fam, dev)
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
        states.append(r.cpu())
        frs.append(float(r.mean()))
    return torch.cat(states), sum(frs) / len(frs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--per-fam", type=int, default=64)
    ap.add_argument("--pairs", type=int, default=128)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": {}}
    for name, arm in ARMS.items():
        rows = []
        for sd in range(args.seeds):
            S, fr = collect(sd, arm, args.per_fam, dev)
            h = ((torch.arange(N) % SIDE) >= 16).long()
            A, B = S[:, h == 0].mean(1), S[:, h == 1].mean(1)
            r_lin = float(torch.corrcoef(torch.stack([A, B]))[0, 1])
            mi_half = mi_binned(A, B)
            # 随机神经元对
            g = torch.Generator().manual_seed(sd + 61)
            mis, gs = [], []
            for _ in range(args.pairs):
                i = int(torch.randint(0, N, (1,), generator=g))
                j = int(torch.randint(0, N, (1,), generator=g))
                if i == j:
                    continue
                x, y = S[:, i], S[:, j]
                if float(x.std()) < 1e-9 or float(y.std()) < 1e-9:
                    continue
                mis.append(mi_binned(x, y, 6))
                gs.append(gauss_mi(float(torch.corrcoef(torch.stack([x, y]))[0, 1])))
            row = {
                "seed": sd,
                "firing": fr,
                "r_lin": r_lin,
                "mi_half": mi_half,
                "mi_half_gauss": gauss_mi(r_lin),
                "mi_pair": sum(mis) / max(len(mis), 1),
                "mi_pair_gauss": sum(gs) / max(len(gs), 1),
            }
            rows.append(row)
            print(
                f"  {name:16s} seed {sd}: 线性 r={r_lin:+.3f}  "
                f"半球 MI {mi_half:.4f}（高斯界 {row['mi_half_gauss']:.4f}，"
                f"{mi_half / max(row['mi_half_gauss'], 1e-9):.2f}×）  "
                f"对 MI {row['mi_pair']:.5f}（高斯 {row['mi_pair_gauss']:.5f}，"
                f"{row['mi_pair'] / max(row['mi_pair_gauss'], 1e-9):.2f}×）",
                flush=True,
            )
        m = lambda k, _r=rows: sum(r[k] for r in _r) / len(_r)  # noqa: E731
        v = {k: m(k) for k in rows[0]}
        v["rows"] = rows
        res["cells"][name] = v
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1, default=float),
            encoding="utf-8",
            newline=chr(10),
        )

    print("")
    print("判：")
    cc = res["cells"]
    ok = [k for k, v in cc.items() if 0.01 <= v["firing"] <= 0.20]
    print(f"  AV-0 发放在窗内：{len(ok)}/{len(cc)} => {'过' if ok else '不过 => 作废'}")
    best = (
        max(ok, key=lambda k: cc[k]["mi_half"] / max(cc[k]["mi_half_gauss"], 1e-9)) if ok else None
    )
    if best:
        v = cc[best]
        ratio = v["mi_half"] / max(v["mi_half_gauss"], 1e-9)
        print(
            f"  AV-1 互信息超过高斯界（>1.5×）：{best} 半球 MI 比 {ratio:.2f}× "
            f"=> {'过' if ratio > 1.5 else '不过'}"
        )
    for k, v in cc.items():
        print(
            f"    {k:16s} 半球 MI {v['mi_half']:.4f}  成对 MI {v['mi_pair']:.5f}  "
            f"线性 r {v['r_lin']:+.3f}"
        )
    if "high_dim(gl=2)" in cc and "global" in cc:
        print(
            f"  AV-2 排序：高维 MI {cc['high_dim(gl=2)']['mi_half']:.4f} vs "
            f"全局 {cc['global']['mi_half']:.4f} => "
            f"{'高维更整合（度量假象）' if cc['high_dim(gl=2)']['mi_half'] > cc['global']['mi_half'] else '全局更整合'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

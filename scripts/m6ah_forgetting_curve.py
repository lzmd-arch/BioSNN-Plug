"""M6ah：**遗忘曲线 —— 拟合时间常数 τ，并测它可不可调**。

## 用户定的原则（2026-11-16）

> 记忆衰减在真实生物是存在的（艾宾浩斯曲线），但千岁的记忆既然有稳定介质，
> 那么符合遗忘规律和比人类更好的记忆就不再冲突。

⇒ **可及性衰减 ≠ 信息丢失。** 生物里两者耦合是因为介质不稳；硅基介质稳定，可以解耦。
所以本实验**不把衰减当缺陷**，而是量它的**形状**与**可调性**。

## 方法

- 在 `t ∈ {64, 128, 256, 512, 1024, 2048}` 上量解码准确率（输入停止后的步数）；
- 拟合 `acc(t) = f + A·exp(−t/τ)`，`f` 取底板；τ 用网格搜索（最小化 SSE）；
- 报 `R²` 与 τ；并扫全局场强度，看 τ 是否随它变。

## 判据

- **AH-0** 发放在窗内；
- **AH-1 有一条指数遗忘曲线** ★：拟合 `R² > 0.9` 且 `τ` 有限；
- **AH-2 τ 可调** ★：不同 `g_field` 下 τ 的极差 > 2×。
"""

from __future__ import annotations

# ruff: noqa: N806
import argparse
import json
import math
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.hist_families import make_family_hist, spatial_bins
from scripts.instrument import majority_baseline, probe
from scripts.m6_pools import build_pools, run_spont

OUT = _pathlib.Path("sweep_results/m6ah_forgetting_curve")
K = 4
T_HIST = 1024
FAMS = list(range(8))


@torch.no_grad()
def decode_at(net, seed: int, t_spont: int, per_fam: int, g_field: float, dev: str):
    feats, labels, frs = [], [], []
    for fam in FAMS:
        toks = make_family_hist(fam, seed * 173 + fam * 29 + 3, T_HIST, per_fam, dev)
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
            t_spont=t_spont,
            net=net,
            g_field=g_field,
        )[0]
        feats.append(spatial_bins(r).cpu())
        labels.append(torch.full((r.shape[0],), fam))
        frs.append(float(r.mean()))
    X, Y = torch.cat(feats), torch.cat(labels)
    perm = torch.randperm(Y.shape[0], generator=torch.Generator().manual_seed(7 + seed))
    X, Y = X[perm], Y[perm]
    cut = int(Y.shape[0] * 0.75)
    acc, _ = probe(X, Y, cut, len(FAMS))
    return acc, majority_baseline(Y[:cut], Y[cut:], len(FAMS)), sum(frs) / len(frs)


def fit_tau(ts, accs, floor: float) -> tuple[float, float]:
    """网格搜索 τ（对数格），拟合 `acc = floor + A·exp(-t/τ)`；返回 `(τ, R²)`。"""
    ys = [a - floor for a in accs]
    ss_tot = sum((y - sum(ys) / len(ys)) ** 2 for y in ys) or 1e-12
    best_tau, best_r2 = float("nan"), -1e9
    for k in range(1, 400):
        tau = 10.0 * (1.05**k)
        es = [math.exp(-t / tau) for t in ts]
        se = sum(e * e for e in es) or 1e-12
        A = sum(y * e for y, e in zip(ys, es, strict=False)) / se
        resid = sum((y - A * e) ** 2 for y, e in zip(ys, es, strict=False))
        r2 = 1.0 - resid / ss_tot
        if r2 > best_r2:
            best_r2, best_tau = r2, tau
    return best_tau, best_r2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--ts", default="64,128,256,512,1024,2048")
    ap.add_argument("--g-fields", default="0.5,1.0")
    ap.add_argument("--per-fam", type=int, default=16)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    ts = [int(x) for x in args.ts.split(",")]
    res: dict = {"cells": {}}
    for gf in [float(x) for x in args.g_fields.split(",")]:
        curves, floors, frs = [], [], []
        for sd in range(args.seeds):
            net = build_pools(sd, dev, K, 0.0)
            accs_s, fl_s, fr_s = [], [], []
            for t in ts:
                acc, fl, fr = decode_at(net, sd, t, args.per_fam, gf, dev)
                accs_s.append(acc)
                fl_s.append(fl)
                fr_s.append(fr)
            curves.append(accs_s)
            floors.append(sum(fl_s) / len(fl_s))
            frs.append(sum(fr_s) / len(fr_s))
            print(
                f"  g_field={gf:g} seed {sd}: "
                + "  ".join(f"t={t}:{a:.3f}" for t, a in zip(ts, accs_s, strict=False)),
                flush=True,
            )
        mean_curve = [sum(c[i] for c in curves) / len(curves) for i in range(len(ts))]
        fl = sum(floors) / len(floors)
        tau, r2 = fit_tau(ts, mean_curve, fl)
        key = f"gf={gf:g}"
        res["cells"][key] = {
            "g_field": gf,
            "ts": ts,
            "curve": mean_curve,
            "floor": fl,
            "tau": tau,
            "r2": r2,
            "firing": sum(frs) / len(frs),
            "rows": curves,
        }
        print(
            f"  g_field={gf:g} 汇总：{', '.join(f'{a:.3f}' for a in mean_curve)}  "
            f"τ={tau:.0f} 步  R²={r2:.3f}  发放 {sum(frs) / len(frs):.4f}",
            flush=True,
        )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )

    print("")
    print("判：")
    cc = res["cells"]
    ok = [k for k, v in cc.items() if 0.01 <= v["firing"] <= 0.20]
    print(f"  AH-0 发放在窗内：{len(ok)}/{len(cc)} 档 => {'过' if ok else '不过 => 作废'}")
    if not ok:
        return 0
    fits = [k for k in ok if cc[k]["r2"] > 0.9 and math.isfinite(cc[k]["tau"])]
    print(
        f"  AH-1 有指数遗忘曲线（R²>0.9）：{len(fits)}/{len(ok)} 档 => {'过' if fits else '不过'}"
    )
    taus = [cc[k]["tau"] for k in ok if math.isfinite(cc[k]["tau"])]
    if len(taus) >= 2:
        spread = max(taus) / max(min(taus), 1e-9)
        print(
            f"  AH-2 τ 可调（极差 > 2×）：{[round(t) for t in taus]} 极差 {spread:.2f}× "
            f"=> {'过' if spread > 2 else '不过'}"
        )
    else:
        print("  AH-2 τ 可调：档数不足")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

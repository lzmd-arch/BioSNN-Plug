"""M6q：慢增益 —— 让"个体"有一个比测量窗更慢的变量。

预注册：`research/sel/prereg-m6q-slow-gain-v1.md`

M6p 的判决是"总体有分布、个体没状态"。诊断是**模型里没有任何量的时间常数 ≥ 测量窗**：
`v` τ=20 步、`ref` ~2 步、`gain` 在 eta=2 时 τ≈2 步、eta=-1 时干脆不施加。
测量窗在输入停止后第 192–255 步。

本轮只改一个数：`eta_gain` 的剂量，把增益时间常数从 ~20 步拉到 ~800 步
（`τ ≈ 4/eta`）。**判据口径全部换成逐个体** —— 那正是 M6p 失败的一列。
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

OUT = _pathlib.Path("sweep_results/m6q_slow_gain")
K = 4
ETA_LADDER = [0.2, 0.05, 0.02, 0.01, 0.005]
TAU = {e: max(4.0 / e, 10.0) for e in ETA_LADDER}


def resid(R: torch.Tensor) -> torch.Tensor:
    return R - R.mean(1, keepdim=True)


def spearman(xs, ys) -> float:
    """两个等长序列的秩相关（无并列时足够）。"""

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=False))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / max(vx * vy, 1e-12)


@torch.no_grad()
def stats(seed: int, g_inh: float, eta: float, dev: str, scale: float, p: float) -> dict:
    """一个 (seed, 臂) 的全部量。**逐个体为主，总体只作附注。**"""
    pid = pool_id(dev, K)
    Rb, Rs = [], []
    for alpha, hs in ((A_LOW, 100 + seed), (A_HIGH, 100 + seed), (A_LOW, 900 + seed)):
        r, snaps, _e = run_spont(seed, alpha, hs, scale, p, dev, K, 0.0, g_inh, eta)
        Rb.append(pool_rates(r, pid, K))
        Rs.append(pool_rates(snaps[:, :, 32:].mean(2), pid, K))  # 稳态窗**后** 32 步
    Ra, Rb2, Rn = Rb
    Sa, Sb2, _Sn = Rs
    n = Ra.shape[0]
    d_hi = sum(rel(resid(Ra)[i], resid(Rb2)[i]) for i in range(n)) / n
    d_nu = sum(rel(resid(Ra)[i], resid(Rn)[i]) for i in range(n)) / n
    p_hi = rel(resid(Ra.mean(0, keepdim=True))[0], resid(Rb2.mean(0, keepdim=True))[0])
    p_nu = rel(resid(Ra.mean(0, keepdim=True))[0], resid(Rn.mean(0, keepdim=True))[0])
    # Q-4：稳态窗内前 32（Sa）vs 后 32（Sb2），逐个体
    st = sum(rel(resid(Sa)[i], resid(Sb2)[i]) for i in range(n)) / n
    return {
        "firing": float(Ra.mean()),
        "d_hist_ind": d_hi,
        "d_null_ind": d_nu,
        "ratio_ind": d_hi / max(d_nu, 1e-12),
        "d_hist_pop": p_hi,
        "d_null_pop": p_nu,
        "ratio_pop": p_hi / max(p_nu, 1e-12),
        "stationarity": st,
        "settled": 1 if st < d_hi else 0,
        "win": 1 if d_hi > d_nu else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--ginhs", default="0,2")
    ap.add_argument("--scale", type=float, default=0.7)
    ap.add_argument("--p", type=float, default=0.02)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    arms: list[tuple[float, float, str]] = [(2.0, -1.0, "基线(eta=-1)")]
    for g in [float(x) for x in args.ginhs.split(",")]:
        for e in ETA_LADDER:
            arms.append((g, e, f"eta={e:g}"))

    res: dict = {"arms": {}, "n_seeds": args.seeds, "K": K, "tau": TAU}
    for g, e, tag in arms:
        rows = [stats(sd, g, e, dev, args.scale, args.p) for sd in range(args.seeds)]
        agg = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
        agg["wins"] = sum(int(r["win"]) for r in rows)
        agg["settled"] = sum(int(r["settled"]) for r in rows)
        agg["rows"] = rows
        agg["g_inh"], agg["eta"], agg["tau"] = g, e, TAU.get(e, float("nan"))
        res["arms"][f"g={g:g}/{tag}"] = agg
        print(
            f"  g={g:g} {tag:12s} τ={agg['tau']:6.1f}  发放 {agg['firing']:.4f}  "
            f"**逐个体 {agg['ratio_ind']:.2f}x({agg['wins']}/{args.seeds})**  "
            f"总体 {agg['ratio_pop']:.2f}x  停住 {agg['settled']}/{args.seeds}",
            flush=True,
        )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )

    print("")
    print("判：")
    cc = res["arms"]
    base = cc["g=2/基线(eta=-1)"]
    q0 = abs(base["ratio_pop"] - 3.14) <= 0.15 and abs(base["ratio_ind"] - 1.02) <= 0.05
    print(
        f"  Q-0 仪器复现 M6p（总体 3.14±0.15、逐个体 1.02±0.05）："
        f"{base['ratio_pop']:.2f} / {base['ratio_ind']:.2f} => {'过' if q0 else '不过 => 作废'}"
    )
    if not q0:
        return 0
    lad = [k for k in cc if "eta=" in k]
    ok1 = [k for k in lad if 0.01 <= cc[k]["firing"] <= 0.20]
    print(f"  Q-1 发放在窗内：{len(ok1)}/{len(lad)} 档 => {'过' if ok1 else '不过 => 作废'}")
    if not ok1:
        return 0
    kb = max(ok1, key=lambda k: cc[k]["ratio_ind"])
    v = cc[kb]
    q2 = v["ratio_ind"] > 1.5 and v["wins"] >= 5
    print(
        f"  Q-2 逐个体带历史（>1.5, ≥5/6）：{kb} **{v['ratio_ind']:.2f}x "
        f"{v['wins']}/{v['n_seeds']}** => {'过' if q2 else '不过'}"
    )
    for g in [float(x) for x in args.ginhs.split(",")]:
        ks = [k for k in ok1 if cc[k]["g_inh"] == g]
        ks.sort(key=lambda k: -cc[k]["tau"])
        if len(ks) < 3:
            continue
        rho = spearman([cc[k]["tau"] for k in ks], [cc[k]["ratio_ind"] for k in ks])
        print(
            f"  Q-3 剂量–反应 (g={g:g})："
            f"{[round(cc[k]['ratio_ind'], 2) for k in ks]}  Spearman={rho:+.2f} => "
            f"{'过' if rho >= 0.9 else '不过'}"
        )
    q4 = v["settled"] >= 5
    print(f"  Q-4 状态在窗内停住：{kb} {v['settled']}/{v['n_seeds']} => {'过' if q4 else '不过'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

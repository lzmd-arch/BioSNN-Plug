"""M6r：历史时长 × 增益接线 —— 两个不换机制的候选。

预注册：`research/sel/prereg-m6r-history-wiring-v1.md`

M6q 给出方向正确但饱和的剂量–反应（0.87 → 1.33 → 1.35）。
本轮问它卡在 1.35 是因为**慢变量攒得不够久**（`t_hist`），
还是因为**增益进不了动力学的主干**（它只乘 `(pace − 场)`，不乘 `rec`）。
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

OUT = _pathlib.Path("sweep_results/m6r_history_wiring")
K = 4
G_INH = 0.0


def resid(R: torch.Tensor) -> torch.Tensor:
    return R - R.mean(1, keepdim=True)


def mean(rows, key):
    return sum(r[key] for r in rows) / len(rows)


@torch.no_grad()
def stats(seed: int, eta: float, t_hist: int, route: str, dev: str) -> dict:
    pid = pool_id(dev, K)
    Rb, Ss = [], []
    for alpha, hs in ((A_LOW, 100 + seed), (A_HIGH, 100 + seed), (A_LOW, 900 + seed)):
        r, snaps, early = run_spont(
            seed, alpha, hs, 0.7, 0.02, dev, K, 0.0, G_INH, eta, t_hist, route
        )
        Rb.append(pool_rates(r, pid, K))
        Ss.append(pool_rates(snaps[:, :, 32:].mean(2), pid, K))
    Ra, Rb2, Rn = Rb
    Sa, Sb2, _Sn = Ss
    n = Ra.shape[0]
    d_hi = sum(rel(resid(Ra)[i], resid(Rb2)[i]) for i in range(n)) / n
    d_nu = sum(rel(resid(Ra)[i], resid(Rn)[i]) for i in range(n)) / n
    st = sum(rel(resid(Sa)[i], resid(Sb2)[i]) for i in range(n)) / n
    return {
        "firing": float(Ra.mean()),
        "d_hist_ind": d_hi,
        "d_null_ind": d_nu,
        "stationarity": st,
        "win": 1 if d_hi > d_nu else 0,
        "settled": 1 if st < d_hi else 0,
        "early_firing": float(early.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--t-hists", default="1024,4096")
    ap.add_argument("--routes", default="field,all")
    ap.add_argument("--etas", default="0.01,0.005")
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"arms": {}, "n_seeds": args.seeds, "K": K, "g_inh": G_INH}
    for t_hist in [int(x) for x in args.t_hists.split(",")]:
        for route in args.routes.split(","):
            for eta in [float(x) for x in args.etas.split(",")]:
                rows = [stats(sd, eta, t_hist, route, dev) for sd in range(args.seeds)]
                key = f"T={t_hist}/r={route}/eta={eta:g}"
                v = {
                    "t_hist": t_hist,
                    "route": route,
                    "eta": eta,
                    "firing": mean(rows, "firing"),
                    "d_hist_ind": mean(rows, "d_hist_ind"),
                    "d_null_ind": mean(rows, "d_null_ind"),
                    "stationarity": mean(rows, "stationarity"),
                    "wins": sum(int(r["win"]) for r in rows),
                    "settled": sum(int(r["settled"]) for r in rows),
                    "rows": rows,
                }
                v["ratio_ind"] = v["d_hist_ind"] / max(v["d_null_ind"], 1e-12)
                res["arms"][key] = v
                print(
                    f"  {key:26s} 发放 {v['firing']:.4f}  "
                    f"**逐个体 {v['ratio_ind']:.2f}x({v['wins']}/{args.seeds})**  "
                    f"停住 {v['settled']}/{args.seeds}",
                    flush=True,
                )
                (args.out / "result.json").write_text(
                    json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
                )

    print("")
    print("判：")
    cc = res["arms"]
    base = cc["T=1024/r=field/eta=0.01"]
    r0 = abs(base["ratio_ind"] - 1.33) <= 0.10 and base["wins"] >= 4
    print(
        f"  R-0 仪器复现 M6q（1.33±0.10、≥4/6）：{base['ratio_ind']:.2f} "
        f"{base['wins']}/{args.seeds} => {'过' if r0 else '不过 => 作废'}"
    )
    if not r0:
        return 0
    ok = [k for k, v in cc.items() if 0.01 <= v["firing"] <= 0.20]
    print(f"  R-1 发放在窗内：{len(ok)}/{len(cc)} 档 => {'过' if ok else '不过 => 作废'}")
    if not ok:
        return 0
    etas = [float(x) for x in args.etas.split(",")]
    hits2 = 0
    for e in etas:
        a, b = cc[f"T=1024/r=field/eta={e:g}"], cc[f"T=4096/r=field/eta={e:g}"]
        hit = b["ratio_ind"] > a["ratio_ind"]
        hits2 += int(hit)
        print(
            f"  R-2 历史时长 eta={e:g}：1024 {a['ratio_ind']:.2f}x → 4096 {b['ratio_ind']:.2f}x "
            f"=> {'同向' if hit else '反向'}"
        )
    print(f"       两个 eta 同向 {hits2}/{len(etas)} => {'过' if hits2 == len(etas) else '不过'}")
    hits3 = 0
    for e in etas:
        a, b = cc[f"T=1024/r=field/eta={e:g}"], cc[f"T=1024/r=all/eta={e:g}"]
        hit = b["ratio_ind"] > a["ratio_ind"]
        hits3 += int(hit)
        print(
            f"  R-3 接线 eta={e:g}：field {a['ratio_ind']:.2f}x → all {b['ratio_ind']:.2f}x "
            f"=> {'同向' if hit else '反向'}"
        )
    print(f"       两个 eta 同向 {hits3}/{len(etas)} => {'过' if hits3 == len(etas) else '不过'}")
    kb = max(ok, key=lambda k: cc[k]["ratio_ind"])
    v = cc[kb]
    r4 = v["ratio_ind"] > 1.5 and v["wins"] >= 5
    print(
        f"  R-4 过线（>1.5, ≥5/6）：{kb} **{v['ratio_ind']:.2f}x {v['wins']}/{args.seeds}** "
        f"=> {'过' if r4 else '不过'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

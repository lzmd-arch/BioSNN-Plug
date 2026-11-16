"""M6q 判读：从已存的逐种子行重算，**比值取"先平均再相除"**。

`m6q_slow_gain.py` 的聚合用的是"逐种子比值再平均"，那是错的：
它被分母小的种子主导。M6p/M7 的口径是 `mean(d_hist)/mean(d_null)`。
这里按正确口径重算，判据本身一个字不改（预注册 v26）。
"""

from __future__ import annotations

import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))


def spearman(xs, ys) -> float:
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


def mean(rows, key):
    return sum(r[key] for r in rows) / len(rows)


def main() -> int:
    path = _pathlib.Path(
        _sys.argv[1] if len(_sys.argv) > 1 else "sweep_results/m6q_slow_gain/result.json"
    )
    d = json.loads(path.read_text(encoding="utf-8"))
    arms = d["arms"]
    rows = {}
    for k, v in arms.items():
        rows[k] = {
            "g": v["g_inh"],
            "eta": v["eta"],
            "tau": v["tau"],
            "firing": v["firing"],
            "d_hist_ind": mean(v["rows"], "d_hist_ind"),
            "d_null_ind": mean(v["rows"], "d_null_ind"),
            "d_hist_pop": mean(v["rows"], "d_hist_pop"),
            "d_null_pop": mean(v["rows"], "d_null_pop"),
            "stationarity": mean(v["rows"], "stationarity"),
            "settled": sum(1 for r in v["rows"] if r["settled"]),
            "wins": sum(1 for r in v["rows"] if r["win"]),
            "n": len(v["rows"]),
        }
        rows[k]["ratio_ind"] = rows[k]["d_hist_ind"] / max(rows[k]["d_null_ind"], 1e-12)
        rows[k]["ratio_pop"] = rows[k]["d_hist_pop"] / max(rows[k]["d_null_pop"], 1e-12)
    for k, v in rows.items():
        print(
            f"  {k:26s} 发放 {v['firing']:.4f}  **逐个体 {v['ratio_ind']:.2f}x"
            f"({v['wins']}/{v['n']})**  总体 {v['ratio_pop']:.2f}x  "
            f"停住 {v['settled']}/{v['n']}"
        )
    print("")
    print("判：")
    base = rows["g=2/基线(eta=-1)"]
    q0 = abs(base["ratio_pop"] - 3.14) <= 0.15 and abs(base["ratio_ind"] - 1.02) <= 0.05
    print(
        f"  Q-0 仪器复现 M6p（总体 3.14±0.15、逐个体 1.02±0.05）："
        f"{base['ratio_pop']:.2f} / {base['ratio_ind']:.2f} => {'过' if q0 else '不过 => 作废'}"
    )
    if not q0:
        return 0
    lad = [k for k in rows if rows[k]["eta"] >= 0]
    ok1 = [k for k in lad if 0.01 <= rows[k]["firing"] <= 0.20]
    print(f"  Q-1 发放在窗内：{len(ok1)}/{len(lad)} 档 => {'过' if ok1 else '不过 => 作废'}")
    if not ok1:
        return 0
    kb = max(ok1, key=lambda k: rows[k]["ratio_ind"])
    v = rows[kb]
    q2 = v["ratio_ind"] > 1.5 and v["wins"] >= 5
    print(
        f"  Q-2 逐个体带历史（>1.5, ≥5/6）：{kb} **{v['ratio_ind']:.2f}x "
        f"{v['wins']}/{v['n']}** => {'过' if q2 else '不过'}"
    )
    for g in sorted({rows[k]["g"] for k in ok1}):
        ks = sorted([k for k in ok1 if rows[k]["g"] == g], key=lambda k: -rows[k]["tau"])
        if len(ks) < 3:
            continue
        rho = spearman([rows[k]["tau"] for k in ks], [rows[k]["ratio_ind"] for k in ks])
        print(
            f"  Q-3 剂量–反应 (g={g:g})：{[round(rows[k]['ratio_ind'], 2) for k in ks]}  "
            f"Spearman={rho:+.2f} => {'过' if rho >= 0.9 else '不过'}"
        )
    q4 = v["settled"] >= 5
    print(f"  Q-4 状态在窗内停住：{kb} {v['settled']}/{v['n']} => {'过' if q4 else '不过'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

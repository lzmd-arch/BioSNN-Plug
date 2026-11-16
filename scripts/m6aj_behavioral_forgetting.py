"""M6aj：**行为的遗忘曲线** —— 闭环能记住起点多久。

## 发现

`M6ai` 在闭环 128 步时给 2.19×（行为带历史），512 步时塌到 0.52–0.97×。
**行为层的记忆随时间消失** —— 和 `M6ah` 的神经层遗忘曲线是同一件事的两个投影。

## 用户的原则

> 千岁的记忆既然有稳定介质，那么符合遗忘规律和比人类更好的记忆就不再冲突。

所以本实验**不把行为遗忘当缺陷**，而是量它的**时间常数**：
闭环能记住起点多久，以及这个常数可不可调。

## 判据

- **AJ-0** 各档闭环都不熄火不饱和（≥80% 步在窗内）；
- **AJ-1 有行为遗忘曲线** ★：`ratio(短) > 1.5` 且 `ratio(长) < 1.2`（真的掉了）；
- 报 `τ_behavior`（在 ratio 上拟合单指数到 1.0 的回落）。
"""

from __future__ import annotations

import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.hist_families import make_family_hist
from scripts.m6_pools import build_pools
from scripts.m6ai_closed_loop import entropy_norm, rate_dist, run_closed

OUT = _pathlib.Path("sweep_results/m6aj_behavioral_forgetting")
K = 4
T_HIST = 1024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--ts", default="32,64,128,256,512")
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    ts = [int(x) for x in args.ts.split(",")]
    res: dict = {"cells": {}}
    for t in ts:
        rows = []
        for sd in range(args.seeds):
            net = build_pools(sd, dev, K, 0.0)
            ha = make_family_hist(0, sd * 97 + 1, T_HIST, 1, dev)
            ha2 = make_family_hist(0, sd * 97 + 11, T_HIST, 1, dev)
            hb = make_family_hist(5, sd * 97 + 2, T_HIST, 1, dev)
            sa, ra = run_closed(net, sd, ha, dev, t)
            _s2, ra2 = run_closed(net, sd, ha2, dev, t)
            _sb, rb = run_closed(net, sd, hb, dev, t)
            inwin = sum(1 for r in ra if 0.01 <= r <= 0.20) / len(ra)
            dh, dn = rate_dist(ra, rb), rate_dist(ra, ra2)
            rows.append(
                {
                    "seed": sd,
                    "d_hist": dh,
                    "d_null": dn,
                    "ratio": dh / max(dn, 1e-12),
                    "in_window": inwin,
                    "entropy": entropy_norm(sa),
                }
            )
        m = lambda k, _r=rows: sum(r[k] for r in _r) / len(_r)  # noqa: E731
        ratio = m("d_hist") / max(m("d_null"), 1e-12)
        res["cells"][f"t={t}"] = {
            "t": t,
            "ratio": ratio,
            "d_hist": m("d_hist"),
            "d_null": m("d_null"),
            "in_window": m("in_window"),
            "entropy": m("entropy"),
            "wins": sum(1 for r in rows if r["d_hist"] > r["d_null"]),
            "rows": rows,
        }
        v = res["cells"][f"t={t}"]
        print(
            f"  t={t:4d}: 历史 {v['d_hist']:.4f} / 对照 {v['d_null']:.4f} = "
            f"**{ratio:.2f}x**（{v['wins']}/{args.seeds}）窗内 {v['in_window']:.2f} "
            f"熵 {v['entropy']:.2f}",
            flush=True,
        )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )

    print("")
    print("判：")
    cc = res["cells"]
    ks = sorted(cc, key=lambda k: cc[k]["t"])
    ok0 = all(cc[k]["in_window"] >= 0.8 for k in ks)
    print(f"  AJ-0 各档不熄火不饱和：{'过' if ok0 else '不过 => 作废'}")
    if not ok0:
        return 0
    r_short, r_long = cc[ks[0]]["ratio"], cc[ks[-1]]["ratio"]
    aj1 = r_short > 1.5 and r_long < 1.2
    print(
        f"  AJ-1 有行为遗忘曲线：短 {r_short:.2f}x → 长 {r_long:.2f}x => {'过' if aj1 else '不过'}"
    )
    parts = []
    for k in ks:
        parts.append(str(cc[k]["t"]) + ":" + format(cc[k]["ratio"], ".2f"))
    print("       曲线：" + ", ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

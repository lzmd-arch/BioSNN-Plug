"""M6aa：**用长程兴奋性连接买整合，而不是用全局标量**。

## 为什么

M6y 的结论是"高维与整合反相关"：局部场给 PR 39.8 但整合只有 0.12；
全局场给整合 0.83 但 PR 只有 2.59。**因为整合只有一条通路：那个全局标量。**

**可预见的替代**：整合可以由**长程兴奋性投射**承载（皮层就是这样：抑制是局部的，
整合靠远距离兴奋）。这样局部抑制保住维度，长程兴奋补上整合。

## 判据

- **AA-0** 发放在窗内；
- **AA-1 三者兼得** ★：`field_radius=2` 下，存在 `p_long` 使
  **PR > 10 且 整合 > 0.3 且 分离 > 0.5**；
- **AA-2 剂量–反应** ★：整合随 `p_long` 单调上升。
"""

from __future__ import annotations

# ruff: noqa: N806
import argparse
import itertools
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.hist_families import make_family_hist
from scripts.m6_pools import N, build_pools, run_spont
from scripts.m6v_dimension import pr_of

OUT = _pathlib.Path("sweep_results/m6aa_longrange")
K, SIDE = 4, 32
T_HIST, T_SPONT = 1024, 2048
RADIUS = 2


@torch.no_grad()
def cell(seed: int, p_long: float, n_fam: int, per_fam: int, dev: str) -> dict:
    w_in, pre, wrec, pid = build_pools(seed, dev, K, 0.0, p_long=p_long)
    net = (w_in, pre, wrec, pid)
    states = []
    for fam in range(n_fam):
        toks = make_family_hist(fam, seed * 601 + fam * 23 + 9, T_HIST, per_fam, dev)
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
            field_radius=RADIUS,
            g_field=0.0,
            g_local=1.0,
        )[0]
        states.append(r.cpu())
    S = torch.cat(states)
    h = ((torch.arange(N) % SIDE) >= 16).long()
    A, B = S[:, h == 0], S[:, h == 1]
    a, b = A.mean(1) - A.mean(), B.mean(1) - B.mean()
    integ = abs(float((a * b).sum() / (a.norm() * b.norm() + 1e-12)))
    segs = [float((H.std(1) / (H.mean(1) + 1e-9)).mean()) for H in (A, B)]
    return {"pr": pr_of(S), "integ": integ, "seg": sum(segs) / 2, "firing": float(S.mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--plongs", default="0,0.125,0.25,0.5")
    ap.add_argument("--n-fam", type=int, default=4)
    ap.add_argument("--per-fam", type=int, default=16)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": {}, "radius": RADIUS}
    for pl in [float(x) for x in args.plongs.split(",")]:
        rows = [cell(sd, pl, args.n_fam, args.per_fam, dev) for sd in range(args.seeds)]
        m = lambda k, _r=rows: sum(r[k] for r in _r) / len(_r)  # noqa: E731
        key = f"p_long={pl:g}"
        res["cells"][key] = {
            "p_long": pl,
            "pr": m("pr"),
            "integ": m("integ"),
            "seg": m("seg"),
            "firing": m("firing"),
            "rows": rows,
        }
        v = res["cells"][key]
        print(
            f"  p_long={pl:5.3f}: PR {v['pr']:6.2f}  整合 {v['integ']:.3f}  "
            f"分离 {v['seg']:.3f}  发放 {v['firing']:.4f}",
            flush=True,
        )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )

    print("")
    print("判：")
    cc = res["cells"]
    ok = [k for k, v in cc.items() if 0.01 <= v["firing"] <= 0.20]
    print(f"  AA-0 发放在窗内：{len(ok)}/{len(cc)} 档 => {'过' if ok else '不过 => 作废'}")
    if not ok:
        return 0
    hit = [k for k in ok if cc[k]["pr"] > 10 and cc[k]["integ"] > 0.3 and cc[k]["seg"] > 0.5]
    print(
        f"  AA-1 三者兼得（PR>10 且 整合>0.3 且 分离>0.5）：{hit or '无'} "
        f"=> {'过' if hit else '不过'}"
    )
    ks = sorted(ok, key=lambda k: cc[k]["p_long"])
    vals = [cc[k]["integ"] for k in ks]
    mono = all(x <= y + 1e-9 for x, y in itertools.pairwise(vals))
    print(
        f"  AA-2 整合随 p_long {[round(x, 3) for x in vals]} => {'单调上升' if mono else '不单调'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

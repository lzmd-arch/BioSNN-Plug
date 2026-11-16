"""M6y：**双尺度场** —— 全局给整合，局部给分离，两者不冲突。

## 假设（用户 2026-11-16 提出）

M11 的反常是：切断跨半球突触后**整合反而更高**。解释是**整合只由全局场承载**，
剪掉突触也没用；而**分离只由局部场承载**。一次只有一个场，所以两者不可兼得。

**两个场叠加**应该给出：全局场 → 长程相干（整合），局部场 → 空间分化（分离）。
这恰好是 M11 那条判据要的"整合高**且**分离保住"，单场架构在原理上做不到。

## 判据

- **Y-0** 发放在窗内；
- **Y-1 两者兼得** ★：存在某档 **PR > 10 且 整合 > 0.3 且 分离 > 0.5**；
- **Y-2 全局场负责整合** ★：固定 `g_local` 时，整合随 `g_field` 单调上升。
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

OUT = _pathlib.Path("sweep_results/m6y_two_scale")
K, SIDE = 4, 32
T_HIST, T_SPONT = 1024, 2048
RADIUS = 2


def halves() -> torch.Tensor:
    return ((torch.arange(N) % SIDE) >= 16).long()


@torch.no_grad()
def cell(seed: int, g_field: float, g_local: float, n_fam: int, per_fam: int, dev: str) -> dict:
    net = build_pools(seed, dev, K, 0.0)
    states = []
    for fam in range(n_fam):
        toks = make_family_hist(fam, seed * 503 + fam * 19 + 5, T_HIST, per_fam, dev)
        r, _, _ = run_spont(
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
            g_field=g_field,
            g_local=g_local,
        )
        states.append(r.cpu())
    S = torch.cat(states)
    h = halves()
    A, B = S[:, h == 0], S[:, h == 1]
    a = A.mean(1) - A.mean()
    b = B.mean(1) - B.mean()
    integ = abs(float((a * b).sum() / (a.norm() * b.norm() + 1e-12)))
    segs = []
    for H in (A, B):
        segs.append(float((H.std(1) / (H.mean(1) + 1e-9)).mean()))
    return {
        "pr": pr_of(S),
        "integ": integ,
        "seg": sum(segs) / 2,
        "firing": float(S.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--gf", default="0,0.5,1.0")
    ap.add_argument("--gl", default="0,0.5,1.0")
    ap.add_argument("--n-fam", type=int, default=4)
    ap.add_argument("--per-fam", type=int, default=16)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": {}, "radius": RADIUS}
    for gl in [float(x) for x in args.gl.split(",")]:
        for gf in [float(x) for x in args.gf.split(",")]:
            rows = [cell(sd, gf, gl, args.n_fam, args.per_fam, dev) for sd in range(args.seeds)]
            m = lambda k, _r=rows: sum(r[k] for r in _r) / len(_r)  # noqa: E731
            key = f"gf={gf:g}/gl={gl:g}"
            res["cells"][key] = {
                "g_field": gf,
                "g_local": gl,
                "pr": m("pr"),
                "integ": m("integ"),
                "seg": m("seg"),
                "firing": m("firing"),
                "rows": rows,
            }
            v = res["cells"][key]
            print(
                f"  gf={gf:3.1f} gl={gl:3.1f}: PR {v['pr']:6.2f}  整合 {v['integ']:.3f}  "
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
    print(f"  Y-0 发放在窗内：{len(ok)}/{len(cc)} 档 => {'过' if ok else '不过 => 作废'}")
    if not ok:
        return 0
    hit = [k for k in ok if cc[k]["pr"] > 10 and cc[k]["integ"] > 0.3 and cc[k]["seg"] > 0.5]
    print(
        f"  Y-1 三者兼得（PR>10 且 整合>0.3 且 分离>0.5）：{hit or '无'} "
        f"=> {'过' if hit else '不过'}"
    )
    for gl in sorted({cc[k]["g_local"] for k in ok}):
        ks = sorted([k for k in ok if cc[k]["g_local"] == gl], key=lambda k: cc[k]["g_field"])
        if len(ks) < 3:
            continue
        vals = [cc[k]["integ"] for k in ks]
        mono = all(x <= y + 1e-9 for x, y in itertools.pairwise(vals))
        print(
            f"  Y-2 gl={gl:g}：整合随 gf {[round(x, 3) for x in vals]} "
            f"=> {'单调上升' if mono else '不单调'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

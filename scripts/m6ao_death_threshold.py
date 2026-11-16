"""M6ao：**死亡阈值** —— 内增益控制器在多大损伤下会饱和。

## 从 M6an 的失败推出来

存续量乘性机制边际不稳定 ⇒ **系统已有的边界是内增益稳态控制器**
（`gain ← gain·exp(η(0.08 − f_i))`，钳在 [0.2, 5.0]）。

**它好到从不失效 ⇒ 无脆性。** 但控制器有**钳位** —— 损伤超过它的调节能力就会饱和，
那之后**发放会离开工作窗**。**那就是这个系统真正的死亡阈值。**

## 判据

- **AO-0** 无损伤时发放在窗内；
- **AO-1 有阈值** ★：存在一个损伤比例，使发放离开 [0.01, 0.20]；
- **AO-2 单调** ★：发放随损伤单调下降（控制器逐步耗尽）。
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.hist_families import make_family_hist
from scripts.m6_pools import N, build_pools, run_spont

OUT = _pathlib.Path("sweep_results/m6ao_death_threshold")
K = 4
T_HIST, T_SPONT = 1024, 2048


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--fracs", default="0,0.1,0.25,0.5,0.75,0.9,0.98")
    ap.add_argument("--per", type=int, default=16)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": {}}
    for fr in [float(x) for x in args.fracs.split(",")]:
        rows = []
        for sd in range(args.seeds):
            net = build_pools(sd, dev, K, 0.0)
            toks = make_family_hist(0, sd * 83 + 5, T_HIST, args.per, dev)
            g = torch.Generator().manual_seed(sd + 401)
            keep = round((1.0 - fr) * N)
            mask = torch.zeros(N, device=dev)
            mask[torch.randperm(N, generator=g).to(dev)[:keep]] = 1.0
            r = run_spont(
                sd,
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
                alive=mask,
            )[0]
            rows.append({"seed": sd, "firing": float(r.mean())})
        m = lambda k, _r=rows: sum(r[k] for r in _r) / len(_r)  # noqa: E731
        res["cells"][f"lesion={fr:g}"] = {
            "frac": fr,
            "firing": m("firing"),
            "rows": rows,
        }
        v = res["cells"][f"lesion={fr:g}"]
        ok = 0.01 <= v["firing"] <= 0.20
        print(
            f"  损伤 {fr:5.0%}: 发放 {v['firing']:.4f} {'✅窗内' if ok else '❌出窗'}", flush=True
        )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )

    print("")
    print("判：")
    cc = res["cells"]
    ks = sorted(cc, key=lambda k: cc[k]["frac"])
    f0 = cc[ks[0]]["firing"]
    print(f"  AO-0 无损伤在窗内：{f0:.4f} => {'过' if 0.01 <= f0 <= 0.20 else '不过 => 作废'}")
    bad = [k for k in ks if not (0.01 <= cc[k]["firing"] <= 0.20)]
    print(
        f"  AO-1 有死亡阈值：第一档出窗 = {cc[bad[0]]['frac']:.0%}"
        if bad
        else "  AO-1 无阈值（全部在窗内）"
    )
    vals = [cc[k]["firing"] for k in ks]
    mono = all(a >= b - 1e-9 for a, b in itertools.pairwise(vals))
    print(f"  AO-2 单调？{[round(v, 4) for v in vals]} => {'是' if mono else '否'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

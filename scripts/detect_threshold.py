"""可检测下界：d′，以及"数量通道是否比强度通道更灵敏"。

预注册：`research/sel/prereg-detect-threshold-v1.md`

AUC 只回答"有没有信号"；d′ 回答"信号相对自身噪声有多大"。
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

from scripts.damage_awareness import measure

OUT = _pathlib.Path("sweep_results/detect_threshold")


def dprime(a: list[float], b: list[float]) -> float:
    """`a` = 损伤组，`b` = 完好组。"""
    n = len(a)
    m = len(b)
    ma, mb = sum(a) / n, sum(b) / m
    va = sum((x - ma) ** 2 for x in a) / max(n - 1, 1)
    vb = sum((x - mb) ** 2 for x in b) / max(m - 1, 1)
    pooled = math.sqrt(max((va + vb) / 2.0, 1e-24))
    return abs(ma - mb) / pooled


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--fracs", default="0.01,0.02,0.05,0.1,0.25,0.5")
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda"
    fracs = [float(x) for x in args.fracs.split(",")]
    res: dict = {"n_seeds": args.seeds, "cells": {}}

    base = [measure(sd, 0.0, False, dev) for sd in range(args.seeds)]
    fr0 = [b[0] for b in base]
    si0 = [b[1] for b in base]

    for frac in fracs:
        les = [measure(sd, frac, False, dev) for sd in range(args.seeds)]
        fr1 = [b[0] for b in les]
        si1 = [b[1] for b in les]
        d_fr = dprime(fr1, fr0)
        d_si = dprime(si1, si0)
        res["cells"][f"{frac:g}"] = {
            "d_firing": d_fr,
            "d_silent": d_si,
            "firing_intact": sum(fr0) / len(fr0),
            "firing_lesion": sum(fr1) / len(fr1),
            "silent_intact": sum(si0) / len(si0),
            "silent_lesion": sum(si1) / len(si1),
        }
        print(
            f"  损伤 {frac:5.2f}  d'发放 {d_fr:6.2f}  d'静默 {d_si:6.2f}   "
            f"静默 {sum(si0) / len(si0):.3f}->{sum(si1) / len(si1):.3f}",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
    )

    print("")
    print("判（d'=2 为清楚可判）：")
    cc = res["cells"]

    def first_above(ch):
        for f in fracs:
            if cc[f"{f:g}"][ch] > 2.0:
                return f
        return None

    c0 = all(cc["0.5"][k] > 2.0 for k in ("d_firing", "d_silent"))
    print(f"  C-0 50% 时两通道 d'>2 => {'过' if c0 else '不过 => 作废'}")
    ff, fs = first_above("d_firing"), first_above("d_silent")
    print(
        f"  C-1 下界：强度 {ff}  数量 {fs}"
        f" => {'过（数量更灵敏）' if (fs is not None and (ff is None or fs <= ff)) else '不过'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

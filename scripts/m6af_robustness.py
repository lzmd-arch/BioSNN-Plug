"""M6af：**结论对两个手拍常数的稳健性** —— `p`（起搏比例）与 `scale`（输入增益）。

## 为什么

今晚所有结果都建立在 `p=0.02`、`scale=0.7` 上，而这两个数**从头到尾没扫过**。
如果"状态可解码 0.536"只在这一点成立，那它是轶事，不是性质。

## 判据

- **AF-0** 发放在窗内；
- **AF-1 稳健** ★：在 `p ∈ {0.01, 0.02, 0.04}` × `scale ∈ {0.5, 0.7, 1.0}` 里，
  **至少 6/9 格**的解码 > 底板 + 0.15；
- **AF-2 不是单点**：最好格与最差格的差距 < 0.30（不是悬崖）。
"""

from __future__ import annotations

# ruff: noqa: N806
import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.hist_families import make_family_hist, spatial_bins
from scripts.instrument import majority_baseline, probe
from scripts.m6_pools import build_pools, run_spont

OUT = _pathlib.Path("sweep_results/m6af_robustness")
K = 4
T_HIST, T_SPONT = 1024, 2048
FAMS = list(range(8))


@torch.no_grad()
def cell(seed: int, p: float, scale: float, per_fam: int, dev: str) -> dict:
    net = build_pools(seed, dev, K, 0.0)
    feats, labels, frs = [], [], []
    for fam in FAMS:
        toks = make_family_hist(fam, seed * 149 + fam * 23 + 8, T_HIST, per_fam, dev)
        r = run_spont(
            seed,
            0.5,
            0,
            scale,
            p,
            dev,
            K,
            0.0,
            0.0,
            -1.0,
            hist_in=toks,
            t_spont=T_SPONT,
            net=net,
        )[0]
        feats.append(spatial_bins(r).cpu())
        labels.append(torch.full((r.shape[0],), fam))
        frs.append(float(r.mean()))
    X, Y = torch.cat(feats), torch.cat(labels)
    perm = torch.randperm(Y.shape[0], generator=torch.Generator().manual_seed(2 + seed))
    X, Y = X[perm], Y[perm]
    cut = int(Y.shape[0] * 0.75)
    acc, _ = probe(X, Y, cut, len(FAMS))
    return {
        "acc": acc,
        "floor": majority_baseline(Y[:cut], Y[cut:], len(FAMS)),
        "firing": sum(frs) / len(frs),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ps", default="0.01,0.02,0.04")
    ap.add_argument("--scales", default="0.5,0.7,1.0")
    ap.add_argument("--per-fam", type=int, default=16)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": {}}
    ok_arms = 0
    accs = []
    for p in [float(x) for x in args.ps.split(",")]:
        for sc in [float(x) for x in args.scales.split(",")]:
            rows = [cell(sd, p, sc, args.per_fam, dev) for sd in range(args.seeds)]
            m = lambda k, _r=rows: sum(r[k] for r in _r) / len(_r)  # noqa: E731
            key = f"p={p:g}/s={sc:g}"
            v = {
                "p": p,
                "scale": sc,
                "acc": m("acc"),
                "floor": m("floor"),
                "firing": m("firing"),
                "rows": rows,
            }
            inwin = 0.01 <= v["firing"] <= 0.20
            good = inwin and v["acc"] > v["floor"] + 0.15
            ok_arms += int(good)
            accs.append(v["acc"] if inwin else 0.0)
            res["cells"][key] = v
            print(
                f"  {key:14s} 解码 {v['acc']:.3f}（底板 {v['floor']:.3f}）"
                f"发放 {v['firing']:.4f} => {'可解码' if good else '不可'}",
                flush=True,
            )
            (args.out / "result.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
            )

    print("")
    print("判：")
    n = len(res["cells"])
    print(f"  AF-1 至少 6/9 格可解码：{ok_arms}/{n} => {'过' if ok_arms >= 6 else '不过'}")
    spread = max(accs) - min(accs)
    print(f"  AF-2 最好与最差差 {spread:.3f}（要 <0.30）=> {'过' if spread < 0.30 else '不过'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

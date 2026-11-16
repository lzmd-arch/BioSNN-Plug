"""M6x：**高维 + 带历史 + 停得住，同时成立吗？**

M6w 已证：`field_radius` 从 32 降到 2，状态流形的 PR 从 **2.49** 涨到 **~70**。
但**高维本身不是目标**。一个"个体状态"要同时满足三条：

| 要素 | 判据 |
| :--- | :--- |
| **高维** | PR > 10 |
| **可解码（带历史）** | 8 族解码 > 底板 + 0.15 |
| **停得住** | 同一初值 T=1024 与 T=2048 的终态相对距离中位 < 0.05 |

三条必须**在同一档同时成立**，否则不作数 —— 这是本轮的核心判据。
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
from scripts.m6_pools import build_pools, rel, run_spont
from scripts.m6v_dimension import pr_of

OUT = _pathlib.Path("sweep_results/m6x_local_life")
K = 4
T_HIST, T_SPONT = 1024, 2048
FAMS = list(range(8))


@torch.no_grad()
def cell(seed: int, radius: int, per_fam: int, dev: str, scale: float = 0.7) -> dict:
    net = build_pools(seed, dev, K, 0.0)
    feats, labels, states, fin = [], [], [], []
    for fam in FAMS:
        toks = make_family_hist(fam, seed * 409 + fam * 17 + 3, T_HIST, per_fam, dev)
        r = run_spont(
            seed,
            0.5,
            0,
            scale,
            0.02,
            dev,
            K,
            0.0,
            0.0,
            -1.0,
            hist_in=toks,
            t_spont=T_SPONT,
            net=net,
            field_radius=radius,
            **field_args(radius),
        )[0]
        feats.append(spatial_bins(r).cpu())
        labels.append(torch.full((r.shape[0],), fam))
        states.append(r.cpu())
        if fam == 0:  # 停没停只测一族，省一半调用
            r2 = run_spont(
                seed,
                0.5,
                0,
                scale,
                0.02,
                dev,
                K,
                0.0,
                0.0,
                -1.0,
                hist_in=toks,
                t_spont=1024,
                net=net,
                field_radius=radius,
                **field_args(radius),
            )[0]
            fin.append((r.cpu(), r2.cpu()))
    X, Y = torch.cat(feats), torch.cat(labels)
    perm = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(9 + seed))
    X, Y = X[perm], Y[perm]
    cut = int(X.shape[0] * 0.75)
    acc, _ = probe(X, Y, cut, len(FAMS))
    floor = majority_baseline(Y[:cut], Y[cut:], len(FAMS))
    pers = sum(sum(rel(a[i], b[i]) for i in range(a.shape[0])) / a.shape[0] for a, b in fin) / len(
        fin
    )
    S = torch.cat(states)
    return {
        "acc": acc,
        "floor": floor,
        "pr": pr_of(S),
        "persist": pers,
        "firing": float(S.mean()),
    }


def field_args(radius: int) -> dict:
    """`field_radius` 与 `g_field/g_local` 的对应关系。

    **2026-11-16 踩过**：加了 `g_local` 之后全局项永远施加，只传 `field_radius`
    不会打开局部场 —— M6x/M6z 第一版因此全部跑成了全局场（PR 2.34 与 radius=32 一致）。
    """
    if radius >= 32:
        return {"g_field": 1.0, "g_local": 0.0}
    return {"g_field": 0.0, "g_local": 1.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--radii", default="2,4,8,32")
    ap.add_argument("--scales", default="0.7,1.4")  # 发放匹配对照：同一半径两个输入增益
    ap.add_argument("--per-fam", type=int, default=16)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": {}}
    for rad, scale in [
        (int(r), float(sc)) for r in args.radii.split(",") for sc in args.scales.split(",")
    ]:
        rows = [cell(sd, rad, args.per_fam, dev, scale) for sd in range(args.seeds)]
        m = lambda k, _r=rows: sum(r[k] for r in _r) / len(_r)  # noqa: E731
        key = f"r={rad}/s={scale:g}"
        res["cells"][key] = {
            "radius": rad,
            "scale": scale,
            "acc": m("acc"),
            "floor": m("floor"),
            "pr": m("pr"),
            "persist": m("persist"),
            "firing": m("firing"),
            "rows": rows,
        }
        v = res["cells"][key]
        print(
            f"  radius={rad:2d}: 解码 {v['acc']:.3f}（底板 {v['floor']:.3f}）  "
            f"PR {v['pr']:6.2f}  持久 {v['persist']:.4f}  发放 {v['firing']:.4f}",
            flush=True,
        )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )

    print("")
    print("判（**三条必须同时成立**）：")
    cc = res["cells"]
    any_ok = False
    for k, v in cc.items():
        ok_f = 0.01 <= v["firing"] <= 0.20
        ok_d = v["acc"] > v["floor"] + 0.15
        ok_p = v["pr"] > 10.0
        ok_s = v["persist"] < 0.05
        both = ok_f and ok_d and ok_p and ok_s
        any_ok = any_ok or both
        print(
            f"  {k:6s} 高维 {'✅' if ok_p else '❌'}  可解码 {'✅' if ok_d else '❌'}  "
            f"停住 {'✅' if ok_s else '❌'}  窗口 {'✅' if ok_f else '❌'}  "
            f"=> {'**同时成立**' if both else '不成立'}"
        )
    print(f"  X-1 存在某档三条同时成立：{'过' if any_ok else '不过'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

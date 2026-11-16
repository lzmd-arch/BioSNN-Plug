"""M6ak：**生命配置有没有？—— 在那个"高维 + 整合 + 分离"的格子上测可解码性**。

## 为什么这是今晚最关键的一格

`M6aa` 第一次让三种性状同时成立：`radius=2` + `g_local=1.0` + **`p_long=0.5`**
给出 PR **47.04**、整合 **0.421**、分离 **3.05**、发放 0.025。

但那格**没测可解码**。若可解码也成立，那就是今晚要找的东西；
若不可解码，则"高维/整合"与"可读"仍然互斥，只是这次是用**接线**换来的整合也一样。

## 判据（五条同时成立才算"生命配置"）

| 判据 | 线 |
| :--- | :--- |
| K-0 发放在窗 | [0.01, 0.20] |
| K-1 高维 | PR > 10 |
| K-2 整合 | > 0.3 |
| K-3 分离 | > 0.5 |
| **K-4 可解码** ★★ | 线性 **或** MLP **或** 1-NN > 底板 + 0.15 |

**K-4 用三种读出**，因为线性对高维编码先验就会失败（用户 2026-11-16 的更正）。
"""

from __future__ import annotations

# ruff: noqa: N803, N806
import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.hist_families import make_family_hist, spatial_bins
from scripts.instrument import majority_baseline, mlp_probe, probe
from scripts.m6_pools import N, build_pools, run_spont
from scripts.m6v_dimension import pr_of

OUT = _pathlib.Path("sweep_results/m6ak_life_config")
K, SIDE = 4, 32
T_HIST, T_SPONT = 1024, 2048
FAMS = list(range(8))
RADIUS = 2


def knn_acc(Xtr, Ytr, Xte, Yte) -> float:
    d = torch.cdist(Xte, Xtr)
    pred = Ytr[d.argmin(1)]
    return float((pred == Yte).float().mean())


@torch.no_grad()
def cell(
    seed: int, p_long: float, per_fam: int, dev: str, pace: float = 0.02, g_local: float = 1.0
) -> dict:
    w_in, pre, wrec, pid = build_pools(seed, dev, K, 0.0, p_long=p_long)
    net = (w_in, pre, wrec, pid)
    full, binned, labels, states, frs = [], [], [], [], []
    for fam in FAMS:
        toks = make_family_hist(fam, seed * 733 + fam * 31 + 12, T_HIST, per_fam, dev)
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
        full.append(r.cpu())
        binned.append(spatial_bins(r).cpu())
        labels.append(torch.full((r.shape[0],), fam))
        states.append(r.cpu())
        frs.append(float(r.mean()))
    Xf, Xb, Y = torch.cat(full), torch.cat(binned), torch.cat(labels)
    S = torch.cat(states)
    h = ((torch.arange(N) % SIDE) >= 16).long()
    A, B = S[:, h == 0], S[:, h == 1]
    a, b = A.mean(1) - A.mean(), B.mean(1) - B.mean()
    integ = abs(float((a * b).sum() / (a.norm() * b.norm() + 1e-12)))
    segs = [float((H.std(1) / (H.mean(1) + 1e-9)).mean()) for H in (A, B)]
    return {
        "pr": pr_of(S),
        "integ": integ,
        "seg": sum(segs) / 2,
        "firing": sum(frs) / len(frs),
        "Xf": Xf,
        "Xb": Xb,
        "Y": Y,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--plongs", default="0,0.25,0.5,0.75")
    ap.add_argument("--ps", default="0.02")
    ap.add_argument("--glocals", default="1.0")
    ap.add_argument("--per-fam", type=int, default=32)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": {}}
    hits = []
    for gl in [float(x) for x in args.glocals.split(",")]:
        for pace in [float(x) for x in args.ps.split(",")]:
            for pl in [float(x) for x in args.plongs.split(",")]:
                rows = [cell(sd, pl, args.per_fam, dev, pace, gl) for sd in range(args.seeds)]
                m = lambda k, _r=rows: sum(r[k] for r in _r) / len(_r)  # noqa: E731
                # **探针必须在 no_grad 之外**（MLP 需要梯度）—— 2026-11-16 踩过
                for i, r in enumerate(rows):
                    perm = torch.randperm(
                        r["Y"].shape[0], generator=torch.Generator().manual_seed(3 + i)
                    )
                    cut = int(r["Y"].shape[0] * 0.75)
                    r["linear"] = probe(r["Xb"][perm], r["Y"][perm], cut, len(FAMS))[0]
                    r["knn"] = knn_acc(
                        r["Xb"][perm][:cut],
                        r["Y"][perm][:cut],
                        r["Xb"][perm][cut:],
                        r["Y"][perm][cut:],
                    )
                    r["mlp"] = mlp_probe(r["Xf"][perm], r["Y"][perm], cut, len(FAMS))[0]
                    r["floor"] = majority_baseline(
                        r["Y"][perm][:cut], r["Y"][perm][cut:], len(FAMS)
                    )
                    # 张量不能进 JSON
                    r.pop("Xf", None)
                    r.pop("Xb", None)
                    r.pop("Y", None)
                key = f"gl={gl:g}/p={pace:g}/pl={pl:g}"
                v = {
                    "p_long": pl,
                    "pace": pace,
                    "g_local": gl,
                    "pr": m("pr"),
                    "integ": m("integ"),
                    "seg": m("seg"),
                    "linear": m("linear"),
                    "knn": m("knn"),
                    "mlp": m("mlp"),
                    "floor": m("floor"),
                    "firing": m("firing"),
                    "rows": rows,
                }
                best_read = max(v["linear"], v["knn"], v["mlp"])
                flags = {
                    "窗口": 0.01 <= v["firing"] <= 0.20,
                    "高维": v["pr"] > 10,
                    "整合": v["integ"] > 0.3,
                    "分离": v["seg"] > 0.5,
                    "可解码": best_read > v["floor"] + 0.15,
                }
                v["flags"] = flags
                v["all5"] = all(flags.values())
                res["cells"][key] = v
                if v["all5"]:
                    hits.append(key)
                print(
                    f"  {key:9s} PR {v['pr']:6.2f} 整合 {v['integ']:.3f} 分离 {v['seg']:.2f} "
                    f"发放 {v['firing']:.4f} | 线性 {v['linear']:.3f} 1-NN {v['knn']:.3f} "
                    f"MLP {v['mlp']:.3f}（底 {v['floor']:.3f}）| "
                    + "".join("✅" if b else "❌" for b in flags.values())
                    + (" **五条全过**" if v["all5"] else ""),
                    flush=True,
                )
                (args.out / "result.json").write_text(
                    json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
                )

    print("")
    print("判：")
    print(f"  K-1..K-4 五条同时成立的配置：{hits or '无'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

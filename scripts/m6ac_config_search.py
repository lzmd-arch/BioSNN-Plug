"""M6ac：**生命配置搜索 —— 五个性状同时量**。

个体状态要"活着"，需要五条同时成立。今晚每一条都单独出现过，从没同时出现过：

| 性状 | 判据 |
| :--- | :--- |
| 高维 | PR > 10 |
| 停住 | 同一初值 T=1024 与 T=2048 的相对距离 < 0.05 |
| 可解码 | 8 族解码 > 底板 + 0.15 |
| 整合 | 两半球平均活动的 |相关| > 0.3 |
| 分离 | 半球内空间变异系数 > 0.5 |

**搜索空间**：`field_radius × g_local × p_long`。
- `field_radius` 控制抑制的空间尺度（小 = 局部，给维度）；
- `g_local` 控制抑制强度（决定操作点/发放率）；
- `p_long` 是长程兴奋比例（`build_pools`），用接线买整合。

每一格都跑 8 个历史族 × `per_fam` 条，五个性状在**同一批状态**上量。
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
from scripts.m6_pools import N, build_pools, rel, run_spont
from scripts.m6v_dimension import pr_of

OUT = _pathlib.Path("sweep_results/m6ac_config_search")
K, SIDE = 4, 32
T_HIST, T_SPONT, T_SHORT = 1024, 2048, 1024
FAMS = list(range(8))


@torch.no_grad()
def cell(seed: int, radius: int, g_local: float, p_long: float, per_fam: int, dev: str) -> dict:
    w_in, pre, wrec, pid = build_pools(seed, dev, K, 0.0, p_long=p_long)
    net = (w_in, pre, wrec, pid)
    feats, labels, states = [], [], []
    persist = []
    for fam in FAMS:
        toks = make_family_hist(fam, seed * 809 + fam * 29 + 7, T_HIST, per_fam, dev)
        kw = {
            "hist_in": toks,
            "net": net,
            "field_radius": radius,
            "g_field": 0.0,
            "g_local": g_local,
        }
        r = run_spont(seed, 0.5, 0, 0.7, 0.02, dev, K, 0.0, 0.0, -1.0, t_spont=T_SPONT, **kw)[0]
        feats.append(spatial_bins(r).cpu())
        labels.append(torch.full((r.shape[0],), fam))
        states.append(r.cpu())
        if fam == 0:
            r2 = run_spont(seed, 0.5, 0, 0.7, 0.02, dev, K, 0.0, 0.0, -1.0, t_spont=T_SHORT, **kw)[
                0
            ]
            persist = sum(rel(r[i], r2[i]) for i in range(r.shape[0])) / r.shape[0]
    X, Y = torch.cat(feats), torch.cat(labels)
    perm = torch.randperm(Y.shape[0], generator=torch.Generator().manual_seed(3 + seed))
    X, Y = X[perm], Y[perm]
    cut = int(Y.shape[0] * 0.75)
    acc, _ = probe(X, Y, cut, len(FAMS))
    floor = majority_baseline(Y[:cut], Y[cut:], len(FAMS))
    S = torch.cat(states)
    h = ((torch.arange(N) % SIDE) >= 16).long()
    A, B = S[:, h == 0], S[:, h == 1]
    a, b = A.mean(1) - A.mean(), B.mean(1) - B.mean()
    integ = abs(float((a * b).sum() / (a.norm() * b.norm() + 1e-12)))
    segs = [float((H.std(1) / (H.mean(1) + 1e-9)).mean()) for H in (A, B)]
    return {
        "pr": pr_of(S),
        "acc": acc,
        "floor": floor,
        "integ": integ,
        "seg": sum(segs) / 2,
        "persist": persist,
        "firing": float(S.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--radii", default="2,8")
    ap.add_argument("--glocals", default="0.5,1,2,4")
    ap.add_argument("--plongs", default="0,0.25")
    ap.add_argument("--per-fam", type=int, default=16)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": {}}
    hits = []
    for rad in [int(x) for x in args.radii.split(",")]:
        for gl in [float(x) for x in args.glocals.split(",")]:
            for pl in [float(x) for x in args.plongs.split(",")]:
                rows = [cell(sd, rad, gl, pl, args.per_fam, dev) for sd in range(args.seeds)]
                m = lambda k, _r=rows: sum(r[k] for r in _r) / len(_r)  # noqa: E731
                key = f"r={rad}/gl={gl:g}/pl={pl:g}"
                v = {
                    "radius": rad,
                    "g_local": gl,
                    "p_long": pl,
                    "pr": m("pr"),
                    "acc": m("acc"),
                    "floor": m("floor"),
                    "integ": m("integ"),
                    "seg": m("seg"),
                    "persist": m("persist"),
                    "firing": m("firing"),
                    "rows": rows,
                }
                flags = {
                    "高维": v["pr"] > 10,
                    "停住": v["persist"] < 0.05,
                    "可解码": v["acc"] > v["floor"] + 0.15,
                    "整合": v["integ"] > 0.3,
                    "分离": v["seg"] > 0.5,
                    "窗口": 0.01 <= v["firing"] <= 0.20,
                }
                v["flags"] = flags
                v["all5"] = all(flags.values())
                res["cells"][key] = v
                if v["all5"]:
                    hits.append(key)
                print(
                    f"  {key:20s} PR {v['pr']:6.2f} 解码 {v['acc']:.3f}(底 {v['floor']:.3f}) "
                    f"整合 {v['integ']:.3f} 分离 {v['seg']:.2f} 持久 {v['persist']:.4f} "
                    f"发放 {v['firing']:.4f} | "
                    + "".join("✅" if b else "❌" for b in flags.values())
                    + (" **五条全过**" if v["all5"] else ""),
                    flush=True,
                )
                (args.out / "result.json").write_text(
                    json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
                )

    print("")
    print("判：")
    print(f"  五条同时成立的配置：{hits or '无'}")
    cc = res["cells"]
    for name, key in (
        ("高维", "pr"),
        ("可解码", "acc"),
        ("整合", "integ"),
        ("分离", "seg"),
    ):
        best = max(cc.values(), key=lambda v: v[key] if key != "acc" else v["acc"] - v["floor"])
        val = best[key] if key != "acc" else best["acc"] - best["floor"]
        print(
            f"  单项最好 {name}: r={best['radius']} gl={best['g_local']:g} "
            f"pl={best['p_long']:g} → {val:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

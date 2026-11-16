"""M6ar：**激素层能不能让三轴共存**（用户 2026-11-17 提案）。

## 三层时间尺度

> 用户：3 轴共存也许需要激素和基因的帮助。

```
基因     设 g_field / g_local / p_long / 半径 / 起搏      （一生不变）
激素     h ← h + (φ − 0.09)/τ_h，再把 g_field 乘 (1+βh)、g_local 乘 (1−βh)
          （分钟级 —— 比门控再慢一个量级）
场       实际的抑制                                    （毫秒级）
```

**激素的关键区别**：它不改场本身，改的是**哪一层在那一刻占主导**。
高活动 ⇒ h 上升 ⇒ 全局权重上升（压住、整合）；低活动 ⇒ h 下降 ⇒ 局部权重上升（分化）。

## 判据

- **AR-0** 发放在窗内；
- **AR-1 兼得** ★★：某档 **PR > 10 且 整合 > 0.3 且 分离 > 0.5**；
- **AR-2 可读** ：该档解码 > 底板 + 0.15。
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
from scripts.instrument import majority_baseline, mlp_probe, probe
from scripts.m6_pools import N, build_pools, run_spont
from scripts.m6v_dimension import pr_of

OUT = _pathlib.Path("sweep_results/m6ar_hormone")
K, SIDE = 4, 32
T_HIST, T_SPONT = 1024, 2048
FAMS = list(range(8))
P_LONG = 0.5
BASE = {"field_radius": 2, "g_field": 1.0, "g_local": 2.0}
GATE = {"gate_w": 0.02, "gate_phi0": 0.08, "gate_tau": 20.0}

ARMS = {
    "static": dict(BASE),
    "gate_only": dict(BASE, **GATE),
    # **自主节律**：锯齿波把门控阈值扫过两个 regime（局部 φ≈0.014、全局 φ≈0.13）
    "sweep512": dict(BASE, **GATE, hormone_sweep=512.0, hormone_shift=0.07),
    "sweep1024": dict(BASE, **GATE, hormone_sweep=1024.0, hormone_shift=0.07),
}


@torch.no_grad()
def cell(seed: int, arm: dict, per_fam: int, dev: str) -> dict:
    w_in, pre, wrec, pid = build_pools(seed, dev, K, 0.0, p_long=P_LONG)
    net = (w_in, pre, wrec, pid)
    full, binned, labels, states, frs = [], [], [], [], []
    for fam in FAMS:
        toks = make_family_hist(fam, seed * 941 + fam * 41 + 17, T_HIST, per_fam, dev)
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
            **arm,
        )[0]
        full.append(r.cpu())
        binned.append(spatial_bins(r).cpu())
        labels.append(torch.full((r.shape[0],), fam))
        states.append(r.cpu())
        frs.append(float(r.mean()))
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
        "Xf": torch.cat(full),
        "Xb": torch.cat(binned),
        "Y": torch.cat(labels),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--per-fam", type=int, default=32)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": {}}
    hits = []
    for name, arm in ARMS.items():
        rows = [cell(sd, arm, args.per_fam, dev) for sd in range(args.seeds)]
        m = lambda k, _r=rows: sum(r[k] for r in _r) / len(_r)  # noqa: E731
        for i, r in enumerate(rows):
            perm = torch.randperm(r["Y"].shape[0], generator=torch.Generator().manual_seed(41 + i))
            cut = int(r["Y"].shape[0] * 0.75)
            r["linear"] = probe(r["Xb"][perm], r["Y"][perm], cut, len(FAMS))[0]
            r["mlp"] = mlp_probe(r["Xf"][perm], r["Y"][perm], cut, len(FAMS))[0]
            r["floor"] = majority_baseline(r["Y"][perm][:cut], r["Y"][perm][cut:], len(FAMS))
            r.pop("Xf", None)
            r.pop("Xb", None)
            r.pop("Y", None)
        v = {
            "arm": arm,
            "pr": m("pr"),
            "integ": m("integ"),
            "seg": m("seg"),
            "firing": m("firing"),
            "linear": m("linear"),
            "mlp": m("mlp"),
            "floor": m("floor"),
            "rows": rows,
        }
        best = max(v["linear"], v["mlp"])
        v["readable"] = bool(best > v["floor"] + 0.15)
        v["all3"] = bool(v["pr"] > 10 and v["integ"] > 0.3 and v["seg"] > 0.5)
        if v["all3"]:
            hits.append(name)
        res["cells"][name] = v
        print(
            f"  {name:12s} PR {v['pr']:6.2f}  整合 {v['integ']:.3f}  分离 {v['seg']:.2f}  "
            f"发放 {v['firing']:.4f}  线性 {v['linear']:.3f} MLP {v['mlp']:.3f}"
            f"（底 {v['floor']:.3f}）=> {'可读' if v['readable'] else '不可读'}"
            f"{'  **三轴全过**' if v['all3'] else ''}",
            flush=True,
        )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1, default=float),
            encoding="utf-8",
            newline=chr(10),
        )

    print("")
    print("判：")
    print(f"  AR-1 三轴兼得的配置：{hits or '无'}")
    for k, v in res["cells"].items():
        ok = 0.01 <= v["firing"] <= 0.20
        print(f"    {k:12s} 窗口 {'✅' if ok else '❌'}  可读 {'✅' if v['readable'] else '❌'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

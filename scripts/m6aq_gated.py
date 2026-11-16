"""M6aq：**状态门控的场** —— 两种场按状态交替，而不是静态混合（用户 2026-11-17 提案）。

## 提案

> 全局场和局部场也许并不会同时出现，而是不同状态下不同场这种协同补充的形式。

一直以来的做法是 `场 = 全局 + 局部` 的**静态混合**，所以每一刻都被迫二选一（交换曲线）。
**按状态交替**则不同：系统在时间上两个 regime 都走到，**整体性质可以兼得，因为不是同一时刻要。**

生物学上也更对：脑的耦合结构随状态变（振荡、神经调质、睡眠/清醒），不是固定混合比。

## 实现

```
λ = clip((φ − φ0)/w, 0, 1)          # 由**当前活动**决定，不是排程
全局场权重 = 1 − λ；局部场权重 = λ
```

高活动 ⇒ 走局部（压住分化）；低活动 ⇒ 走全局（整合）。

## 判据

- **AQ-0** 发放在窗内；
- **AQ-1 兼得** ★★：门控档 **PR > 10 且 整合 > 0.3 且 分离 > 0.5**（静态档做不到）；
- **AQ-2 可读不塌** ：门控档的解码 > 底板 + 0.15。
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

OUT = _pathlib.Path("sweep_results/m6aq_gated")
K, SIDE = 4, 32
T_HIST, T_SPONT = 1024, 2048
FAMS = list(range(8))

ARMS = {
    "static_global": {"field_radius": 32, "g_field": 1.0, "g_local": 0.0, "gate_w": 0.0},
    # **慢门控**：阈值设在网络会穿过的位置（全局 regime 下 φ≈0.13、局部下 ≈0.02），
    # 这样 λ 才会在两个 regime 之间来回摆，而不是停在不动点。
    "gated_t50": {
        "field_radius": 2,
        "g_field": 1.0,
        "g_local": 2.0,
        "gate_w": 0.04,
        "gate_phi0": 0.10,
        "gate_tau": 50.0,
    },
    "gated_t200": {
        "field_radius": 2,
        "g_field": 1.0,
        "g_local": 2.0,
        "gate_w": 0.04,
        "gate_phi0": 0.10,
        "gate_tau": 200.0,
    },
}
P_LONG = 0.5


@torch.no_grad()
def cell(seed: int, arm: dict, per_fam: int, dev: str) -> dict:
    w_in, pre, wrec, pid = build_pools(seed, dev, K, 0.0, p_long=P_LONG)
    net = (w_in, pre, wrec, pid)
    full, binned, labels, states, frs = [], [], [], [], []
    for fam in FAMS:
        toks = make_family_hist(fam, seed * 877 + fam * 37 + 15, T_HIST, per_fam, dev)
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
    for name, arm in ARMS.items():
        rows = [cell(sd, arm, args.per_fam, dev) for sd in range(args.seeds)]
        m = lambda k, _r=rows: sum(r[k] for r in _r) / len(_r)  # noqa: E731
        for i, r in enumerate(rows):
            perm = torch.randperm(r["Y"].shape[0], generator=torch.Generator().manual_seed(31 + i))
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
        v["readable"] = best > v["floor"] + 0.15
        res["cells"][name] = v
        print(
            f"  {name:14s} PR {v['pr']:6.2f}  整合 {v['integ']:.3f}  分离 {v['seg']:.2f}  "
            f"发放 {v['firing']:.4f}  线性 {v['linear']:.3f} MLP {v['mlp']:.3f}"
            f"（底 {v['floor']:.3f}）=> {'可读' if v['readable'] else '不可读'}",
            flush=True,
        )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1, default=float),
            encoding="utf-8",
            newline=chr(10),
        )

    print("")
    print("判：")
    cc = res["cells"]
    g = cc["gated_hyst"]
    ok0 = 0.01 <= g["firing"] <= 0.20
    print(f"  AQ-0 门控档发放在窗内：{g['firing']:.4f} => {'过' if ok0 else '不过 => 作废'}")
    aq1 = g["pr"] > 10 and g["integ"] > 0.3 and g["seg"] > 0.5
    print(
        f"  AQ-1 兼得（PR>10 且 整合>0.3 且 分离>0.5）：PR {g['pr']:.1f}、整合 {g['integ']:.3f}、"
        f"分离 {g['seg']:.2f} => {'过' if aq1 else '不过'}"
    )
    print(f"  AQ-2 可读不塌（>底板+0.15）：{'过' if g['readable'] else '不过'}")
    print("")
    print("  对照：")
    for k in ("static_global",):
        v = cc[k]
        print(
            f"    {k:14s} PR {v['pr']:6.2f}  整合 {v['integ']:.3f}  分离 {v['seg']:.2f}  "
            f"{'可读' if v['readable'] else '不可读'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

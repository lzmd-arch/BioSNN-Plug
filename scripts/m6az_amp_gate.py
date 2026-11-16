"""M6az：**连续振幅门控 —— 三轴兼得的第一个配置**（1 种子冒烟已过五条，本脚本做 6 种子）

## 为什么它和二值门控不同

| | 二值门控（今晚五版全失败） | **连续振幅门控** |
| :--- | :--- | :--- |
| 控制变量 | φ（瞬时发放率） | **两模振幅的相对相位** |
| 切换 | 二值（0/1） | **连续旋转** |
| λ 取值 | 饱和在 0 或 1 | **连续，|a_g|²** |
| PR | 1.0（锁死） | **58–67** |

机制：两个场是**同一个两模态的两种投影**；活动偏离目标时振幅做**旋转**（等价于干涉），
`λ = |a_g|²` 连续变化 ⇒ **结构上不会锁死**。

## 判据（沿用 `M6ak` 的五条，不另立）

发放在窗、PR>10、整合>0.3、分离>0.5、可读（>底板+0.15）。**五条同时成立。**
**≥6 种子。**
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

OUT = _pathlib.Path("sweep_results/m6az_amp_gate")
K, SIDE = 4, 32
T_HIST, T_SPONT = 1024, 2048
FAMS = list(range(8))
P_LONG = 0.5


@torch.no_grad()
def cell(seed: int, amp: float, per_fam: int, dev: str) -> dict:
    w_in, pre, wrec, pid = build_pools(seed, dev, K, 0.0, p_long=P_LONG)
    net = (w_in, pre, wrec, pid)
    full, binned, labels, states, frs = [], [], [], [], []
    for fam in FAMS:
        toks = make_family_hist(fam, seed * 809 + fam * 43 + 21, T_HIST, per_fam, dev)
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
            field_radius=2,
            g_field=1.0,
            g_local=2.0,
            amp_gate=amp,
        )[0]
        full.append(r.cpu())
        binned.append(spatial_bins(r).cpu())
        labels.append(torch.full((r.shape[0],), fam))
        states.append(r.cpu())
        frs.append(float(r.mean()))
    S = torch.cat(states)
    h = (torch.arange(N) % SIDE >= 16).long()
    A, B = S[:, h == 0], S[:, h == 1]
    a_, b_ = A.mean(1) - A.mean(), B.mean(1) - B.mean()
    integ = abs(float((a_ * b_).sum() / (a_.norm() * b_.norm() + 1e-12)))
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
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--amps", default="2.0")
    ap.add_argument("--per-fam", type=int, default=32)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": {}}
    for amp in [float(x) for x in args.amps.split(",")]:
        rows = [cell(sd, amp, args.per_fam, dev) for sd in range(args.seeds)]
        m = lambda k, _r=rows: sum(r[k] for r in _r) / len(_r)  # noqa: E731
        for i, r in enumerate(rows):
            perm = torch.randperm(r["Y"].shape[0], generator=torch.Generator().manual_seed(51 + i))
            cut = int(r["Y"].shape[0] * 0.75)
            r["linear"] = probe(r["Xb"][perm], r["Y"][perm], cut, len(FAMS))[0]
            r["mlp"] = mlp_probe(r["Xf"][perm], r["Y"][perm], cut, len(FAMS))[0]
            r["floor"] = majority_baseline(r["Y"][perm][:cut], r["Y"][perm][cut:], len(FAMS))
            r.pop("Xf", None)
            r.pop("Xb", None)
            r.pop("Y", None)
        v = {k: m(k) for k in rows[0]}
        v["rows"] = rows
        v["amp"] = amp
        v["best_read"] = max(v["linear"], v["mlp"])
        v["all5"] = bool(
            0.01 <= v["firing"] <= 0.20
            and v["pr"] > 10
            and v["integ"] > 0.3
            and v["seg"] > 0.5
            and v["best_read"] > v["floor"] + 0.15
        )
        res["cells"][f"amp={amp:g}"] = v
        print(
            f"  amp={amp:g}: PR {v['pr']:6.2f}  整合 {v['integ']:.3f}  分离 {v['seg']:.2f}  "
            f"线性 {v['linear']:.3f}  MLP {v['mlp']:.3f}（底 {v['floor']:.3f}）  "
            f"发放 {v['firing']:.4f}  => {'**五条全过**' if v['all5'] else '不成立'}",
            flush=True,
        )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1, default=float),
            encoding="utf-8",
            newline=chr(10),
        )

    print("")
    print("判：")
    for k, v in res["cells"].items():
        print(f"  {k}: {'**五条同时成立**' if v['all5'] else '不成立'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

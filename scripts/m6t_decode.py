"""M6t（Round A）：**从终态解码它自己的历史**。

判据写在 `research/sel/night-2026-11-16.md`（跑之前）。

M6s 已证：`g_inh=0` 时状态是**连续吸引子**，1024→2048 只漂 1.5%。
那么该问的就变成：**这个状态里读不读得出它自己的历史？**
读不出来，"状态由历史决定"就只是确定性的同义反复，不是记忆。

若能读出来，**解码准确率本身就是"能力"指标** —— M4/M5/M8 三根柱子都要用它。
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
from scripts.m6_pools import run_spont

OUT = _pathlib.Path("sweep_results/m6t_decode")
K = 4
T_HIST = 1024
T_SPONT = 1024
B_BATCH = 32


@torch.no_grad()
def collect(
    seed: int, dev: str, conc: float, n_fam: int, per_fam: int
) -> tuple[torch.Tensor, torch.Tensor, float]:
    feats, labels, frs = [], [], []
    for fam in range(n_fam):
        for chunk in range(per_fam // B_BATCH):
            toks = make_family_hist(
                fam, seed * 100003 + fam * 101 + chunk, T_HIST, B_BATCH, dev, conc=conc
            )
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
            )
            feats.append(spatial_bins(r).cpu())
            labels.append(torch.full((r.shape[0],), fam))
            frs.append(float(r.mean()))
    return torch.cat(feats), torch.cat(labels), sum(frs) / len(frs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--concs", default="0.7")
    ap.add_argument("--n-fam", type=int, default=8)
    ap.add_argument("--per-fam", type=int, default=32)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": {}, "n_fam": args.n_fam, "per_fam": args.per_fam}
    for conc in [float(x) for x in args.concs.split(",")]:
        accs, ctrls, floors, frs = [], [], [], []
        for sd in range(args.seeds):
            X, Y, fr = collect(sd, dev, conc, args.n_fam, args.per_fam)
            frs.append(fr)
            perm = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(4242 + sd))
            X, Y = X[perm], Y[perm]
            cut = int(X.shape[0] * 0.75)
            acc, _lam = probe(X, Y, cut, args.n_fam)
            accs.append(acc)
            floors.append(majority_baseline(Y[:cut], Y[cut:], args.n_fam))
            Ys = Y[torch.randperm(Y.shape[0], generator=torch.Generator().manual_seed(99 + sd))]
            ctrl, _ = probe(X, Ys, cut, args.n_fam)
            ctrls.append(ctrl)
            print(
                f"  conc={conc:g} seed {sd}: 发放 {fr:.4f}  解码 {acc:.3f}  "
                f"对照 {ctrl:.3f}  底板 {floors[-1]:.3f}",
                flush=True,
            )
        m = lambda v: sum(v) / len(v)  # noqa: E731
        v = {
            "acc": accs,
            "ctrl": ctrls,
            "floor": floors,
            "firing": frs,
            "acc_mean": m(accs),
            "ctrl_mean": m(ctrls),
            "floor_mean": m(floors),
            "firing_mean": m(frs),
            "seeds": args.seeds,
        }
        res["cells"][f"conc={conc:g}"] = v
        print(
            f"  conc={conc:g} 汇总：解码 {v['acc_mean']:.3f}  对照 {v['ctrl_mean']:.3f}  "
            f"底板 {v['floor_mean']:.3f}  发放 {v['firing_mean']:.4f}",
            flush=True,
        )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )

    print("")
    print("判：")
    cc = res["cells"]
    best = max(cc.values(), key=lambda x: x["acc_mean"])
    t0 = 0.01 <= best["firing_mean"] <= 0.20
    print(f"  T-0 发放在窗内：{best['firing_mean']:.4f} => {'过' if t0 else '不过 => 作废'}")
    if not t0:
        return 0
    t1 = best["acc_mean"] > best["floor_mean"] + 0.15
    print(
        f"  T-1 解码（> 底板+0.15）：{best['acc_mean']:.3f} vs {best['floor_mean']:.3f} "
        f"=> {'过' if t1 else '不过'}"
    )
    t2 = abs(best["ctrl_mean"] - best["floor_mean"]) <= 0.05
    print(
        f"  T-2 对照（打乱标签回基线 ±0.05）：{best['ctrl_mean']:.3f} vs "
        f"{best['floor_mean']:.3f} => {'过' if t2 else '不过'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

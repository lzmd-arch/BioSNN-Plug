"""M12（Round I）：**自我模型 —— 我现在已经知道我会变成什么**。

判据在 `research/sel/night-2026-11-16.md`（含**修订 4c**）。

## 修订 4c（开跑前）

原设计"用 t 步的 v 预测 t+1 步的 z"在 `z = 1[v > θ]` 下 **AUC 必然 ≈1** ——
那测的是阈值函数存在，不是自我模型（`trivial-solution-passes-the-bar` 那个坑）。

改成：**从 t=64 的状态预测 t=2048 的盆地**。端点由轨迹决定，不由当步状态直接给出。
对照：打乱 (特征, 盆地) 配对。
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
from scripts.instrument import probe
from scripts.m6_pools import build_pools, rel, run_spont

OUT = _pathlib.Path("sweep_results/m12_selfmodel")
K = 4
T_HIST = 1024
T_LONG = 2048
T_EARLY = 64
TOL = 0.05


def basin_ids(S: torch.Tensor) -> list[int]:
    reps: list[int] = []
    out = [-1] * S.shape[0]
    for i in range(S.shape[0]):
        hit = -1
        for c, r in enumerate(reps):
            if rel(S[i], S[r]) < TOL:
                hit = c
                break
        if hit < 0:
            reps.append(i)
            hit = len(reps) - 1
        out[i] = hit
    return out


@torch.no_grad()
def main_run() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--per-fam", type=int, default=16)
    ap.add_argument("--n-fam", type=int, default=8)
    ap.add_argument("--g-inh", type=float, default=1.0)  # M6u 已证这档是 10-14 个盆
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": []}
    for sd in range(args.seeds):
        net = build_pools(sd, dev, K, 0.0)
        feats, finals = [], []
        for fam in range(args.n_fam):
            for chunk in range(max(1, args.per_fam // 32)):
                toks = make_family_hist(fam, sd * 331 + fam * 7 + chunk, T_HIST, 32, dev)
                early, _, _ = run_spont(
                    sd,
                    0.5,
                    0,
                    0.7,
                    0.02,
                    dev,
                    K,
                    0.0,
                    args.g_inh,
                    -1.0,
                    hist_in=toks,
                    t_spont=T_EARLY,
                    net=net,
                )
                final, _, _ = run_spont(
                    sd,
                    0.5,
                    0,
                    0.7,
                    0.02,
                    dev,
                    K,
                    0.0,
                    args.g_inh,
                    -1.0,
                    hist_in=toks,
                    t_spont=T_LONG,
                    net=net,
                )
                feats.append(spatial_bins(early).cpu())
                finals.append(final.cpu())
        X = torch.cat(feats)
        S = torch.cat(finals)
        bids = basin_ids(S)
        Y = torch.tensor(bids)
        n_cls = len(set(bids))
        perm = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(11 + sd))
        X, Y = X[perm], Y[perm]
        cut = int(X.shape[0] * 0.75)
        acc, _ = probe(X, Y, cut, n_cls)
        Ys = Y[torch.randperm(Y.shape[0], generator=torch.Generator().manual_seed(77 + sd))]
        ctrl, _ = probe(X, Ys, cut, n_cls)
        row = {"seed": sd, "n_basins": n_cls, "acc": acc, "ctrl": ctrl, "chance": 1.0 / n_cls}
        res["cells"].append(row)
        print(
            f"  seed {sd}: 盆 {n_cls}  自预测 {acc:.3f}  打乱 {ctrl:.3f}  机会 {row['chance']:.3f}",
            flush=True,
        )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )

    print("")
    print("判：")
    rows = res["cells"]

    def norm(r, key):
        return (r[key] - r["chance"]) / max(1.0 - r["chance"], 1e-9)

    a = sum(norm(r, "acc") for r in rows) / len(rows)
    c = sum(norm(r, "ctrl") for r in rows) / len(rows)
    m1 = a > 0.75
    print(f"  M12-1 归一化自预测 > 0.75：{a:.3f} => {'过' if m1 else '不过'}")
    m2 = abs(c) <= 0.10
    print(f"  M12-2 打乱配对 ≈ 机会水平（|·|≤0.10）：{c:+.3f} => {'过' if m2 else '不过'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_run())

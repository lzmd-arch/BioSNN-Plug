"""M5（Round D）：**再生** —— 切掉一部分神经元，能力能不能自己回来。

判据在 `research/sel/night-2026-11-16.md`（跑之前）。
能力指标 = 解码历史族的准确率（Round A 建立）。

两臂：`plastic`（继续 STDP）vs `frozen`（冻结突触学习，稳态仍在）。
**M5-2 是这条柱子的核心**：恢复必须依赖自身活动，而不是"切了也没事"。
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

OUT = _pathlib.Path("sweep_results/m5_regeneration")
K = 4
T_HIST = 1024
T_MEAS = 1024
T_TRAIN = 64
FAMS = list(range(8))


@torch.no_grad()
def decode(net, alive, seed: int, per_fam: int, dev: str):
    feats, labels = [], []
    for fam in FAMS:
        toks = make_family_hist(fam, seed * 41 + fam * 11 + 3, T_HIST, per_fam, dev)
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
            t_spont=T_MEAS,
            net=net,
            alive=alive,
        )
        feats.append(spatial_bins(r).cpu())
        labels.append(torch.full((r.shape[0],), fam))
    X = torch.cat(feats)
    Y = torch.cat(labels)
    perm = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(5 + seed))
    X, Y = X[perm], Y[perm]
    cut = int(X.shape[0] * 0.75)
    acc, _ = probe(X, Y, cut, len(FAMS))
    return acc, majority_baseline(Y[:cut], Y[cut:], len(FAMS))


@torch.no_grad()
def train(net, alive, seed: int, batches: int, stdp: float, dev: str) -> None:
    for i in range(batches):
        fam = FAMS[i % len(FAMS)]
        toks = make_family_hist(fam, seed * 613 + i * 17 + 2, T_HIST, 32, dev)
        run_spont(
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
            t_spont=T_TRAIN,
            net=net,
            stdp=stdp,
            alive=alive,
        )


@torch.no_grad()
def lesion_mask(net, seed: int, frac: float, dev: str) -> torch.Tensor:
    """按历史期活跃度切掉最低的 `frac`。**切结构，不是切权重。**"""
    toks = make_family_hist(0, seed * 101 + 9, T_HIST, 32, dev)
    r, _, _ = run_spont(
        seed, 0.5, 0, 0.7, 0.02, dev, K, 0.0, 0.0, -1.0, hist_in=toks, t_spont=256, net=net
    )
    act = r.mean(0)
    thr = torch.quantile(act, frac)
    return torch.where(act <= thr, torch.zeros_like(act), torch.ones_like(act))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--fracs", default="0.1,0.5")
    ap.add_argument("--per-fam", type=int, default=16)
    ap.add_argument("--train-batches", type=int, default=4)
    ap.add_argument("--stdp", type=float, default=0.05)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": []}
    for sd in range(args.seeds):
        base_net = build_pools(sd, dev, K, 0.0)
        alive0 = torch.ones(base_net[0].shape[1], device=dev)
        acc0, floor = decode(base_net, alive0, sd, args.per_fam, dev)
        print(f"  seed {sd}: 基线解码 {acc0:.3f}（底板 {floor:.3f}）", flush=True)
        for frac in [float(x) for x in args.fracs.split(",")]:
            mask = lesion_mask(base_net, sd, frac, dev)
            post, _ = decode(base_net, mask, sd, args.per_fam, dev)
            row = {"seed": sd, "frac": frac, "baseline": acc0, "post": post, "arms": {}}
            for arm, stdp in (("plastic", args.stdp), ("frozen", 0.0)):
                net = (base_net[0], base_net[1], base_net[2].clone(), base_net[3])
                train(net, mask, sd, args.train_batches, stdp, dev)
                rec, _ = decode(net, mask, sd, args.per_fam, dev)
                row["arms"][arm] = rec
            res["cells"].append(row)
            print(
                f"    frac={frac:g}: 切后 {post:.3f} → plastic {row['arms']['plastic']:.3f} "
                f"/ frozen {row['arms']['frozen']:.3f}",
                flush=True,
            )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )

    print("")
    print("判：")
    rows = res["cells"]
    m5_1 = [(r, r["arms"]["plastic"] / max(r["baseline"], 1e-9)) for r in rows]
    ok1 = sum(1 for _, q in m5_1 if q >= 0.70)
    print(
        f"  M5-1 恢复到切前 70%：{[round(q, 2) for _, q in m5_1]} "
        f"=> {ok1}/{len(m5_1)} {'过' if ok1 == len(m5_1) else '不过'}"
    )
    diffs = [r["arms"]["plastic"] - r["arms"]["frozen"] for r in rows]
    wins = sum(1 for d in diffs if d > 0)
    print(
        f"  M5-2 依赖自身活动（plastic > frozen，≥5/6）：差 "
        f"{[round(d, 3) for d in diffs]} => {wins}/{len(diffs)} "
        f"{'过' if wins >= max(5, 0.83 * len(diffs)) else '不过'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

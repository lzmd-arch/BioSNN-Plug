"""M4（Round C）：**终身可塑** —— 学新本领，旧本领退多少。

判据在 `research/sel/night-2026-11-16.md`（跑之前）。

能力指标 = **从终态解码历史族的准确率**（Round A 建立）。
流程：基线测量 → 用族 0–3 做 STDP 训练（旧本领）→ 测量 → 用族 4–7 训练（新本领）→ 测量。

`wrec` 是**同一张量**跨阶段原地累积（`run_spont(net=..., stdp=...)`）。
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

OUT = _pathlib.Path("sweep_results/m4_plasticity")
K = 4
T_HIST = 1024
T_MEAS = 1024
T_TRAIN = 64
OLD = [0, 1, 2, 3]
NEW = [4, 5, 6, 7]


@torch.no_grad()
def measure(net, seed: int, fams: list[int], per_fam: int, dev: str):
    feats, labels = [], []
    for fam in fams:
        toks = make_family_hist(fam, seed * 31 + fam * 7 + 5, T_HIST, per_fam, dev)
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
        )
        feats.append(spatial_bins(r).cpu())
        labels.append(torch.full((r.shape[0],), fams.index(fam)))
    return torch.cat(feats), torch.cat(labels)


def decode(net, seed: int, fams: list[int], per_fam: int, dev: str):
    X, Y = measure(net, seed, fams, per_fam, dev)
    n_cls = len(fams)
    perm = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(7 + seed))
    X, Y = X[perm], Y[perm]
    cut = int(X.shape[0] * 0.75)
    acc, _ = probe(X, Y, cut, n_cls)
    return acc, majority_baseline(Y[:cut], Y[cut:], n_cls)


@torch.no_grad()
def train(net, seed: int, fams: list[int], batches: int, stdp: float, dev: str) -> None:
    for i in range(batches):
        fam = fams[i % len(fams)]
        toks = make_family_hist(fam, seed * 977 + i * 13 + 1, T_HIST, 32, dev)
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
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--per-fam", type=int, default=16)
    ap.add_argument("--train-batches", type=int, default=6)
    ap.add_argument("--stdp", type=float, default=0.05)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": [], "stdp": args.stdp, "train_batches": args.train_batches}
    for sd in range(args.seeds):
        net = build_pools(sd, dev, K, 0.0)
        a0_old, f0_old = decode(net, sd, OLD, args.per_fam, dev)
        a0_new, _ = decode(net, sd, NEW, args.per_fam, dev)
        train(net, sd, OLD, args.train_batches, args.stdp, dev)
        a1_old, _ = decode(net, sd, OLD, args.per_fam, dev)
        a1_new, _ = decode(net, sd, NEW, args.per_fam, dev)
        train(net, sd, NEW, args.train_batches, args.stdp, dev)
        a2_old, f2_old = decode(net, sd, OLD, args.per_fam, dev)
        a2_new, f2_new = decode(net, sd, NEW, args.per_fam, dev)
        row = {
            "seed": sd,
            "old": [a0_old, a1_old, a2_old],
            "new": [a0_new, a1_new, a2_new],
            "floor_old": f0_old,
            "floor_new": f2_new,
            "floor_old_end": f2_old,
        }
        res["cells"].append(row)
        print(
            f"  seed {sd}: 旧族 {a0_old:.3f} → {a1_old:.3f} → {a2_old:.3f} | "
            f"新族 {a0_new:.3f} → {a1_new:.3f} → {a2_new:.3f} | "
            f"底板 {f0_old:.3f}/{f2_new:.3f}",
            flush=True,
        )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )

    print("")
    print("判：")
    rows = res["cells"]
    m = lambda k, i: sum(r[k][i] for r in rows) / len(rows)  # noqa: E731
    base_old = m("old", 1)
    floor = sum(r["floor_old"] for r in rows) / len(rows)
    m0 = base_old > floor + 0.15
    print(
        f"  M4-0 训练前旧族可解码（> 底板+0.15）：{base_old:.3f} vs {floor:.3f} "
        f"=> {'过' if m0 else '不过 => 作废'}"
    )
    if not m0:
        return 0
    drop = base_old - m("old", 2)
    gain = m("new", 2) - m("new", 1)
    m1 = drop < 0.10 and m("new", 2) > floor + 0.15
    print(
        f"  M4-1 旧族退化 {drop:+.3f}（要 <0.10）、新族终值 {m('new', 2):.3f} "
        f"（要 >{floor + 0.15:.3f}）=> {'过' if m1 else '不过'}"
    )
    print(f"       新族在学习旧族后 {m('new', 1):.3f} → 学新后 {m('new', 2):.3f}（{gain:+.3f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

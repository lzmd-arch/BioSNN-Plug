"""M5b：**切谁**才叫损伤 —— 低活跃 / 高活跃 / 随机。

## 为什么补这一轮

M5（Round D）发现"切掉低活跃神经元之后解码**变好**"⇒ 它切的是噪声，不是组织。
**真正的检验是切高活跃的那批**：如果那样也不掉，这个系统就是彻底不脆的；
如果掉，那么"脆性存在，但只在特定靶子上" —— 那是完全不同的一条结论。

**这不是预注册判据，是对 M5 那个混淆的补救**（写于 2026-11-16 夜，M5 结果之后、本轮跑之前）。
判据在这里现写，跑之前落盘：

- **L-0**：发放在窗内；
- **L-1 靶子有差别** ★：`high` 的切后解码 **显著低于** `low`（配对，≥5/6）；
- **L-2 随机在中间**：`low ≤ random ≤ high` 至少 4/6 种子成立。
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
from scripts.m6_pools import N, build_pools, run_spont

OUT = _pathlib.Path("sweep_results/m5b_lesion_target")
K = 4
T_HIST, T_MEAS = 1024, 1024
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
    X, Y = torch.cat(feats), torch.cat(labels)
    perm = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(5 + seed))
    X, Y = X[perm], Y[perm]
    cut = int(X.shape[0] * 0.75)
    acc, _ = probe(X, Y, cut, len(FAMS))
    return acc, majority_baseline(Y[:cut], Y[cut:], len(FAMS))


@torch.no_grad()
def act_of(net, seed: int, dev: str) -> torch.Tensor:
    toks = make_family_hist(0, seed * 101 + 9, T_HIST, 32, dev)
    r, _, _ = run_spont(
        seed, 0.5, 0, 0.7, 0.02, dev, K, 0.0, 0.0, -1.0, hist_in=toks, t_spont=256, net=net
    )
    return r.mean(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--per-fam", type=int, default=16)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": [], "frac": args.frac}
    for sd in range(args.seeds):
        net = build_pools(sd, dev, K, 0.0)
        ones = torch.ones(N, device=dev)
        base, floor = decode(net, ones, sd, args.per_fam, dev)
        act = act_of(net, sd, dev)
        q = torch.quantile(act, args.frac)
        g = torch.Generator().manual_seed(sd + 21)
        masks = {
            "low": torch.where(act <= q, torch.zeros_like(act), torch.ones_like(act)),
            "high": torch.where(act >= q, torch.zeros_like(act), torch.ones_like(act)),
            "random": (torch.randperm(N, generator=g).to(dev) >= int(args.frac * N)).float(),
        }
        row = {"seed": sd, "baseline": base, "floor": floor, "arms": {}}
        for name, m in masks.items():
            acc, _ = decode(net, m, sd, args.per_fam, dev)
            row["arms"][name] = acc
        res["cells"].append(row)
        print(
            f"  seed {sd}: 基线 {base:.3f}（底板 {floor:.3f}）| low {row['arms']['low']:.3f}  "
            f"random {row['arms']['random']:.3f}  **high {row['arms']['high']:.3f}**",
            flush=True,
        )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )

    print("")
    print("判：")
    rows = res["cells"]
    ok = [r for r in rows if 0.01 <= r["baseline"] <= 1.0]
    w1 = sum(1 for r in ok if r["arms"]["high"] < r["arms"]["low"])
    print(
        f"  L-1 high < low（≥5/6）：{[round(r['arms']['high'] - r['arms']['low'], 3) for r in ok]} "
        f"=> {w1}/{len(ok)} {'过' if w1 >= max(5, 0.83 * len(ok)) else '不过'}"
    )
    w2 = sum(
        1
        for r in ok
        if r["arms"]["low"] <= r["arms"]["random"] <= r["arms"]["high"]
        or r["arms"]["high"] <= r["arms"]["random"] <= r["arms"]["low"]
    )
    print(f"  L-2 随机在中间：{w2}/{len(ok)} {'过' if w2 >= max(4, 0.66 * len(ok)) else '不过'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

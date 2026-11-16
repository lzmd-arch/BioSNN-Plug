"""M6ad2：**顺序判别** —— 状态能不能分清"A 然后 B"和"B 然后 A"。

## 为什么换这个任务

`M6ad` 的复合任务（4 族 × 4 族 = 16 类，128 样本）**基线就不可解码**（0.000 / 0.031），
测不了脆性。

**顺序判别是最小的时间整合检验**：两段历史、只有顺序不同。
- (0,0) 与 (1,1)：单调，不需要整合；
- **(0,1) 与 (1,0)：只有先后不同** —— 状态必须携带"顺序"这个时间信息。

4 类、128 样本，读出可行。**而且它依赖网络完整性**（顺序信息最容易在损伤下丢失），
所以它同时是脆性的候选检验。

## 判据

- **AO-0** 发放在窗内；
- **AO-1 能分单调对**：`(0,0)` vs `(1,1)` 的二分类 > 0.75（前提检查）；
- **AO-2 能分顺序对** ★★：`(0,1)` vs `(1,0)` 的二分类 > 机会 + 0.15；
- **AO-3 有脆性** ★：切 50% 后 AO-2 的二分类掉到机会附近。
"""

from __future__ import annotations

# ruff: noqa: N803, N806, SIM300
import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.hist_families import make_family_hist, spatial_bins
from scripts.instrument import probe
from scripts.m6_pools import N, build_pools, run_spont

OUT = _pathlib.Path("sweep_results/m6ad2_order")
K = 4
T_HIST, T_SPONT = 1024, 2048
HALF = T_HIST // 2


def ordered_hist(a: int, b: int, seed: int, per: int, dev: str) -> torch.Tensor:
    h1 = make_family_hist(a, seed * 1013 + a * 7 + 1, HALF, per, dev)
    h2 = make_family_hist(b, seed * 1013 + b * 11 + 2, HALF, per, dev)
    return torch.cat([h1, h2], dim=1)


@torch.no_grad()
def measure(net, alive, seed: int, per: int, dev: str):
    X, Y = [], []
    for idx, (a, b) in enumerate([(0, 0), (1, 1), (0, 1), (1, 0)]):
        toks = ordered_hist(a, b, seed, per, dev)
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
            alive=alive,
        )[0]
        X.append(spatial_bins(r).cpu())
        Y.append(torch.full((r.shape[0],), idx))
    return torch.cat(X), torch.cat(Y)


def two_class_acc(X, Y, keep: tuple[int, int]) -> float:
    m = (Y == keep[0]) | (Y == keep[1])
    X, Y = X[m], Y[m]
    Y = (Y == keep[1]).long()
    perm = torch.randperm(Y.shape[0], generator=torch.Generator().manual_seed(17))
    X, Y = X[perm], Y[perm]
    cut = int(Y.shape[0] * 0.75)
    return probe(X, Y, cut, 2)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--per", type=int, default=32)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": []}
    for sd in range(args.seeds):
        net = build_pools(sd, dev, K, 0.0)
        ones = torch.ones(N, device=dev)
        X0, Y0 = measure(net, ones, sd, args.per, dev)
        g = torch.Generator().manual_seed(sd + 211)
        mask = (torch.randperm(N, generator=g).to(dev) >= int(args.frac * N)).float()
        X1, Y1 = measure(net, mask, sd, args.per, dev)
        mono0 = two_class_acc(X0, Y0, (0, 1))
        ord0 = two_class_acc(X0, Y0, (2, 3))
        mono1 = two_class_acc(X1, Y1, (0, 1))
        ord1 = two_class_acc(X1, Y1, (2, 3))
        row = {"seed": sd, "mono": mono0, "order": ord0, "mono_post": mono1, "order_post": ord1}
        res["cells"].append(row)
        print(
            f"  seed {sd}: 单调对 {mono0:.3f}  顺序对 {ord0:.3f} | "
            f"切后 单调 {mono1:.3f}  顺序 {ord1:.3f}",
            flush=True,
        )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )

    print("")
    print("判：")
    rows = res["cells"]
    m = lambda k, _r=rows: sum(r[k] for r in _r) / len(_r)  # noqa: E731
    print(
        f"  AO-1 单调对可解码（>0.75）：{m('mono'):.3f} => {'过' if m('mono') > 0.75 else '不过'}"
    )
    ao2 = m("order") > 0.65
    print(
        f"  AO-2 顺序可判别（>0.65，即机会 0.5 +0.15）：{m('order'):.3f} "
        f"=> {'过' if ao2 else '不过'}"
    )
    drop = m("order") - m("order_post")
    print(
        f"  AO-3 切 50% 后顺序判别下降：{m('order'):.3f} → {m('order_post'):.3f} "
        f"（{drop:+.3f}）=> {'有脆性' if drop > 0.10 else '无脆性'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

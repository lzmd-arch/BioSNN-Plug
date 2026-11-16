"""M6am：**双群体 + 慢耦合** —— 架构提案的核心预测。

## 提案（目标文档 §32）

实测：一个群体不能同时具备"高维 / 整合 / 可读 / 时间整合"——
两个 regime 擅长的事完全不重叠，而且三性状有单调交换曲线。

**所以建两个群体**：
- **A（全局场）**：低维、可解码 0.61–0.73、整合 0.83、**顺序 0.698**；
- **B（局部场 + 长程）**：高维 PR 47–108。
- 用**慢耦合**连起来（不是即时标量：M6y 证明即时标量加到 16 倍也补不回整合）。

## 判据

- **AM-0** 两个群体的发放都在窗内；
- **AM-1 联合状态高维** ★：联合（A 与 B 拼接）的 PR > 10；
- **AM-2 联合状态仍能判别顺序** ★★：顺序对 > 0.65（机会 0.5）；
- **AM-3 耦合不破坏** ★：`c=0.1` 时 AM-1 与 AM-2 仍成立（与 `c=0` 相比不塌）。

**AM-2 与 AM-3 一起才说明"双群体真的同时具备两件事"，而不是简单拼接。**
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
from scripts.m6_pools import N, build_pools
from scripts.m6v_dimension import pr_of

OUT = _pathlib.Path("sweep_results/m6am_two_pop")
K, SIDE = 4, 32
T_HIST, T_SPONT = 1024, 2048
HALF = T_HIST // 2
TAU_SLOW = 100.0


@torch.no_grad()
def run_two(netA, netB, seed: int, hist, dev: str, t_spont: int, c_couple: float):
    """两个群体同步演化，靠**慢变量**互相耦合。返回 `(rateA, rateB)`。"""
    wA, preA, wrecA, _ = netA
    wA = wA * 0.7
    netB = (netB[0] * 0.7, netB[1], netB[2], netB[3])
    wB, preB, wrecB, _ = netB
    B = hist.shape[0]
    vA = torch.zeros((B, N), device=dev)
    zA = torch.zeros((B, N), device=dev)
    rA = torch.zeros((B, N), device=dev)
    vB = torch.zeros((B, N), device=dev)
    zB = torch.zeros((B, N), device=dev)
    rB = torch.zeros((B, N), device=dev)
    g = torch.Generator().manual_seed(seed + 5150)
    P_ab = (torch.randn(N, N, generator=g) / N**0.5).to(dev) * 0.1
    P_ba = (torch.randn(N, N, generator=g) / N**0.5).to(dev) * 0.1
    # **起搏电流**：run_spont 里有，这里漏过一次，结果 B 群体直接死掉（发放 0.0000）
    pace = torch.rand(N, generator=torch.Generator().manual_seed(seed + 31337)) < 0.02
    pace = pace.float().to(dev) * 0.5
    slowA = torch.full((N,), 0.02, device=dev)
    slowB = torch.full((N,), 0.02, device=dev)
    a = 1.0 / TAU_SLOW
    rateA = torch.zeros((B, N), device=dev)
    rateB = torch.zeros((B, N), device=dev)

    # 群体 B 的局部场矩阵
    ii = torch.arange(N, device=dev)
    xi, yi = (ii % SIDE).float(), (ii // SIDE).float()
    dx = torch.minimum(
        (xi.unsqueeze(1) - xi.unsqueeze(0)).abs(), SIDE - (xi.unsqueeze(1) - xi.unsqueeze(0)).abs()
    )
    dy = torch.minimum(
        (yi.unsqueeze(1) - yi.unsqueeze(0)).abs(), SIDE - (yi.unsqueeze(1) - yi.unsqueeze(0)).abs()
    )
    near = (torch.sqrt(dx * dx + dy * dy) <= 2.0).float()
    fmat = near / near.sum(1, keepdim=True).clamp(min=1.0)

    n_steps = hist.shape[1] + t_spont
    for t in range(n_steps):
        ext = hist[:, t] if t < hist.shape[1] else None
        inA = wA[ext] if ext is not None else torch.zeros((B, N), device=dev)
        inB = wB[ext] if ext is not None else torch.zeros((B, N), device=dev)
        if ext is None:
            inA = pace.unsqueeze(0).expand(B, -1)
            inB = pace.unsqueeze(0).expand(B, -1)
        # 慢耦合
        cA = c_couple * (slowB @ P_ba) if c_couple else 0.0
        cB = c_couple * (slowA @ P_ab) if c_couple else 0.0
        recA = torch.zeros_like(zA)
        for j in range(preA.shape[1]):
            recA = recA + zA[:, preA[:, j]] * wrecA[:, j].unsqueeze(0)
        recB = torch.zeros_like(zB)
        for j in range(preB.shape[1]):
            recB = recB + zB[:, preB[:, j]] * wrecB[:, j].unsqueeze(0)
        phiA = float(zA.mean())
        curA = inA - 1.0 * phiA + cA
        nvA = 0.95 * vA + curA + recA - zA * 0.5
        nzA = torch.where(rA > 0, torch.zeros_like(nvA), ((nvA - 0.5) / 0.5 > 0).float())
        rA = torch.clamp(rA + 2.0 * nzA - 1.0, 0.0, 2.0)
        vA, zA = nvA, nzA
        locB = zB @ fmat
        curB = inB - 2.0 * locB + cB
        nvB = 0.95 * vB + curB + recB - zB * 0.5
        nzB = torch.where(rB > 0, torch.zeros_like(nvB), ((nvB - 0.5) / 0.5 > 0).float())
        rB = torch.clamp(rB + 2.0 * nzB - 1.0, 0.0, 2.0)
        vB, zB = nvB, nzB
        slowA = (1 - a) * slowA + a * zA.mean(0)
        slowB = (1 - a) * slowB + a * zB.mean(0)
        if t >= n_steps - 64:
            rateA += zA
            rateB += zB
    return rateA / 64, rateB / 64


def two_class_acc(X, Y, keep: tuple[int, int]) -> float:
    m = (Y == keep[0]) | (Y == keep[1])
    X, Y = X[m], Y[m]
    Y = (Y == keep[1]).long()
    perm = torch.randperm(Y.shape[0], generator=torch.Generator().manual_seed(23))
    X, Y = X[perm], Y[perm]
    cut = int(Y.shape[0] * 0.75)
    return probe(X, Y, cut, 2)[0]


def ordered_hist(a: int, b: int, seed: int, per: int, dev: str) -> torch.Tensor:
    h1 = make_family_hist(a, seed * 1013 + a * 7 + 1, HALF, per, dev)
    h2 = make_family_hist(b, seed * 1013 + b * 11 + 2, HALF, per, dev)
    return torch.cat([h1, h2], dim=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--per", type=int, default=32)
    ap.add_argument("--cs", default="0,0.1")
    ap.add_argument("--p-long", type=float, default=0.5)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": {}}
    for c in [float(x) for x in args.cs.split(",")]:
        rows = []
        for sd in range(args.seeds):
            netA = build_pools(sd, dev, K, 0.0)
            netB = build_pools(sd, dev, K, 0.0, p_long=args.p_long)
            X, Y, fA, fB, states = [], [], [], [], []
            for idx, (a_, b_) in enumerate([(0, 0), (1, 1), (0, 1), (1, 0)]):
                toks = ordered_hist(a_, b_, sd, args.per, dev)
                rA, rB = run_two(netA, netB, sd, toks, dev, T_SPONT, c)
                # 两个群体各 1024 神经元，**分别分桶再拼**（128 维）
                feat = torch.cat([spatial_bins(rA).cpu(), spatial_bins(rB).cpu()], dim=1)
                X.append(feat)
                Y.append(torch.full((rA.shape[0],), idx))
                fA.append(float(rA.mean()))
                fB.append(float(rB.mean()))
                # **逐群体标准化再拼**：A 的发放是 B 的 7 倍，不归一化时协方差被 A 吃光
                na = (rA - rA.mean()) / (rA.std() + 1e-9)
                nb = (rB - rB.mean()) / (rB.std() + 1e-9)
                states.append(torch.cat([na.cpu(), nb.cpu()], dim=1))
            Xc, Yc = torch.cat(X), torch.cat(Y)
            order = two_class_acc(Xc, Yc, (2, 3))
            rows.append(
                {
                    "seed": sd,
                    "order": order,
                    "firingA": sum(fA) / len(fA),
                    "firingB": sum(fB) / len(fB),
                    "pr": pr_of(torch.cat(states)),
                }
            )
            print(
                f"  c={c:g} seed {sd}: 顺序 {order:.3f}  发放 A {rows[-1]['firingA']:.4f} "
                f"B {rows[-1]['firingB']:.4f}  PR {rows[-1]['pr']:.1f}",
                flush=True,
            )
        m = lambda k, _r=rows: sum(r[k] for r in _r) / len(_r)  # noqa: E731
        res["cells"][f"c={c:g}"] = {
            "c": c,
            "order": m("order"),
            "firingA": m("firingA"),
            "firingB": m("firingB"),
            "rows": rows,
        }
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )

    print("")
    print("判：")
    cc = res["cells"]
    for k, v in cc.items():
        inwin = 0.01 <= v["firingA"] <= 0.20 and 0.01 <= v["firingB"] <= 0.20
        print(
            f"  {k}: 顺序 {v['order']:.3f}  窗内 {'✅' if inwin else '❌'} "
            f"=> {'过' if inwin and v['order'] > 0.65 else '不过'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

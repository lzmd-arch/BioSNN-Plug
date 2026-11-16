"""给高维补上生物学：**轴突延迟**与**兴奋/抑制平衡**。

## 为什么"三维不如一维"可能是我实现的问题

生物皮层确实是三维的，而三维在生物体里之所以有用，靠的**不只是"邻居多"**：

| 生物特征 | 我们的网格 | 后果 |
| :--- | :--- | :--- |
| **轴突传导延迟 ∝ 距离** | 所有连接**同一时刻**到（延迟=1） | 空间距离在时间上没有体现，等于白摆 |
| **兴奋/抑制平衡**（皮层约 8:2） | 递归**全兴奋** | 网络容易饱和，高维更甚 |
| 距离衰减的**概率**连接 | 固定 ±1 邻居 | 没有小世界性 |

**延迟是最被低估的一条**：它把"空间距离"转成"时间结构"，储层计算里延迟扩展记忆容量是有明确证据的。
没有延迟的三维，只是"一个邻居更多的图"，不是三维。

## 本实验

| 维度 | 形状 |
| :--- | :--- |
| 1D | 4096（半径 4：±1..±4） |
| 3D | 16×16×16（半径 1：26 邻居） |
| 6D | 4×4×4×4×4×4（半径 1：12 邻居） |

四档生物学：**基线 / +延迟 / +E-I / +延迟 +E-I**。

| 条件 | 做法 |
| :--- | :--- |
| **延迟** | 每步维护 `D=4` 步的脉冲历史；连接延迟 = `round(欧氏距离)`，截到 `[1, D]` |
| **E-I** | 随机 20% 的递归权重取负 |

其余全部沿用 `dim_ladder`：神经元数 4096、递归增益归一 1.20、稠密随机输入、全局非线性读出、3 种子。

## 判据

| 编号 | 判据 |
| :--- | :--- |
| **B-1** | 任一高维在任一生物学条件下 > 1D 同条件 **≥5pp**，且三种子同向 |
| **B-2** | 同一维度下，加生物学特征比基线高 ≥5pp |

**B-1 不过 ⇒ 三维收口，理由升级为"补了延迟与 E-I 仍然不赢"。**
"""

from __future__ import annotations

# ruff: noqa: N806  —— b, T, n 是这类循环的惯例命名

import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.dim_ladder import mlp, ridge
from scripts.structure_ladder import V, make_pref, make_task, window_features

OUT = _pathlib.Path("sweep_results/dim_biology")
TAUS = (2.0, 16.0)
DIMS = {"1D": (4096,), "3D": (16, 16, 16), "6D": (4, 4, 4, 4, 4, 4)}
RADIUS = {"1D": 4, "3D": 1, "6D": 1}


def coords_of(n: int, dims: tuple[int, ...]) -> torch.Tensor:
    idx = torch.arange(n)
    out, stride = [], 1
    for d in reversed(dims):
        out.append((idx // stride) % d)
        stride *= d
    return torch.stack(out[::-1], 1)


def lin_of(c: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    out, stride = torch.zeros(c.shape[0], dtype=torch.long), 1
    for k in range(len(dims) - 1, -1, -1):
        out = out + c[:, k] * stride
        stride *= dims[k]
    return out


def build(name: str, n: int, gain: float, seed: int, dev: str, max_delay: int):
    """返回 (w_in, pre, wrec, delay)。`delay[n,k]` 是第 k 条连接要回看的步数。"""
    g = torch.Generator().manual_seed(seed)
    dims, r = DIMS[name], RADIUS[name]
    w_in = (torch.randn(V, n, generator=g) / n**0.5).to(dev)
    ci = coords_of(n, dims)
    # 半径 r 内的全部邻居（含环绕），同时记下欧氏距离
    offs = torch.cartesian_prod(*[torch.arange(-r, r + 1) for _ in range(len(dims))])
    offs = offs.reshape(-1, len(dims))  # 1D 时 cartesian_prod 返回一维，先摊成 (M, ndim)
    offs = offs[(offs != 0).any(dim=1)]
    # 高维的半径 1 邻域会爆炸（6D 是 3^6-1=728）。**封顶到 32 个邻居**，
    # 否则既慢又不再是"局部"连接。固定抽样，保证可复现。
    if offs.shape[0] > 32:
        g2 = torch.Generator().manual_seed(11)
        keep = torch.randperm(offs.shape[0], generator=g2)[:32]
        offs = offs[keep]
    dist = offs.float().norm(dim=1)
    nb, dl = [], []
    for j in range(offs.shape[0]):
        c2 = (ci + offs[j]) % torch.tensor(dims)
        nb.append(lin_of(c2, dims))
        dl.append(dist[j].round().clamp(1, max_delay).long().expand(n))
    pre = torch.stack(nb, 1)
    delay = torch.stack(dl, 1).clamp(1, max_delay)
    wrec = torch.rand(n, pre.shape[1], generator=g).to(dev)
    wrec = wrec * (gain / wrec.sum(1, keepdim=True))
    return w_in, pre, wrec, delay


@torch.no_grad()
def run(w_in, pre, wrec, delay, toks, dev, max_delay, ei_frac, stride=8):
    b, T = toks.shape
    n = w_in.shape[1]
    v = torch.zeros((b, n), device=dev)
    z = torch.zeros((b, n), device=dev)
    ref = torch.zeros((b, n), device=dev)
    decays = torch.tensor([1.0 / t for t in TAUS], device=dev)
    read = torch.zeros((b, len(TAUS), n), device=dev)
    hist = [torch.zeros_like(z) for _ in range(max_delay)]  # hist[0] = 上一步
    sign = torch.ones(n, pre.shape[1], device=dev)
    if ei_frac > 0:
        g = torch.Generator().manual_seed(7)
        mask = torch.rand(n, pre.shape[1], generator=g).to(dev) < ei_frac
        sign = torch.where(mask, -torch.ones_like(sign), sign)
    weff = wrec * sign
    feats = []
    for t in range(T):
        rec = torch.zeros_like(z)
        for k in range(pre.shape[1]):
            pk, dk = pre[:, k], delay[:, k]
            for d in range(1, max_delay + 1):
                m = dk == d
                if m.any():
                    rec[:, m] += hist[d - 1][:, pk[m]] * weff[m, k].unsqueeze(0)
        nv = 0.95 * v + w_in[toks[:, t]] + rec - z * 0.5
        fresh = ((nv - 0.5) / 0.5 > 0).float()
        nz = torch.where(ref > 0, torch.zeros_like(fresh), fresh)
        ref = torch.clamp(ref + 2.0 * nz - 1.0, 0.0, 2.0)
        v, z = nv, nz
        hist = [z, *hist[:-1]]
        read = read * decays.view(1, -1, 1) + z.unsqueeze(1)
        if t % stride == 0:
            feats.append(read.reshape(b, -1))
    return torch.stack(feats, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=96)
    ap.add_argument("--train-seq", type=int, default=256)
    ap.add_argument("--test-seq", type=int, default=64)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--gain", type=float, default=1.2)
    ap.add_argument("--max-delay", type=int, default=4)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(x) for x in args.seeds.split(",")]
    res: dict = {"max_delay": args.max_delay, "cells": {}}

    conds = [
        ("base", 1, 0.0),
        ("delay", args.max_delay, 0.0),
        ("ei", 1, 0.2),
        ("delay+ei", args.max_delay, 0.2),
    ]

    ctl = []
    for sd in seeds:
        pref = make_pref(0, 3.0).to(dev)
        tr, trs, _ = make_task(args.train_seq, args.seq, 10 + sd, dev, pref)
        te, tes, _ = make_task(args.test_seq, args.seq, 20 + sd, dev, pref)
        ctl.append(
            ridge(
                window_features(torch.cat([tr, te]), 32, args.stride),
                torch.cat([trs, tes])[:, :: args.stride],
                args.train_seq,
            )
        )
    res["control"] = sum(ctl) / len(ctl)
    print(f"  窗口32 对照 {res['control']:.3f}", flush=True)

    for name in DIMS:
        for cname, md, ei in conds:
            accs = []
            for sd in seeds:
                pref = make_pref(0, 3.0).to(dev)
                tr, trs, _ = make_task(args.train_seq, args.seq, 10 + sd, dev, pref)
                te, tes, _ = make_task(args.test_seq, args.seq, 20 + sd, dev, pref)
                toks = torch.cat([tr, te])
                st = torch.cat([trs, tes])[:, :: args.stride]
                w_in, pre, wrec, delay = build(name, 4096, args.gain, sd, dev, md)
                feat = run(w_in, pre, wrec, delay, toks, dev, md, ei, args.stride)
                accs.append(mlp(feat, st, args.train_seq))
            key = f"{name}/{cname}"
            res["cells"][key] = {"acc": sum(accs) / len(accs), "runs": accs}
            print(
                f"  {name:3s} {cname:9s} {sum(accs) / len(accs):.3f}  "
                f"({', '.join(f'{a:.3f}' for a in accs)})",
                flush=True,
            )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print("\n判：")
    best = max(res["cells"], key=lambda k: res["cells"][k]["acc"])
    b = res["cells"][best]
    ref = res["cells"]["1D/" + best.split("/")[1]]
    gain = b["acc"] - ref["acc"]
    same = all(x >= y - 1e-9 for x, y in zip(b["runs"], ref["runs"], strict=False))
    print(
        f"  B-1 最佳 {best} {b['acc']:.3f} − 1D同条件 {ref['acc']:.3f} = {gain:+.3f}；"
        f"三种子同向 {'是' if same else '否'} ⇒ {'过' if gain >= 0.05 and same else '不过'}"
    )
    for name in DIMS:
        base = res["cells"][f"{name}/base"]["acc"]
        best_c = max(res["cells"][f"{name}/{c}"]["acc"] for c in ("delay", "ei", "delay+ei"))
        print(f"  B-2 {name} 生物学增益 {best_c - base:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

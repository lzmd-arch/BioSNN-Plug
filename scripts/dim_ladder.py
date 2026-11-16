"""维度阶梯：**等神经元数、等递归增益、随机输入**，只让"空间是几维"变。

## 为什么这样设计

上一轮（`structure_matrix.py`）证明了三件事：

1. 递归驱动必须**归一**——高维拓扑天然有更多邻居，不归一就是在比"谁电流大"；
2. 读出必须**非线性**——MLP 比线性普遍高 3.7–8.2pp；
3. 读出必须**全局**——局部块读出把高维的空间结构抹平了。

所以这里固定：`n=4096`、递归增益 **1.20**、输入**稠密随机**（不引入地形映射这个第三变量）、
读出**全局**（线性 / MLP 两档）。**唯一变量是网格的维数。**

## 维度与形状（n = 4096 = 2^12，各维尽量等长）

| 维数 | 形状 |
| :--- | :--- |
| 1D | 4096 |
| 2D | 64 × 64 |
| 3D | 16 × 16 × 16 |
| 4D | 8 × 8 × 8 × 8 |
| 6D | 4 × 4 × 4 × 4 × 4 × 4 |

（5D 无法在 2^12 上取等长，跳过；4D→6D 已经能看出趋势。）

## 判据

| 编号 | 判据 |
| :--- | :--- |
| **D-0** | 窗口对照 ≥ 随机 + 20pp（任务可解） |
| **D-1** | 最佳维数比 1D 高 **≥5pp**，且**三种子都同向** |
| **D-2** | 相邻维数之间的增益 <2pp 即视为**饱和**，取饱和点之前那一维 |
"""

from __future__ import annotations

import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.structure_ladder import K, V, make_pref, make_task, window_features

OUT = _pathlib.Path("sweep_results/dim_ladder")
DIMS = [
    ("1D", (4096,)),
    ("2D", (64, 64)),
    ("3D", (16, 16, 16)),
    ("4D", (8, 8, 8, 8)),
    ("6D", (4, 4, 4, 4, 4, 4)),
]
TAUS = (2.0, 16.0)


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


def build(n: int, dims: tuple[int, ...], gain: float, seed: int, dev: str):
    g = torch.Generator().manual_seed(seed)
    w_in = (torch.randn(V, n, generator=g) / n**0.5).to(dev)  # 稠密随机输入：唯一的变量是连接拓扑
    ci = coords_of(n, dims)
    nb = []
    for ax in range(len(dims)):
        for delta in (1, -1):
            c2 = ci.clone()
            c2[:, ax] = (c2[:, ax] + delta) % dims[ax]
            nb.append(lin_of(c2, dims))
    pre = torch.stack(nb, 1)  # (n, 2·ndim)
    wrec = torch.rand(n, pre.shape[1], generator=g).to(dev)
    wrec = wrec * (gain / wrec.sum(1, keepdim=True))  # **归一**
    return w_in, pre, wrec


@torch.no_grad()
def run(w_in, pre, wrec, toks, dev, stride=8):
    b, T = toks.shape
    n = w_in.shape[1]
    v = torch.zeros((b, n), device=dev)
    z = torch.zeros((b, n), device=dev)
    ref = torch.zeros((b, n), device=dev)
    decays = torch.tensor([1.0 / t for t in TAUS], device=dev)
    read = torch.zeros((b, len(TAUS), n), device=dev)
    feats = []
    for t in range(T):
        cur = w_in[toks[:, t]]
        rec = torch.zeros_like(z)
        for s in range(pre.shape[1]):
            rec = rec + z[:, pre[:, s]] * wrec[:, s].unsqueeze(0)
        nv = 0.95 * v + cur + rec - z * 0.5
        fresh = ((nv - 0.5) / 0.5 > 0).float()
        nz = torch.where(ref > 0, torch.zeros_like(fresh), fresh)
        ref = torch.clamp(ref + 2.0 * nz - 1.0, 0.0, 2.0)
        v, z = nv, nz
        read = read * decays.view(1, -1, 1) + z.unsqueeze(1)
        if t % stride == 0:
            feats.append(read.reshape(b, -1))
    return torch.stack(feats, 1)


def ridge(feats, st, n_train):
    X = feats.reshape(-1, feats.shape[-1])
    Y = st.reshape(-1)
    cut = n_train * st.shape[1]
    Xtr, Xte = X[:cut], X[cut:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Yh = torch.nn.functional.one_hot(Y[:cut], K).float()
    A = Xtr.T @ Xtr + 1.0 * torch.eye(Xtr.shape[1], device=X.device)
    W = torch.linalg.solve(A, Xtr.T @ Yh)
    return float(((Xte @ W).argmax(-1) == Y[cut:]).float().mean())


def mlp(feats, st, n_train, steps=600, hidden=256):
    X = feats.reshape(-1, feats.shape[-1])
    Y = st.reshape(-1)
    cut = n_train * st.shape[1]
    Xtr, Xte = X[:cut], X[cut:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Ytr, Yte = Y[:cut], Y[cut:]
    net = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.GELU(), torch.nn.Linear(hidden, K)
    ).to(X.device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(steps):
        perm = torch.randperm(Xtr.shape[0], device=X.device)[:8192]
        opt.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(net(Xtr[perm]), Ytr[perm]).backward()
        opt.step()
    with torch.no_grad():
        return float((net(Xte).argmax(-1) == Yte).float().mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=96)
    ap.add_argument("--train-seq", type=int, default=256)
    ap.add_argument("--test-seq", type=int, default=64)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--gain", type=float, default=1.2)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(x) for x in args.seeds.split(",")]

    res: dict = {"gain": args.gain, "seeds": seeds, "cells": {}}
    wf = None
    ctl = []
    for sd in seeds:
        pref = make_pref(0, 3.0).to(dev)
        tr, trs, _ = make_task(args.train_seq, args.seq, 10 + sd, dev, pref)
        te, tes, _ = make_task(args.test_seq, args.seq, 20 + sd, dev, pref)
        toks = torch.cat([tr, te])
        st = torch.cat([trs, tes])[:, :: args.stride]
        wf = window_features(toks, L=32, stride=args.stride)
        ctl.append(ridge(wf, st, args.train_seq))
    res["control"] = {"window32_mean": sum(ctl) / len(ctl), "runs": ctl}
    print(f"  窗口32 对照 {sum(ctl) / len(ctl):.3f}", flush=True)

    for name, dims in DIMS:
        for dec_name, dec in (("linear", ridge), ("mlp", mlp)):
            accs = []
            for sd in seeds:
                pref = make_pref(0, 3.0).to(dev)
                tr, trs, _ = make_task(args.train_seq, args.seq, 10 + sd, dev, pref)
                te, tes, _ = make_task(args.test_seq, args.seq, 20 + sd, dev, pref)
                toks = torch.cat([tr, te])
                st = torch.cat([trs, tes])[:, :: args.stride]
                w_in, pre, wrec = build(4096, dims, args.gain, sd, dev)
                feat = run(w_in, pre, wrec, toks, dev, args.stride)
                accs.append(dec(feat, st, args.train_seq))
            key = f"{name}/{dec_name}"
            res["cells"][key] = {
                "acc": sum(accs) / len(accs),
                "runs": accs,
                "range_pp": (max(accs) - min(accs)) * 100,
            }
            print(
                f"  {name:3s} {dec_name:6s} {sum(accs) / len(accs):.3f}  "
                f"({', '.join(f'{a:.3f}' for a in accs)})",
                flush=True,
            )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )

    print("\n判：")
    best_key = max(res["cells"], key=lambda k: res["cells"][k]["acc"])
    b = res["cells"][best_key]
    base = res["cells"]["1D/" + best_key.split("/")[1]]
    print(
        f"  D-0 窗口32 {res['control']['window32_mean']:.3f} ≥ {1 / K + 0.2:.3f} ⇒ "
        f"{'过' if res['control']['window32_mean'] >= 1 / K + 0.2 else '不过'}"
    )
    gain = b["acc"] - base["acc"]
    same = all(x >= y - 1e-9 for x, y in zip(b["runs"], base["runs"]))
    print(
        f"  D-1 最佳 {best_key} {b['acc']:.3f} − 1D {base['acc']:.3f} = {gain:+.3f}；"
        f"三种子同向 {'是' if same else '否'} ⇒ {'过' if gain >= 0.05 and same else '不过'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

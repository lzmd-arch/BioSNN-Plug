"""四个部件同时高维：连接度数 / 神经元 τ / 读出时间基底 / 读出空间。

## 为什么是这四个

之前所有结构实验都只动**一个**部件，而且读出那一侧一直是低维的：
2 个 τ 的 EMA 把 T=96 步的脉冲序列压成两维，再全局求和。**任何超过这两个 τ 的时间结构、
任何神经元之间的差异，在这条读出里都不存在。** 用这样的读出比较布线，等于蒙着眼睛比。

| 部件 | 低维版 | 高维版 |
| :--- | :--- | :--- |
| **连接** | 环，度数 4（0.1%） | **稠密随机，度数 4096（100%）** |
| **神经元 τ** | 全部 α=0.95 | **逐神经元异质 τ，log-uniform [5, 500]** |
| **读出时间基底** | 2 个 τ | **8 个 τ** |
| **读出空间** | 全局求和后 MLP | 全神经元（8192→32768 维）后 MLP |

输入本来就是稠密随机（高维），不动。

## 消融阶梯（防止又只测一个变量）

| 臂 | 连接 | τ | 读出 |
| :--- | :--- | :--- | :--- |
| `low` | 环 4 | 均一 | 2 τ |
| `+conn` | **稠密** | 均一 | 2 τ |
| `+tau` | 环 4 | **异质** | 2 τ |
| `+read` | 环 4 | 均一 | **8 τ** |
| **`all`** | **稠密** | **异质** | **8 τ** |

## 判据

| 编号 | 判据 |
| :--- | :--- |
| **H-0** | 每臂发放率 ∈ [0.03, 0.20] |
| **H-1** | acc ≥ 多数类 + 15pp |
| **H-2** | `all` 比 `low` 高 **≥10pp**，三种子同向 |
| **H-3** | 四个单因素臂各自相对 `low` 的增益都要报（判断是否可加） |
"""

from __future__ import annotations

import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import math

import torch

from scripts.dim_ladder import mlp
from scripts.structure_ladder import K, V, make_pref, make_task

OUT = _pathlib.Path("sweep_results/highdim_all")
TAUS8 = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
TAUS2 = (2.0, 16.0)


def build_conn(kind: str, n: int, g: float, seed: int, dev: str):
    gen = torch.Generator().manual_seed(seed)
    if kind == "ring":
        off = torch.tensor([-1, -2, 1, 2])
        pre = (torch.arange(n).unsqueeze(1) + off.unsqueeze(0)) % n
        w = torch.rand(n, 4, generator=gen).to(dev)
        # **增益必须走参数**：之前写死 1.2，导致"扫增益"其实没扫，
        # 环永远自激在 20%、稠密永远安静，两边没有可比的工作区间。
        w = w * (g / w.sum(1, keepdim=True))
        return pre, w
    if kind == "dense":
        # **度数 = 全部神经元**：真正的高维混合。按 sqrt(n) 缩放后再乘增益。
        w = torch.randn(n, n, generator=gen).to(dev) / n**0.5
        return None, w * g
    raise ValueError(kind)


def build_alpha(kind: str, n: int, seed: int, dev: str):
    if kind == "uniform":
        return torch.full((n,), 0.95, device=dev)
    gen = torch.Generator().manual_seed(seed + 100)
    lo, hi = math.log(5.0), math.log(500.0)
    tau = torch.exp(torch.rand(n, generator=gen) * (hi - lo) + lo)
    return torch.exp(-1.0 / tau).to(dev)


@torch.no_grad()
def run_ring(w_in, pre, wrec, alpha, toks, taus, dev, stride=8):
    b, T = toks.shape
    n = w_in.shape[1]
    v = torch.zeros((b, n), device=dev)
    z = torch.zeros((b, n), device=dev)
    ref = torch.zeros((b, n), device=dev)
    decays = torch.tensor([1.0 / t for t in taus], device=dev)
    read = torch.zeros((b, len(taus), n), device=dev)
    feats, rate = [], 0.0
    for t in range(T):
        cur = w_in[toks[:, t]]
        rec = torch.zeros_like(z)
        for k in range(pre.shape[1]):
            rec = rec + z[:, pre[:, k]] * wrec[:, k].unsqueeze(0)
        nv = alpha * v + cur + rec - z * 0.5
        fresh = ((nv - 0.5) / 0.5 > 0).float()
        nz = torch.where(ref > 0, torch.zeros_like(fresh), fresh)
        ref = torch.clamp(ref + 2.0 * nz - 1.0, 0.0, 2.0)
        v, z = nv, nz
        rate += float(z.mean())
        read = read * decays.view(1, -1, 1) + z.unsqueeze(1)
        if t % stride == 0:
            feats.append(read.reshape(b, -1))
    return torch.stack(feats, 1), rate / T


@torch.no_grad()
def run_dense(w_in, wmat, alpha, toks, taus, dev, stride=8):
    b, T = toks.shape
    n = w_in.shape[1]
    v = torch.zeros((b, n), device=dev)
    z = torch.zeros((b, n), device=dev)
    ref = torch.zeros((b, n), device=dev)
    decays = torch.tensor([1.0 / t for t in taus], device=dev)
    read = torch.zeros((b, len(taus), n), device=dev)
    feats, rate = [], 0.0
    for t in range(T):
        cur = w_in[toks[:, t]]
        rec = z @ wmat
        nv = alpha * v + cur + rec - z * 0.5
        fresh = ((nv - 0.5) / 0.5 > 0).float()
        nz = torch.where(ref > 0, torch.zeros_like(fresh), fresh)
        ref = torch.clamp(ref + 2.0 * nz - 1.0, 0.0, 2.0)
        v, z = nv, nz
        rate += float(z.mean())
        read = read * decays.view(1, -1, 1) + z.unsqueeze(1)
        if t % stride == 0:
            feats.append(read.reshape(b, -1))
    return torch.stack(feats, 1), rate / T


def calibrate(conn, tkind, taus, dev, target=0.08):
    """**每个臂单独标定输入尺度**，让发放率落在同一档。

    不标定就是在比"谁的电流大"——这是本仓已经踩过的坑（递推/输入 2.16× ⇒ 掉 4.9pp）。
    """
    best = None
    for sc in (2, 4, 8, 12, 16, 24, 32):
        pref = make_pref(0, 3.0).to(dev)
        toks, _, _ = make_task(16, 48, 1, dev, pref)
        g = torch.Generator().manual_seed(0)
        w_in = (torch.randn(V, 4096, generator=g) / 4096**0.5 * sc).to(dev)
        alpha = build_alpha(tkind, 4096, 0, dev)
        pre, wm = build_conn(conn, 4096, 0.15, 0, dev)
        if conn == "ring":
            _, fr = run_ring(w_in, pre, wm, alpha, toks, taus, dev)
        else:
            _, fr = run_dense(w_in, wm, alpha, toks, taus, dev)
        if 0.03 <= fr <= 0.20:
            score = abs(fr - target)
            if best is None or score < best[2]:
                best = (sc, fr, score)
    return best if best else (16, float("nan"), 9e9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=96)
    ap.add_argument("--train-seq", type=int, default=256)
    ap.add_argument("--test-seq", type=int, default=64)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--input-scale", type=float, default=2.0)
    ap.add_argument("--dense-gain", type=float, default=0.15)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(x) for x in args.seeds.split(",")]
    n = 4096
    res: dict = {"cells": {}}

    maj = []
    for sd in seeds:
        pref = make_pref(0, 3.0).to(dev)
        _, tes, _ = make_task(args.test_seq, args.seq, 20 + sd, dev, pref)
        st = tes[:, :: args.stride].reshape(-1)
        maj.append(float(torch.bincount(st, minlength=K).max()) / st.numel())
    res["majority"] = sum(maj) / len(maj)
    print(f"  多数类基线 {res['majority']:.3f}", flush=True)

    arms = [
        ("low", "ring", "uniform", TAUS2),
        ("+conn", "dense", "uniform", TAUS2),
        ("+tau", "ring", "hetero", TAUS2),
        ("+read", "ring", "uniform", TAUS8),
        ("all", "dense", "hetero", TAUS8),
    ]
    for name, conn, tkind, taus in arms:
        sc, probe_fr, _ = calibrate(conn, tkind, taus, dev)
        print(f"  {name:6s} 标定输入尺度 {sc}（探测发放 {probe_fr:.4f}）", flush=True)
        accs, frs = [], []
        for sd in seeds:
            pref = make_pref(0, 3.0).to(dev)
            tr, trs, _ = make_task(args.train_seq, args.seq, 10 + sd, dev, pref)
            te, tes, _ = make_task(args.test_seq, args.seq, 20 + sd, dev, pref)
            toks = torch.cat([tr, te])
            st = torch.cat([trs, tes])[:, :: args.stride]
            g = torch.Generator().manual_seed(sd)
            w_in = (torch.randn(V, n, generator=g) / n**0.5 * sc).to(dev)
            alpha = build_alpha(tkind, n, sd, dev)
            pre, wm = build_conn(conn, n, args.dense_gain, sd, dev)
            if conn == "ring":
                feat, fr = run_ring(w_in, pre, wm, alpha, toks, taus, dev, args.stride)
            else:
                feat, fr = run_dense(w_in, wm, alpha, toks, taus, dev, args.stride)
            accs.append(mlp(feat, st, args.train_seq))
            frs.append(fr)
            del feat
            torch.cuda.empty_cache()
        res["cells"][name] = {
            "acc": sum(accs) / len(accs),
            "runs": accs,
            "firing": sum(frs) / len(frs),
        }
        print(
            f"  {name:6s} acc {sum(accs) / len(accs):.3f}  发放 {sum(frs) / len(frs):.4f}  "
            f"({', '.join(f'{a:.3f}' for a in accs)})",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print("\n判：")
    floor = res["majority"] + 0.15
    for k, v in res["cells"].items():
        print(
            f"  H-0 {k:6s} 发放 {v['firing']:.4f} {'✓' if 0.03 <= v['firing'] <= 0.20 else '✗'}   "
            f"H-1 acc {v['acc']:.3f} vs {floor:.3f} {'✓' if v['acc'] >= floor else '✗'}"
        )
    lo = res["cells"]["low"]
    al = res["cells"]["all"]
    gain = al["acc"] - lo["acc"]
    same = all(x >= y - 1e-9 for x, y in zip(al["runs"], lo["runs"], strict=False))
    print(
        f"  H-2 all {al['acc']:.3f} − low {lo['acc']:.3f} = {gain:+.3f}；"
        f"三种子同向 {'是' if same else '否'} ⇒ {'过' if gain >= 0.10 and same else '不过'}"
    )
    for k in ("+conn", "+tau", "+read"):
        print(f"  H-3 {k:6s} 相对 low 增益 {res['cells'][k]['acc'] - lo['acc']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

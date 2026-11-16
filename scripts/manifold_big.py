"""高维 + 超欧几何 + **全程无非线性以外的可学部件**。

## 三条约束（用户 2026-11-16）

1. **神经元要多**：2,048 → **16,384**，度数 1,024（不是 4）；
2. **不要线性**：递归聚合用**树突亚基**（分束求和 → 逐束非线性 → 合并），
   读出**零可学参数**（类均值在流形上解析给出，按测地距离分类）；
3. **几何要更适配**：不只双曲，加**球面**与**产品空间 H×S**——
   对话数据是混合结构（层级话题 + 周期节律），单因子几何强迫它只有一种结构。

## 流形上的类原型（全是闭式，无梯度）

| 几何 | 类原型 |
| :--- | :--- |
| 欧氏 | 算术均值 |
| 球面 | 均值后归一化 |
| 双曲（Poincaré） | **爱因斯坦中点**（Klein 坐标下加权平均，再映回 Poincaré） |
| 产品 H×S | 两个因子各按上面算，距离相加 |

## 判据

| 编号 | 判据 |
| :--- | :--- |
| **G-0** | 发放率 ∈ [0.03, 0.20]，且**每个几何都 > 随机 + 15pp**（否则比较在地板上，无效） |
| **G-1** | 最佳几何比欧氏 ≥ **+5pp**，三种子同向 |
| **G-2** | **产品空间 ≥ 单因子最好者** |
| **G-3** | 可学参数 = 0 |
"""

from __future__ import annotations

# ruff: noqa: N803, N806  -- 矩阵推导里 X/W/A/T/K 是惯例
import argparse
import json
import math
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.instrument import TARGET_FIRING, calibrate, seeded_dirichlet

OUT = _pathlib.Path("sweep_results/manifold_big")
TAUS = (2.0, 16.0)


# ---------------------------------------------------------------- 任务
def make_task(n_seq: int, T: int, K: int, seed: int, dev: str, vocab: int = 64):
    g = torch.Generator().manual_seed(seed)
    pref = seeded_dirichlet(3.0, K).to(dev)
    seg = max(1, T // K)
    phase = torch.randint(0, K, (n_seq,), generator=g).to(dev)
    st = torch.zeros(n_seq, T, dtype=torch.long, device=dev)
    toks = torch.zeros(n_seq, T, dtype=torch.long, device=dev)
    for t in range(T):
        s = (phase + t // seg) % K
        st[:, t] = s
        toks[:, t] = torch.multinomial(pref[s].cpu(), 1, generator=g).squeeze(-1).to(dev)
    return toks, st


# ---------------------------------------------------------------- 网络
def build_net(n: int, deg: int, vocab: int, seed: int, dev: str, subunits: int = 8):
    """稀疏递归（CSR）+ 树突亚基。返回 (w_in, 每亚基的稀疏矩阵列表)。"""
    g = torch.Generator().manual_seed(seed)
    w_in = (torch.randn(vocab, n, generator=g) / n**0.5).to(dev)
    per = deg // subunits
    mats = []
    for _ in range(subunits):
        idx = torch.randint(0, n, (per * n,), generator=g)
        rows = torch.arange(n).repeat_interleave(per)
        w = torch.rand(per * n, generator=g)
        m = torch.sparse_coo_tensor(torch.stack([rows, idx]), w, (n, n)).coalesce()
        mats.append(m.to_sparse_csr().to(dev))
    return w_in, mats


@torch.no_grad()
def run_net(w_in, mats, toks, dev, alpha=0.95, theta=0.5, stride=8):
    b, T = toks.shape
    n = w_in.shape[1]
    v = torch.zeros((b, n), device=dev)
    z = torch.zeros((b, n), device=dev)
    ref = torch.zeros((b, n), device=dev)
    decays = torch.tensor([1.0 / t for t in TAUS], device=dev)
    read = torch.zeros((b, len(TAUS), n), device=dev)
    feats, rate = [], 0.0
    for t in range(T):
        cur = w_in[toks[:, t]]
        # **树突亚基**：每束线性求和后过一次非线性，再相加 —— 聚合本身非线性
        rec = torch.zeros_like(z)
        for m in mats:
            sub = torch.sparse.mm(m, z.T).T / math.sqrt(max(1, m._nnz() // n))
            rec = rec + torch.sign(sub) * torch.sqrt(sub.abs() + 1e-6)
        nv = alpha * v + cur + rec - z * theta
        fresh = ((nv - theta) / theta > 0).float()
        nz = torch.where(ref > 0, torch.zeros_like(fresh), fresh)
        ref = torch.clamp(ref + 2.0 * nz - 1.0, 0.0, 2.0)
        v, z = nv, nz
        rate += float(z.mean())
        read = read * decays.view(1, -1, 1) + z.unsqueeze(1)
        if t % stride == 0:
            feats.append(read.reshape(b, -1))
    return torch.stack(feats, 1), rate / T


# ---------------------------------------------------------------- 流形
def unit(x):
    return x / x.norm(dim=1, keepdim=True).clamp(min=1e-9)


def to_ball(x, kappa=1.0):
    """压进 Poincaré 球（半径随模长变化，不能是常数）。"""
    nrm = x.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return unit(x) * torch.tanh(kappa * nrm) * 0.95


def d_euclid(a, b):
    return torch.cdist(a, b)


def d_sphere(a, b):
    return torch.acos((unit(a) @ unit(b).T).clamp(-1 + 1e-6, 1 - 1e-6))


def d_poincare(a, b):
    na = a.norm(dim=1, keepdim=True).pow(2).clamp(max=1 - 1e-5)
    nb = b.norm(dim=1, keepdim=True).pow(2).clamp(max=1 - 1e-5)
    d2 = (a[:, None, :] - b[None]).pow(2).sum(-1)
    c = 1 + 2 * d2 / ((1 - na) * (1 - nb.T))
    return torch.arccosh(c.clamp(min=1.0 + 1e-6))


def einstein_midpoint(x, kappa=1.0):
    """Klein 坐标下加权平均再映回 Poincaré —— 闭式，无梯度。"""
    nrm = x.norm(dim=1, keepdim=True).clamp(min=1e-9)
    r = torch.tanh(kappa * nrm) / kappa
    klein = unit(x) * r
    mid = klein.mean(0, keepdim=True)
    mn = mid.norm(dim=1, keepdim=True).clamp(max=1 - 1e-4)
    mid = unit(mid) * mn
    # Klein → Poincaré
    return mid / (1 + torch.sqrt((1 - mid.pow(2).sum(1, keepdim=True)).clamp(min=1e-6)))


def centroid(x, how: str):
    if how == "euclid":
        return x.mean(0, keepdim=True)
    if how == "sphere":
        return unit(x.mean(0, keepdim=True))
    if how == "poincare":
        return einstein_midpoint(x)
    raise ValueError(how)


def acc_of(Xtr, Ytr, Xte, Yte, how, K):
    if how == "euclid":
        proto = torch.cat(
            [
                Xtr[Ytr == c].mean(0, keepdim=True)
                if (Ytr == c).any()
                else Xtr.mean(0, keepdim=True)
                for c in range(K)
            ]
        )
        d = d_euclid(Xte, proto)
    elif how == "sphere":
        proto = unit(
            torch.cat(
                [
                    Xtr[Ytr == c].mean(0, keepdim=True)
                    if (Ytr == c).any()
                    else Xtr.mean(0, keepdim=True)
                    for c in range(K)
                ]
            )
        )
        d = d_sphere(Xte, proto)
    elif how == "poincare":
        proto = torch.cat(
            [
                einstein_midpoint(Xtr[Ytr == c]) if (Ytr == c).any() else einstein_midpoint(Xtr)
                for c in range(K)
            ]
        )
        d = d_poincare(Xte, proto)
    else:  # product
        ph = torch.cat(
            [
                einstein_midpoint(Xtr[Ytr == c]) if (Ytr == c).any() else einstein_midpoint(Xtr)
                for c in range(K)
            ]
        )
        ps = unit(
            torch.cat(
                [
                    Xtr[Ytr == c].mean(0, keepdim=True)
                    if (Ytr == c).any()
                    else Xtr.mean(0, keepdim=True)
                    for c in range(K)
                ]
            )
        )
        d = d_poincare(Xte, ph) + d_sphere(Xte, ps)
    return float((d.argmin(1) == Yte).float().mean())


def calibrate_scale(n: int, deg: int, seed: int, dev: str, K: int = 8):
    """**先标定工作点，不达标就不许跑比较**。

    今天已经三次栽在同一处：改 n 之后忘了输入也随 1/√n 缩，网络静默、
    读出全零、"准确率"退化成多数类。这次把标定写进主流程。
    """
    tr, _ = make_task(32, 48, K, 10 + seed, dev)
    w_in, mats = build_net(n, deg, 64, seed, dev)  # 只建一次：同 seed 逐位相同
    # 细网格：树突的 sign·√|·| 让小信号被放大，阈值附近很陡，粗网格会跨过去
    ok, sc, fr = calibrate(lambda s: run_net(w_in * s, mats, tr, dev)[1])
    if not ok:
        return None
    return (sc, fr, abs(fr - TARGET_FIRING))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--deg", type=int, default=1024)
    ap.add_argument("--ks", default="8,16")
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--train-seq", type=int, default=256)
    ap.add_argument("--test-seq", type=int, default=64)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(x) for x in args.seeds.split(",")]
    res: dict = {"n": args.n, "deg": args.deg, "cells": {}}

    cal = calibrate_scale(args.n, args.deg, seeds[0], dev)
    if cal is None:
        print("✗ 没有任何输入尺度能把发放率标进 [0.03, 0.20] —— **中止，不做比较**")
        return 1
    scale, probe_fr, _ = cal
    print(f"  **标定输入尺度 {scale}（探测发放 {probe_fr:.4f}）**", flush=True)

    for K in [int(x) for x in args.ks.split(",")]:
        for how in ("euclid", "sphere", "poincare", "product"):
            accs, frs = [], []
            for sd in seeds:
                tr, trs = make_task(args.train_seq, args.seq, K, 10 + sd, dev)
                te, tes = make_task(args.test_seq, args.seq, K, 20 + sd, dev)
                toks = torch.cat([tr, te])
                st = torch.cat([trs, tes])[:, ::8]
                w_in, mats = build_net(args.n, args.deg, 64, sd, dev)
                feat, fr = run_net(w_in * scale, mats, toks, dev)
                X = feat.reshape(-1, feat.shape[-1])
                Y = st.reshape(-1)
                cut = args.train_seq * st.shape[1]
                mu, sdv = X[:cut].mean(0), X[:cut].std(0) + 1e-6
                X = (X - mu) / sdv
                Xtr, Xte, Ytr, Yte = X[:cut], X[cut:], Y[:cut], Y[cut:]
                Xtr_p, Xte_p = to_ball(Xtr), to_ball(Xte)
                accs.append(acc_of(Xtr_p, Ytr, Xte_p, Yte, how, K))
                frs.append(fr)
                del feat, X, Xtr, Xte, w_in, mats
                torch.cuda.empty_cache()
            key = f"K{K}/{how}"
            res["cells"][key] = {
                "acc": sum(accs) / len(accs),
                "runs": accs,
                "firing": sum(frs) / len(frs),
            }
            print(
                f"  K={K:2d} {how:8s} acc {sum(accs) / len(accs):.3f}  "
                f"发放 {sum(frs) / len(frs):.4f}",
                flush=True,
            )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print("\n判：")
    for K in [int(x) for x in args.ks.split(",")]:
        chance = 1.0 / K
        for how in ("euclid", "sphere", "poincare", "product"):
            v = res["cells"][f"K{K}/{how}"]
            ok = v["acc"] >= chance + 0.15 and 0.03 <= v["firing"] <= 0.20
            print(
                f"  G-0 K={K} {how:8s} acc {v['acc']:.3f} vs 门槛 {chance + 0.15:.3f} "
                f"发放 {v['firing']:.4f} ⇒ {'✓' if ok else '✗'}"
            )
        e = res["cells"][f"K{K}/euclid"]["acc"]
        for how in ("sphere", "poincare", "product"):
            print(f"  G-1 K={K} {how:8s} − 欧氏 = {res['cells'][f'K{K}/{how}']['acc'] - e:+.3f}")
        best_single = max(res["cells"][f"K{K}/{h}"]["acc"] for h in ("sphere", "poincare"))
        prod = res["cells"][f"K{K}/product"]["acc"]
        print(
            f"  G-2 K={K} 产品 {prod:.3f} − 单因子最好 {best_single:.3f} = "
            f"{prod - best_single:+.3f} ⇒ {'过' if prod >= best_single else '不过'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

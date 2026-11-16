"""网络的**空间结构维数**：等神经元、等入度、等增益，只变格点维数。

## 之前测过，但全部作废

1D/2D/3D/4D/6D 那几轮跑在**静默网络**上（发放 0.0001），"准确率"其实是多数类。
这一版把三个条件都控住：

- **工作点先标定**（发放率 ∈ [0.03, 0.20]，不达标就中止）；
- **无非线性以外的可学部件**（树突亚基 + 零参数流形读出）；
- **等神经元 16,384、等入度 1,024、等递归总增益**。

## 机制假设：直径

格点直径 ∝ `n^(1/d)`。n=16384 时 1D 的直径是 8192，7D 只有约 7。
**如果空间维数有用，机制应该是"直径短 ⇒ 混合快 ⇒ 同样步数里整合更多历史"。**

所以除了准确率，**每个配置都报实测直径**（BFS 平均最短路径），
用来检验"直径 → 能力"这条链。

## 判据

| 编号 | 判据 |
| :--- | :--- |
| **S-0** | 工作点：发放率 ∈ [0.03, 0.20]，且每个配置 acc ≥ 随机 + 15pp |
| **S-1** | 最佳维数比 1D 高 **≥5pp**，三种子同向 |
| **S-2** | 直径与准确率的**秩相关 ≤ −0.6**（直径越短越强）——否则机制不成立 |
"""

from __future__ import annotations

# ruff: noqa: N806  -- 矩阵推导里 X/W/A/T/K 是惯例
import argparse
import json
import math
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.instrument import calibrate, mlp_probe
from scripts.manifold_big import acc_of, make_task, to_ball
from scripts.structure_ladder import window_features

OUT = _pathlib.Path("sweep_results/spatial_dim")
TAUS = (2.0, 16.0)
LATTICES = {
    "1D": (16384,),
    "2D": (128, 128),
    "3D": (32, 32, 16),
    "7D": (4, 4, 4, 4, 4, 4, 4),
}


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


def neighbor_offsets(dims: tuple[int, ...], k: int) -> torch.Tensor:
    """按格点距离排序取最近 k 个偏移（含环绕）。"""
    r = 1
    while True:
        offs = torch.cartesian_prod(*[torch.arange(-r, r + 1) for _ in dims]).reshape(-1, len(dims))
        offs = offs[(offs != 0).any(dim=1)]
        if offs.shape[0] >= k or r > max(dims):
            break
        r += 1
    dist = offs.float().norm(dim=1)
    order = dist.argsort()
    return offs[order[:k]], dist[order[:k]]


def build_spatial(dims: tuple[int, ...], n: int, k: int, seed: int, dev: str, subunits: int = 8):
    """距离衰减权重的稀疏递归，按亚基分组。"""
    g = torch.Generator().manual_seed(seed)
    ci = coords_of(n, dims)
    offs, dist = neighbor_offsets(dims, k)
    pre = torch.stack([lin_of((ci + o) % torch.tensor(dims), dims) for o in offs], 1)  # (n,k)
    # 距离衰减：近邻权重大
    decay = torch.exp(-dist / max(float(dist.median()), 1e-6))
    per = k // subunits
    mats = []
    for b in range(subunits):
        cols = pre[:, b * per : (b + 1) * per].reshape(-1)
        rows = torch.arange(n).repeat_interleave(per)
        # **距离衰减**：同一亚基内，越远的邻居权重越小
        w = (torch.rand(n, per, generator=g) * decay[b * per : (b + 1) * per].unsqueeze(0)).reshape(
            -1
        )
        coo = torch.sparse_coo_tensor(torch.stack([rows, cols]), w, (n, n)).coalesce()
        mats.append(coo.to_sparse_csr().to(dev))
    return mats, pre, dist


def mean_degree_diameter(pre: torch.Tensor, dims: tuple[int, ...], n: int, n_src: int = 4):
    """**精确** BFS 的平均最短路径（向量化，层数有限）。

    这里踩过两个坑，都记下来：

    1. 旧实现只给被抽样的源点填邻接表，BFS 走到别处时邻接表是空的 ——
       几乎所有点对都报 2 跳；
    2. 改成按需取邻居之后仍然是 2.x，因为搜索在 `cap` 个节点处截断，
       而 k=1024 时**第 3 层就已经越过 cap**，平均值被前两层主导。

    现在是全图精确 BFS：每层用 `pre[frontier]` 一次取出整层的邻居。
    总代价 O(n·k)，n=16384、k=1024 也只有 1.7e7 次索引。
    """
    srcs = torch.linspace(0, n - 1, n_src).long()
    tot = 0.0
    for s in srcs:
        dist = torch.full((n,), -1, dtype=torch.long)
        dist[int(s)] = 0
        frontier = torch.zeros(n, dtype=torch.bool)
        frontier[int(s)] = True
        d = 0
        while bool(frontier.any()):
            d += 1
            idx = frontier.nonzero().squeeze(1)
            cand = torch.zeros(n, dtype=torch.bool)
            cand[pre[idx].reshape(-1)] = True
            cand &= dist < 0
            if not bool(cand.any()):
                break
            dist[cand] = d
            frontier = cand
        tot += float(dist[dist >= 0].float().mean())
    return tot / len(srcs)


@torch.no_grad()
def run_spatial(mats, w_in, toks, dev, alpha=0.95, theta=0.5, stride=8):
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
        rec = torch.zeros_like(z)
        for m in mats:
            sub = torch.sparse.mm(m, z.T).T
            rec = rec + torch.sign(sub) * torch.sqrt(sub.abs() + 1e-6) / math.sqrt(128)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--ks-class", type=int, default=8)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--train-seq", type=int, default=384)
    ap.add_argument("--test-seq", type=int, default=96)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--geometry", default="product")
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(x) for x in args.seeds.split(",")]
    n = 16384
    K = args.ks_class
    res: dict = {"n": n, "k": args.k, "geometry": args.geometry, "cells": {}}

    # ---- 正向对照：最近 8 个 token 的 one-hot，完全不经过网络 ----
    ceilings, majors = [], []
    for sd in seeds:
        trw, trsw = make_task(args.train_seq, args.seq, K, 10 + sd, dev)
        tew, tesw = make_task(args.test_seq, args.seq, K, 20 + sd, dev)
        toks_w = torch.cat([trw, tew])
        st_w = torch.cat([trsw, tesw])[:, ::8]
        wf = window_features(toks_w, L=8, stride=8)
        Xw = wf.reshape(-1, wf.shape[-1])
        Yw = st_w.reshape(-1)
        cutw = args.train_seq * st_w.shape[1]
        mw, sw = Xw[:cutw].mean(0), Xw[:cutw].std(0) + 1e-6
        Xw = (Xw - mw) / sw
        ceilings.append(
            acc_of(to_ball(Xw[:cutw]), Yw[:cutw], to_ball(Xw[cutw:]), Yw[cutw:], args.geometry, K)
        )
        trp, tep = st_w[: args.train_seq], st_w[args.train_seq :]
        top = int(torch.bincount(trp.reshape(-1), minlength=K).argmax())
        majors.append(float((tep == top).float().mean()))
        del toks_w, st_w, wf, Xw
    res["ceiling"] = sum(ceilings) / len(ceilings)
    res["majority"] = sum(majors) / len(majors)
    print(
        f"  窗口天花板 {res['ceiling']:.3f}   底板(多数类) {res['majority']:.3f}",
        flush=True,
    )
    torch.cuda.empty_cache()

    for name, dims in LATTICES.items():
        # ---- 标定工作点 ----
        tr, _ = make_task(32, args.seq, K, 10, dev)  # T 必须与正式跑一致
        mats, pre, _dist = build_spatial(dims, n, args.k, 0, dev)
        w_in = (torch.randn(64, n, generator=torch.Generator().manual_seed(0)) / n**0.5).to(dev)
        ok_cal, sc, _pfr = calibrate(
            lambda s, _m=mats, _w=w_in, _t=tr: run_spatial(_m, _w * s, _t, dev)[1]
        )
        if not ok_cal:
            print(f"  {name:3s} ✗ 标定失败，该臂无效", flush=True)
            continue
        diam = mean_degree_diameter(pre, dims, n)
        # ---- 正式跑 ----
        accs, frs, protos = [], [], []
        for sd in seeds:
            tr, trs = make_task(args.train_seq, args.seq, K, 10 + sd, dev)
            te, tes = make_task(args.test_seq, args.seq, K, 20 + sd, dev)
            toks = torch.cat([tr, te])
            st = torch.cat([trs, tes])[:, ::8]
            wi = (torch.randn(64, n, generator=torch.Generator().manual_seed(sd)) / n**0.5 * sc).to(
                dev
            )
            feat, fr = run_spatial(mats, wi, toks, dev)
            X = feat.reshape(-1, feat.shape[-1])
            Y = st.reshape(-1)
            cut = args.train_seq * st.shape[1]
            mu, sdv = X[:cut].mean(0), X[:cut].std(0) + 1e-6
            X = (X - mu) / sdv
            Xtr, Xte = to_ball(X[:cut]), to_ball(X[cut:])
            protos.append(acc_of(Xtr, Y[:cut], Xte, Y[cut:], args.geometry, K))
            # **判据用的读出**：与 dim_ladder.mlp / instrument.mlp_probe 逐字同超参。
            # 四条臂的读出完全相同 ⇒ 臂间差异只能来自特征本身。
            m, _ = mlp_probe(X, Y, cut, K, batch=1024)
            accs.append(m)
            frs.append(fr)
            del feat, X, Xtr, Xte
            torch.cuda.empty_cache()
        fr_mean = sum(frs) / len(frs)
        if not (0.03 <= fr_mean <= 0.20):
            print(f"  {name:3s} 发放 {fr_mean:.4f} 出窗，该臂无效", flush=True)
            continue
        res["cells"][name] = {
            "acc": sum(accs) / len(accs),
            "runs": accs,
            "proto": sum(protos) / len(protos),
            "proto_runs": protos,
            "firing": sum(frs) / len(frs),
            "diameter": diam,
            "scale": sc,
        }
        print(
            f"  {name:3s} 直径 {diam:7.1f}  发放 {sum(frs) / len(frs):.4f}  "
            f"mlp {sum(accs) / len(accs):.3f} ({', '.join(f'{a:.3f}' for a in accs)})  "
            f"proto {sum(protos) / len(protos):.3f}",
            flush=True,
        )
        del mats
        torch.cuda.empty_cache()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print("\n判：")
    chance = 1.0 / K
    ok0 = all(v["firing"] >= 0.03 and v["acc"] >= chance + 0.15 for v in res["cells"].values())
    print(f"  S-0 工作点 + 真解码：{'过' if ok0 else '不过'}")
    if "1D" in res["cells"]:
        base = res["cells"]["1D"]["acc"]
        best = max(res["cells"], key=lambda k: res["cells"][k]["acc"])
        b = res["cells"][best]
        gain = b["acc"] - base
        same = all(
            x >= y - 1e-9 for x, y in zip(b["runs"], res["cells"]["1D"]["runs"], strict=False)
        )
        print(
            f"  S-1 最佳 {best} {b['acc']:.3f} − 1D {base:.3f} = {gain:+.3f}；"
            f"三种子同向 {'是' if same else '否'} ⇒ {'过' if gain >= 0.05 and same else '不过'}"
        )
        names = list(res["cells"])
        d = torch.tensor([res["cells"][x]["diameter"] for x in names])
        a = torch.tensor([res["cells"][x]["acc"] for x in names])
        rho = float(
            (
                (d.argsort().argsort().float() - d.argsort().argsort().float().mean())
                * (a.argsort().argsort().float() - a.argsort().argsort().float().mean())
            ).mean()
            / (d.argsort().argsort().float().std() * a.argsort().argsort().float().std() + 1e-9)
        )
        print(f"  S-2 直径 vs 准确率秩相关 {rho:+.2f} ≤ −0.6 ⇒ {'过' if rho <= -0.6 else '不过'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

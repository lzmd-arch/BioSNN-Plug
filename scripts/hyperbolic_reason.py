"""双曲**推理**：树操作任务，**同一套读出**（预注册 H-I）。

## 任务

$B$ 叉树、深度 $D$。给**两个叶子**，**预测它们的最近共同祖先**（LCA 的内部节点 id）。

这是真正的树操作：不是查表，而是"把两个叶子的位置合成到共同上级"。

**输入对两臂完全相同**：叶子的固定随机嵌入 `[e_A, e_B, e_A − e_B]`。
模型必须**自己学出层级**——嵌入里没有层级信息。

## 两臂

| 臂 | 读出 |
| :--- | :--- |
| 欧氏 | `Linear → 2 维 → 欧氏距离到 21 个原型` |
| 双曲 | `Linear → 2 维 Poincaré 圆盘 → 双曲距离到 21 个原型` |

**维度、参数量、输入全部相同，只有几何不同。**

## 判据

| 编号 | 判据 |
| :--- | :--- |
| **HI-1** | 双曲 ≥ 欧氏 **+5pp** |
| **HI-2** | 扁平对照（同样的架构、预测一个**没有层级**的目标）上增益消失 |
| **HI-3** | 参数量差 <5% |

**HI-2 是关键**：如果扁平任务上双曲也赢，那赢的不是层级，是这层几何本身的某种偏置。
"""

from __future__ import annotations

# ruff: noqa: N803, N806  -- 矩阵推导里 X/W/A/T/K 是惯例
import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

OUT = _pathlib.Path("sweep_results/hyperbolic_reason")


def tree_paths(branch: int, depth: int) -> torch.Tensor:
    """每个叶子的 base-B 路径，(n_leaf, depth)。"""
    n = branch**depth
    idx = torch.arange(n)
    return torch.stack([(idx // (branch ** (depth - 1 - d))) % branch for d in range(depth)], 1)


def lca_index(paths: torch.Tensor, branch: int, depth: int) -> torch.Tensor:
    """两两 LCA 在"内部节点编号"里的下标。内部节点按层编号：层 d 的第 i 个 → 前缀和 + i。"""
    n = paths.shape[0]
    lcp = torch.zeros(n, n, dtype=torch.long)
    alive = torch.ones(n, n, dtype=torch.bool)
    for d in range(depth):
        alive = alive & (paths[:, d].unsqueeze(1) == paths[:, d].unsqueeze(0))
        lcp += alive.long()
    # 层 d 有 branch^d 个节点；内部节点总数 = sum_{d<depth} branch^d
    offs = torch.tensor([sum(branch**k for k in range(d)) for d in range(depth)])
    # LCA 的层 = lcp（叶子层是 depth，故 LCA 层 = lcp，且 lcp ≤ depth-1 对两个不同叶子）
    node_in_level = torch.zeros(n, n, dtype=torch.long)
    for d in range(depth):
        acc = torch.zeros(n, n, dtype=torch.long)
        for j in range(d):
            acc = acc * branch + paths[:, j].unsqueeze(1).expand(n, n)
        node_in_level = torch.where(lcp == d, acc, node_in_level)
    return offs[lcp.clamp(max=depth - 1)] + node_in_level


def hyper_dist(a, b):
    na = (a**2).sum(1, keepdim=True).clamp(max=1 - 1e-6)
    nb = (b**2).sum(1, keepdim=True).clamp(max=1 - 1e-6)
    d2 = ((a.unsqueeze(1) - b.unsqueeze(0)) ** 2).sum(-1)
    return torch.arccosh((1 + 2 * d2 / ((1 - na) * (1 - nb.T))).clamp(min=1.0 + 1e-7))


class Head(torch.nn.Module):
    def __init__(self, fin, n_cls, hyperbolic):
        super().__init__()
        self.proj = torch.nn.Linear(fin, 2)
        self.proto = torch.nn.Parameter(torch.randn(n_cls, 2) * 0.1)
        self.hyperbolic = hyperbolic

    def forward(self, f):
        z = self.proj(f)
        if self.hyperbolic:
            z = torch.tanh(z) * 0.95
            p = torch.tanh(self.proto) * 0.95
            return -hyper_dist(z, p)
        return -torch.cdist(z, self.proto)


def run_arm(Xtr, Ytr, Xte, Yte, n_cls, hyperbolic, dev, steps=600):
    net = Head(Xtr.shape[1], n_cls, hyperbolic).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(steps):
        idx = torch.randperm(Xtr.shape[0], device=dev)[:8192]
        opt.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(net(Xtr[idx]), Ytr[idx]).backward()
        opt.step()
    with torch.no_grad():
        acc = float((net(Xte).argmax(-1) == Yte).float().mean())
    return acc, sum(p.numel() for p in net.parameters())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", type=int, default=4)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(x) for x in args.seeds.split(",")]
    B, D = args.branch, args.depth
    n_leaf = B**D
    paths = tree_paths(B, D)
    n_inner = sum(B**k for k in range(D))
    res: dict = {"branch": B, "depth": D, "n_leaf": n_leaf, "n_inner": n_inner, "cells": {}}

    for name, flat in (("tree", False), ("flat_control", True)):
        accs = {"euclid": [], "hyperbolic": []}
        params = {}
        for sd in seeds:
            g = torch.Generator().manual_seed(sd)
            emb = torch.randn(n_leaf, args.dim, generator=g).to(dev)  # **固定随机嵌入，无层级**
            ia = torch.randint(0, n_leaf, (args.train + args.test,), generator=g).to(dev)
            ib = torch.randint(0, n_leaf, (args.train + args.test,), generator=g).to(dev)
            same = ia == ib
            ib = torch.where(same, (ib + 1) % n_leaf, ib)
            Y = lca_index(paths, B, D)[ia.cpu(), ib.cpu()].to(dev)
            if flat:
                # **扁平对照**：目标与层级无关（叶子 A 的 id 对 n_inner 取模）
                Y = (ia % n_inner).to(dev)
            X = torch.cat([emb[ia], emb[ib], emb[ia] - emb[ib]], 1)
            cut = args.train
            for arm, hyp in (("euclid", False), ("hyperbolic", True)):
                acc, pr = run_arm(X[:cut], Y[:cut], X[cut:], Y[cut:], n_inner, hyp, dev)
                accs[arm].append(acc)
                params[arm] = pr
        for arm in ("euclid", "hyperbolic"):
            res["cells"][f"{name}/{arm}"] = {
                "acc": sum(accs[arm]) / len(accs[arm]),
                "runs": accs[arm],
                "params": params[arm],
            }
        e = res["cells"][f"{name}/euclid"]["acc"]
        h = res["cells"][f"{name}/hyperbolic"]["acc"]
        print(f"  {name:12s} 欧氏 {e:.3f}  双曲 {h:.3f}  差 {h - e:+.3f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    e = res["cells"]["tree/euclid"]["acc"]
    h = res["cells"]["tree/hyperbolic"]["acc"]
    fe = res["cells"]["flat_control/euclid"]["acc"]
    fh = res["cells"]["flat_control/hyperbolic"]["acc"]
    print("\n判：")
    print(f"  HI-1 树任务 双曲 − 欧氏 = {h - e:+.3f} ≥0.05 ⇒ {'过' if h - e >= 0.05 else '不过'}")
    print(
        f"  HI-2 扁平对照增益 {fh - fe:+.3f} 应≈0 ⇒ "
        f"{'过（增益来自层级）' if abs(fh - fe) < 0.05 else '不过（增益与层级无关）'}"
    )
    p0 = res["cells"]["tree/euclid"]["params"]
    p1 = res["cells"]["tree/hyperbolic"]["params"]
    print(
        f"  HI-3 参数量差 {abs(p0 - p1) / max(p0, p1):.1%} <5% ⇒ "
        f"{'过' if abs(p0 - p1) / max(p0, p1) < 0.05 else '不过'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

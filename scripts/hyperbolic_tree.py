"""**正确地**实现 Sarkar 树嵌入，重测双曲的层级优势。

## 上一版错在哪

`hyperbolic_memory.py` 把**所有叶子放在同一半径**上，只用了 Sarkar 公式算叶子半径，
内部的"深度→半径"结构整个丢掉了。**那等于"一圈等距点"，不是树嵌入。**

## 这一版

$B$ 叉树、深度 $D$，`n = (B^{D+1}-1)/(B-1)` 个节点。

**半径（从叶子往上推）**

| 几何 | 递推 |
| :--- | :--- |
| **双曲（Sarkar）** | `tanh(r_{d-1}/2) = q · tanh(r_d/2)`，`q = 1/(1+B)` |
| 欧氏（对照） | `r_{d-1} = q · r_d` |

**角度**：深度 $d$、序号 $i$ 的节点取 `(i + 0.5) · 2π / B^d`——即父节点的扇区内均分。

**关键**：双曲里 `tanh` 递推让根附近极慢、靠近边界极快 ⇒ **深层节点的"周长"指数增长**；
欧氏只有线性增长。这是层级容量优势的来源。

## 判据

| 编号 | 判据 |
| :--- | :--- |
| **HT-1** | 全节点两两树距离 vs 嵌入距离的**秩相关**：双曲 > 欧氏 **≥0.05** |
| **HT-2** | **最大相对失真** `max|d_embed/d_tree − 1|`：双曲 ≤ 欧氏的一半 |
| **HT-3** | 打乱层级（同半径、随机角度）后优势消失 |
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

OUT = _pathlib.Path("sweep_results/hyperbolic_tree")


def build_tree(branch: int, depth: int):
    """返回 (每节点深度, 每节点路径下标, 每节点扇区中心角)。"""
    depths, idxs = [], []
    for d in range(depth + 1):
        for i in range(branch**d):
            depths.append(d)
            idxs.append(i)
    n = len(depths)
    theta = torch.tensor(
        [(i + 0.5) * 2 * math.pi / (branch**d) for d, i in zip(depths, idxs, strict=False)]
    )
    return torch.tensor(depths), torch.tensor(idxs), theta, n


def lca_depth(branch: int, depth: int):
    """两两 LCA 深度。节点的路径 = 它的 base-B 下标补零到 depth 位。"""
    dep, idxs = [], []
    for d in range(depth + 1):
        for i in range(branch**d):
            dep.append(d)
            idxs.append(i)
    n = len(dep)
    # 路径：把下标左对齐成 depth 位 base-B 数字
    paths = torch.zeros(n, depth, dtype=torch.long)
    for k, (d, i) in enumerate(zip(dep, idxs, strict=False)):
        if d == 0:
            continue
        for j in range(d):
            paths[k, j] = (i // (branch ** (d - 1 - j))) % branch
    lcp = torch.zeros(n, n, dtype=torch.long)
    alive = torch.ones(n, n, dtype=torch.bool)
    for j in range(depth):
        alive = alive & (paths[:, j].unsqueeze(1) == paths[:, j].unsqueeze(0))
        lcp += alive.long()
    dep_t = torch.tensor(dep)
    lca = torch.minimum(lcp, torch.minimum(dep_t.unsqueeze(1), dep_t.unsqueeze(0)))
    return dep_t.unsqueeze(1) + dep_t.unsqueeze(0) - 2 * lca


def radii(branch: int, depth: int, r_leaf: float, hyperbolic: bool) -> list[float]:
    q = 1.0 / (1.0 + branch)
    out = [r_leaf]
    cur = r_leaf
    for _ in range(depth):
        cur = 2 * math.atanh(q * math.tanh(cur / 2)) if hyperbolic else q * cur
        out.append(cur)
    return out[::-1]  # 下标 0 = 根


def embed(branch: int, depth: int, r_leaf: float, hyperbolic: bool, shuffle: bool = False):
    dep, _idxs, theta, n = build_tree(branch, depth)
    rs = radii(branch, depth, r_leaf, hyperbolic)
    r = torch.tensor([rs[int(d)] for d in dep])
    if shuffle:
        theta = theta[torch.randperm(n)]
    if hyperbolic:
        return torch.stack([r, theta], 1)
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], 1)


def dmat(X: torch.Tensor, hyperbolic: bool) -> torch.Tensor:
    if hyperbolic:
        ra, ta = X[:, 0:1], X[:, 1:2]
        c = torch.cosh(ra) * torch.cosh(ra.T) - torch.sinh(ra) * torch.sinh(ra.T) * torch.cos(
            ta - ta.T
        )
        return torch.arccosh(c.clamp(min=1.0 + 1e-7))
    return torch.cdist(X, X)


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = (ra - ra.mean()) / (ra.std() + 1e-9)
    rb = (rb - rb.mean()) / (rb.std() + 1e-9)
    return float((ra * rb).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", type=int, default=4)
    ap.add_argument("--depths", default="2,3,4")
    ap.add_argument("--r-leaf", type=float, default=6.0)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    res: dict = {"branch": args.branch, "r_leaf": args.r_leaf, "rows": {}}

    for depth in [int(x) for x in args.depths.split(",")]:
        D = lca_depth(args.branch, depth).float()
        n = D.shape[0]
        iu = torch.triu_indices(n, n, 1)
        dt = D[iu]
        m = dt > 0
        row = {"n": n}
        for name, hyp in (("euclid", False), ("hyperbolic", True)):
            X = embed(args.branch, depth, args.r_leaf, hyp)
            de = dmat(X, hyp)[iu]
            # 标度对齐（两种几何的绝对尺度不同）
            de = de / (de[m].median() / dt[m].median())
            row[name + "_rho"] = spearman(de[m], dt[m])
            row[name + "_maxrel"] = float(((de[m] - dt[m]).abs() / dt[m]).max())
            Xs = embed(args.branch, depth, args.r_leaf, hyp, shuffle=True)
            row[name + "_shuf_rho"] = spearman(dmat(Xs, hyp)[iu][m], dt[m])
        res["rows"][str(depth)] = row
        print(
            f"  D={depth} n={n:4d}  秩相关 欧 {row['euclid_rho']:.3f} / 双 {row['hyperbolic_rho']:.3f}"
            f"   最大相对失真 欧 {row['euclid_maxrel']:.2f} / 双 {row['hyperbolic_maxrel']:.2f}"
            f"   打乱后 欧 {row['euclid_shuf_rho']:.3f} / 双 {row['hyperbolic_shuf_rho']:.3f}",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print("\n判：")
    ok1 = ok2 = True
    for k, v in res["rows"].items():
        d1 = v["hyperbolic_rho"] - v["euclid_rho"]
        d2 = v["euclid_maxrel"] / max(v["hyperbolic_maxrel"], 1e-9)
        ok1 &= d1 >= 0.05
        ok2 &= d2 >= 2.0
        print(
            f"  D={k}  HT-1 秩相关差 {d1:+.3f} ≥0.05 {'✓' if d1 >= 0.05 else '✗'}   "
            f"HT-2 失真比 {d2:.2f}× ≥2 {'✓' if d2 >= 2.0 else '✗'}"
        )
    print(f"  HT-1 {'过' if ok1 else '不过'}   HT-2 {'过' if ok2 else '不过'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

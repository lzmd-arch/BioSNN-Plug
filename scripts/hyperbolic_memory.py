"""双曲记忆：**层级容量**的对照实验（预注册 H-M）。

## 要测的命题

层级数据（会话→轮次→句子→事实）在**双曲**空间里的可分辨容量随维度**指数增长**，
欧氏只有多项式。若成立，同一维度下双曲能撑住更多条记忆。

## 构造（Sarkar 的树嵌入 + 它的欧氏对照）

平衡树，分支 B、深度 D ⇒ N = B^D 个叶子。

| 几何 | 叶子的半径 | 父节点的半径 |
| :--- | :--- | :--- |
| **双曲** | `r_leaf = R_max` | `tanh(r_parent/2) = q · tanh(r_child/2)` |
| 欧氏 | `r_leaf = R_max` | `r_parent = q · r_child` |

`q = 1/(1+B)`。角度：每个节点在自己父节点的扇区内均分。

**关键差别**：在双曲里，半径 R 处的"周长"是 `sinh(R)`（指数），欧氏是 `2πR`（线性）。
所以同样 N 个叶子，双曲的**角间隔对应的弧长**大得多。

## 查询与判据

查询 = 某个叶子 + 噪声（角度按叶宽的比例、半径按固定 σ）。用**各自的原生度量**找最近邻。

| 编号 | 判据 | 通过线 |
| :--- | :--- | :--- |
| **HM-1** | `hit@1` 随 N 增长的保持长度 | 双曲 ≥ 欧氏 **2×** |
| **HM-2** | N=1024 时 `hit@1` | 双曲 ≥ 欧氏 **+10pp** |
| **HM-3** | **打乱树的层级**（同节点、随机父子）后增益消失 | 归因成立 |
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

OUT = _pathlib.Path("sweep_results/hyperbolic_memory")


def tree_embed(branch: int, depth: int, r_max: float, hyperbolic: bool):
    """返回 (叶子角度, 叶子半径)。层级决定扇区与半径，几何决定半径的递推。"""
    n_leaf = branch**depth
    q = 1.0 / (1.0 + branch)
    # 从叶子往上推半径
    radii = [r_max]
    cur = r_max
    for _ in range(depth - 1):
        if hyperbolic:
            cur = 2 * math.atanh(q * math.tanh(cur / 2))
        else:
            cur = q * cur
        radii.append(cur)
    radii.reverse()  # 下标 0 = 根
    theta = torch.arange(n_leaf).float() * (2 * math.pi / n_leaf)
    return theta, radii[-1]  # 叶子都在最外圈


def positions(theta: torch.Tensor, r: float, hyperbolic: bool):
    """(N,2)：双曲返回 (r, θ)，欧氏返回 (x, y)。"""
    if hyperbolic:
        return torch.stack([torch.full_like(theta, r), theta], 1)
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], 1)


def dist_matrix(a: torch.Tensor, b: torch.Tensor, hyperbolic: bool) -> torch.Tensor:
    if hyperbolic:
        ra, ta = a[:, 0:1], a[:, 1:2]
        rb, tb = b[:, 0:1], b[:, 1:2]
        c = torch.cosh(ra) * torch.cosh(rb.T) - torch.sinh(ra) * torch.sinh(rb.T) * torch.cos(
            ta - tb.T
        )
        return torch.arccosh(c.clamp(min=1.0 + 1e-7))
    return torch.cdist(a, b)


def evaluate(
    branch: int,
    depth: int,
    hyperbolic: bool,
    sigma_ang: float,
    sigma_r: float,
    seed: int,
    shuffle_tree: bool = False,
) -> float:
    g = torch.Generator().manual_seed(seed)
    n_leaf = branch**depth
    theta, r_leaf = tree_embed(branch, depth, r_max=6.0, hyperbolic=hyperbolic)
    if shuffle_tree:
        # **HM-3**：保留同样的半径分布，但角度随机打乱 ⇒ 层级结构消失
        theta = theta[torch.randperm(n_leaf, generator=g)]
    base = positions(theta, r_leaf, hyperbolic)
    width = 2 * math.pi / n_leaf
    nq = min(256, n_leaf)
    idx = torch.randperm(n_leaf, generator=g)[:nq]
    if hyperbolic:
        qa = torch.stack(
            [
                r_leaf + torch.randn(nq, generator=g) * sigma_r,
                theta[idx] + torch.randn(nq, generator=g) * sigma_ang,
            ],
            1,
        )
    else:
        ang = theta[idx] + torch.randn(nq, generator=g) * sigma_ang
        rr = r_leaf + torch.randn(nq, generator=g) * sigma_r
        qa = torch.stack([rr * torch.cos(ang), rr * torch.sin(ang)], 1)
    d = dist_matrix(qa, base, hyperbolic)
    pred = d.argmin(1)
    return float((pred == idx).float().mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", type=int, default=4)
    ap.add_argument("--depths", default="3,4,5,6")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--sigma-ang", type=float, default=2e-3, help="**绝对**角度噪声（弧度）")
    ap.add_argument("--sigma-r", type=float, default=0.05)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    res: dict = {"branch": args.branch, "curve": {}}

    for depth in [int(x) for x in args.depths.split(",")]:
        n_leaf = args.branch**depth
        row = {"n": n_leaf}
        for name, hyp in (("euclid", False), ("hyperbolic", True)):
            accs = [
                evaluate(args.branch, depth, hyp, args.sigma_ang, args.sigma_r, sd) for sd in seeds
            ]
            shr = [
                evaluate(
                    args.branch, depth, hyp, args.sigma_ang, args.sigma_r, sd, shuffle_tree=True
                )
                for sd in seeds
            ]
            row[name] = sum(accs) / len(accs)
            row[name + "_shuffled"] = sum(shr) / len(shr)
        res["curve"][str(depth)] = row
        print(
            f"  N={n_leaf:5d}  欧氏 {row['euclid']:.3f}  双曲 {row['hyperbolic']:.3f}  "
            f"（打乱层级：欧氏 {row['euclid_shuffled']:.3f} / 双曲 {row['hyperbolic_shuffled']:.3f}）",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    last = res["curve"][str(max(int(x) for x in args.depths.split(",")))]
    print("\n判：")
    print(
        f"  HM-2 N={last['n']} 双曲 {last['hyperbolic']:.3f} − 欧氏 {last['euclid']:.3f} = "
        f"{last['hyperbolic'] - last['euclid']:+.3f} ≥0.10 ⇒ "
        f"{'过' if last['hyperbolic'] - last['euclid'] >= 0.10 else '不过'}"
    )
    d_h = last["hyperbolic"] - last["hyperbolic_shuffled"]
    d_e = last["euclid"] - last["euclid_shuffled"]
    print(
        f"  HM-3 打乱层级后：双曲掉 {d_h:+.3f}，欧氏掉 {d_e:+.3f} ⇒ "
        f"{'过（增益来自层级）' if d_h > d_e else '不过（增益与层级无关）'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

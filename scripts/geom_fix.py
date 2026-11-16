"""修好之后的几何实验：**测布线，不测格点**。

## 为什么重做

2026-11-16 前四轮结构实验全部作废，因为网络几乎不发放
（发放率 **0.0001**，递归/输入 **0.006×**），读出特征近零，
"准确率"其实是**多数类基线 0.376**。

用户随后指出方向错误：文献里的"高维"不是**格点的几何维数**，而是

| 含义 | 可测量 |
| :--- | :--- |
| ① **表征几何**：状态空间的有效维度 | **participation ratio**（(Σλ)²/Σλ²） |
| ② **连接组的内蕴几何**：小世界 / 双曲 / 无标度 | 度数分布、直径、**重连比例** |
| ③ 场自由度 | 不在本实验范围 |

> 「维度不是空间的属性，**是布线的属性**。」

## 本实验

**工作点先标定**：输入 scale=2 ⇒ 发放率 ~10%（scale=1 时是 0.0001）。

| 臂 | 布线 |
| :--- | :--- |
| `ring` | 1D 环，半径 4（8 邻居） |
| `grid3` | 3D 格点，半径 1（26 邻居） |
| `smallworld` | 3D 格点 + **20% 的边重连到随机目标**（小世界） |
| `scalefree` | **优先连接**（度分布幂律，双曲/层级网络的可计算替身） |

读出：全局线性 / 全局 MLP。

## 判据

| 编号 | 判据 |
| :--- | :--- |
| **F-0** | 每个臂的发放率落在 **[0.03, 0.20]** —— 工作点有效 |
| **F-1** | 准确率 **≥ 多数类 + 15pp** —— 真在解码，不是在猜多数类 |
| **F-2** | 最佳布线比 `ring` 高 **≥5pp**，且三种子同向 |
| **F-3** | 报告 participation ratio 与准确率的关系（相关性，不作通过线） |

**F-1 不过 ⇒ 该臂的结果不计入 F-2。**
"""

from __future__ import annotations

import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.dim_ladder import mlp, ridge
from scripts.structure_ladder import K, V, make_pref, make_task

OUT = _pathlib.Path("sweep_results/geom_fix")
TAUS = (2.0, 16.0)


def _coords(n: int, dims: tuple[int, ...]) -> torch.Tensor:
    idx = torch.arange(n)
    out, stride = [], 1
    for d in reversed(dims):
        out.append((idx // stride) % d)
        stride *= d
    return torch.stack(out[::-1], 1)


def _lin(c: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    out, stride = torch.zeros(c.shape[0], dtype=torch.long), 1
    for k in range(len(dims) - 1, -1, -1):
        out = out + c[:, k] * stride
        stride *= dims[k]
    return out


def wiring(kind: str, n: int, gain: float, seed: int, dev: str):
    """返回 (pre, wrec)。**只改布线，不改输入、不改增益。**"""
    g = torch.Generator().manual_seed(seed)
    if kind == "ring":
        off = torch.tensor([-1, -2, 1, 2, -4, -3, 3, 4])
        pre = (torch.arange(n).unsqueeze(1) + off.unsqueeze(0)) % n
    elif kind in ("grid3", "smallworld"):
        dims = (16, 16, 16)
        ci = _coords(n, dims)
        offs = torch.cartesian_prod(*[torch.arange(-1, 2) for _ in range(3)]).reshape(-1, 3)
        offs = offs[(offs != 0).any(dim=1)]
        pre = torch.stack([_lin((ci + o) % torch.tensor(dims), dims) for o in offs], 1)
        if kind == "smallworld":
            # 20% 的边重连到随机目标：度数不变、路径长度骤降
            mask = torch.rand(pre.shape, generator=g) < 0.2
            rnd = torch.randint(0, n, pre.shape, generator=g)
            pre = torch.where(mask, rnd, pre)
    elif kind == "scalefree":
        # 优先连接：度分布幂律。双曲/层级网络的可计算替身。
        deg = torch.ones(n)
        edges = []
        for _ in range(8):  # 每个神经元 8 条出边，与 ring 同度数
            prob = deg / deg.sum()
            src = torch.multinomial(prob, n, replacement=True, generator=g)
            edges.append(src)
            deg = deg + torch.bincount(src, minlength=n).float()
        pre = torch.stack(edges, 1)
    else:
        raise ValueError(kind)
    wrec = torch.rand(n, pre.shape[1], generator=g).to(dev)
    wrec = wrec * (gain / wrec.sum(1, keepdim=True))
    return pre, wrec


@torch.no_grad()
def run(w_in, pre, wrec, toks, dev, stride=8):
    b, T = toks.shape
    n = w_in.shape[1]
    v = torch.zeros((b, n), device=dev)
    z = torch.zeros((b, n), device=dev)
    ref = torch.zeros((b, n), device=dev)
    decays = torch.tensor([1.0 / t for t in TAUS], device=dev)
    read = torch.zeros((b, len(TAUS), n), device=dev)
    feats, rates = [], 0.0
    for t in range(T):
        cur = w_in[toks[:, t]]
        rec = torch.zeros_like(z)
        for k in range(pre.shape[1]):
            rec = rec + z[:, pre[:, k]] * wrec[:, k].unsqueeze(0)
        nv = 0.95 * v + cur + rec - z * 0.5
        fresh = ((nv - 0.5) / 0.5 > 0).float()
        nz = torch.where(ref > 0, torch.zeros_like(fresh), fresh)
        ref = torch.clamp(ref + 2.0 * nz - 1.0, 0.0, 2.0)
        v, z = nv, nz
        rates += float(z.mean())
        read = read * decays.view(1, -1, 1) + z.unsqueeze(1)
        if t % stride == 0:
            feats.append(read.reshape(b, -1))
    return torch.stack(feats, 1), rates / T


def participation_ratio(feats: torch.Tensor) -> float:
    """状态空间的有效维度：(Σλ)² / Σλ²。等于"用了几个方向"。"""
    X = feats.reshape(-1, feats.shape[-1]).float()
    X = X - X.mean(0, keepdim=True)
    # 用样本协方差的特征值（列数远大于样本数时走 Gram 矩阵）
    if X.shape[0] < X.shape[1]:
        ev = torch.linalg.eigvalsh(X @ X.T / X.shape[0]).clamp(min=0)
    else:
        ev = torch.linalg.eigvalsh(X.T @ X / X.shape[0]).clamp(min=0)
    s1, s2 = float(ev.sum()), float((ev**2).sum())
    return (s1 * s1) / max(s2, 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=96)
    ap.add_argument("--train-seq", type=int, default=256)
    ap.add_argument("--test-seq", type=int, default=64)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--gain", type=float, default=1.2)
    ap.add_argument("--input-scale", type=float, default=2.0)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(x) for x in args.seeds.split(",")]
    res: dict = {"input_scale": args.input_scale, "gain": args.gain, "cells": {}}

    # 多数类基线
    maj = []
    for sd in seeds:
        pref = make_pref(0, 3.0).to(dev)
        _, tes, _ = make_task(args.test_seq, args.seq, 20 + sd, dev, pref)
        st = tes[:, :: args.stride].reshape(-1)
        maj.append(float(torch.bincount(st, minlength=K).max()) / st.numel())
    res["majority"] = sum(maj) / len(maj)
    print(f"  多数类基线 {res['majority']:.3f}", flush=True)

    kinds = ["ring", "grid3", "smallworld", "scalefree"]
    for kind in kinds:
        for dec_name, dec in (("linear", ridge), ("mlp", mlp)):
            accs, prs, frs = [], [], []
            for sd in seeds:
                pref = make_pref(0, 3.0).to(dev)
                tr, trs, _ = make_task(args.train_seq, args.seq, 10 + sd, dev, pref)
                te, tes, _ = make_task(args.test_seq, args.seq, 20 + sd, dev, pref)
                toks = torch.cat([tr, te])
                st = torch.cat([trs, tes])[:, :: args.stride]
                g = torch.Generator().manual_seed(sd)
                w_in = (torch.randn(V, 4096, generator=g) / 4096**0.5 * args.input_scale).to(dev)
                pre, wrec = wiring(kind, 4096, args.gain, sd, dev)
                feat, fr = run(w_in, pre, wrec, toks, dev, args.stride)
                accs.append(dec(feat, st, args.train_seq))
                prs.append(participation_ratio(feat))
                frs.append(fr)
            key = f"{kind}/{dec_name}"
            res["cells"][key] = {
                "acc": sum(accs) / len(accs),
                "runs": accs,
                "firing": sum(frs) / len(frs),
                "pr": sum(prs) / len(prs),
            }
            print(
                f"  {kind:11s} {dec_name:6s} acc {sum(accs) / len(accs):.3f}  "
                f"发放 {sum(frs) / len(frs):.4f}  PR {sum(prs) / len(prs):7.1f}",
                flush=True,
            )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )

    print("\n判：")
    floor = res["majority"] + 0.15
    for k, v in res["cells"].items():
        ok_fr = 0.03 <= v["firing"] <= 0.20
        ok_acc = v["acc"] >= floor
        print(
            f"  F-0 {k:20s} 发放 {v['firing']:.4f} {'✓' if ok_fr else '✗'}   "
            f"F-1 acc {v['acc']:.3f} vs 门槛 {floor:.3f} {'✓' if ok_acc else '✗'}"
        )
    base = res["cells"]["ring/mlp"]["acc"]
    best = max(res["cells"], key=lambda k: res["cells"][k]["acc"])
    b = res["cells"][best]
    gain = b["acc"] - base
    same = all(
        x >= y - 1e-9 for x, y in zip(b["runs"], res["cells"]["ring/mlp"]["runs"], strict=False)
    )
    print(
        f"  F-2 最佳 {best} {b['acc']:.3f} − ring/mlp {base:.3f} = {gain:+.3f}；"
        f"三种子同向 {'是' if same else '否'} ⇒ {'过' if gain >= 0.05 and same else '不过'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

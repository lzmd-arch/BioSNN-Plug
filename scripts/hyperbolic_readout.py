"""双曲**读出**：在同一读出维度下，能分开多少个类别（预注册 H-R）。

## 设计

读出维度**固定 d=2**，扫类别数 K。

| 臂 | 读出 |
| :--- | :--- |
| 欧氏 | 特征 → 2 维欧氏平面 → **欧氏距离**到原型 |
| 双曲 | 特征 → **Poincaré 圆盘**（tanh 压缩）→ **双曲距离**到原型 |

两臂的投影层参数量相同、维度相同，**唯一差别是几何**。

## 为什么这条能测出东西

欧氏 2 维里，K 个原型的可分性受周长约束（线性）；双曲 2 维的"周长"随半径指数增长。
所以 **K 越大，双曲的相对优势应该越明显**。

## 判据

| 编号 | 判据 |
| :--- | :--- |
| **HR-1** | K=32 时双曲 ≥ 欧氏 **+5pp** |
| **HR-2** | 增益随 K 单调（K=4 时≈0，K=32 时最大）——证明确实是容量而非别的 |
| **HR-3** | 两臂参数量差 <5% |
"""

from __future__ import annotations

# ruff: noqa: N803, N806  -- 矩阵推导里 X/W/A/T/K 是惯例
import argparse
import itertools
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.instrument import seeded_dirichlet
from scripts.structure_ladder import V

OUT = _pathlib.Path("sweep_results/hyperbolic_readout")
TAUS = (2.0, 16.0)


def make_task_k(n_seq: int, T: int, K: int, seed: int, dev: str):
    """**相位随机的确定性循环**：s_t = (phase + t // seg) % K。

    之前用 p_stay=0.97 的随机游走，初始态固定为 0 ⇒ 状态分布极度偏斜，
    K≥16 时多数类在训练集里根本不出现 ⇒ 均值 NaN ⇒ 准确率恒为 0。

    改后：每个状态都出现 `seg` 步，且**相位每条序列随机** ⇒ 位置不能用来作弊。
    """
    g = torch.Generator().manual_seed(seed)
    pref = seeded_dirichlet(3.0, K).to(dev)
    seg = max(1, T // K)
    phase = torch.randint(0, K, (n_seq,), generator=g).to(dev)
    states = torch.zeros(n_seq, T, dtype=torch.long, device=dev)
    toks = torch.zeros(n_seq, T, dtype=torch.long, device=dev)
    for t in range(T):
        s = (phase + t // seg) % K
        states[:, t] = s
        toks[:, t] = torch.multinomial(pref[s].cpu(), 1, generator=g).squeeze(-1).to(dev)
    return toks, states


@torch.no_grad()
def reservoir(toks, dev, n=2048, stride=8):
    g = torch.Generator().manual_seed(0)
    w_in = (torch.randn(V, n, generator=g) / n**0.5 * 16.0).to(dev)
    off = torch.tensor([-1, -2, 1, 2])
    pre = (torch.arange(n).unsqueeze(1) + off.unsqueeze(0)) % n
    wrec = torch.rand(n, 4, generator=g).to(dev)
    wrec = wrec * (0.8 / wrec.sum(1, keepdim=True))
    b, T = toks.shape
    v = torch.zeros((b, n), device=dev)
    z = torch.zeros((b, n), device=dev)
    ref = torch.zeros((b, n), device=dev)
    decays = torch.tensor([1.0 / t for t in TAUS], device=dev)
    read = torch.zeros((b, len(TAUS), n), device=dev)
    feats = []
    for t in range(T):
        rec = torch.zeros_like(z)
        for k in range(4):
            rec = rec + z[:, pre[:, k]] * wrec[:, k].unsqueeze(0)
        nv = 0.95 * v + w_in[toks[:, t]] + rec - z * 0.5
        fresh = ((nv - 0.5) / 0.5 > 0).float()
        nz = torch.where(ref > 0, torch.zeros_like(fresh), fresh)
        ref = torch.clamp(ref + 2.0 * nz - 1.0, 0.0, 2.0)
        v, z = nv, nz
        read = read * decays.view(1, -1, 1) + z.unsqueeze(1)
        if t % stride == 0:
            feats.append(read.reshape(b, -1))
    return torch.stack(feats, 1), float(z.mean())


def hyper_dist(a, b):
    """a,b: (N,2) 圆盘坐标 → 双曲距离。"""
    na = (a**2).sum(1, keepdim=True).clamp(max=1 - 1e-6)
    nb = (b**2).sum(1, keepdim=True).clamp(max=1 - 1e-6)
    d2 = ((a.unsqueeze(1) - b.unsqueeze(0)) ** 2).sum(-1)
    c = 1 + 2 * d2 / ((1 - na) * (1 - nb.T))
    return torch.arccosh(c.clamp(min=1.0 + 1e-7))


class Readout(torch.nn.Module):
    def __init__(self, fin, K, hyperbolic: bool, d: int = 2):
        super().__init__()
        self.proj = torch.nn.Linear(fin, d)
        self.proto = torch.nn.Parameter(torch.randn(K, d) * 0.1)
        self.hyperbolic = hyperbolic

    def forward(self, f):
        z = self.proj(f)
        if self.hyperbolic:
            z = torch.tanh(z) * 0.95  # 压进 Poincaré 圆盘
            p = torch.tanh(self.proto) * 0.95
            return -hyper_dist(z, p)
        return -torch.cdist(z, self.proto)


def train_eval(feat, st, n_train, K, hyperbolic, dev, steps=400):
    X = feat.reshape(-1, feat.shape[-1])
    Y = st.reshape(-1)
    cut = n_train * st.shape[1]
    Xtr, Xte, Ytr, Yte = X[:cut], X[cut:], Y[:cut], Y[cut:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    net = Readout(Xtr.shape[1], K, hyperbolic).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(steps):
        idx = torch.randperm(Xtr.shape[0], device=dev)[:8192]
        opt.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(net(Xtr[idx]), Ytr[idx]).backward()
        opt.step()
    with torch.no_grad():
        return float((net(Xte).argmax(-1) == Yte).float().mean()), sum(
            p.numel() for p in net.parameters()
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="4,8,16,32")
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--train-seq", type=int, default=192)
    ap.add_argument("--test-seq", type=int, default=48)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(x) for x in args.seeds.split(",")]
    res: dict = {"cells": {}}

    for K in [int(x) for x in args.ks.split(",")]:
        out = {}
        for name, hyp in (("euclid", False), ("hyperbolic", True)):
            accs, params = [], 0
            for sd in seeds:
                tr, trs = make_task_k(args.train_seq, args.seq, K, 10 + sd, dev)
                te, tes = make_task_k(args.test_seq, args.seq, K, 20 + sd, dev)
                toks = torch.cat([tr, te])
                st = torch.cat([trs, tes])[:, ::8]
                feat, fr = reservoir(toks, dev)
                a, params = train_eval(feat, st, args.train_seq, K, hyp, dev)
                accs.append(a)
                del feat
                torch.cuda.empty_cache()
            out[name] = {"acc": sum(accs) / len(accs), "runs": accs, "params": params, "firing": fr}
        out["K"] = K
        res["cells"][str(K)] = out
        print(
            f"  K={K:3d} 欧氏 {out['euclid']['acc']:.3f}  双曲 {out['hyperbolic']['acc']:.3f}  "
            f"差 {out['hyperbolic']['acc'] - out['euclid']['acc']:+.3f}  "
            f"（参数量 欧 {out['euclid']['params']} / 双 {out['hyperbolic']['params']}）",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print("\n判：")
    ks = sorted(res["cells"], key=int)
    top = res["cells"][ks[-1]]
    d = top["hyperbolic"]["acc"] - top["euclid"]["acc"]
    print(f"  HR-1 K={ks[-1]} 双曲 − 欧氏 = {d:+.3f} ≥0.05 ⇒ {'过' if d >= 0.05 else '不过'}")
    diffs = [res["cells"][k]["hyperbolic"]["acc"] - res["cells"][k]["euclid"]["acc"] for k in ks]
    单调 = all(b >= a - 1e-9 for a, b in itertools.pairwise(diffs))
    print(f"  HR-2 增益随 K 单调：{[f'{x:+.3f}' for x in diffs]} ⇒ {'过' if 单调 else '不过'}")
    p0, p1 = top["euclid"]["params"], top["hyperbolic"]["params"]
    print(
        f"  HR-3 参数量差 {abs(p0 - p1) / max(p0, p1):.1%} <5% ⇒ "
        f"{'过' if abs(p0 - p1) / max(p0, p1) < 0.05 else '不过'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

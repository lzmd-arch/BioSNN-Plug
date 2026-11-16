"""双曲 vs 欧氏：**在原始特征空间、原生维度、无可学投影**下比。

## 修的两个设计缺陷

上一轮（`hyperbolic_readout.py` / `hyperbolic_reason.py`）两处都错在同一个地方：

1. **维度被压到 2**：网络状态是 8192 维，压到 2 维的信息损失淹没了几何差异；
2. **中间有一层可学 Linear**：它可以把空间任意扭曲，几何差异归不到几何上。

**这一版**：几何直接作用在**原始特征**上，维度可配置（2 / 32 / 8192 = 原生），
**没有任何可学投影**，分类器就是**最近原型**（原型 = 类均值）。

## 两个臂共用的坐标，只换度量

```
x = tanh(α·‖f‖) · f/‖f‖        # 半径随特征模长变化 —— 双曲的容量来自半径差
```

- **欧氏臂**：`‖x − μ_c‖₂`
- **双曲臂**：`arccosh(1 + 2‖x−μ_c‖² / ((1−‖x‖²)(1−‖μ_c‖²)))`

**同一个 x、同一组原型，只换距离。** 若把点都放到同一半径上，双曲距离会退化成角距离，
那测的是余弦相似度——所以半径必须随模长变化。

## 判据

| 编号 | 判据 |
| :--- | :--- |
| **HP-1** | 原生维度（8192）上，双曲 ≥ 欧氏 **+5pp** |
| **HP-2** | 增益随维度**单调上升**（2 → 32 → 8192）——证明确实是"高维里几何才发力" |
| **HP-3** | 打乱类别标签后两者都掉到随机（防止标签泄漏） |
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

from scripts.hyperbolic_readout import make_task_k, reservoir

OUT = _pathlib.Path("sweep_results/hyperbolic_raw")


def project(f: torch.Tensor, alpha: float) -> torch.Tensor:
    """半径随模长变化：‖x‖ = tanh(α‖f‖)。**不能是常数半径**，否则双曲退化成角距离。"""
    nrm = f.norm(dim=1, keepdim=True).clamp(min=1e-9)
    u = f / nrm
    return u * torch.tanh(alpha * nrm)


def _centroids(Xtr, Ytr, K):
    """空类回退到全局均值 —— 否则 mean 出 NaN，所有预测塌成一类。"""
    glob = Xtr.mean(0, keepdim=True)
    return torch.cat(
        [(Xtr[Ytr == c].mean(0, keepdim=True) if (Ytr == c).any() else glob) for c in range(K)]
    )


def euclid_acc(Xtr, Ytr, Xte, Yte, K):
    mu = _centroids(Xtr, Ytr, K)
    return float(((Xte[:, None, :] - mu[None]).pow(2).sum(-1).argmin(1) == Yte).float().mean())


def hyper_acc(Xtr, Ytr, Xte, Yte, K):
    mu = _centroids(Xtr, Ytr, K)
    # 原型也要压回球内
    mn = mu.norm(dim=1, keepdim=True).clamp(max=1 - 1e-3)
    mu = mu / mu.norm(dim=1, keepdim=True).clamp(min=1e-9) * mn
    nx = Xte.norm(dim=1, keepdim=True).clamp(max=1 - 1e-4)
    Xte = Xte / Xte.norm(dim=1, keepdim=True).clamp(min=1e-9) * nx
    d2 = (Xte[:, None, :] - mu[None]).pow(2).sum(-1)
    c = 1 + 2 * d2 / ((1 - nx.pow(2)) * (1 - mu.norm(dim=1, keepdim=True).pow(2).T))
    return float((torch.arccosh(c.clamp(min=1.0 + 1e-7)).argmin(1) == Yte).float().mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="8,16,32")
    ap.add_argument("--dims", default="2,32,8192")
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--train-seq", type=int, default=192)
    ap.add_argument("--test-seq", type=int, default=48)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(x) for x in args.seeds.split(",")]
    res: dict = {"alpha": args.alpha, "cells": {}}

    for K in [int(x) for x in args.ks.split(",")]:
        cache = []
        for sd in seeds:
            tr, trs = make_task_k(args.train_seq, args.seq, K, 10 + sd, dev)
            te, tes = make_task_k(args.test_seq, args.seq, K, 20 + sd, dev)
            toks = torch.cat([tr, te])
            st = torch.cat([trs, tes])[:, ::8]
            feat, _ = reservoir(toks, dev)
            X = feat.reshape(-1, feat.shape[-1])
            Y = st.reshape(-1)
            cache.append((X, Y, args.train_seq * st.shape[1]))
            del feat
            torch.cuda.empty_cache()
        for d in [int(x) for x in args.dims.split(",")]:
            e_accs, h_accs = [], []
            for X, Y, cut in cache:
                if d < X.shape[1]:  # 固定随机投影，**不学习**
                    g = torch.Generator().manual_seed(7)
                    P = (torch.randn(X.shape[1], d, generator=g) / X.shape[1] ** 0.5).to(dev)
                    Xd = X @ P
                else:
                    Xd = X
                mu, sd_ = Xd[:cut].mean(0), Xd[:cut].std(0) + 1e-6
                Xd = (Xd - mu) / sd_
                Xtr, Xte, Ytr, Yte = Xd[:cut], Xd[cut:], Y[:cut], Y[cut:]
                Xtr_p, Xte_p = (
                    project(Xtr, args.alpha),
                    project(Xte, args.alpha),
                )
                e_accs.append(euclid_acc(Xtr_p, Ytr, Xte_p, Yte, K))
                h_accs.append(hyper_acc(Xtr_p, Ytr, Xte_p, Yte, K))
            key = f"K{K}/d{d}"
            res["cells"][key] = {
                "euclid": sum(e_accs) / len(e_accs),
                "hyperbolic": sum(h_accs) / len(h_accs),
                "runs_e": e_accs,
                "runs_h": h_accs,
            }
            print(
                f"  K={K:3d} d={d:5d}  欧氏 {sum(e_accs) / len(e_accs):.3f}  "
                f"双曲 {sum(h_accs) / len(h_accs):.3f}  "
                f"差 {sum(h_accs) / len(h_accs) - sum(e_accs) / len(e_accs):+.3f}",
                flush=True,
            )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print("\n判：")
    for K in [int(x) for x in args.ks.split(",")]:
        native = res["cells"][f"K{K}/d{max(int(x) for x in args.dims.split(','))}"]
        d = native["hyperbolic"] - native["euclid"]
        print(
            f"  HP-1 K={K} 原生维度 双曲 − 欧氏 = {d:+.3f} ≥0.05 ⇒ {'过' if d >= 0.05 else '不过'}"
        )
    for K in [int(x) for x in args.ks.split(",")]:
        diffs = [
            res["cells"][f"K{K}/d{d}"]["hyperbolic"] - res["cells"][f"K{K}/d{d}"]["euclid"]
            for d in [int(x) for x in args.dims.split(",")]
        ]
        mono = all(b >= a - 1e-9 for a, b in itertools.pairwise(diffs))
        print(
            f"  HP-2 K={K} 增益随维度 {[f'{x:+.3f}' for x in diffs]} 单调 ⇒ {'过' if mono else '不过'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

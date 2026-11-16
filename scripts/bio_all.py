"""把生物机制**全部加回去**，测超可加性（预注册：全开 > 单项之和）。

## 用户 2026-11-16 的假设

> 也许只有**几乎完整模拟生物脑的所有角度加起来**才可以实现涌现。

这是一个可证伪的强假设。仓库里已经逐条测过，结论是「改标量/学习的机制**全部为零或负**」
（16 次实验）。**"单独无效、合起来有效"就是超可加。**

## 操作化

| 全开增益 vs 单项增益之和 | 判定 |
| :--- | :--- |
| **全开 > 和** | **超可加** —— 交互真实，假设成立 |
| 全开 ≈ max(单项) | 只是最好那一项在起作用，**没有涌现** |
| 全开 < 和 | 互相干扰 |

## 六个开关

| 开关 | 内容 |
| :--- | :--- |
| `alif` | 阈值适应 `θ + β·a`，`a ← ρa + z` |
| `tau` | 逐神经元异质 τ（log-uniform [5, 500]） |
| `delay` | 轴突延迟 = `round(格点距离)`，截到 [1,4] |
| `ei` | 20% 的递归权重取负 |
| `dend` | 输入分 4 束，每束求和后过 `sign·√|·|` 再合并 |
| `inhib` | 4 个池，池间侧抑制 |

## 纪律

- **每个臂单独标定输入尺度**（发放率 ∈ [0.03, 0.20]），并报出来；
- 每臂同时报**多数类基线**与**准确率**；
- 读出统一用 MLP（**它是仪器，不是产品**）——在这些开关还没证明有价值之前，
  不用"零可学参数"给自己加难度。
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

from scripts.dim_ladder import mlp
from scripts.instrument import calibrate, seeded_dirichlet

OUT = _pathlib.Path("sweep_results/bio_all")
TAUS = (2.0, 16.0)
V, K_DEF = 64, 8


def make_task(n_seq: int, T: int, K: int, seed: int, dev: str):
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


def build(n: int, switches: set[str], seed: int, dev: str, deg: int = 32):
    g = torch.Generator().manual_seed(seed)
    w_in = (torch.randn(V, n, generator=g) / n**0.5).to(dev)
    pre = torch.randint(0, n, (n, deg), generator=g).to(dev)
    wrec = torch.rand(n, deg, generator=g).to(dev)
    wrec = wrec * (1.2 / wrec.sum(1, keepdim=True))
    if "ei" in switches:
        mask = torch.rand(n, deg, generator=g).to(dev) < 0.2
        wrec = torch.where(mask, -wrec, wrec)
    alpha = (
        torch.exp(
            -1.0
            / torch.exp(torch.rand(n, generator=g) * (math.log(500) - math.log(5)) + math.log(5))
        )
        if "tau" in switches
        else torch.full((n,), 0.95)
    ).to(dev)
    delay = (
        (torch.randint(1, 5, (n, deg), generator=g))
        if "delay" in switches
        else torch.ones(n, deg, dtype=torch.long)
    ).to(dev)  # **必须上设备**，否则 mask 索引会炸
    # 池划分（侧抑制用）
    pool_id = (torch.arange(n) % 4).to(dev) if "inhib" in switches else None
    return w_in, pre, wrec, alpha, delay, pool_id


@torch.no_grad()
def run(x, toks, dev, stride=8):
    w_in, pre, wrec, alpha, delay, pool_id, sw = x
    b, T = toks.shape
    n = w_in.shape[1]
    v = torch.zeros((b, n), device=dev)
    z = torch.zeros((b, n), device=dev)
    ref = torch.zeros((b, n), device=dev)
    a = torch.zeros((b, n), device=dev)
    hist = [torch.zeros_like(z) for _ in range(4)]
    decays = torch.tensor([1.0 / t for t in TAUS], device=dev)
    read = torch.zeros((b, len(TAUS), n), device=dev)
    feats, rate = [], 0.0
    for t in range(T):
        cur = w_in[toks[:, t]]
        rec = torch.zeros_like(z)
        for k in range(pre.shape[1]):
            pk = pre[:, k]
            if "delay" in sw:
                for d in range(1, 5):
                    m = delay[:, k] == d
                    if m.any():
                        rec[:, m] += hist[d - 1][:, pk[m]] * wrec[m, k].unsqueeze(0)
            else:
                rec = rec + z[:, pk] * wrec[:, k].unsqueeze(0)
        if "inhib" in sw:
            # 池间侧抑制：减去其他池上一步的平均发放
            pm = torch.zeros(b, 4, device=dev)
            for p in range(4):
                pm[:, p] = z[:, pool_id == p].mean(1)
            rec = rec - 0.5 * pm[:, pool_id]
        theta = 0.5 + (0.4 * a if "alif" in sw else 0.0)
        nv = alpha * v + cur + rec - z * 0.5
        fresh = ((nv - theta) / theta > 0).float()
        nz = torch.where(ref > 0, torch.zeros_like(fresh), fresh)
        ref = torch.clamp(ref + 2.0 * nz - 1.0, 0.0, 2.0)
        if "alif" in sw:
            a = 0.99 * a + nz
        v, z = nv, nz
        hist = [z, *hist[:-1]]
        rate += float(z.mean())
        read = read * decays.view(1, -1, 1) + z.unsqueeze(1)
        if t % stride == 0:
            feats.append(read.reshape(b, -1))
    return torch.stack(feats, 1), rate / T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--deg", type=int, default=32)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--train-seq", type=int, default=320)
    ap.add_argument("--test-seq", type=int, default=64)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(x) for x in args.seeds.split(",")]
    K = K_DEF
    singles = ["alif", "tau", "delay", "ei", "inhib"]
    arms = [("off", set())] + [(s, {s}) for s in singles] + [("all", set(singles))]
    res: dict = {"n": args.n, "deg": args.deg, "cells": {}}

    for name, sw in arms:
        # 标定
        tr0, _ = make_task(32, 48, K, 10, dev)
        w_in, pre, wrec, alpha, delay, pool_id = build(args.n, sw, 0, dev, args.deg)
        cur = (w_in, pre, wrec, alpha, delay, pool_id, sw)
        # 标定走仪器：等比细网格 + 窗口判定（粗网格会整段跨过窗口）
        ok_cal, sc, _pfr = calibrate(
            lambda s, _w=w_in, _rest=cur[1:], _t=tr0: run((_w * s, *_rest), _t, dev)[1]
        )
        if not ok_cal:
            print(f"  {name:6s} ✗ 标定失败，该臂无效", flush=True)
            continue
        accs, frs = [], []
        for sd in seeds:
            tr, trs = make_task(args.train_seq, args.seq, K, 10 + sd, dev)
            te, tes = make_task(args.test_seq, args.seq, K, 20 + sd, dev)
            toks = torch.cat([tr, te])
            st = torch.cat([trs, tes])[:, ::8]
            w2 = build(args.n, sw, sd, dev, args.deg)
            feat, fr = run((w2[0] * sc, *w2[1:], sw), toks, dev)
            accs.append(mlp(feat, st, args.train_seq))
            frs.append(fr)
            del feat
            torch.cuda.empty_cache()
        res["cells"][name] = {
            "acc": sum(accs) / len(accs),
            "runs": accs,
            "firing": sum(frs) / len(frs),
            "scale": sc,
        }
        print(
            f"  {name:6s} scale {sc:4.1f}  发放 {sum(frs) / len(frs):.4f}  "
            f"acc {sum(accs) / len(accs):.3f}",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print("\n判：")
    if "off" in res["cells"] and "all" in res["cells"]:
        off = res["cells"]["off"]["acc"]
        gains = {s: res["cells"][s]["acc"] - off for s in singles if s in res["cells"]}
        allg = res["cells"]["all"]["acc"] - off
        ssum = sum(gains.values())
        print(f"  关：{off:.3f}   全开：{res['cells']['all']['acc']:.3f}（增益 {allg:+.3f}）")
        for s, g in gains.items():
            print(f"    单项 {s:6s} 增益 {g:+.3f}")
        print(f"  单项之和 {ssum:+.3f}   max 单项 {max(gains.values()):+.3f}")
        print(
            f"  **超可加判定**：全开 {allg:+.3f} vs 和 {ssum:+.3f} ⇒ "
            f"{'超可加（交互成立）' if allg > ssum + 0.02 else '不超可加'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

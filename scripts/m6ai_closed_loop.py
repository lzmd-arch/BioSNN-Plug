"""M6ai：**闭环自输入** —— 让网络的输出决定它下一个看到什么。

## 为什么这是今晚最该建的东西

前面所有测量（解码、PR、整合、持久、遗忘曲线）都是在**被动系统**上做的：
给一段输入，看终态。**没有一次让系统自己的行为改变它的输入。**

有状态 ≠ 有行为。闭环是"个体在做自己的事"的最小实现：
`下一个输入符号 = 从当前状态经一个固定"运动映射"选出来的`。

## 设计

- 一段历史之后**关掉外部输入**，改为 `symbol_t = argmax(W_motor · rate_{t-1})`，
  `W_motor` 是固定随机的 `8 × N` 映射（"运动皮层到输入"的接线，不学习）；
- 记录：发放是否维持在窗内、符号序列的复杂度、以及**闭环轨迹是否依赖初始历史**。

## 判据

- **AI-0 不熄火也不饱和**：闭环中 ≥80% 的步发放 ∈ [0.01, 0.20]；
- **AI-1 行为非平凡** ★：符号序列的归一化熵 > 0.5（不是恒定、也不是二选一）；
- **AI-2 带自己的历史** ★★：两条不同历史出发的闭环轨迹，差异 > 同历史两次重复的差异。
"""

from __future__ import annotations

# ruff: noqa: N806
import argparse
import json
import math
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.hist_families import make_family_hist
from scripts.m6_pools import N, build_pools

OUT = _pathlib.Path("sweep_results/m6ai_closed_loop")
K = 4
T_HIST, T_CLOSED = 1024, 512
N_SYM = 8


def motor_map(seed: int, dev: str) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed + 4242)
    return (torch.randn(N_SYM, N, generator=g) / N**0.5).to(dev)


@torch.no_grad()
def run_closed(net, seed: int, hist, dev: str, t_closed: int, leak_scale: float = 1.0):
    """先跑历史，再闭环。返回 `(符号序列, 每步发放)`。"""
    w_in, pre, wrec, _pid = net
    w_in = w_in * leak_scale
    B = hist.shape[0]
    alive = torch.ones(N, device=dev)
    v = torch.zeros((B, N), device=dev)
    z = torch.zeros((B, N), device=dev)
    ref = torch.zeros((B, N), device=dev)
    f_i = torch.full((N,), 0.08, device=dev)
    f_run = 0.08
    wmot = motor_map(seed, dev)

    for t in range(hist.shape[1]):
        rec = torch.zeros_like(z)
        for j in range(pre.shape[1]):
            rec = rec + z[:, pre[:, j]] * wrec[:, j].unsqueeze(0)
        phi = float(z.mean())
        cur = w_in[hist[:, t]] - 1.0 * (f_run / 0.08) * phi
        nv = 0.95 * v + cur + rec - z * 0.5
        nz = torch.where(ref > 0, torch.zeros_like(nv), ((nv - 0.5) / 0.5 > 0).float()) * alive
        ref = torch.clamp(ref + 2.0 * nz - 1.0, 0.0, 2.0)
        v, z = nv, nz
        f_i = 0.9 * f_i + 0.1 * z.mean(0)
        f_run = 0.9 * f_run + 0.1 * float(z.mean())

    syms, rates = [], []
    sym = hist[:, -1]
    for _t in range(t_closed):
        rec = torch.zeros_like(z)
        for j in range(pre.shape[1]):
            rec = rec + z[:, pre[:, j]] * wrec[:, j].unsqueeze(0)
        phi = float(z.mean())
        cur = w_in[sym] - 1.0 * phi
        nv = 0.95 * v + cur + rec - z * 0.5
        nz = torch.where(ref > 0, torch.zeros_like(nv), ((nv - 0.5) / 0.5 > 0).float()) * alive
        ref = torch.clamp(ref + 2.0 * nz - 1.0, 0.0, 2.0)
        v, z = nv, nz
        rates.append(float(z.mean()))
        # **运动映射**：当前状态 -> 下一个符号（8 类里选一个，再映射到词表）
        logits = wmot @ z.mean(0)
        sym = logits.argmax().expand(B)
        syms.append(int(logits.argmax()))
    return syms, rates


def hamming(a, b) -> float:
    """两条符号序列的汉明距离占比 —— 闭环轨迹的"行为差异"。"""
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] != b[i]) / max(n, 1)


def entropy_norm(syms) -> float:
    n = len(syms)
    cnt: dict = {}
    for s in syms:
        cnt[s] = cnt.get(s, 0) + 1
    h = -sum((c / n) * math.log(c / n) for c in cnt.values())
    return h / math.log(N_SYM)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--t-closed", type=int, default=T_CLOSED)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": []}
    for sd in range(args.seeds):
        net = build_pools(sd, dev, K, 0.0)
        # A 与 A'：**同族、不同随机流**（同统计）；B：不同族 ⇒ 不同统计
        ha = make_family_hist(0, sd * 97 + 1, T_HIST, 1, dev)
        ha2 = make_family_hist(0, sd * 97 + 11, T_HIST, 1, dev)
        hb = make_family_hist(5, sd * 97 + 2, T_HIST, 1, dev)
        ha_rep = make_family_hist(0, sd * 97 + 1, T_HIST, 1, dev)  # 逐位重复，查确定性
        sa, ra = run_closed(net, sd, ha, dev, args.t_closed)
        sa2, _ = run_closed(net, sd, ha2, dev, args.t_closed)
        sb, _ = run_closed(net, sd, hb, dev, args.t_closed)
        srep, _ = run_closed(net, sd, ha_rep, dev, args.t_closed)
        inwin = sum(1 for r in ra if 0.01 <= r <= 0.20) / len(ra)
        d_hist = hamming(sa, sb)
        d_null = hamming(sa, sa2)
        row = {
            "seed": sd,
            "in_window_frac": inwin,
            "entropy_a": entropy_norm(sa),
            "entropy_b": entropy_norm(sb),
            "same_history_identical": sa == srep,
            "d_hist": d_hist,
            "d_null": d_null,
            "ratio": d_hist / max(d_null, 1e-9),
            "firing_mean": sum(ra) / len(ra),
        }
        res["cells"].append(row)
        print(
            f"  seed {sd}: 窗内 {inwin:.2f}  发放 {row['firing_mean']:.4f}  "
            f"熵 A {row['entropy_a']:.2f} B {row['entropy_b']:.2f}  "
            f"轨迹距离 历史 {d_hist:.3f} / 对照 {d_null:.3f} = {row['ratio']:.2f}x  "
            f"逐位重复 {row['same_history_identical']}",
            flush=True,
        )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )

    print("")
    print("判：")
    rows = res["cells"]
    m = lambda k, _r=rows: sum(r[k] for r in _r) / len(_r)  # noqa: E731
    ai0 = m("in_window_frac") >= 0.8
    print(
        f"  AI-0 闭环不熄火不饱和（≥80% 步在窗内）：{m('in_window_frac'):.2f} "
        f"=> {'过' if ai0 else '不过'}"
    )
    ent = max(m("entropy_a"), m("entropy_b"))
    ai1 = ent > 0.5
    print(f"  AI-1 行为非平凡（归一化熵 >0.5）：{ent:.2f} => {'过' if ai1 else '不过'}")
    wins = sum(1 for r in rows if r["d_hist"] > r["d_null"])
    ratio = m("d_hist") / max(m("d_null"), 1e-9)
    ai2 = ratio > 1.5 and wins == len(rows)
    print(
        f"  AI-2 闭环轨迹带历史（>1.5，逐种子同号）：{ratio:.2f}x {wins}/{len(rows)} "
        f"=> {'过' if ai2 else '不过'}"
    )
    print(
        f"       确定性：同历史逐位相同 "
        f"{sum(1 for r in rows if r['same_history_identical'])}/{len(rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

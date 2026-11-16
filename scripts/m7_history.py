"""M7 个体史：同权重、不同历史 ⇒ 同一探针上响应不同，且在静默间隔后仍不同。

预注册：`research/sel/prereg-m7-history-v1.md`

关键设计：**探针对 A、B 完全相同** ⇒ 任何可区分性都只能来自内部状态，
不是来自输入。
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.bio_all import make_task
from scripts.instrument import FIRING_WINDOW, scale_grid
from scripts.turnover import build_net

OUT = _pathlib.Path("sweep_results/m7_history")
N = 1024
#: 稳态场强度（v15 里 -0.5~-1 最优，这里取正值配合负号）
G_FIELD = 1.0
#: 结构变化比例：历史中活跃最低的这批神经元被永久静默
STRUCT_Q = 0.2
T_HIST = 96
T_GAP = 96
T_PROBE = 32


@torch.no_grad()
def run_seg(w_in, pre, wrec, toks, dev, state, alive, gain, steps):
    """跑一段，**状态跨段传递**。返回 `(新状态, 本段平均发放)`。

    `run_block` 每次都把 `v/z/ref` 重置为零 —— 那样历史和探针根本不在同一个个体上跑。
    M7 要测的正是闭环，所以必须自己管状态。
    """
    v, z, ref = state
    b = toks.shape[0]
    total, n = 0.0, 0
    read = torch.zeros((b, 2, N), device=dev)
    decays = torch.tensor([0.5, 0.0625], device=dev)  # τ = 2, 16
    f_i = torch.full((N,), 0.08, device=dev)
    f_run = 0.08
    for t in range(min(steps, toks.shape[1])):
        rec = torch.zeros_like(z)
        for j in range(pre.shape[1]):
            rec = rec + z[:, pre[:, j]] * wrec[:, j].unsqueeze(0)
        cur = w_in[toks[:, t]] * gain.unsqueeze(0)
        # **稳态场**：递归饱和压不住（输入增益替代不了递归驱动），
        # 所以用全局抑制反馈把群体活动按住（v15 验证过的那条）
        phi = float(z.mean())
        cur = cur - G_FIELD * (f_run / 0.08) * phi
        f_run = 0.9 * f_run + 0.1 * phi
        nv = 0.95 * v + cur + rec - z * 0.5
        fresh = ((nv - 0.5) / 0.5 > 0).float()
        nz = torch.where(ref > 0, torch.zeros_like(fresh), fresh) * alive
        ref = torch.clamp(ref + 2.0 * nz - 1.0, 0.0, 2.0)
        v, z = nv, nz
        # **内增益稳态**：没有它，持久状态下的网络会在 96 步内爬到饱和（实测 0.5）
        f_i = 0.9 * f_i + 0.1 * z.mean(0)
        if t % 4 == 3:
            gain = torch.clamp(gain * torch.exp(2.0 * (0.08 - f_i)), 0.2, 5.0)
        read = read * decays.view(1, -1, 1) + z.unsqueeze(1)
        total += float(z.sum() / (alive.sum() * b))
        n += 1
    return (v, z, ref), total / max(n, 1), read.reshape(b, -1).mean(0)


@torch.no_grad()
def response(
    seed: int,
    alpha_hist: float,
    gap: int,
    dev: str,
    hist_seed: int,
    scale: float,
    struct: bool = False,
):
    """跑历史 → 静默 `gap` 步 → 同一探针。返回探针窗内的每存活神经元平均发放。

    `hist_seed` 相同 ⇒ 历史流逐位相同（M7-3 靠这条）。
    `scale` 由 `calibrate_scale` 给出 —— **不标定就会饱和**（第一次跑 0.309，历史完全无影响）。
    """
    w_in, pre, wrec = build_net(seed, dev)
    w_in = w_in * scale
    alive = torch.ones(N, device=dev)
    gain = torch.ones(N, device=dev)
    state = (
        torch.zeros((32, N), device=dev),
        torch.zeros((32, N), device=dev),
        torch.zeros((32, N), device=dev),
    )
    hist, _ = make_task(32, T_HIST, 8, hist_seed, dev, alpha_hist)
    state, _, _ = run_seg(w_in, pre, wrec, hist, dev, state, alive, gain, T_HIST)
    if struct:
        # **历史写进结构**：把历史中活跃最低的那批神经元永久静默
        # （状态会被后续输入冲掉，结构不会被冲掉）
        act = (state[1] != 0).float().mean(0)
        thr = torch.quantile(act, STRUCT_Q)
        alive = torch.where(act <= thr, torch.zeros_like(alive), alive)
    # 静默间隔：**同一状态接着跑**，只喂中立输入
    gap_toks, _ = make_task(32, max(gap, 1), 8, 4242, dev, 3.0)
    if gap > 0:
        state, _, _ = run_seg(w_in, pre, wrec, gap_toks, dev, state, alive, gain, gap)
    probe, _ = make_task(32, T_PROBE, 8, 7777, dev, 3.0)
    _, fr, feat = run_seg(w_in, pre, wrec, probe, dev, state, alive, gain, T_PROBE)
    del w_in, pre, wrec, hist, probe
    torch.cuda.empty_cache()
    return fr, feat


def calibrate_scale(seed: int, dev: str) -> float:
    """把发放标进工作窗。**标定也必须走 `run_seg`** ——
    用会自动重置状态的 `run_block` 标出来的尺度，在持久状态下会饱和（实测打到 0.5）。
    """
    w_in, pre, wrec = build_net(seed, dev)
    toks, _ = make_task(32, 48, 8, 7777, dev, 3.0)
    alive = torch.ones(N, device=dev)
    gain = torch.ones(N, device=dev)
    for s in scale_grid(0.05, 2.0, 1.15):
        st = (
            torch.zeros((32, N), device=dev),
            torch.zeros((32, N), device=dev),
            torch.zeros((32, N), device=dev),
        )
        _, fr, _ = run_seg(w_in * s, pre, wrec, toks, dev, st, alive, gain, 48)
        if FIRING_WINDOW[0] <= fr <= FIRING_WINDOW[1]:
            del w_in, pre, wrec, toks
            torch.cuda.empty_cache()
            return s
    raise RuntimeError(f"seed {seed} 没有任何尺度能落在 {FIRING_WINDOW} —— 该臂无效")


def dprime(a: list[float], b: list[float]) -> float:
    n, m = len(a), len(b)
    ma, mb = sum(a) / n, sum(b) / m
    va = sum((x - ma) ** 2 for x in a) / max(n - 1, 1)
    vb = sum((x - mb) ** 2 for x in b) / max(m - 1, 1)
    return abs(ma - mb) / math.sqrt(max((va + vb) / 2.0, 1e-24))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--gaps", default="0,16,64")
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res: dict = {"n_seeds": args.seeds, "cells": {}}

    scales = [calibrate_scale(sd, dev) for sd in range(args.seeds)]
    res["scales"] = scales
    print(f"  标定尺度：{[round(x, 3) for x in scales]}", flush=True)

    for struct in (False, True):
        for gap in [int(x) for x in args.gaps.split(",")]:
            rel_hist, rel_null = [], []
            for sd in range(args.seeds):
                _, fa = response(sd, 1.0, gap, dev, 100 + sd, scales[sd], struct)
                _, fb = response(sd, 8.0, gap, dev, 100 + sd, scales[sd], struct)
                _, fa2 = response(sd, 1.0, gap, dev, 900 + sd, scales[sd], struct)
                rel_hist.append(float((fa - fb).norm() / (0.5 * (fa.norm() + fb.norm()) + 1e-12)))
                rel_null.append(float((fa - fa2).norm() / (0.5 * (fa.norm() + fa2.norm()) + 1e-12)))
            m_hist = sum(rel_hist) / len(rel_hist)
            m_null = sum(rel_null) / len(rel_null)
            ratio = m_hist / max(m_null, 1e-12)
            key = f"{'struct' if struct else 'state'}/gap={gap}"
            res["cells"][key] = {
                "mag_history": m_hist,
                "mag_null": m_null,
                "ratio": ratio,
                "per_seed_hist": rel_hist,
                "per_seed_null": rel_null,
            }
            n_win = sum(1 for a, b in zip(rel_hist, rel_null, strict=False) if a > b)
            print(
                f"  {key:16s} 历史 {m_hist * 100:5.2f}%  对照 {m_null * 100:5.2f}%  "
                f"比 {ratio:.2f}x  更大 {n_win}/{len(rel_hist)}",
                flush=True,
            )

    # M7-3：同历史两次运行必须逐位相同
    _, r1 = response(0, 1.0, 0, dev, 500, scales[0])
    _, r2 = response(0, 1.0, 0, dev, 500, scales[0])
    res["m7_3_bitwise_equal"] = bool(torch.equal(r1, r2))
    print(f"  M7-3 同历史两次逐位相同：{res['m7_3_bitwise_equal']}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
    )

    print("")
    print("判：")
    cc = res["cells"]

    for k in cc:
        c = cc[k]
        wins = sum(1 for a, b in zip(c["per_seed_hist"], c["per_seed_null"], strict=False) if a > b)
        ok = c["ratio"] > 1.5 and wins == len(c["per_seed_hist"])
        print(
            f"  {k:16s} 比 {c['ratio']:.2f}x  {wins}/{len(c['per_seed_hist'])} => {'过' if ok else '不过'}"
        )
    print(f"  M7-3 同历史逐位相同：{'过' if res['m7_3_bitwise_equal'] else '不过'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""折叠的皮层薄片：物理 3 维嵌入是否起作用。

预注册：`research/sel/prereg-cortical-fold-v1.md`

## 设计

`4096` 个神经元排在 `64 × 64` 的**环面薄片**上，折叠 `z = A·sin(2πkx/L)`。
每个神经元取 32 个前突触伙伴，概率 `∝ exp(−d/λ)`。

| 臂 | `d` | 折叠的影响 |
| :--- | :--- | :--- |
| `sheet` | 薄片（2D 环面）距离 | **无** —— 图与不折叠完全相同（内置零对照） |
| `space` | 3 维距离 `√(dx²+dy²+dz²)` | 有 —— 折叠改变了谁和谁连 |

测的是**线索保持曲线**（`memory_depth.run_md`）：循环任务的准确率已四次证明
对自变量不敏感。
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

from scripts.instrument import calibrate, mlp_probe
from scripts.memory_depth import K_CUE, auc_of, make_delay_task, run_md, tau_of

OUT = _pathlib.Path("sweep_results/cortical_fold")
V = 64
L = 64  # 薄片边长
N = L * L
A_FOLD = 16.0
K_FOLD = 1
LAM = 4.0
DEG = 32
R = 12  # 候选窗半径（薄片坐标）


def sheet_coords(dev: str):
    idx = torch.arange(N)
    x = (idx % L).float()
    y = (idx // L).float()
    z = A_FOLD * torch.sin(2.0 * math.pi * K_FOLD * x / L)
    return x.to(dev), y.to(dev), z.to(dev)


@torch.no_grad()
def build_fold(seed: int, dev: str, use_3d: bool):
    """按薄片距离或 3 维距离建接线。返回 `(w_in, pre, wrec)`。"""
    g = torch.Generator().manual_seed(seed)
    x, y, z = sheet_coords(dev)
    ax = torch.arange(-R, R + 1).float()
    ox, oy = torch.meshgrid(ax, ax, indexing="ij")
    ox, oy = ox.reshape(-1).to(dev), oy.reshape(-1).to(dev)  # (C,)
    C = ox.numel()

    cx = (x.unsqueeze(1) + ox.unsqueeze(0)) % L
    cy = (y.unsqueeze(1) + oy.unsqueeze(0)) % L
    zc = A_FOLD * torch.sin(2.0 * math.pi * K_FOLD * cx / L)

    d2 = (ox.unsqueeze(0) ** 2 + oy.unsqueeze(0) ** 2).expand(N, C)
    d = d2 + ((z.unsqueeze(1) - zc) ** 2 if use_3d else 0.0)
    w = torch.exp(-torch.sqrt(d) / LAM)
    # 排除自己（偏移 (0,0)）
    self_mask = (ox == 0) & (oy == 0)
    w[:, self_mask] = 0.0
    score = w * (1.0 + 0.5 * torch.rand(N, C, generator=g).to(dev))
    idx = score.topk(DEG, dim=1).indices
    lin = (cy * L + cx).long()
    pre = lin.gather(1, idx)

    w_in = (torch.randn(V, N, generator=g) / N**0.5).to(dev)
    wrec = torch.rand(N, DEG, generator=g).to(dev)
    wrec = wrec * (1.2 / wrec.sum(1, keepdim=True))
    return w_in, pre, wrec


def jaccard(pa: torch.Tensor, pb: torch.Tensor) -> float:
    rows = torch.arange(pa.shape[0], device=pa.device).unsqueeze(1)
    ka = (rows * N + pa).reshape(-1)
    kb = (rows * N + pb).reshape(-1)
    inter = float(torch.isin(ka, kb).sum())
    union = float(ka.numel() + kb.numel() - inter)
    return inter / max(union, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--train-seq", type=int, default=1536)
    ap.add_argument("--test-seq", type=int, default=384)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6")
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(x) for x in args.seeds.split(",")]
    res: dict = {"L": L, "N": N, "A": A_FOLD, "lam": LAM, "deg": DEG, "cells": {}}

    majors = []
    for sd in seeds:
        _, cue = make_delay_task(args.train_seq + args.test_seq, args.seq, 10 + sd, dev)
        trp, tep = cue[: args.train_seq], cue[args.train_seq :]
        top = int(torch.bincount(trp, minlength=K_CUE).argmax())
        majors.append(float((tep == top).float().mean()))
    floor = sum(majors) / len(majors)
    res["majority"] = floor
    print(f"  底板(多数类) {floor:.3f}", flush=True)

    graphs = {}
    for arm, use_3d in (("sheet", False), ("space", True)):
        curves, frs = [], []
        for sd in seeds:
            w_in, pre, wrec = build_fold(sd, dev, use_3d)
            graphs.setdefault(arm, []).append(pre.clone())
            cal_toks, cal_cue = make_delay_task(32, args.seq, 30 + sd, dev)

            def _rate(s, _w=w_in, _p=pre, _rw=wrec, _t=cal_toks, _c=cal_cue):
                return run_md(_w * s, _p, _rw, _t, _c, dev, 0.0, 0.0, 0.0)[1]

            ok_cal, sc, _pfr = calibrate(_rate)
            if not ok_cal:
                print(f"  {arm:5s} seed {sd} ✗ 标定失败", flush=True)
                continue
            tr, cue_tr = make_delay_task(args.train_seq, args.seq, 10 + sd, dev)
            te, cue_te = make_delay_task(args.test_seq, args.seq, 20 + sd, dev)
            toks = torch.cat([tr, te])
            cue = torch.cat([cue_tr, cue_te])
            feat, fr = run_md(w_in * sc, pre, wrec, toks, cue, dev, 0.0, 0.0, 0.0)
            curve = [
                mlp_probe(feat[:, i], cue, args.train_seq, K_CUE)[0] for i in range(feat.shape[1])
            ]
            curves.append(curve)
            frs.append(fr)
            del feat, toks, cue, w_in, pre, wrec
            torch.cuda.empty_cache()
        if not curves:
            continue
        n_s = len(curves[0])
        mean_curve = [sum(cv[i] for cv in curves) / len(curves) for i in range(n_s)]
        res["cells"][arm] = {
            "curve": mean_curve,
            "curves": curves,
            "auc": [auc_of(cv, floor) for cv in curves],
            "tau": [tau_of(cv, floor) for cv in curves],
            "peak": [max(cv) for cv in curves],
            "firing": sum(frs) / len(frs),
        }
        print(
            f"  {arm:5s} 发放 {sum(frs) / len(frs):.4f}  曲线 {[round(a, 3) for a in mean_curve]}",
            flush=True,
        )

    if graphs.get("sheet") and graphs.get("space"):
        js = [
            jaccard(graphs["sheet"][i], graphs["space"][i])
            for i in range(min(len(graphs["sheet"]), len(graphs["space"])))
        ]
        res["jaccard"] = js
        print(f"  Jaccard(sheet, space) = {[round(x, 3) for x in js]}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
    )

    print("")
    print("判：")
    cc = res["cells"]
    if "sheet" not in cc or "space" not in cc:
        print("  ✗ 有效臂不足")
        return 1
    thresh = floor + 0.10
    d0 = all(max(v["peak"]) >= thresh and 0.03 <= v["firing"] <= 0.20 for v in cc.values())
    print(f"  P-0 峰值 ≥ {thresh:.3f}、工作点 => {'过' if d0 else '不过 => 作废'}")
    if not d0:
        return 0
    jmax = max(res.get("jaccard", [1.0]))
    print(
        f"  P-1 折叠改了接线：Jaccard 最大 {jmax:.3f}（要 <0.8）=> {'过' if jmax < 0.8 else '不过'}"
    )

    def sign(diffs):
        n = len(diffs)
        pos = sum(1 for d in diffs if d > 0)
        return max(pos, n - pos), n

    for key, label in (("peak", "P-2 峰值"), ("auc", "P-3 AUC")):
        diffs = [b - a for a, b in zip(cc["sheet"][key], cc["space"][key], strict=False)]
        k, n = sign(diffs)
        rel = (sum(cc["space"][key]) / len(cc["space"][key])) / (
            sum(cc["sheet"][key]) / len(cc["sheet"][key])
        ) - 1.0
        print(f"  {label}：逐种子 space-sheet = {[round(d, 4) for d in diffs]}")
        print(f"        同号 {k}/{n} => {'过' if k == n else '不过'}；效应量 {rel * 100:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

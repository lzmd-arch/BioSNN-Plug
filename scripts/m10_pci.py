"""M10（Round G）：**扰动复杂度（PCI 的代理）** + 扰动传播广度。

判据在 `research/sel/night-2026-11-16.md`（含修订 5 —— 对照与归一化都定死了）。

扰动方式：在**输入停止的那一刻**给一个空间象限的膜电位加 `v_add`（`run_spont(v_add=...)`），
其余神经元不动。响应 `Δ = z_扰动 − z_基线`，同一历史、同一初值。
"""

from __future__ import annotations

import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.hist_families import make_family_hist
from scripts.m6_pools import N, build_pools, run_spont

OUT = _pathlib.Path("sweep_results/m10_pci")
K = 4
T_HIST = 1024
T_SPONT = 1024
REC = 256
SIDE, BLK = 32, 8


def lz(seq: torch.Tensor) -> float:
    """LZ76 复杂度，按长度归一。`seq` 是 0/1 的 1-D 张量。"""
    s = "".join("1" if v else "0" for v in seq.tolist())
    n = len(s)
    if n == 0:
        return 0.0
    i, c, pos = 0, 1, 1
    k, kmax = 1, 1
    while True:
        if s[i + k - 1] == s[pos + k - 1]:
            k += 1
            if pos + k > n:
                c += 1
                break
        else:
            kmax = max(kmax, k)
            i += 1
            if i == pos:
                c += 1
                pos += kmax
                if pos + 1 > n:
                    break
                i, k, kmax = 0, 1, 1
            else:
                k = 1
    return c / n


def self_check() -> bool:
    """已知复杂度的合成序列必须给出已知的**顺序**。

    2026-11-16 补：原版没有自检，而 PCI 恒为 0.000 —— 事后查明是归一化写错
    （`LZ76/n` 的上限约 0.07-0.15，永远到不了 1，`(c-c_rand)/(1-c_rand)` 必然≈0）。
    """
    n = 4000
    g = torch.Generator().manual_seed(0)
    cases = {
        "全零": torch.zeros(n),
        "周期3": (torch.arange(n) % 3 == 0).float(),
        "随机": (torch.rand(n, generator=g) < 0.5).float(),
    }
    vals = {k: lz(v) for k, v in cases.items()}
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in vals.items()), flush=True)
    return bool(vals["全零"] < vals["周期3"] < vals["随机"])


def to_bins(zt: torch.Tensor) -> torch.Tensor:
    """`(N, T)` → `(64, T)` 空间桶。"""
    _n, t = zt.shape
    step = SIDE // BLK
    return zt.view(SIDE, SIDE, t).view(BLK, step, BLK, step, t).mean(dim=(1, 3)).reshape(-1, t)


@torch.no_grad()
def arm(net, seed: int, fam: int, dev: str, v_add=None):
    toks = make_family_hist(fam, seed * 91 + fam * 23 + 6, T_HIST, 32, dev)
    _, snaps, _ = run_spont(
        seed,
        0.5,
        0,
        0.7,
        0.02,
        dev,
        K,
        0.0,
        0.0,
        -1.0,
        hist_in=toks,
        t_spont=T_SPONT,
        record_len=REC,
        net=net,
        v_add=v_add,
    )
    return snaps.mean(0)  # (N, REC)


def quadrant_mask(dev: str, q: int) -> torch.Tensor:
    idx = torch.arange(N, device=dev)  # **必须在 dev 上建**，否则 where 会设备不匹配
    x, y = idx % SIDE, idx // SIDE
    qx, qy = q % 2, q // 2
    inside = (x >= 16) == bool(qx)
    inside = inside & ((y >= 16) == bool(qy))
    return torch.where(inside, torch.full((N,), 2.0, device=dev), torch.zeros(N, device=dev))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    print("M10-0 自检（LZ 的已知顺序）：", flush=True)
    sc = self_check()
    print(f"  => {'过' if sc else '**不过 ⇒ PCI 全部作废**'}", flush=True)
    res: dict = {"self_check": sc, "cells": []}
    if not sc:
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )
        return 1
    for sd in range(args.seeds):
        net = build_pools(sd, dev, K, 0.0)
        pcis, spreads = [], []
        for q in range(4):
            base = arm(net, sd, 0, dev)
            pert = arm(net, sd, 0, dev, v_add=quadrant_mask(dev, q).unsqueeze(0))
            d = to_bins(pert - base).cpu()
            b = (d > 0).reshape(-1)
            c = lz(b)
            g = torch.Generator().manual_seed(sd * 13 + q)
            cr = lz(b[torch.randperm(b.numel(), generator=g)])
            # **比值，不是 (c-c_rand)/(1-c_rand)**：后者的 1 是错的上界
            pcis.append(c / max(cr, 1e-12))
            # 传播：受扰象限之外有响应的桶占比
            dq = (d.mean(1) != 0).float().view(BLK, BLK)
            out = dq.clone()
            out[16 // (SIDE // BLK) :, :] = 0
            out[:, 16 // (SIDE // BLK) :] = 0
            spreads.append(float(out.mean()))
        m = lambda v: sum(v) / len(v)  # noqa: E731
        row = {"seed": sd, "pci": m(pcis), "spread": m(spreads)}
        res["cells"].append(row)
        print(f"  seed {sd}: PCI {row['pci']:.3f}  传播 {row['spread']:.3f}", flush=True)
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )

    print("")
    print("判：")
    rows = res["cells"]
    p = sum(r["pci"] for r in rows) / len(rows)
    w = sum(1 for r in rows if r["pci"] > 1.2)
    print(
        f"  M10-1 PCI > 0.3：{p:.3f} => {w}/{len(rows)} {'过' if w >= max(5, 0.83 * len(rows)) else '不过'}"
    )
    s = sum(r["spread"] for r in rows) / len(rows)
    print(f"  M10-2 传播 ≥ 0.5：{s:.3f} => {'过' if s >= 0.5 else '不过'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

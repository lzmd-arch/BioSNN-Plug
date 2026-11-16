"""损伤自知：强度通道 vs 数量通道。

预注册：`research/sel/prereg-damage-awareness-v1.md`

预测：
- **强度通道**（每存活神经元平均发放）被内增益补偿拉回 ⇒ 看不见损伤；
- **数量通道**（长期静默神经元比例）不被掩盖 ⇒ 看得见。

**诚实标注**：50% 损伤下静默比例是"平凡可判"的（死单元恒静默）。
有信息量的是 10% / 25% 两档。
"""

from __future__ import annotations

import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.bio_all import make_task
from scripts.turnover import build_net, run_block

OUT = _pathlib.Path("sweep_results/damage_awareness")
N = 1024


@torch.no_grad()
def measure(seed: int, frac: float, comp: bool, dev: str):
    """返回 `(每存活神经元发放, 长期静默比例, 平均增益)`。"""
    toks, _ = make_task(32, 48, 8, 10 + seed, dev)
    w_in, pre, wrec = build_net(seed, dev)
    gen = torch.Generator().manual_seed(9000 + seed)
    alive = (torch.rand(N, generator=gen) >= frac).float().to(dev)
    gain = torch.ones(N, device=dev)
    if comp:
        # 内增益：逐神经元把发放拉向 f*
        for _ in range(4):
            act, _fr = run_block(w_in, pre, wrec, toks, dev, alive, gain)
            gain = torch.clamp(gain * torch.exp(2.0 * (0.08 - act)), 0.2, 5.0)
    act, fr = run_block(w_in, pre, wrec, toks, dev, alive, gain)
    silent = float((act < 1e-6).float().mean())
    del w_in, pre, wrec, toks
    torch.cuda.empty_cache()
    return fr, silent, float(gain.mean())


def auc(pos: list[float], neg: list[float]) -> float:
    """排序指标：正类样本大于负类样本的概率。"""
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 for p in pos for n in neg if p > n)
    ties = sum(0.5 for p in pos for n in neg if p == n)
    return (wins + ties) / (len(pos) * len(neg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--fracs", default="0.1,0.25,0.5")
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res: dict = {"n_seeds": args.seeds, "cells": {}}

    for frac in [float(x) for x in args.fracs.split(",")]:
        for comp in (False, True):
            fr_int, fr_les, si_int, si_les, gm = [], [], [], [], []
            for sd in range(args.seeds):
                a = measure(sd, 0.0, comp, dev)
                b = measure(sd, frac, comp, dev)
                fr_int.append(a[0])
                fr_les.append(b[0])
                si_int.append(a[1])
                si_les.append(b[1])
                gm.append(b[2])
            key = f"frac={frac:g}/comp={int(comp)}"
            res["cells"][key] = {
                "auc_firing": auc(fr_les, fr_int),
                "auc_silent": auc(si_les, si_int),
                "firing_intact": sum(fr_int) / len(fr_int),
                "firing_lesion": sum(fr_les) / len(fr_les),
                "silent_intact": sum(si_int) / len(si_int),
                "silent_lesion": sum(si_les) / len(si_les),
                "gain_mean": sum(gm) / len(gm),
            }
            c = res["cells"][key]
            print(
                f"  {key:20s} AUC 发放 {c['auc_firing']:.3f} 静默 {c['auc_silent']:.3f}  "
                f"发放 {c['firing_intact']:.4f}->{c['firing_lesion']:.4f}  "
                f"静默 {c['silent_intact']:.3f}->{c['silent_lesion']:.3f}",
                flush=True,
            )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
    )

    print("")
    print("判：")
    cc = res["cells"]
    a0 = cc["frac=0.5/comp=0"]["auc_firing"]
    print(f"  A-0 no_comp@50% 发放 AUC {a0:.3f}（要 >0.8）=> {'过' if a0 > 0.8 else '不过'}")
    a1_no = cc["frac=0.5/comp=0"]["auc_firing"]
    a1_yes = cc["frac=0.5/comp=1"]["auc_firing"]
    print(
        f"  A-1 补偿掩盖信号：{a1_yes:.3f} vs {a1_no:.3f}，降 {a1_no - a1_yes:.3f}"
        f"（要 ≥0.2）=> {'过' if a1_no - a1_yes >= 0.2 else '不过'}"
    )
    keys = list(cc)
    a2 = all(cc[k]["auc_silent"] > 0.8 for k in keys)
    print(
        f"  A-2 数量通道不被掩盖：静默 AUC {[round(cc[k]['auc_silent'], 3) for k in keys]} "
        f"=> {'过' if a2 else '不过'}"
    )
    print("  ⚠️ 50% 档的静默可判是**平凡的**（死单元恒静默），有信息量的是 10%/25%：")
    for k in keys:
        print(f"     {k:20s} 静默 AUC {cc[k]['auc_silent']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

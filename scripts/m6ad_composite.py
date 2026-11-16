"""M6ad：**复合历史 —— 造一个会依赖网络完整性的能力**。

## 为什么

今晚有一条反复出现的负结果：**任何去相关手段都让解码变好**
（切掉 50%、加噪声、换局部抑制）。诊断是**任务本身**：
解码"哪一族历史"是一个静态标签，读的是低阶统计量，去相关自然有益。

**要测脆性，能力必须依赖完整性。** 复合历史要求网络**在时间上整合并保持两段信息**：
前半段族 A、后半段族 B，从终态**同时**解码两者。

## 判据

- **AD-0**：简单任务（只看后半段族 B）可解码 > 底板 + 0.15（前提）；
- **AD-1 复合任务可解码** ★：同时解码 (A, B) 的准确率 > 2 族机会水平 + 0.15；
- **AD-2 它有脆性** ★★：随机切 50% 之后，复合任务准确率**低于**基线（≥2/3 种子）。
  —— 这一条如果成立，就是今晚第一次测到脆性；
- **AD-3 对照**：同一批数据上，简单任务的切后**不低于**基线（复现 M5b）。
  两条同时成立才说明"复合任务测的是完整性，简单任务不是"。
"""

from __future__ import annotations

# ruff: noqa: N806
import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.hist_families import spatial_bins
from scripts.instrument import majority_baseline, probe
from scripts.m6_pools import N, build_pools, run_spont

OUT = _pathlib.Path("sweep_results/m6ad_composite")
K = 4
T_HIST, T_SPONT = 1024, 2048
N_FAM = 4  # 4 族 -> 复合任务 4x4=16 类
PAIRS = [(a, b) for a in range(N_FAM) for b in range(N_FAM)]


def composite_hist(a: int, b: int, seed: int, per: int, dev: str) -> torch.Tensor:
    """前半段族 a、后半段族 b。两段各 `per//2` 条，拼起来。"""
    from scripts.hist_families import make_family_hist

    h1 = make_family_hist(a, seed * 1013 + a * 7 + 1, T_HIST // 2, per, dev)
    h2 = make_family_hist(b, seed * 1013 + b * 11 + 2, T_HIST // 2, per, dev)
    return torch.cat([h1, h2], dim=1)


@torch.no_grad()
def measure(net, alive, seed: int, per: int, dev: str):
    feats, y_comp, y_simple = [], [], []
    for idx, (a, b) in enumerate(PAIRS):
        toks = composite_hist(a, b, seed, per, dev)
        r = run_spont(
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
            net=net,
            alive=alive,
        )[0]
        feats.append(spatial_bins(r).cpu())
        y_comp.append(torch.full((r.shape[0],), idx))
        y_simple.append(torch.full((r.shape[0],), b))
    X = torch.cat(feats)
    Yc, Ys = torch.cat(y_comp), torch.cat(y_simple)
    perm = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(8 + seed))
    X, Yc, Ys = X[perm], Yc[perm], Ys[perm]
    cut = int(X.shape[0] * 0.75)
    acc_c, _ = probe(X, Yc, cut, len(PAIRS))
    acc_s, _ = probe(X, Ys, cut, N_FAM)
    fc = majority_baseline(Yc[:cut], Yc[cut:], len(PAIRS))
    fs = majority_baseline(Ys[:cut], Ys[cut:], N_FAM)
    return acc_c, fc, acc_s, fs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": []}
    for sd in range(args.seeds):
        net = build_pools(sd, dev, K, 0.0)
        ones = torch.ones(N, device=dev)
        c0, fc0, s0, fs0 = measure(net, ones, sd, args.per, dev)
        g = torch.Generator().manual_seed(sd + 131)
        mask = (torch.randperm(N, generator=g).to(dev) >= int(args.frac * N)).float()
        c1, _fc1, s1, _fs1 = measure(net, mask, sd, args.per, dev)
        row = {
            "seed": sd,
            "comp_base": c0,
            "comp_post": c1,
            "comp_floor": fc0,
            "simple_base": s0,
            "simple_post": s1,
            "simple_floor": fs0,
        }
        res["cells"].append(row)
        print(
            f"  seed {sd}: 复合 {c0:.3f} → 切后 {c1:.3f}（{c1 - c0:+.3f}，底板 {fc0:.3f}）| "
            f"简单 {s0:.3f} → {s1:.3f}（{s1 - s0:+.3f}）",
            flush=True,
        )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )

    print("")
    print("判：")
    rows = res["cells"]
    m = lambda k, _r=rows: sum(r[k] for r in _r) / len(_r)  # noqa: E731
    chance = 1.0 / len(PAIRS)
    ad0 = m("simple_base") > m("simple_floor") + 0.15
    print(
        f"  AD-0 简单任务可解码：{m('simple_base'):.3f} vs {m('simple_floor'):.3f} "
        f"=> {'过' if ad0 else '不过 => 作废'}"
    )
    if not ad0:
        return 0
    ad1 = m("comp_base") > chance + 0.15
    print(
        f"  AD-1 复合任务可解码（> 机会 {chance:.3f} +0.15）：{m('comp_base'):.3f} "
        f"=> {'过' if ad1 else '不过'}"
    )
    hurt = sum(1 for r in rows if r["comp_post"] < r["comp_base"])
    print(
        f"  AD-2 复合任务有脆性（切后更低，≥2/3）：{hurt}/{len(rows)} "
        f"（{m('comp_base'):.3f} → {m('comp_post'):.3f}）=> {'过' if hurt >= 2 else '不过'}"
    )
    hurt_s = sum(1 for r in rows if r["simple_post"] < r["simple_base"])
    print(
        f"  AD-3 简单任务切后不低于基线（≥2/3）：{len(rows) - hurt_s}/{len(rows)} "
        f"=> {'过' if len(rows) - hurt_s >= 2 else '不过'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

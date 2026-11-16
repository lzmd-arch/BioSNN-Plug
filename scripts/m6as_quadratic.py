"""M6as：**二次读出能不能救回高维状态** —— 量子提案的机制内核。

## 用户 2026-11-17 指出我那条"量子被等距性挡住"的论证有两个漏洞

**① 等距性只挡住状态距离，挡不住可观测量。**
酉演化保距（`‖Ua−Ub‖ = ‖a−b‖`），所以没有指数发散。但可观测量是 `|ψ|²` ——
**二次的、含交叉项**：`|a + b·e^{iδ}|² = |a|² + |b|² + 2|a||b|cos δ`。
**极小的相位差会在干涉项里直接显形。**
⇒ "状态距离不变"不等于"观测不到差别"。我原来的论证把两者混为一谈了。

**② 观察者效应在我们架构里不是外部干预，是系统自己的动作。**
闭环里测量者就是系统自己 ⇒ **坍缩是那个打破等距性的非线性** —— 正好补上"缺少放大"。

## 但先测机制内核，不直接搭量子模块

量子那条路的机制主张是：**二次的可观测量能读出线性读不出的东西。**
这条**不需要量子就能测**：把高维状态的读出从线性换成**二次（成对乘积）**。

## 判据

- **AS-0** 发放在窗内；
- **AS-1 二次读出救回** ★★：`gl=2.0/pl=0.5`（PR ~64、线性 0.37）上，
  二次读出 > 线性 + 0.15；
- **AS-2 对照**：同一个二次读出在**全局场**（低维）上不应显著更好（否则是特征数的功劳）。
"""

from __future__ import annotations

# ruff: noqa: N803, N806
import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.hist_families import make_family_hist, spatial_bins
from scripts.instrument import majority_baseline, mlp_probe, probe
from scripts.m6_pools import build_pools, run_spont

OUT = _pathlib.Path("sweep_results/m6as_quadratic")
K = 4
T_HIST, T_SPONT = 1024, 2048
FAMS = list(range(8))
P_LONG = 0.5

ARMS = {
    "high_dim(gl=2,pl=0.5)": {
        "field_radius": 2,
        "g_field": 0.0,
        "g_local": 2.0,
    },
    "global(gl=0)": {"field_radius": 32, "g_field": 1.0, "g_local": 0.0},
}


def quad_feat(X: torch.Tensor) -> torch.Tensor:
    """二次特征：所有成对乘积（含平方）。`d → d(d+1)/2`。"""
    Xc = X - X.mean(0, keepdim=True)
    Xc = Xc / (Xc.std(0, keepdim=True) + 1e-6)
    d = Xc.shape[1]
    iu = torch.triu_indices(d, d)
    return (Xc[:, iu[0]] * Xc[:, iu[1]]).cpu()


@torch.no_grad()
def collect(seed: int, arm: dict, per_fam: int, dev: str):
    w_in, pre, wrec, pid = build_pools(seed, dev, K, 0.0, p_long=P_LONG)
    net = (w_in, pre, wrec, pid)
    feats, labels, frs = [], [], []
    for fam in FAMS:
        toks = make_family_hist(fam, seed * 811 + fam * 43 + 19, T_HIST, per_fam, dev)
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
            **arm,
        )[0]
        feats.append(spatial_bins(r).cpu())
        labels.append(torch.full((r.shape[0],), fam))
        frs.append(float(r.mean()))
    return torch.cat(feats), torch.cat(labels), sum(frs) / len(frs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--per-fam", type=int, default=64)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": {}}
    for name, arm in ARMS.items():
        rows = []
        for sd in range(args.seeds):
            X, Y, fr = collect(sd, arm, args.per_fam, dev)
            perm = torch.randperm(Y.shape[0], generator=torch.Generator().manual_seed(13 + sd))
            X, Y = X[perm], Y[perm]
            cut = int(Y.shape[0] * 0.75)
            lin, _ = probe(X, Y, cut, len(FAMS))
            quad, _ = probe(quad_feat(X), Y, cut, len(FAMS))
            mlp = mlp_probe(X, Y, cut, len(FAMS))[0]
            floor = majority_baseline(Y[:cut], Y[cut:], len(FAMS))
            rows.append(
                {"seed": sd, "linear": lin, "quad": quad, "mlp": mlp, "floor": floor, "firing": fr}
            )
            print(
                f"  {name:22s} seed {sd}: 线性 {lin:.3f}  二次 {quad:.3f}  MLP {mlp:.3f}  "
                f"底板 {floor:.3f}  发放 {fr:.4f}",
                flush=True,
            )
        m = lambda k, _r=rows: sum(r[k] for r in _r) / len(_r)  # noqa: E731
        res["cells"][name] = {
            "linear": m("linear"),
            "quad": m("quad"),
            "mlp": m("mlp"),
            "floor": m("floor"),
            "firing": m("firing"),
            "rows": rows,
        }
        v = res["cells"][name]
        print(
            f"  {name:22s} 均值：线性 {v['linear']:.3f}  二次 {v['quad']:.3f}  "
            f"MLP {v['mlp']:.3f}  底板 {v['floor']:.3f}",
            flush=True,
        )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1, default=float),
            encoding="utf-8",
            newline=chr(10),
        )

    print("")
    print("判：")
    cc = res["cells"]
    h = cc["high_dim(gl=2,pl=0.5)"]
    ok0 = 0.01 <= h["firing"] <= 0.20
    print(f"  AS-0 高维档发放在窗内：{h['firing']:.4f} => {'过' if ok0 else '不过 => 作废'}")
    gain = h["quad"] - h["linear"]
    print(
        f"  AS-1 二次读出救回（> 线性 +0.15）：线性 {h['linear']:.3f} → 二次 {h['quad']:.3f} "
        f"（{gain:+.3f}）=> {'过' if gain > 0.15 else '不过'}"
    )
    g = cc["global(gl=0)"]
    ggain = g["quad"] - g["linear"]
    print(f"  AS-2 对照（全局场上的二次增益应更小）：{ggain:+.3f} （高维档是 {gain:+.3f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

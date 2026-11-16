"""M6ae：**信息在时间上衰不衰减 —— 审计查出的最关键缺口**。

## 为什么必须补

M6t 只测了一个时间点：输入停止后第 1024 步（窗口 960–1023），解码 0.536。
**没有测更早的时间点。** 如果 t=64 就已经 0.53，那"它记住了历史"这个说法
缺少时间维度的证据 —— 只能说"终态可解码"。

**正确的问法**：解码准确率随"输入停止后的步数"怎么变？

- **平**（t=64 与 t=2048 差不多）⇒ 信息是**状态**携带的，不依赖残留输入 ⇒ 记忆成立；
- **单调下降** ⇒ 信息在衰减，1024 步的数字只是残留 ⇒ "记忆"的说法要收回。

## 判据

- **AE-0** 发放在窗内；
- **AE-1 不是残留** ★：`t=64` 与 `t=2048` 的解码都 > 底板 + 0.15；
- **AE-2 没有显著衰减** ★：`t=2048` 的解码 ≥ `t=64` 的 **0.8 倍**。
"""

from __future__ import annotations

# ruff: noqa: N806
import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.hist_families import make_family_hist, spatial_bins
from scripts.instrument import majority_baseline, probe
from scripts.m6_pools import build_pools, run_spont

OUT = _pathlib.Path("sweep_results/m6ae_time_ladder")
K = 4
T_HIST = 1024
FAMS = list(range(8))


@torch.no_grad()
def decode_at(net, seed: int, t_spont: int, per_fam: int, dev: str, g_inh: float):
    feats, labels, frs = [], [], []
    for fam in FAMS:
        toks = make_family_hist(fam, seed * 137 + fam * 19 + 6, T_HIST, per_fam, dev)
        r = run_spont(
            seed,
            0.5,
            0,
            0.7,
            0.02,
            dev,
            K,
            0.0,
            g_inh,
            -1.0,
            hist_in=toks,
            t_spont=t_spont,
            net=net,
        )[0]
        feats.append(spatial_bins(r).cpu())
        labels.append(torch.full((r.shape[0],), fam))
        frs.append(float(r.mean()))
    X, Y = torch.cat(feats), torch.cat(labels)
    perm = torch.randperm(Y.shape[0], generator=torch.Generator().manual_seed(4 + seed))
    X, Y = X[perm], Y[perm]
    cut = int(Y.shape[0] * 0.75)
    acc, _ = probe(X, Y, cut, len(FAMS))
    return acc, majority_baseline(Y[:cut], Y[cut:], len(FAMS)), sum(frs) / len(frs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--ts", default="64,256,1024,2048")
    ap.add_argument("--g-inh", type=float, default=0.0)
    ap.add_argument("--per-fam", type=int, default=16)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": {}, "g_inh": args.g_inh}
    for t in [int(x) for x in args.ts.split(",")]:
        rows = []
        for sd in range(args.seeds):
            net = build_pools(sd, dev, K, 0.0)
            acc, floor, fr = decode_at(net, sd, t, args.per_fam, dev, args.g_inh)
            rows.append({"seed": sd, "acc": acc, "floor": floor, "firing": fr})
        m = lambda k, _r=rows: sum(r[k] for r in _r) / len(_r)  # noqa: E731
        res["cells"][f"t={t}"] = {
            "t": t,
            "acc": m("acc"),
            "floor": m("floor"),
            "firing": m("firing"),
            "rows": rows,
        }
        v = res["cells"][f"t={t}"]
        print(
            f"  t={t:5d}: 解码 {v['acc']:.3f}（底板 {v['floor']:.3f}）发放 {v['firing']:.4f}",
            flush=True,
        )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )

    print("")
    print("判：")
    cc = res["cells"]
    ts = sorted(cc, key=lambda k: cc[k]["t"])
    early, late = cc[ts[0]], cc[ts[-1]]
    ok0 = all(0.01 <= cc[k]["firing"] <= 0.20 for k in ts)
    print(f"  AE-0 发放在窗内：{'过' if ok0 else '不过 => 作废'}")
    if not ok0:
        return 0
    ok1 = early["acc"] > early["floor"] + 0.15 and late["acc"] > late["floor"] + 0.15
    print(
        f"  AE-1 首末都可解码：t={early['t']} {early['acc']:.3f}、"
        f"t={late['t']} {late['acc']:.3f} => {'过' if ok1 else '不过'}"
    )
    ratio = late["acc"] / max(early["acc"], 1e-9)
    print(
        f"  AE-2 没有显著衰减（末 ≥ 首的 0.8）：{ratio:.2f} => {'过' if ratio >= 0.8 else '不过'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

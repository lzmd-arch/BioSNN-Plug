"""M6ab：**"高维但读不出"是特征分辨率的锅，还是状态真的不带历史？**

M6x 第一格：`field_radius=2` 给出 PR 59.6、持久 0.049，但解码只有 0.073（底板 0.062）。
两种解释：

1. **特征分辨率**：读出用的是 8×8 空间桶（64 维），而状态有 ~60 维 —— 桶把信号抹平了；
2. **真的没信息**：高维状态是噪声，历史不在里面。

用**全分辨率**（逐神经元发放，1024 维）重测可以分开这两条。

判据：
- **AB-1**：`radius=2` 下，全分辨率解码 > 底板 + 0.15 ⇒ 是特征的锅；
- 若也不过 ⇒ **高维状态确实不携带可线性读出的历史**。
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

OUT = _pathlib.Path("sweep_results/m6ab_decode_resolution")
K = 4
T_HIST, T_SPONT = 1024, 2048
FAMS = list(range(8))


def field_args(radius: int) -> dict:
    return {"g_field": 1.0, "g_local": 0.0} if radius >= 32 else {"g_field": 0.0, "g_local": 1.0}


@torch.no_grad()
def collect(seed: int, radius: int, per_fam: int, dev: str):
    net = build_pools(seed, dev, K, 0.0)
    full, binned, labels = [], [], []
    for fam in FAMS:
        toks = make_family_hist(fam, seed * 271 + fam * 13 + 4, T_HIST, per_fam, dev)
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
            field_radius=radius,
            **field_args(radius),
        )[0]
        full.append(r.cpu())
        binned.append(spatial_bins(r).cpu())
        labels.append(torch.full((r.shape[0],), fam))
    return torch.cat(full), torch.cat(binned), torch.cat(labels)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--radii", default="2,32")
    ap.add_argument("--per-fam", type=int, default=16)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": {}}
    for rad in [int(x) for x in args.radii.split(",")]:
        rows = []
        for sd in range(args.seeds):
            Xf, Xb, Y = collect(sd, rad, args.per_fam, dev)
            perm = torch.randperm(Y.shape[0], generator=torch.Generator().manual_seed(6 + sd))
            cut = int(Y.shape[0] * 0.75)
            acc_f, _ = probe(Xf[perm], Y[perm], cut, len(FAMS))
            acc_b, _ = probe(Xb[perm], Y[perm], cut, len(FAMS))
            floor = majority_baseline(Y[perm][:cut], Y[perm][cut:], len(FAMS))
            rows.append({"seed": sd, "full": acc_f, "bins": acc_b, "floor": floor})
            print(
                f"  radius={rad:2d} seed {sd}: 全分辨率 {acc_f:.3f}  64 桶 {acc_b:.3f}  "
                f"底板 {floor:.3f}",
                flush=True,
            )
        m = lambda k, _r=rows: sum(r[k] for r in _r) / len(_r)  # noqa: E731
        res["cells"][f"r={rad}"] = {
            "radius": rad,
            "full": m("full"),
            "bins": m("bins"),
            "floor": m("floor"),
            "rows": rows,
        }
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )

    print("")
    print("判：")
    cc = res["cells"]
    if "r=2" in cc:
        v = cc["r=2"]
        ok = v["full"] > v["floor"] + 0.15
        print(
            f"  AB-1 局部场下全分辨率可解码（> 底板+0.15）：{v['full']:.3f} vs {v['floor']:.3f} "
            f"=> {'过（是特征的锅）' if ok else '不过（高维状态确实不携带可线性读出的历史）'}"
        )
        print(f"       对照：64 桶给出 {v['bins']:.3f}")
    if "r=32" in cc:
        v = cc["r=32"]
        print(f"  全局场：全分辨率 {v['full']:.3f}、64 桶 {v['bins']:.3f}、底板 {v['floor']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

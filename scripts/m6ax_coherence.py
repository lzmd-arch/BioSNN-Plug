"""M6ax：**相干性的间接测量** —— 用互信息（纠缠熵的代理）看它衰不衰减、能不能恢复。

## 用户 2026-11-17 的提议

> 她的载体是相干性，而"维持它"就是代谢 —— 那条我没有能力测。
> 我有一个间接测法：用纠缠熵进行读出。

## 为什么成立

对纯的联合态，`I(A;B) = 2·S(A)` —— **互信息就是纠缠熵**。
`m6av` 已测：全局 MI **1.135**、高维 MI **0.067**（17 倍）。

## 而它真正能回答的是**代谢问题**

"维持相干性"是一个**有时间维度**的量：

- 若损伤后 **S 能自己回来** ⇒ 她在主动维持 ⇒ **代谢成立**；
- 若 **S 只单调下掉、并在死亡阈值处崩塌** ⇒ 相干性耗散、维持不了 ⇒ 那条路关闭，但有证据。

## 判据

- **AX-0** 发放在窗内（至少无损档）；
- **AX-1 相干性随损伤单调下降** ★：`S(lesion)` 单调降；
- **AX-2 相干性的崩塌点与死亡阈值重合** ★★：`S` 掉到无损值 50% 以下的损伤比例
  与发放出窗的比例相差 ≤ 一档；
- **AX-3 能否恢复** ★：短窗（256）与长窗（2048）的 `S` 差 ≥ 10% ⇒ 有恢复趋势；
  否则 ⇒ 只耗散不恢复。
"""

from __future__ import annotations

# ruff: noqa: N806
import argparse
import itertools
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.hist_families import make_family_hist
from scripts.m6_pools import N, build_pools, run_spont

OUT = _pathlib.Path("sweep_results/m6ax_coherence")
K, SIDE = 4, 32
T_HIST = 1024
FAMS = list(range(8))


def mi_binned(a: torch.Tensor, b: torch.Tensor, nb: int = 8) -> float:
    qa = torch.quantile(a.float(), torch.linspace(0, 1, nb + 1)[1:-1])
    qb = torch.quantile(b.float(), torch.linspace(0, 1, nb + 1)[1:-1])
    ia = torch.bucketize(a.float().contiguous(), qa)
    ib = torch.bucketize(b.float().contiguous(), qb)
    n = a.shape[0]
    joint = torch.zeros(nb, nb)
    for x, y in zip(ia.tolist(), ib.tolist(), strict=False):
        joint[x, y] += 1
    joint = joint / n
    pa, pb = joint.sum(1, keepdim=True), joint.sum(0, keepdim=True)
    outer = pa @ pb
    mask = joint > 0
    return float((joint[mask] * (joint[mask] / outer[mask]).log()).sum())


@torch.no_grad()
def measure(seed: int, frac: float, t_spont: int, per_fam: int, dev: str):
    net = build_pools(seed, dev, K, 0.0)
    g = torch.Generator().manual_seed(seed + 901)
    keep = round((1.0 - frac) * N)
    mask = torch.zeros(N, device=dev)
    mask[torch.randperm(N, generator=g).to(dev)[:keep]] = 1.0
    states, frs = [], []
    for fam in FAMS:
        toks = make_family_hist(fam, seed * 661 + fam * 61 + 37, T_HIST, per_fam, dev)
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
            t_spont=t_spont,
            net=net,
            alive=mask,
        )[0]
        states.append(r.cpu())
        frs.append(float(r.mean()))
    S = torch.cat(states)
    h = ((torch.arange(N) % SIDE) >= 16).long()
    alive_mask = mask.cpu() > 0.5
    A = S[:, (h.cpu() == 0) & alive_mask]
    B = S[:, (h.cpu() == 1) & alive_mask]
    if A.shape[1] == 0 or B.shape[1] == 0:
        return None
    mi = mi_binned(A.mean(1), B.mean(1))
    return {"mi": mi, "firing": sum(frs) / len(frs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--fracs", default="0,0.1,0.25,0.5,0.65,0.75")
    ap.add_argument("--ts", default="256,2048")
    ap.add_argument("--per-fam", type=int, default=32)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": {}}
    for t in [int(x) for x in args.ts.split(",")]:
        for fr in [float(x) for x in args.fracs.split(",")]:
            rows = [
                m
                for sd in range(args.seeds)
                if (m := measure(sd, fr, t, args.per_fam, dev)) is not None
            ]
            if not rows:
                continue
            mean = lambda k, _r=rows: sum(x[k] for x in _r) / len(_r)  # noqa: E731
            key = f"t={t}/lesion={fr:g}"
            res["cells"][key] = {
                "t": t,
                "frac": fr,
                "mi": mean("mi"),
                "firing": mean("firing"),
                "rows": rows,
            }
            print(f"  {key:22s} S≈MI {mean('mi'):.4f}  发放 {mean('firing'):.4f}", flush=True)
            (args.out / "result.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=1, default=float),
                encoding="utf-8",
                newline=chr(10),
            )

    print("")
    print("判：")
    cc = res["cells"]
    ts = sorted({v["t"] for v in cc.values()})
    frs = sorted({v["frac"] for v in cc.values()})
    t0 = ts[-1]
    base = cc.get(f"t={t0}/lesion=0")
    if base is None:
        print("  AX-0 缺无损档 => 作废")
        return 0
    ok0 = 0.01 <= base["firing"] <= 0.20
    print(f"  AX-0 无损档发放在窗内：{base['firing']:.4f} => {'过' if ok0 else '不过 => 作废'}")
    seq = [cc[f"t={t0}/lesion={fr:g}"]["mi"] for fr in frs if f"t={t0}/lesion={fr:g}" in cc]
    mono = all(a >= b - 1e-9 for a, b in itertools.pairwise(seq))
    print(
        f"  AX-1 相干性随损伤单调下降：{[round(x, 4) for x in seq]} => {'过' if mono else '不过'}"
    )
    half = [fr for fr in frs if cc[f"t={t0}/lesion={fr:g}"]["mi"] < 0.5 * base["mi"]]
    print(f"  AX-2 相干性崩塌点：{half[0]:.0%}" if half else "  AX-2 未崩塌")
    if len(ts) >= 2:
        pairs = []
        for fr in frs:
            a, b = cc.get(f"t={ts[0]}/lesion={fr:g}"), cc.get(f"t={ts[-1]}/lesion={fr:g}")
            if a and b and b["mi"] > 0:
                pairs.append(b["mi"] / a["mi"] - 1.0)
        print(
            f"  AX-3 长窗相对短窗的变化：{[round(x, 3) for x in pairs]} "
            f"=> {'有恢复趋势' if any(x > 0.10 for x in pairs) else '只耗散不恢复'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

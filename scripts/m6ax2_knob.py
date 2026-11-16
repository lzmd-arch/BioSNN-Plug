"""追试 ①：**拐点跟不跟「相干供给旋钮」走**（预注册 `research/sel/prereg-coherence-collapse-knob-v1.md`）

## 要回答的

`m6ax` 测到半球互信息的崩塌点在 65%，与死亡阈值重合 —— 但**重合只是相关**。
要升级成机制，必须证明**崩塌点能被一个旋钮推动**。

旋钮 = `g_field`（全局抑制场强度）—— 全局场正是制造共同模态的那一项
（`m6av`：全局 MI 1.135 vs 高维 0.067）。

**`g_field` 越大 ⇒ 共同模态越强 ⇒ 有更多"供给"维持关联 ⇒ 预测崩塌点向更高损伤移动。**

## 判据（照预注册，一个字不改）

- **K-0** 各档无损时 MI>0.3 且发放在窗 ⇒ 否则作废；
- **K-1 崩塌点跟旋钮走** ★★：至少 **3 个不同**的崩塌点；
- **K-2 崩塌点与死亡阈值同步** ★：每档内两者相差 ≤ 一档；
- **K-3 若不跟** ⇒ 退回双机制并存，不强行解释。

**纪律：≥6 种子。**
"""

from __future__ import annotations

# ruff: noqa: N806
import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.hist_families import make_family_hist
from scripts.m6_pools import N, build_pools, run_spont

OUT = _pathlib.Path("sweep_results/m6ax2_knob")
K, SIDE = 4, 32
T_HIST, T_SPONT = 1024, 2048


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
def measure(seed: int, g_field: float, frac: float, per_fam: int, dev: str):
    net = build_pools(seed, dev, K, 0.0)
    g = torch.Generator().manual_seed(seed + 901)
    keep = round((1.0 - frac) * N)
    alive = torch.zeros(N, device=dev)
    alive[torch.randperm(N, generator=g).to(dev)[:keep]] = 1.0
    states, frs = [], []
    for fam in range(8):
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
            t_spont=T_SPONT,
            net=net,
            alive=alive,
            g_field=g_field,
        )[0]
        states.append(r.cpu())
        frs.append(float(r.mean()))
    S = torch.cat(states)
    h = (torch.arange(N) % SIDE >= 16).long()
    am = alive.cpu() > 0.5
    A, B = S[:, (h == 0) & am], S[:, (h == 1) & am]
    if A.shape[1] < 2 or B.shape[1] < 2:
        return None
    return {"mi": mi_binned(A.mean(1), B.mean(1)), "firing": sum(frs) / len(frs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--gfs", default="0.25,0.5,1.0,2.0")
    ap.add_argument("--fracs", default="0,0.1,0.25,0.5,0.6,0.65,0.7,0.75")
    ap.add_argument("--per-fam", type=int, default=32)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": {}}
    for gf in [float(x) for x in args.gfs.split(",")]:
        curve = []
        for fr in [float(x) for x in args.fracs.split(",")]:
            rows = [
                m
                for sd in range(args.seeds)
                if (m := measure(sd, gf, fr, args.per_fam, dev)) is not None
            ]
            if not rows:
                continue
            mean = lambda k, _r=rows: sum(x[k] for x in _r) / len(_r)  # noqa: E731
            key = f"gf={gf:g}/lesion={fr:g}"
            res["cells"][key] = {
                "g_field": gf,
                "frac": fr,
                "mi": mean("mi"),
                "firing": mean("firing"),
            }
            curve.append((fr, mean("mi"), mean("firing")))
            print(f"  {key:26s} MI {mean('mi'):.4f}  发放 {mean('firing'):.4f}", flush=True)
            (args.out / "result.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=1, default=float),
                encoding="utf-8",
                newline=chr(10),
            )
        res.setdefault("curves", {})[f"{gf:g}"] = curve

    print("")
    print("判：")
    curves = res.get("curves", {})
    ok0 = all(c and c[0][2] >= 0.01 and c[0][1] > 0.3 for c in curves.values() if c)
    print(f"  K-0 各档无损有效（MI>0.3、发放≥0.01）：{'过' if ok0 else '不过 => 作废'}")
    if not ok0:
        return 0
    collapses = {}
    for gf, c in curves.items():
        base = c[0][1]
        hit = next((fr for fr, mi, _ in c if mi < 0.5 * base), None)
        collapses[gf] = hit
        print(f"    g_field={gf}: 崩塌点 {hit if hit is not None else '未崩塌'}")
    distinct = len({v for v in collapses.values() if v is not None})
    print(
        f"  K-1 崩塌点跟旋钮走（≥3 个不同值）：{distinct} 个 => "
        f"{'过（升级为机制）' if distinct >= 3 else '不过'}"
    )
    deaths = {}
    for gf, c in curves.items():
        deaths[gf] = next((fr for fr, _, f in c if not (0.01 <= f <= 0.20)), None)
    close = sum(
        1
        for gf in collapses
        if collapses[gf] is not None
        and deaths.get(gf) is not None
        and abs(collapses[gf] - deaths[gf]) <= 0.1
    )
    print(
        f"  K-2 崩塌点与死亡阈值同步（≤一档）：{close}/{len(collapses)} 档 => "
        f"{'过' if close == len(collapses) else '不过'}"
    )
    if distinct < 3:
        print("  K-3 崩塌点不跟旋钮走 ⇒ **退回双机制并存**，不强行解释。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

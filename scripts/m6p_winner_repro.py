"""M6p-P3 修正：**胜者在"同统计"下可复现吗**（而不是"不同历史下是否不同"）。

## 原判据为什么错

M6p 的 P-3 写的是"不同历史下胜者不恒定"。**那条被噪声平凡满足**：
胜者在关键臂上逐样本一致率只有 0.20–0.48，它本来就在随噪声跳。
"随历史变"与"随噪声变"不可区分。

## 正确的问法

- **对照**：同一条历史统计、两条不同随机流（A vs A'）⇒ 胜者应该**一样**；
- **处理**：不同历史统计（A vs B）⇒ 胜者应该**不同**。

判据：**同统计一致率 > 不同统计一致率**（≥5/6 种子）。这才排除"胜者是噪声"。
"""

from __future__ import annotations

import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.m6_pools import build_pools, pool_id, pool_rates, run_spont

OUT = _pathlib.Path("sweep_results/m6p_winner_repro")
K = 4
T_HIST, T_SPONT = 1024, 2048


@torch.no_grad()
def winner(net, seed: int, alpha: float, hist_seed: int, g_inh: float, dev: str) -> int:
    from scripts.bio_all import make_task

    toks, _ = make_task(32, T_HIST, 8, hist_seed, dev, alpha)
    r = run_spont(
        seed,
        alpha,
        hist_seed,
        0.7,
        0.02,
        dev,
        K,
        0.0,
        g_inh,
        -1.0,
        hist_in=toks,
        t_spont=T_SPONT,
        net=net,
    )[0]
    pr = pool_rates(r.mean(0, keepdim=True), pool_id(dev, K), K)[0]
    return int(pr.argmax())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--ginhs", default="1,2")
    ap.add_argument("--n-rep", type=int, default=4)  # 每条条件重复几条随机流
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": {}}
    for g_inh in [float(x) for x in args.ginhs.split(",")]:
        rows = []
        for sd in range(args.seeds):
            net = build_pools(sd, dev, K, 0.0)
            wa = [winner(net, sd, 0.5, 100 + sd * 10 + i, g_inh, dev) for i in range(args.n_rep)]
            wb = [winner(net, sd, 20.0, 100 + sd * 10 + i, g_inh, dev) for i in range(args.n_rep)]
            same = sum(
                1 for i in range(args.n_rep) for j in range(args.n_rep) if i != j and wa[i] == wa[j]
            ) / max(args.n_rep * (args.n_rep - 1), 1)
            diff = sum(1 for i in range(args.n_rep) for j in range(args.n_rep) if wa[i] == wb[j])
            diff = diff / max(args.n_rep**2, 1)
            rows.append({"seed": sd, "same": same, "diff": diff, "wa": wa, "wb": wb})
            print(
                f"  g={g_inh:g} seed {sd}: A 的胜者 {wa} | B 的胜者 {wb} | "
                f"同统计一致 {same:.2f}  不同统计一致 {diff:.2f}",
                flush=True,
            )
        m = lambda k, _r=rows: sum(r[k] for r in _r) / len(_r)  # noqa: E731
        wins = sum(1 for r in rows if r["same"] > r["diff"])
        res["cells"][f"g={g_inh:g}"] = {
            "same": m("same"),
            "diff": m("diff"),
            "wins": wins,
            "n": args.seeds,
            "rows": rows,
        }
        print(
            f"  g={g_inh:g} 汇总：同统计 {m('same'):.3f} vs 不同统计 {m('diff'):.3f} "
            f"=> {wins}/{args.seeds}",
            flush=True,
        )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )

    print("")
    print("判：")
    cc = res["cells"]
    best = max(cc.values(), key=lambda v: v["wins"])
    ok = best["wins"] >= max(5, 0.83 * best["n"])
    print(
        f"  P3' 同统计一致率 > 不同统计（≥5/6）：最好 "
        f"{best['same']:.3f} vs {best['diff']:.3f}，{best['wins']}/{best['n']} "
        f"=> {'过' if ok else '不过'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

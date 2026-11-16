"""复杂度阶梯：把「机制多一个值多少」画成曲线，找膝点，再逐个拆掉。

预注册：`research/sel/prereg-emergence-ladder-v1.md`

## 为什么

`bio_all` 证明了「全开 > 任何一个单项」（3/3 种子），但按
`prereg-emergence-v1` 的 Q-1…Q-4 还叫不了涌现：**没有膝点（Q-2）**、
**没有逐个拆卸（Q-3）**。

## 两段

1. **难度标定**：扫 `alpha`（Dirichlet 浓度），选一个使**窗口 8-token 天花板**
   落进 [0.45, 0.55] 的。现在的任务全开已经 0.359、天花板 0.356 —— 到顶了，
   在到顶的任务上画阶梯，膝点会是天花板造成的。
2. **阶梯 + 拆卸**：L0…L5 嵌套链，再从 L5 逐个拿掉一个机制。

## 纪律

- 判据在 prereg 里，跑之前落盘；
- 每档单独标定工作点，真实发放率出窗就作废；
- 探针播种（`mlp_probe` 取 3 个种子均值并报每次的值）；
- 底板用**实测**多数类，不用 `1/K`。
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

from scripts.bio_all import build, make_task, run
from scripts.instrument import calibrate, mlp_probe, probe
from scripts.structure_ladder import window_features

OUT = _pathlib.Path("sweep_results/bio_ladder")
V, K = 64, 8

#: 嵌套链，顺序由 bio_all v2 的实测单项增益排定（见 prereg 第一节），跑之前定死
CHAIN = ["inhib", "ei", "alif", "delay", "tau"]
RUNGS = [("L0", [])] + [(f"L{k + 1}", CHAIN[: k + 1]) for k in range(len(CHAIN))]
#: 天花板的目标区间
TARGET_CEIL = (0.40, 0.60)  # amend1：原为 (0.45, 0.55)，两个相邻网格点正好跨在带子两边


def ceiling_at(alpha: float, n_train: int, n_test: int, seq: int, seed: int, dev: str) -> float:
    """窗口 8-token 天花板（不经过网络）。`cut` 用**训练序列数 x 步数**，不是总序列数。"""
    tr, trs = make_task(n_train + n_test, seq, K, 10 + seed, dev, alpha)
    st = trs[:, ::8]
    wf = window_features(tr, L=8, stride=8)
    acc, _ = mlp_probe(wf, st, n_train * st.shape[1], K)
    return acc


def majority_at(alpha: float, n_train: int, n_test: int, seq: int, seed: int, dev: str) -> float:
    _, st = make_task(n_train + n_test, seq, K, 10 + seed, dev, alpha)
    st = st[:, ::8]
    trp, tep = st[:n_train], st[n_train:]
    top = int(torch.bincount(trp.reshape(-1), minlength=K).argmax())
    return float((tep == top).float().mean())


def run_arm(name: str, switches: list[str], alpha: float, args, dev: str):
    """一档：标定 → 3 个种子 → 返回 (accs, firing, scale) 或 None。"""
    sw = set(switches)
    tr0, _ = make_task(32, args.seq, K, 10, dev, alpha)
    w_in, pre, wrec, a_t, delay, pool_id = build(args.n, sw, 0, dev, args.deg)
    rest = (pre, wrec, a_t, delay, pool_id, sw)
    ok_cal, sc, _pfr = calibrate(lambda s, _w=w_in, _r=rest, _t=tr0: run((_w * s, *_r), _t, dev)[1])
    if not ok_cal:
        print(f"  {name:6s} ✗ 标定失败，该档无效", flush=True)
        return None
    accs, frs, lins = [], [], []
    for sd in args.seed_list:
        tr, trs = make_task(args.train_seq, args.seq, K, 10 + sd, dev, alpha)
        te, tes = make_task(args.test_seq, args.seq, K, 20 + sd, dev, alpha)
        toks = torch.cat([tr, te])
        st = torch.cat([trs, tes])[:, ::8]
        w2 = build(args.n, sw, sd, dev, args.deg)
        feat, fr = run((w2[0] * sc, *w2[1:], sw), toks, dev)
        cut = args.train_seq * st.shape[1]
        m, _ = mlp_probe(feat, st, cut, K)
        # **第二个指标**：线性探针（λ 由训练集内部留出选）。L-2 要"膝点在两个
        # 都不同的指标上落在同一处" —— 归一化增益只是准确率的单调变换，不算。
        lin, _lam = probe(feat.reshape(-1, feat.shape[-1]), st.reshape(-1), cut, K)
        accs.append(m)
        lins.append(lin)
        frs.append(fr)
        del feat, w2, toks, st
        torch.cuda.empty_cache()
    fr_mean = sum(frs) / len(frs)
    if not (0.03 <= fr_mean <= 0.20):
        print(f"  {name:6s} 发放 {fr_mean:.4f} 出窗，该档无效", flush=True)
        return None
    return accs, lins, fr_mean, sc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--deg", type=int, default=32)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--train-seq", type=int, default=384)
    ap.add_argument("--test-seq", type=int, default=96)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--alphas", default="3,6,12,24,48")
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.seed_list = [int(x) for x in args.seeds.split(",")]
    res: dict = {"n": args.n, "deg": args.deg, "chain": CHAIN, "cells": {}}

    # ---- 1. 难度标定 ----
    print("难度标定（窗口 8-token 天花板）：", flush=True)
    cal = []
    for a in [float(x) for x in args.alphas.split(",")]:
        c = sum(
            ceiling_at(a, args.train_seq, args.test_seq, args.seq, sd, dev) for sd in args.seed_list
        ) / len(args.seed_list)
        cal.append({"alpha": a, "ceiling": c})
        print(f"  alpha {a:5.1f}  天花板 {c:.3f}", flush=True)
        torch.cuda.empty_cache()
    res["calibration"] = cal
    ok = [c for c in cal if TARGET_CEIL[0] <= c["ceiling"] <= TARGET_CEIL[1]]
    if not ok:
        print(f"\n✗ 没有任何 alpha 能把天花板放进 {TARGET_CEIL} —— **中止，不凑合**")
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
        )
        return 1
    chosen = min(ok, key=lambda c: abs(c["ceiling"] - 0.5))
    alpha = chosen["alpha"]
    res["alpha"] = alpha
    res["ceiling"] = chosen["ceiling"]
    maj = sum(
        majority_at(alpha, args.train_seq, args.test_seq, args.seq, sd, dev)
        for sd in args.seed_list
    ) / len(args.seed_list)
    res["majority"] = maj
    print(f"\n选中 alpha={alpha}（天花板 {chosen['ceiling']:.3f}，底板 {maj:.3f}）\n", flush=True)

    # ---- 2. 阶梯 ----
    for name, sw in RUNGS:
        r = run_arm(name, sw, alpha, args, dev)
        if r is None:
            continue
        accs, lins, fr, sc = r
        res["cells"][name] = {
            "switches": sw,
            "acc": sum(accs) / len(accs),
            "runs": accs,
            "lin_acc": sum(lins) / len(lins),
            "firing": fr,
            "scale": sc,
        }
        print(
            f"  {name:6s} {'+'.join(sw) or '(无)':32s} scale {sc:5.2f} 发放 {fr:.4f}  "
            f"acc {sum(accs) / len(accs):.3f} ({', '.join(f'{a:.3f}' for a in accs)})",
            flush=True,
        )

    # ---- 3. 逐个拆卸（Q-3） ----
    full = CHAIN
    print("", flush=True)
    for drop in full:
        keep = [s for s in full if s != drop]
        r = run_arm(f"-{drop}", keep, alpha, args, dev)
        if r is None:
            continue
        accs, lins, fr, sc = r
        res["cells"][f"-{drop}"] = {
            "switches": keep,
            "acc": sum(accs) / len(accs),
            "runs": accs,
            "lin_acc": sum(lins) / len(lins),
            "firing": fr,
            "scale": sc,
        }
        print(
            f"  -{drop:5s} {'+'.join(keep):32s} scale {sc:5.2f} 发放 {fr:.4f}  "
            f"acc {sum(accs) / len(accs):.3f}",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
    )

    # ---- 4. 判 ----
    print("\n判：", flush=True)
    have = [n for n, _ in RUNGS if n in res["cells"]]
    L0, L5 = "L0", "L5"
    ok0 = L0 in res["cells"] and L5 in res["cells"] and res["cells"][L5]["acc"] >= maj + 0.15
    print(f"  L-0 工作点 + 真解码（L5 ≥ 底板 {maj:.3f}+0.15）：{'过' if ok0 else '不过 ⇒ 作废'}")
    if not ok0 or len(have) < 3:
        return 0
    steps = []
    for a, b in itertools.pairwise(have):
        s = res["cells"][b]["acc"] - res["cells"][a]["acc"]
        steps.append(s)
        print(f"     {a}→{b}: {s * 100:+.1f}pp")
    knee = None
    for i in range(len(steps) - 1):
        if steps[i] < 0.02 and steps[i + 1] < 0.02:
            knee = have[i]
            break
    print(f"  L-1 膝点：{'在 ' + knee if knee else '**没有**（连续两档 <2pp 未出现）'}")

    # L-2：**两个都不同的指标**上膝点是否落在同一处（防 Mirage）
    lsteps = []
    for a, b in itertools.pairwise(have):
        lsteps.append(res["cells"][b]["lin_acc"] - res["cells"][a]["lin_acc"])
        print(f"     [线性读出] {a}→{b}: {lsteps[-1] * 100:+.1f}pp")
    lknee = None
    for i in range(len(lsteps) - 1):
        if lsteps[i] < 0.02 and lsteps[i + 1] < 0.02:
            lknee = have[i]
            break
    same_knee = (knee is None and lknee is None) or (knee == lknee)
    print(
        f"  L-2 膝点一致性：mlp {knee or '无'} / 线性 {lknee or '无'} ⇒ "
        f"{'一致' if same_knee else '**不一致（Mirage 风险）**'}"
    )
    if L5 in res["cells"]:
        base = res["cells"][L5]["acc"]
        drops = {d: base - res["cells"][f"-{d}"]["acc"] for d in CHAIN if f"-{d}" in res["cells"]}
        for d, v in sorted(drops.items(), key=lambda x: -x[1]):
            print(f"     -{d:6s} 掉 {v * 100:+.1f}pp")
        big = [d for d, v in drops.items() if v >= 0.05]
        print(
            f"  L-3 拆卸：掉幅 ≥5pp 的机制 {len(big)} 个 ({', '.join(big) or '无'}) ⇒ "
            f"{'过' if len(big) >= 2 else '不过'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
